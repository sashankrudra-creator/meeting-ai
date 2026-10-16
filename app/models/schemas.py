"""
Pydantic schemas for the Voice-Based MoM pipeline (Section 8).

These are the single source of truth for the shape of data flowing between
pipeline stages, storage and the API. Keeping them in one module (rather
than scattered dataclasses per stage) is what lets `orchestrator.py` pass
a fully-typed object from one stage to the next.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# Processing lifecycle (Section 14)
# ---------------------------------------------------------------------------
class ProcessingStatus(str, Enum):
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    EXTRACTING_AUDIO = "EXTRACTING_AUDIO"
    TRANSCRIBING = "TRANSCRIBING"
    DIARIZING = "DIARIZING"
    ALIGNING = "ALIGNING"
    CALCULATING_STATISTICS = "CALCULATING_STATISTICS"
    GENERATING_SUMMARY = "GENERATING_SUMMARY"
    SAVING_RESULT = "SAVING_RESULT"
    COMPLETED = "COMPLETED"
    # Terminal, non-FAILED state: the transcript, diarization, alignment and
    # speaker statistics all completed successfully, but MoM/summary
    # generation itself failed. A summarization failure must not destroy an
    # otherwise-valid transcript/statistics result (see MeetingOrchestrator
    # and MeetingRepository.save_result_without_summary), so this is
    # recorded as its own outcome rather than as FAILED.
    COMPLETED_WITHOUT_SUMMARY = "COMPLETED_WITHOUT_SUMMARY"
    FAILED = "FAILED"

    @property
    def is_terminal(self) -> bool:
        return self in (
            ProcessingStatus.COMPLETED,
            ProcessingStatus.COMPLETED_WITHOUT_SUMMARY,
            ProcessingStatus.FAILED,
        )


# ---------------------------------------------------------------------------
# Raw stage outputs (internal, pre-alignment)
# ---------------------------------------------------------------------------
class RawTranscriptSegment(BaseModel):
    """One segment as emitted directly by the ASR model, before speaker
    assignment. `start`/`end` are seconds from the start of the audio."""

    start: float
    end: float
    text: str
    language: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    low_confidence: bool = Field(
        default=False,
        description="True when this segment's avg_logprob/no_speech_prob crossed the "
                     "configured ASR quality thresholds (see Settings.asr_quality_filter_mode). "
                     "Only ever set when filtering/flagging is enabled; never removes the "
                     "segment by itself (see FasterWhisperTranscriber.transcribe).",
    )


class DiarizationTurn(BaseModel):
    """One speaker turn as emitted directly by the diarization model,
    before being mapped to a display name."""

    speaker: str  # e.g. "SPEAKER_00"
    start: float
    end: float


class TranscriptionResult(BaseModel):
    segments: list[RawTranscriptSegment] = Field(default_factory=list)
    detected_languages: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    filtered_segment_count: int = Field(
        default=0, description="Number of segments dropped by ASR quality filtering "
                                "(asr_quality_filter_mode='filter'). 0 unless filtering is enabled."
    )
    flagged_segment_count: int = Field(
        default=0, description="Number of segments kept but marked low_confidence by ASR "
                                "quality filtering (asr_quality_filter_mode='flag' or 'filter')."
    )


class DiarizationResult(BaseModel):
    turns: list[DiarizationTurn] = Field(default_factory=list)
    speaker_labels: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Aligned / final transcript (Section 7)
# ---------------------------------------------------------------------------
class TranscriptSegment(BaseModel):
    segment_id: int
    speaker_id: str
    speaker_name: str
    start: float
    end: float
    text: str
    language: str | None = None
    overlapping_speakers: list[str] = Field(
        default_factory=list,
        description="Other speaker_ids whose diarization turns overlapped this ASR "
                     "segment, when the segment was ambiguous. Empty when the "
                     "assignment was unambiguous. See alignment.py for policy.",
    )
    low_confidence: bool = Field(
        default=False,
        description="Carried through from the raw ASR segment (see RawTranscriptSegment); "
                     "True when ASR quality filtering flagged this segment as a likely "
                     "hallucination/low-confidence transcription.",
    )


# ---------------------------------------------------------------------------
# Speaker statistics (Section 9)
# ---------------------------------------------------------------------------
class SpeakerStats(BaseModel):
    speaker_id: str
    name: str
    speaking_time_seconds: float
    speaking_percentage: float
    segment_count: int
    average_segment_duration_seconds: float
    longest_segment_duration_seconds: float


# ---------------------------------------------------------------------------
# Local MoM / summary (Section 11)
# ---------------------------------------------------------------------------
class ActionItem(BaseModel):
    task: str
    responsible: str | None = None
    deadline: str | None = None


class MeetingSummary(BaseModel):
    overview: str = ""
    key_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal problems encountered while generating this summary (e.g. an "
                     "LLM chunk or reduce step returned output that wasn't valid JSON, or "
                     "summarization failed outright for the whole meeting). Empty when nothing "
                     "went wrong. Surfacing this is deliberate: a parse failure silently "
                     "producing an empty-looking-but-'valid' summary would be indistinguishable "
                     "from 'the meeting genuinely had no key points' otherwise.",
    )


# ---------------------------------------------------------------------------
# Final structured result (Section 8)
# ---------------------------------------------------------------------------
class MeetingResult(BaseModel):
    meeting_id: str
    duration_seconds: float
    languages: list[str] = Field(default_factory=list)
    speakers: list[SpeakerStats] = Field(default_factory=list)
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    summary: MeetingSummary = Field(default_factory=MeetingSummary)
    filtered_segment_count: int = Field(
        default=0, description="See TranscriptionResult.filtered_segment_count."
    )
    flagged_segment_count: int = Field(
        default=0, description="See TranscriptionResult.flagged_segment_count."
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# API request/response models
# ---------------------------------------------------------------------------
class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    status: str = "failed"
    error: ErrorDetail


class MeetingCreateResponse(BaseModel):
    meeting_id: str
    status: ProcessingStatus
    filename: str
    message: str = "Meeting accepted for processing."


class MeetingStatusResponse(BaseModel):
    meeting_id: str
    status: ProcessingStatus
    filename: str
    created_at: datetime
    updated_at: datetime
    error: ErrorDetail | None = None
    duration_seconds: float | None = None
    languages: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[misc]
    @property
    def is_complete(self) -> bool:
        return self.status in (ProcessingStatus.COMPLETED, ProcessingStatus.COMPLETED_WITHOUT_SUMMARY)


class TranscriptResponse(BaseModel):
    meeting_id: str
    status: ProcessingStatus
    transcript: list[TranscriptSegment] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    meeting_id: str
    status: ProcessingStatus
    summary: MeetingSummary | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    version: str
    details: dict[str, Any] = Field(default_factory=dict)
