"""
Central pipeline orchestrator (Section 13).

    validate_file()
            |
    extract_audio() / preprocess_audio()
            |
    transcribe()
            |
    diarize()
            |
    align_transcript_and_speakers()
            |
    calculate_statistics()
            |
    generate_mom()
            |
    persist_result()

Each stage is a separate, independently-unit-tested module
(app.pipeline.preprocessing/transcription/diarization/alignment/statistics/
summarization); this file only sequences them and translates any
`PipelineError` raised by a stage into a FAILED status recorded in
storage, with the processing-status column updated (Section 14) before
each stage begins so a client polling GET /meetings/{id} can see exactly
where a long-running job currently is.

Implementation note on Section 13's "extract_audio()" + "preprocess_audio()":
both are implemented as a single ffmpeg invocation
(`app.pipeline.preprocessing.preprocess_audio`) that extracts the audio
stream and normalizes it to mono/16kHz/PCM in one pass, rather than two
separate ffmpeg processes — running ffmpeg twice over the same input would
only add I/O with no benefit. The `EXTRACTING_AUDIO` processing status
covers this combined step.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.models.schemas import (
    MeetingResult,
    MeetingSummary,
    ProcessingStatus,
    SpeakerStats,
    TranscriptionResult,
    TranscriptSegment,
)
from app.pipeline.alignment import align_transcript_and_speakers
from app.pipeline.diarization import Diarizer, PyannoteDiarizer, build_speaker_name_mapping
from app.pipeline.preprocessing import preprocess_audio, validate_media_file
from app.pipeline.statistics import compute_speaker_statistics
from app.pipeline.summarization import Summarizer, build_summarizer
from app.pipeline.transcription import FasterWhisperTranscriber, Transcriber
from app.storage.repository import MeetingRepository
from app.utils.audio import get_audio_duration_seconds
from app.utils.errors import ErrorCode, PipelineError
from app.utils.logging import get_logger

logger = get_logger("pipeline.orchestrator")


class MeetingOrchestrator:
    """Runs the full pipeline for one meeting at a time.

    Transcriber/Diarizer/Summarizer are injected (defaulting to the real
    production implementations) so tests — and the orchestrator's own
    integration test — can substitute lightweight fakes without touching
    this class's control flow.
    """

    def __init__(
        self,
        settings: Settings,
        repository: MeetingRepository,
        transcriber: Transcriber | None = None,
        diarizer: Diarizer | None = None,
        summarizer: Summarizer | None = None,
    ):
        self._settings = settings
        self._repo = repository
        self._transcriber = transcriber or FasterWhisperTranscriber(settings)
        self._diarizer = diarizer or PyannoteDiarizer(settings)
        self._summarizer = summarizer or build_summarizer(settings)

    def process_meeting(self, meeting_id: str) -> None:
        logger.info("Starting pipeline for meeting %s", meeting_id)
        try:
            record = self._repo.get_meeting(meeting_id)
            upload_path = Path(record.stored_upload_path)

            self._set_status(meeting_id, ProcessingStatus.VALIDATING)
            validate_media_file(upload_path, self._settings)

            self._set_status(meeting_id, ProcessingStatus.EXTRACTING_AUDIO)
            audio_path = preprocess_audio(upload_path, meeting_id, self._settings)
            self._repo.set_processed_audio_path(meeting_id, audio_path)
            duration_seconds = get_audio_duration_seconds(audio_path, self._settings)

            self._set_status(meeting_id, ProcessingStatus.TRANSCRIBING)
            transcription = self._transcriber.transcribe(audio_path)

            self._set_status(meeting_id, ProcessingStatus.DIARIZING)
            diarization = self._diarizer.diarize(audio_path)
            speaker_mapping = build_speaker_name_mapping(diarization)

            self._set_status(meeting_id, ProcessingStatus.ALIGNING)
            transcript = align_transcript_and_speakers(
                transcription, diarization, speaker_mapping, self._settings
            )

            self._set_status(meeting_id, ProcessingStatus.CALCULATING_STATISTICS)
            speaker_stats = compute_speaker_statistics(transcript, duration_seconds)
            languages = sorted({seg.language for seg in transcript if seg.language})

            # Persist the transcript and statistics now, independently of
            # whatever happens during summarization below. A summarization
            # failure must not destroy an otherwise-valid transcript/
            # statistics result (Section 13/18) — from this point on,
            # `GET /meetings/{id}/transcript` works even if MoM generation
            # fails next.
            self._repo.save_transcript_and_statistics(meeting_id, transcript, duration_seconds, languages)

            self._set_status(meeting_id, ProcessingStatus.GENERATING_SUMMARY)
            try:
                summary = self._summarizer.summarize(transcript)
            except PipelineError as exc:
                logger.error(
                    "Summarization failed for meeting %s (%s): %s — completing without a "
                    "summary instead of failing the whole meeting.",
                    meeting_id, exc.code, exc.message,
                )
                self._finish_without_summary(
                    meeting_id, transcript, duration_seconds, languages, speaker_stats,
                    exc.code, exc.message, transcription,
                )
                return
            except Exception:  # noqa: BLE001 - summarization must never take down an otherwise-valid result
                logger.exception("Unexpected error during summarization for meeting %s", meeting_id)
                self._finish_without_summary(
                    meeting_id, transcript, duration_seconds, languages, speaker_stats,
                    ErrorCode.INTERNAL_ERROR, "An unexpected internal error occurred during summarization.",
                    transcription,
                )
                return

            self._set_status(meeting_id, ProcessingStatus.SAVING_RESULT)
            result = MeetingResult(
                meeting_id=meeting_id,
                duration_seconds=duration_seconds,
                languages=languages,
                speakers=speaker_stats,
                transcript=transcript,
                summary=summary,
                filtered_segment_count=transcription.filtered_segment_count,
                flagged_segment_count=transcription.flagged_segment_count,
            )
            self._repo.save_result(meeting_id, result)
            logger.info("Meeting %s processed successfully (%d segments)", meeting_id, len(transcript))

        except PipelineError as exc:
            logger.error("Meeting %s failed (%s): %s", meeting_id, exc.code, exc.message)
            self._safe_mark_failed(meeting_id, exc.code, exc.message)
        except Exception:  # noqa: BLE001 - last line of defense, never leak raw tracebacks
            logger.exception("Unexpected error processing meeting %s", meeting_id)
            self._safe_mark_failed(
                meeting_id, ErrorCode.INTERNAL_ERROR, "An unexpected internal error occurred."
            )

    def _finish_without_summary(
        self,
        meeting_id: str,
        transcript: list[TranscriptSegment],
        duration_seconds: float,
        languages: list[str],
        speaker_stats: list[SpeakerStats],
        error_code: str,
        error_message: str,
        transcription: TranscriptionResult,
    ) -> None:
        """Record a meeting as COMPLETED_WITHOUT_SUMMARY (Section 13/18):
        transcript, statistics and a warning-carrying (empty) summary are
        all persisted and retrievable, and the summarization failure is
        recorded for transparency — but the meeting is NOT marked FAILED,
        since transcription/diarization/alignment/statistics all genuinely
        succeeded."""
        result = MeetingResult(
            meeting_id=meeting_id,
            duration_seconds=duration_seconds,
            languages=languages,
            speakers=speaker_stats,
            transcript=transcript,
            summary=MeetingSummary(warnings=[f"Summarization failed ({error_code}): {error_message}"]),
            filtered_segment_count=transcription.filtered_segment_count,
            flagged_segment_count=transcription.flagged_segment_count,
        )
        try:
            self._repo.save_result_without_summary(
                meeting_id, result, error_code=error_code, error_message=error_message
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to persist completed-without-summary result for meeting %s; "
                "marking FAILED instead so the meeting doesn't get stuck mid-pipeline.",
                meeting_id,
            )
            self._safe_mark_failed(meeting_id, error_code, error_message)

    def _set_status(self, meeting_id: str, status: ProcessingStatus) -> None:
        self._repo.update_status(meeting_id, status)

    def _safe_mark_failed(self, meeting_id: str, code: str, message: str) -> None:
        try:
            self._repo.update_status(meeting_id, ProcessingStatus.FAILED, error_code=code, error_message=message)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to even record failure status for meeting %s", meeting_id)
