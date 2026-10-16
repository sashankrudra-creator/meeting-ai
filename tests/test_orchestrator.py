"""
Orchestrator-level tests (Sections 13/14/18): verifies stage sequencing,
processing-status transitions, and that a failure at ANY stage is caught
and recorded as a structured FAILED status rather than propagating as a
raw exception.

FakeTranscriber/FakeDiarizer avoid needing real model downloads (see
tests/test_api.py's docstring for the same caveat); ExtractiveSummarizer
runs for real.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.schemas import (
    DiarizationResult,
    DiarizationTurn,
    MeetingSummary,
    ProcessingStatus,
    RawTranscriptSegment,
    TranscriptionResult,
    TranscriptSegment,
)
from app.pipeline.diarization import Diarizer
from app.pipeline.orchestrator import MeetingOrchestrator
from app.pipeline.summarization import ExtractiveSummarizer, Summarizer
from app.pipeline.transcription import Transcriber
from app.storage.database import Database
from app.storage.repository import MeetingRepository
from app.utils.errors import DiarizationError, ErrorCode, SummarizationError, TranscriptionError


class OkTranscriber(Transcriber):
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        return TranscriptionResult(
            segments=[RawTranscriptSegment(start=0.0, end=1.0, text="hello team", language="en")],
            detected_languages=["en"],
            duration_seconds=1.0,
        )


class OkDiarizer(Diarizer):
    def diarize(self, audio_path: Path) -> DiarizationResult:
        return DiarizationResult(
            turns=[DiarizationTurn(speaker="SPEAKER_00", start=0.0, end=1.0)],
            speaker_labels=["SPEAKER_00"],
        )


class RaisingTranscriber(Transcriber):
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        raise TranscriptionError("simulated ASR crash")


class RaisingDiarizer(Diarizer):
    def diarize(self, audio_path: Path) -> DiarizationResult:
        raise DiarizationError("simulated diarization crash")


class RaisingGenericTranscriber(Transcriber):
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        raise ValueError("totally unexpected bug")


class RaisingSummarizer(Summarizer):
    def summarize(self, transcript: list[TranscriptSegment]) -> MeetingSummary:
        raise SummarizationError("simulated local-LLM crash")


class RaisingGenericSummarizer(Summarizer):
    def summarize(self, transcript: list[TranscriptSegment]) -> MeetingSummary:
        raise ValueError("totally unexpected summarizer bug")


@pytest.fixture()
def repo(settings) -> MeetingRepository:
    return MeetingRepository(Database(settings.database_path), settings)


def _create_meeting_with_real_audio(repo, settings, valid_wav_file, meeting_id="orc-1"):
    stored_path = settings.uploads_dir / f"{meeting_id}.wav"
    stored_path.write_bytes(valid_wav_file.read_bytes())
    return repo.create_meeting(meeting_id, "meeting.wav", stored_path)


def test_happy_path_transitions_through_all_statuses_and_completes(settings, repo, valid_wav_file):
    _create_meeting_with_real_audio(repo, settings, valid_wav_file)

    observed_statuses: list[ProcessingStatus] = []
    original_update_status = repo.update_status

    def spy_update_status(meeting_id, status, **kwargs):
        observed_statuses.append(status)
        return original_update_status(meeting_id, status, **kwargs)

    repo.update_status = spy_update_status  # type: ignore[method-assign]

    orchestrator = MeetingOrchestrator(
        settings, repo,
        transcriber=OkTranscriber(), diarizer=OkDiarizer(), summarizer=ExtractiveSummarizer(settings),
    )
    orchestrator.process_meeting("orc-1")

    assert observed_statuses == [
        ProcessingStatus.VALIDATING,
        ProcessingStatus.EXTRACTING_AUDIO,
        ProcessingStatus.TRANSCRIBING,
        ProcessingStatus.DIARIZING,
        ProcessingStatus.ALIGNING,
        ProcessingStatus.CALCULATING_STATISTICS,
        ProcessingStatus.GENERATING_SUMMARY,
        ProcessingStatus.SAVING_RESULT,
    ]

    record = repo.get_meeting("orc-1")
    assert record.status == ProcessingStatus.COMPLETED
    result = repo.get_result("orc-1")
    assert result.transcript[0].speaker_name == "Person 1"


def test_transcription_failure_marks_meeting_failed_with_correct_code(settings, repo, valid_wav_file):
    _create_meeting_with_real_audio(repo, settings, valid_wav_file, "orc-2")
    orchestrator = MeetingOrchestrator(
        settings, repo,
        transcriber=RaisingTranscriber(), diarizer=OkDiarizer(), summarizer=ExtractiveSummarizer(settings),
    )
    orchestrator.process_meeting("orc-2")

    record = repo.get_meeting("orc-2")
    assert record.status == ProcessingStatus.FAILED
    assert record.error.code == ErrorCode.TRANSCRIPTION_FAILED


def test_diarization_failure_marks_meeting_failed_with_correct_code(settings, repo, valid_wav_file):
    _create_meeting_with_real_audio(repo, settings, valid_wav_file, "orc-3")
    orchestrator = MeetingOrchestrator(
        settings, repo,
        transcriber=OkTranscriber(), diarizer=RaisingDiarizer(), summarizer=ExtractiveSummarizer(settings),
    )
    orchestrator.process_meeting("orc-3")

    record = repo.get_meeting("orc-3")
    assert record.status == ProcessingStatus.FAILED
    assert record.error.code == ErrorCode.DIARIZATION_FAILED


def test_unexpected_exception_is_caught_and_reported_generically(settings, repo, valid_wav_file):
    _create_meeting_with_real_audio(repo, settings, valid_wav_file, "orc-4")
    orchestrator = MeetingOrchestrator(
        settings, repo,
        transcriber=RaisingGenericTranscriber(), diarizer=OkDiarizer(), summarizer=ExtractiveSummarizer(settings),
    )
    # Must never raise out of process_meeting — the background worker thread
    # would otherwise silently die without recording a failure.
    orchestrator.process_meeting("orc-4")

    record = repo.get_meeting("orc-4")
    assert record.status == ProcessingStatus.FAILED
    assert record.error.code == ErrorCode.INTERNAL_ERROR
    assert "traceback" not in record.error.message.lower()
    assert "Traceback" not in record.error.message


def test_summarization_failure_completes_without_summary_and_keeps_transcript(settings, repo, valid_wav_file):
    """Regression for C1: a summarization failure must NOT destroy an
    otherwise-valid transcript/statistics result. Before this fix, the
    single `save_result()` call happened only after summarization
    succeeded, so `get_transcript()`/`get_result()` returned nothing at all
    when summarization raised — even though transcription, diarization,
    alignment and statistics had all genuinely succeeded."""
    _create_meeting_with_real_audio(repo, settings, valid_wav_file, "orc-6")
    orchestrator = MeetingOrchestrator(
        settings, repo,
        transcriber=OkTranscriber(), diarizer=OkDiarizer(), summarizer=RaisingSummarizer(),
    )
    orchestrator.process_meeting("orc-6")

    record = repo.get_meeting("orc-6")
    assert record.status == ProcessingStatus.COMPLETED_WITHOUT_SUMMARY
    assert record.status != ProcessingStatus.FAILED
    assert record.error.code == ErrorCode.SUMMARIZATION_FAILED

    # The transcript must be retrievable even though summarization failed.
    transcript = repo.get_transcript("orc-6")
    assert len(transcript) == 1
    assert transcript[0].text == "hello team"

    # The result (with an empty, warning-carrying summary) must also be
    # retrievable, carrying the speaker statistics that were computed.
    result = repo.get_result("orc-6")
    assert result is not None
    assert len(result.speakers) == 1
    assert result.summary.key_points == []
    assert any("Summarization failed" in w for w in result.summary.warnings)


def test_unexpected_summarizer_exception_also_completes_without_summary(settings, repo, valid_wav_file):
    """Same guarantee as above, but for a non-PipelineError bug inside the
    summarizer — it must not be treated any differently from a 'clean'
    SummarizationError as far as the transcript/statistics are concerned."""
    _create_meeting_with_real_audio(repo, settings, valid_wav_file, "orc-7")
    orchestrator = MeetingOrchestrator(
        settings, repo,
        transcriber=OkTranscriber(), diarizer=OkDiarizer(), summarizer=RaisingGenericSummarizer(),
    )
    orchestrator.process_meeting("orc-7")

    record = repo.get_meeting("orc-7")
    assert record.status == ProcessingStatus.COMPLETED_WITHOUT_SUMMARY
    assert record.error.code == ErrorCode.INTERNAL_ERROR

    transcript = repo.get_transcript("orc-7")
    assert len(transcript) == 1


def test_missing_upload_file_is_handled_gracefully(settings, repo):
    stored_path = settings.uploads_dir / "ghost.wav"  # never actually written to disk
    repo.create_meeting("orc-5", "ghost.wav", stored_path)

    orchestrator = MeetingOrchestrator(
        settings, repo,
        transcriber=OkTranscriber(), diarizer=OkDiarizer(), summarizer=ExtractiveSummarizer(settings),
    )
    orchestrator.process_meeting("orc-5")

    record = repo.get_meeting("orc-5")
    assert record.status == ProcessingStatus.FAILED
    assert record.error.code == ErrorCode.CORRUPT_MEDIA
