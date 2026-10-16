from __future__ import annotations

from pathlib import Path

import pytest

from app.models.schemas import (
    MeetingResult,
    MeetingSummary,
    ProcessingStatus,
    SpeakerStats,
    TranscriptSegment,
)
from app.storage.database import Database
from app.storage.repository import MeetingRepository
from app.utils.errors import ErrorCode, MeetingNotFoundError


@pytest.fixture()
def repo(settings) -> MeetingRepository:
    db = Database(settings.database_path)
    return MeetingRepository(db, settings)


def test_create_and_get_meeting(repo, tmp_path):
    upload_path = tmp_path / "meeting.wav"
    upload_path.write_bytes(b"fake")

    record = repo.create_meeting("m1", "meeting.wav", upload_path)

    assert record.meeting_id == "m1"
    assert record.status == ProcessingStatus.QUEUED
    fetched = repo.get_meeting("m1")
    assert fetched.filename == "meeting.wav"


def test_get_nonexistent_meeting_raises(repo):
    with pytest.raises(MeetingNotFoundError):
        repo.get_meeting("does-not-exist")


def test_get_meeting_or_none(repo):
    assert repo.get_meeting_or_none("nope") is None


def test_update_status_transitions_and_records_error(repo, tmp_path):
    upload_path = tmp_path / "m.wav"
    upload_path.write_bytes(b"fake")
    repo.create_meeting("m2", "m.wav", upload_path)

    repo.update_status("m2", ProcessingStatus.TRANSCRIBING)
    assert repo.get_meeting("m2").status == ProcessingStatus.TRANSCRIBING

    repo.update_status(
        "m2", ProcessingStatus.FAILED,
        error_code=ErrorCode.TRANSCRIPTION_FAILED, error_message="model exploded",
    )
    record = repo.get_meeting("m2")
    assert record.status == ProcessingStatus.FAILED
    assert record.error.code == ErrorCode.TRANSCRIPTION_FAILED
    assert record.error.message == "model exploded"


def test_save_result_writes_files_and_updates_row(repo, tmp_path):
    upload_path = tmp_path / "m3.wav"
    upload_path.write_bytes(b"fake")
    repo.create_meeting("m3", "m3.wav", upload_path)

    result = MeetingResult(
        meeting_id="m3",
        duration_seconds=42.0,
        languages=["en", "hi"],
        speakers=[
            SpeakerStats(
                speaker_id="SPEAKER_00", name="Person 1", speaking_time_seconds=42.0,
                speaking_percentage=100.0, segment_count=1,
                average_segment_duration_seconds=42.0, longest_segment_duration_seconds=42.0,
            )
        ],
        transcript=[
            TranscriptSegment(
                segment_id=1, speaker_id="SPEAKER_00", speaker_name="Person 1",
                start=0.0, end=42.0, text="hello", language="en",
            )
        ],
        summary=MeetingSummary(overview="A short meeting.", key_points=["hello"]),
    )

    repo.save_result("m3", result)

    record = repo.get_meeting("m3")
    assert record.status == ProcessingStatus.COMPLETED
    assert record.duration_seconds == 42.0
    assert record.languages == ["en", "hi"]
    assert Path(record.transcript_path).exists()
    assert Path(record.result_path).exists()

    transcript = repo.get_transcript("m3")
    assert len(transcript) == 1
    assert transcript[0].text == "hello"

    fetched_result = repo.get_result("m3")
    assert fetched_result.duration_seconds == 42.0

    summary = repo.get_summary("m3")
    assert summary.overview == "A short meeting."


def test_save_transcript_and_statistics_is_independent_of_result(repo, tmp_path):
    """Regression for C1: `save_transcript_and_statistics` must make the
    transcript retrievable on its own, without ever touching `status` or
    `result_path` — those belong to whichever of `save_result` /
    `save_result_without_summary` runs afterwards."""
    upload_path = tmp_path / "m5.wav"
    upload_path.write_bytes(b"fake")
    repo.create_meeting("m5", "m5.wav", upload_path)

    transcript = [
        TranscriptSegment(
            segment_id=1, speaker_id="SPEAKER_00", speaker_name="Person 1",
            start=0.0, end=10.0, text="hello", language="en",
        )
    ]
    repo.save_transcript_and_statistics("m5", transcript, duration_seconds=10.0, languages=["en"])

    record = repo.get_meeting("m5")
    assert record.status == ProcessingStatus.QUEUED  # untouched
    assert record.duration_seconds == 10.0
    assert record.languages == ["en"]

    fetched_transcript = repo.get_transcript("m5")
    assert len(fetched_transcript) == 1
    assert fetched_transcript[0].text == "hello"

    # No result yet — only the transcript/statistics were persisted.
    assert repo.get_result("m5") is None


def test_save_result_without_summary_marks_completed_without_summary(repo, tmp_path):
    upload_path = tmp_path / "m6.wav"
    upload_path.write_bytes(b"fake")
    repo.create_meeting("m6", "m6.wav", upload_path)

    transcript = [
        TranscriptSegment(
            segment_id=1, speaker_id="SPEAKER_00", speaker_name="Person 1",
            start=0.0, end=5.0, text="hi", language="en",
        )
    ]
    result = MeetingResult(
        meeting_id="m6",
        duration_seconds=5.0,
        languages=["en"],
        speakers=[
            SpeakerStats(
                speaker_id="SPEAKER_00", name="Person 1", speaking_time_seconds=5.0,
                speaking_percentage=100.0, segment_count=1,
                average_segment_duration_seconds=5.0, longest_segment_duration_seconds=5.0,
            )
        ],
        transcript=transcript,
        summary=MeetingSummary(warnings=["Summarization failed (SUMMARIZATION_FAILED): boom"]),
    )

    repo.save_result_without_summary(
        "m6", result, error_code=ErrorCode.SUMMARIZATION_FAILED, error_message="boom"
    )

    record = repo.get_meeting("m6")
    assert record.status == ProcessingStatus.COMPLETED_WITHOUT_SUMMARY
    assert record.error.code == ErrorCode.SUMMARIZATION_FAILED
    assert record.error.message == "boom"

    fetched_transcript = repo.get_transcript("m6")
    assert len(fetched_transcript) == 1

    fetched_result = repo.get_result("m6")
    assert fetched_result is not None
    assert fetched_result.summary.warnings


def test_get_transcript_before_completion_returns_empty(repo, tmp_path):
    upload_path = tmp_path / "m4.wav"
    upload_path.write_bytes(b"fake")
    repo.create_meeting("m4", "m4.wav", upload_path)
    assert repo.get_transcript("m4") == []
    assert repo.get_result("m4") is None
    assert repo.get_summary("m4") is None


def test_list_meetings_orders_by_created_at_desc(repo, tmp_path):
    for i in range(3):
        p = tmp_path / f"m{i}.wav"
        p.write_bytes(b"fake")
        repo.create_meeting(f"list-{i}", f"m{i}.wav", p)

    records = repo.list_meetings(limit=10)
    ids = [r.meeting_id for r in records]
    assert "list-0" in ids and "list-1" in ids and "list-2" in ids
