"""
API integration tests (Section 20).

Honesty note: `FakeTranscriber`/`FakeDiarizer` below stand in for
`FasterWhisperTranscriber`/`PyannoteDiarizer` because those need model
weights this sandbox cannot download (see README "Known Limitations").
Everything else in this test is REAL and exercised end-to-end through the
actual HTTP API: file upload, real ffmpeg validation + extraction of a
synthetic-but-real WAV file, the real orchestrator sequencing every
pipeline stage, the real (non-LLM) `ExtractiveSummarizer`, real SQLite +
filesystem persistence, and the real FastAPI error-handling layer. This is
the "integration test for the complete pipeline where practical" called
for in Section 20.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_repository, get_settings_dep, get_worker
from app.main import register_routes_and_handlers
from app.models.schemas import DiarizationResult, DiarizationTurn, RawTranscriptSegment, TranscriptionResult
from app.pipeline.diarization import Diarizer
from app.pipeline.orchestrator import MeetingOrchestrator
from app.pipeline.summarization import ExtractiveSummarizer
from app.pipeline.transcription import Transcriber
from app.storage.database import Database
from app.storage.repository import MeetingRepository


class FakeTranscriber(Transcriber):
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        return TranscriptionResult(
            segments=[
                RawTranscriptSegment(start=0.0, end=1.0, text="We will deploy tomorrow morning", language="en"),
                RawTranscriptSegment(start=1.0, end=2.0, text="Person two will handle the testing", language="en"),
            ],
            detected_languages=["en"],
            duration_seconds=4.0,  # deliberately wrong, to prove the orchestrator uses real ffprobe duration instead
        )


class FakeDiarizer(Diarizer):
    def diarize(self, audio_path: Path) -> DiarizationResult:
        return DiarizationResult(
            turns=[
                DiarizationTurn(speaker="SPEAKER_00", start=0.0, end=1.0),
                DiarizationTurn(speaker="SPEAKER_01", start=1.0, end=2.0),
            ],
            speaker_labels=["SPEAKER_00", "SPEAKER_01"],
        )


class SynchronousWorker:
    """Test double for BackgroundWorker: runs the job immediately, in-thread,
    so tests don't need to poll/sleep for a real background thread pool."""

    def __init__(self, orchestrator: MeetingOrchestrator):
        self._orchestrator = orchestrator

    def submit(self, meeting_id: str) -> None:
        self._orchestrator.process_meeting(meeting_id)


@pytest.fixture()
def repo(settings) -> MeetingRepository:
    return MeetingRepository(Database(settings.database_path), settings)


@pytest.fixture()
def client(settings, repo):
    settings.summarizer_backend = "extractive"  # no model download needed
    orchestrator = MeetingOrchestrator(
        settings, repo,
        transcriber=FakeTranscriber(),
        diarizer=FakeDiarizer(),
        summarizer=ExtractiveSummarizer(settings),
    )
    worker = SynchronousWorker(orchestrator)

    app = FastAPI()
    register_routes_and_handlers(app)
    app.dependency_overrides[get_settings_dep] = lambda: settings
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_worker] = lambda: worker

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["details"]["ffmpeg_available"] is True


def test_upload_rejects_unsupported_extension(client):
    resp = client.post(
        "/api/v1/meetings",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "UNSUPPORTED_FILE_EXTENSION"


def test_upload_rejects_empty_file(client):
    resp = client.post(
        "/api/v1/meetings",
        files={"file": ("meeting.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "EMPTY_UPLOAD"


def test_get_nonexistent_meeting_returns_structured_404(client):
    resp = client.get("/api/v1/meetings/does-not-exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "MEETING_NOT_FOUND"


def test_result_not_ready_before_completion_returns_409(client, repo, tmp_path):
    upload_path = tmp_path / "pending.wav"
    upload_path.write_bytes(b"fake")
    repo.create_meeting("pending-1", "pending.wav", upload_path)

    resp = client.get("/api/v1/meetings/pending-1/result")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "RESULT_NOT_READY"


def test_full_pipeline_happy_path_end_to_end(client, valid_wav_file):
    with open(valid_wav_file, "rb") as f:
        resp = client.post(
            "/api/v1/meetings",
            files={"file": ("team_sync.wav", f, "audio/wav")},
        )
    assert resp.status_code == 201
    meeting_id = resp.json()["meeting_id"]
    assert resp.json()["filename"] == "team_sync.wav"

    status_resp = client.get(f"/api/v1/meetings/{meeting_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "COMPLETED"
    # Real ffprobe duration of the synthetic 2s fixture, NOT the fake transcriber's bogus 4.0
    assert status_resp.json()["duration_seconds"] == pytest.approx(2.0, abs=0.3)

    transcript_resp = client.get(f"/api/v1/meetings/{meeting_id}/transcript")
    transcript = transcript_resp.json()["transcript"]
    assert len(transcript) == 2
    assert transcript[0]["speaker_name"] == "Person 1"
    assert transcript[1]["speaker_name"] == "Person 2"
    assert [t["segment_id"] for t in transcript] == [1, 2]

    summary_resp = client.get(f"/api/v1/meetings/{meeting_id}/summary")
    summary = summary_resp.json()["summary"]
    assert any("deploy tomorrow" in d for d in summary["decisions"])
    assert any("handle the testing" in a["task"] for a in summary["action_items"])

    result_resp = client.get(f"/api/v1/meetings/{meeting_id}/result")
    assert result_resp.status_code == 200
    result = result_resp.json()
    assert result["languages"] == ["en"]
    assert len(result["speakers"]) == 2
    speaker_by_name = {s["name"]: s for s in result["speakers"]}
    assert speaker_by_name["Person 1"]["segment_count"] == 1


def test_missing_required_upload_field_returns_structured_422(client):
    """Regression for H7: a malformed request (here, missing the required
    `file` field entirely) must return the SAME structured
    {"status": "failed", "error": {...}} contract as every other error in
    this API, not FastAPI/Starlette's default {"detail": [...]} shape."""
    resp = client.post("/api/v1/meetings")
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "INVALID_REQUEST"
    assert "message" in body["error"] and body["error"]["message"]


def test_upload_aborts_streaming_when_size_exceeds_limit_without_leaving_partial_file(client, settings):
    """Regression for H2: the upload endpoint must enforce the max-size
    limit against a running byte count WHILE STREAMING to disk, not after
    buffering the whole file into memory first. Using a small chunk size
    and a small cap keeps this test cheap while still forcing several
    `await file.read(chunk_size)` calls before the limit is hit."""
    settings.upload_chunk_size_bytes = 1024  # 1 KB chunks -> many reads
    settings.max_upload_size_mb = 1  # 1 MB cap

    content = b"A" * (2 * 1024 * 1024)  # 2 MB, exceeds the 1 MB cap
    resp = client.post(
        "/api/v1/meetings",
        files={"file": ("big.wav", content, "audio/wav")},
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "FILE_TOO_LARGE"
    # the partial file written before the abort must be cleaned up
    assert list(settings.uploads_dir.glob("*.wav")) == []


def test_upload_streams_in_bounded_chunks_and_reassembles_the_file_exactly(client, settings, valid_wav_file):
    """A deliberately tiny, odd chunk size forces many `file.read()` calls;
    the resulting on-disk file must still be byte-identical to the
    original, proving the chunked write path doesn't corrupt or truncate
    data."""
    settings.upload_chunk_size_bytes = 37
    original_bytes = valid_wav_file.read_bytes()

    with open(valid_wav_file, "rb") as f:
        resp = client.post("/api/v1/meetings", files={"file": ("chunked.wav", f, "audio/wav")})
    assert resp.status_code == 201
    meeting_id = resp.json()["meeting_id"]

    stored = list(settings.uploads_dir.glob(f"{meeting_id}.*"))
    assert len(stored) == 1
    assert stored[0].read_bytes() == original_bytes


def test_corrupt_media_upload_ends_in_failed_status_with_structured_error(client, corrupt_media_file):
    with open(corrupt_media_file, "rb") as f:
        resp = client.post(
            "/api/v1/meetings",
            files={"file": ("bad.wav", f, "audio/wav")},
        )
    assert resp.status_code == 201
    meeting_id = resp.json()["meeting_id"]

    status_resp = client.get(f"/api/v1/meetings/{meeting_id}")
    body = status_resp.json()
    assert body["status"] == "FAILED"
    assert body["error"]["code"] == "CORRUPT_MEDIA"

    # /result must refuse cleanly rather than ever expose a raw traceback
    result_resp = client.get(f"/api/v1/meetings/{meeting_id}/result")
    assert result_resp.status_code == 409


def test_result_and_transcript_are_retrievable_end_to_end_when_summarization_fails(settings, repo, valid_wav_file):
    """Regression for C1, exercised through the real HTTP API end-to-end
    (not just the orchestrator unit test in test_orchestrator.py): when
    summarization fails, /transcript and /result must still return the
    otherwise-valid data with a 200, and /status must report
    COMPLETED_WITHOUT_SUMMARY rather than FAILED."""
    from app.pipeline.summarization import Summarizer
    from app.utils.errors import SummarizationError

    class AlwaysFailsSummarizer(Summarizer):
        def summarize(self, transcript):
            raise SummarizationError("simulated local-LLM crash")

    orchestrator = MeetingOrchestrator(
        settings, repo,
        transcriber=FakeTranscriber(), diarizer=FakeDiarizer(), summarizer=AlwaysFailsSummarizer(),
    )
    worker = SynchronousWorker(orchestrator)

    app = FastAPI()
    register_routes_and_handlers(app)
    app.dependency_overrides[get_settings_dep] = lambda: settings
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_worker] = lambda: worker

    with TestClient(app) as client:
        with open(valid_wav_file, "rb") as f:
            resp = client.post("/api/v1/meetings", files={"file": ("sync.wav", f, "audio/wav")})
        assert resp.status_code == 201
        meeting_id = resp.json()["meeting_id"]

        status_resp = client.get(f"/api/v1/meetings/{meeting_id}")
        assert status_resp.json()["status"] == "COMPLETED_WITHOUT_SUMMARY"
        assert status_resp.json()["is_complete"] is True

        transcript_resp = client.get(f"/api/v1/meetings/{meeting_id}/transcript")
        assert transcript_resp.status_code == 200
        assert len(transcript_resp.json()["transcript"]) == 2

        result_resp = client.get(f"/api/v1/meetings/{meeting_id}/result")
        assert result_resp.status_code == 200
        result_body = result_resp.json()
        assert len(result_body["speakers"]) == 2
        assert result_body["summary"]["key_points"] == []
        assert any("Summarization failed" in w for w in result_body["summary"]["warnings"])
