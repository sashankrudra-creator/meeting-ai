"""
Repository layer: the only module that speaks SQL / touches result files
directly. Every other layer (pipeline, API) goes through
`MeetingRepository` so persistence details stay swappable and testable.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings
from app.models.schemas import ErrorDetail, MeetingResult, MeetingSummary, ProcessingStatus, TranscriptSegment
from app.storage.database import Database
from app.utils.errors import MeetingNotFoundError, StorageError
from app.utils.logging import get_logger

logger = get_logger("storage.repository")


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class MeetingRecord:
    meeting_id: str
    filename: str
    stored_upload_path: str
    status: ProcessingStatus
    created_at: str
    updated_at: str
    processed_audio_path: str | None = None
    transcript_path: str | None = None
    result_path: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    duration_seconds: float | None = None
    languages: list[str] | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> MeetingRecord:
        return cls(
            meeting_id=row["meeting_id"],
            filename=row["filename"],
            stored_upload_path=row["stored_upload_path"],
            processed_audio_path=row["processed_audio_path"],
            transcript_path=row["transcript_path"],
            result_path=row["result_path"],
            status=ProcessingStatus(row["status"]),
            error_code=row["error_code"],
            error_message=row["error_message"],
            duration_seconds=row["duration_seconds"],
            languages=json.loads(row["languages"]) if row["languages"] else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @property
    def error(self) -> ErrorDetail | None:
        if self.error_code:
            return ErrorDetail(code=self.error_code, message=self.error_message or "")
        return None


class MeetingRepository:
    def __init__(self, database: Database, settings: Settings):
        self._db = database
        self._settings = settings

    # ------------------------------------------------------------------
    # Create / update lifecycle
    # ------------------------------------------------------------------
    def create_meeting(self, meeting_id: str, filename: str, stored_upload_path: Path) -> MeetingRecord:
        now = _now()
        try:
            with self._db.write_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO meetings
                        (meeting_id, filename, stored_upload_path, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (meeting_id, filename, str(stored_upload_path), ProcessingStatus.QUEUED.value, now, now),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to create meeting record: {exc}") from exc
        return self.get_meeting(meeting_id)

    def update_status(
        self,
        meeting_id: str,
        status: ProcessingStatus,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        try:
            with self._db.write_connection() as conn:
                conn.execute(
                    """
                    UPDATE meetings
                    SET status = ?, error_code = ?, error_message = ?, updated_at = ?
                    WHERE meeting_id = ?
                    """,
                    (status.value, error_code, error_message, _now(), meeting_id),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to update status for meeting {meeting_id}: {exc}") from exc

    def set_processed_audio_path(self, meeting_id: str, path: Path) -> None:
        self._update_field(meeting_id, "processed_audio_path", str(path))

    def _update_field(self, meeting_id: str, field: str, value: str) -> None:
        try:
            with self._db.write_connection() as conn:
                conn.execute(
                    f"UPDATE meetings SET {field} = ?, updated_at = ? WHERE meeting_id = ?",  # noqa: S608
                    (value, _now(), meeting_id),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to update {field} for meeting {meeting_id}: {exc}") from exc

    def save_transcript_and_statistics(
        self,
        meeting_id: str,
        transcript: list[TranscriptSegment],
        duration_seconds: float,
        languages: list[str],
    ) -> None:
        """Persist the transcript to the filesystem and record duration/
        languages in SQLite, INDEPENDENTLY of whether summarization
        succeeds afterwards.

        Called by the orchestrator right after alignment/statistics
        complete, before summarization is even attempted, so that a
        summarization failure can never destroy an otherwise-valid
        transcript: `get_transcript()` works from this point on regardless
        of what happens next. Does not touch `status` or `result_path` —
        those are set by whichever of `save_result` /
        `save_result_without_summary` runs afterwards.
        """
        settings = self._settings
        transcript_path = settings.transcripts_dir / f"{meeting_id}.json"
        try:
            transcript_path.write_text(
                json.dumps([seg.model_dump(mode="json") for seg in transcript], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise StorageError(f"Failed to write transcript file for meeting {meeting_id}: {exc}") from exc

        try:
            with self._db.write_connection() as conn:
                conn.execute(
                    """
                    UPDATE meetings
                    SET transcript_path = ?, duration_seconds = ?, languages = ?, updated_at = ?
                    WHERE meeting_id = ?
                    """,
                    (str(transcript_path), duration_seconds, json.dumps(languages), _now(), meeting_id),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to record transcript metadata for meeting {meeting_id}: {exc}") from exc

    def save_result_without_summary(
        self,
        meeting_id: str,
        result: MeetingResult,
        *,
        error_code: str,
        error_message: str,
    ) -> None:
        """Persist a result whose summarization step failed: the transcript,
        speaker statistics and an (empty, warning-carrying) summary are all
        written to disk, and the meeting is marked
        `COMPLETED_WITHOUT_SUMMARY` — a terminal state distinct from FAILED,
        since everything up to and including statistics genuinely
        succeeded. The failure that prevented summarization is still
        recorded (`error_code`/`error_message`) for transparency, exactly
        as a FAILED meeting would record its failure."""
        settings = self._settings
        transcript_path = settings.transcripts_dir / f"{meeting_id}.json"
        result_path = settings.results_dir / f"{meeting_id}.json"

        try:
            transcript_path.write_text(
                json.dumps([seg.model_dump(mode="json") for seg in result.transcript], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            result_path.write_text(
                result.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            raise StorageError(f"Failed to write result files for meeting {meeting_id}: {exc}") from exc

        try:
            with self._db.write_connection() as conn:
                conn.execute(
                    """
                    UPDATE meetings
                    SET status = ?, transcript_path = ?, result_path = ?,
                        duration_seconds = ?, languages = ?, error_code = ?,
                        error_message = ?, updated_at = ?
                    WHERE meeting_id = ?
                    """,
                    (
                        ProcessingStatus.COMPLETED_WITHOUT_SUMMARY.value,
                        str(transcript_path),
                        str(result_path),
                        result.duration_seconds,
                        json.dumps(result.languages),
                        error_code,
                        error_message,
                        _now(),
                        meeting_id,
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to record result metadata for meeting {meeting_id}: {exc}") from exc

    def save_result(self, meeting_id: str, result: MeetingResult) -> None:
        """Persist the transcript and the full result to the filesystem
        (Section 17), then record their paths + summary metadata in SQLite,
        and mark the meeting COMPLETED — all as one logical step."""
        settings = self._settings
        transcript_path = settings.transcripts_dir / f"{meeting_id}.json"
        result_path = settings.results_dir / f"{meeting_id}.json"

        try:
            transcript_path.write_text(
                json.dumps([seg.model_dump(mode="json") for seg in result.transcript], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            result_path.write_text(
                result.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            raise StorageError(f"Failed to write result files for meeting {meeting_id}: {exc}") from exc

        try:
            with self._db.write_connection() as conn:
                conn.execute(
                    """
                    UPDATE meetings
                    SET status = ?, transcript_path = ?, result_path = ?,
                        duration_seconds = ?, languages = ?, error_code = NULL,
                        error_message = NULL, updated_at = ?
                    WHERE meeting_id = ?
                    """,
                    (
                        ProcessingStatus.COMPLETED.value,
                        str(transcript_path),
                        str(result_path),
                        result.duration_seconds,
                        json.dumps(result.languages),
                        _now(),
                        meeting_id,
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to record result metadata for meeting {meeting_id}: {exc}") from exc

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_meeting(self, meeting_id: str) -> MeetingRecord:
        try:
            with self._db.connect() as conn:
                row = conn.execute(
                    "SELECT * FROM meetings WHERE meeting_id = ?", (meeting_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to fetch meeting {meeting_id}: {exc}") from exc

        if row is None:
            raise MeetingNotFoundError(f"No meeting found with id '{meeting_id}'.")
        return MeetingRecord.from_row(row)

    def get_meeting_or_none(self, meeting_id: str) -> MeetingRecord | None:
        try:
            return self.get_meeting(meeting_id)
        except MeetingNotFoundError:
            return None

    def get_transcript(self, meeting_id: str) -> list[TranscriptSegment]:
        record = self.get_meeting(meeting_id)
        if not record.transcript_path or not Path(record.transcript_path).exists():
            return []
        raw = json.loads(Path(record.transcript_path).read_text(encoding="utf-8"))
        return [TranscriptSegment.model_validate(item) for item in raw]

    def get_result(self, meeting_id: str) -> MeetingResult | None:
        record = self.get_meeting(meeting_id)
        if not record.result_path or not Path(record.result_path).exists():
            return None
        raw = json.loads(Path(record.result_path).read_text(encoding="utf-8"))
        return MeetingResult.model_validate(raw)

    def get_summary(self, meeting_id: str) -> MeetingSummary | None:
        result = self.get_result(meeting_id)
        return result.summary if result else None

    def list_meetings(self, limit: int = 50) -> list[MeetingRecord]:
        try:
            with self._db.connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM meetings ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to list meetings: {exc}") from exc
        return [MeetingRecord.from_row(row) for row in rows]
