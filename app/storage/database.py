"""
SQLite connection/schema management (Section 17).

Design choices:
- A fresh `sqlite3` connection is opened per operation rather than kept
  open for the process lifetime. SQLite handles this cheaply and it
  avoids any cross-thread connection-sharing issues between the FastAPI
  request thread(s) and the background processing worker.
- WAL journal mode is enabled so a long-running background write (saving
  a big result) does not block concurrent status-polling reads.
- A module-level `threading.Lock` serializes writes, since SQLite allows
  only one writer at a time and we would rather queue briefly than raise
  "database is locked" errors under the (deliberately simple, Section 16)
  in-process background-worker model.
- Only metadata, status and *paths* to on-disk artifacts are stored in
  SQLite; the actual transcript/result JSON and audio files live on the
  filesystem (Section 17: "Avoid putting huge audio blobs directly into
  SQLite").
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from app.utils.errors import StorageError
from app.utils.logging import get_logger

logger = get_logger("storage.database")

_write_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    meeting_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    stored_upload_path TEXT NOT NULL,
    processed_audio_path TEXT,
    transcript_path TEXT,
    result_path TEXT,
    status TEXT NOT NULL,
    error_code TEXT,
    error_message TEXT,
    duration_seconds REAL,
    languages TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_meetings_status ON meetings(status);
"""


class Database:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        try:
            with self.connect() as conn:
                conn.executescript(SCHEMA)
                conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to initialize database schema: {exc}") from exc

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def write_connection(self):
        """A connection guarded by the module write lock, for INSERT/UPDATE."""
        with _write_lock, self.connect() as conn:
            yield conn
