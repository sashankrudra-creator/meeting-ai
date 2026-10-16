"""Tests for the simple in-process background worker (Section 16)."""

from __future__ import annotations

import threading
import time

from app.pipeline.worker import BackgroundWorker


class _FakeOrchestrator:
    def __init__(self, delay: float = 0.05):
        self.processed: list[str] = []
        self._delay = delay
        self._lock = threading.Lock()

    def process_meeting(self, meeting_id: str) -> None:
        time.sleep(self._delay)
        with self._lock:
            self.processed.append(meeting_id)


class _CrashingOrchestrator:
    def process_meeting(self, meeting_id: str) -> None:
        raise RuntimeError("boom")


def test_submit_runs_in_background_not_blocking_caller(settings):
    fake = _FakeOrchestrator(delay=0.2)
    worker = BackgroundWorker(settings, fake)

    start = time.monotonic()
    worker.submit("m1")
    elapsed = time.monotonic() - start

    assert elapsed < 0.1  # submit() returns immediately, well before the 0.2s job finishes
    assert fake.processed == []  # not yet done

    worker.shutdown(wait=True)
    assert fake.processed == ["m1"]


def test_multiple_jobs_all_complete(settings):
    settings.max_concurrent_jobs = 2
    fake = _FakeOrchestrator(delay=0.05)
    worker = BackgroundWorker(settings, fake)

    for i in range(5):
        worker.submit(f"m{i}")

    worker.shutdown(wait=True)
    assert sorted(fake.processed) == [f"m{i}" for i in range(5)]


def test_worker_survives_orchestrator_exception(settings):
    worker = BackgroundWorker(settings, _CrashingOrchestrator())
    worker.submit("will-crash")
    # Should not raise / hang — the worker thread swallows and logs the exception.
    worker.shutdown(wait=True)
