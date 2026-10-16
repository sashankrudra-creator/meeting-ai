"""
Minimal in-process background worker (Section 16).

The spec is explicit: don't introduce Celery/Redis/Kafka unless there's a
clear technical reason, and keep the architecture easy to run locally. A
bounded `ThreadPoolExecutor` is enough to (a) return the upload endpoint's
response immediately and (b) cap how many meetings are transcribed/
diarized/summarized at once, which matters because those stages are
CPU/GPU-bound and unbounded concurrency would thrash a single machine.

This is a single-process solution: it does not survive an application
restart (a QUEUED/IN-PROGRESS meeting at restart time stays at whatever
status it was last in — see the README's Known Limitations). That's an
explicit, documented trade-off for the "keep infrastructure simple" goal,
not an oversight.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from app.config import Settings
from app.pipeline.orchestrator import MeetingOrchestrator
from app.utils.logging import get_logger

logger = get_logger("pipeline.worker")


class BackgroundWorker:
    def __init__(self, settings: Settings, orchestrator: MeetingOrchestrator):
        self._orchestrator = orchestrator
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.max_concurrent_jobs),
            thread_name_prefix="mom-pipeline-worker",
        )

    def submit(self, meeting_id: str) -> None:
        logger.info("Queuing meeting %s for background processing", meeting_id)
        self._executor.submit(self._run, meeting_id)

    def _run(self, meeting_id: str) -> None:
        try:
            self._orchestrator.process_meeting(meeting_id)
        except Exception:  # noqa: BLE001 - orchestrator already handles PipelineError internally
            logger.exception("Background worker thread crashed processing meeting %s", meeting_id)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
