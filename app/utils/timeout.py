"""
Windows-compatible execution-timeout helper.

Root-cause context (production bug, see app/pipeline/transcription.py):
`signal.alarm`/`SIGALRM`-based timeouts don't exist on Windows, and the
blocking calls this guards (`WhisperModel(...)`, `Pipeline.from_pretrained(...)`,
`AutoModelForCausalLM.from_pretrained(...)`, and the Hugging Face Hub network
calls those make internally) are native/C-extension code that doesn't check
for cooperative cancellation anyway. The only portable way to bound an
arbitrary blocking call is to run it on a worker thread and stop waiting on
it after a deadline, which is what `run_with_timeout` does.

This does NOT kill the underlying call if it times out — Python has no safe
API to forcibly terminate a running thread, and the calls being guarded
(native network/IO code) generally can't be interrupted anyway. The
timed-out thread keeps running in the background (and is abandoned, not
joined); the caller simply stops waiting for it and can report a clean
failure instead of hanging forever. That's a deliberate, documented
trade-off: an occasional abandoned thread is strictly better than the
previous behavior (an indefinite hang with near-zero CPU usage, no
exception, and no way to ever recover the request).
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeoutError
from typing import TypeVar

T = TypeVar("T")


def run_with_timeout(fn: Callable[[], T], timeout_seconds: float | None, description: str) -> T:
    """Run `fn()` to completion and return its result.

    Raises the built-in `TimeoutError` if `fn()` has not finished within
    `timeout_seconds` — callers translate that into the stage-appropriate
    `PipelineError` subclass so the failure is reported through the normal
    structured-error path instead of hanging.

    `timeout_seconds=None` disables the bound entirely (runs `fn()` on the
    calling thread, the original untimed behavior) — used by callers/tests
    that want a plain blocking call.
    """
    if timeout_seconds is None:
        return fn()

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="timeout-guard")
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_seconds)
    except _FutureTimeoutError as exc:
        raise TimeoutError(
            f"{description} did not complete within {timeout_seconds:.0f}s."
        ) from exc
    finally:
        # wait=False: never block here waiting for an already-timed-out (or
        # otherwise still-running) call to finish. See module docstring.
        executor.shutdown(wait=False)
