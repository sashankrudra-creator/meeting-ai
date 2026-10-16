"""
Unit tests for app.utils.timeout.run_with_timeout — the Windows-compatible
(no signal.alarm) execution-timeout helper used to bound model-loading calls
that can otherwise hang indefinitely (see app/pipeline/transcription.py for
the production incident this exists to prevent).
"""

from __future__ import annotations

import threading
import time

import pytest

from app.utils.timeout import run_with_timeout


def test_returns_result_when_fn_completes_within_timeout():
    assert run_with_timeout(lambda: 42, 5.0, "quick call") == 42


def test_raises_timeout_error_when_fn_is_too_slow():
    def slow():
        time.sleep(1.0)
        return "done"

    with pytest.raises(TimeoutError, match="slow call did not complete within"):
        run_with_timeout(slow, 0.05, "slow call")


def test_propagates_the_original_exception_when_fn_raises():
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        run_with_timeout(boom, 5.0, "erroring call")


def test_none_timeout_disables_the_bound_and_runs_on_the_calling_thread():
    calling_thread = threading.current_thread()
    seen: dict[str, threading.Thread] = {}

    def fn():
        seen["thread"] = threading.current_thread()
        return "ok"

    assert run_with_timeout(fn, None, "unbounded call") == "ok"
    assert seen["thread"] is calling_thread
