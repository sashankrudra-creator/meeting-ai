"""
Unit tests for app.utils.windows_dll.configure_windows_dll_search_path.

Root-cause context: pyannote.audio 4.x decodes audio via TorchCodec, which
on Windows needs a *shared* FFmpeg build's DLLs registered with
`os.add_dll_directory()` before use (PATH alone is not enough since Python
3.8's DLL-loading changes) — see app/pipeline/diarization.py and the
module docstring in app/utils/windows_dll.py for the full explanation.

This test suite runs in a non-Windows CI/dev environment, so every test
simulates Windows by monkeypatching `sys.platform` and (since
`os.add_dll_directory` does not exist at all outside Windows) by injecting
a fake `os.add_dll_directory` attribute — the real Windows DLL-loading
behavior itself is NOT exercised here; only the pure-Python
config-driven/idempotency/no-op logic is.
"""

from __future__ import annotations

import os
import sys

import pytest

from app.utils import windows_dll
from app.utils.windows_dll import configure_windows_dll_search_path


@pytest.fixture(autouse=True)
def _reset_configured_flag(monkeypatch):
    """The module's "already attempted" flag is process-global state (by
    design — see the module docstring), so each test needs its own clean
    slate rather than inheriting whatever an earlier test left behind."""
    monkeypatch.setattr(windows_dll, "_configured", False)


def test_noop_on_non_windows_even_when_configured(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    calls = []
    monkeypatch.setattr(os, "add_dll_directory", lambda p: calls.append(p), raising=False)

    settings.ffmpeg_shared_dll_dir = tmp_path  # a real, existing directory
    configure_windows_dll_search_path(settings)

    assert calls == []


def test_noop_on_windows_when_not_configured(settings, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    calls = []
    monkeypatch.setattr(os, "add_dll_directory", lambda p: calls.append(p), raising=False)

    settings.ffmpeg_shared_dll_dir = None
    configure_windows_dll_search_path(settings)

    assert calls == []


def test_noop_on_windows_when_configured_directory_does_not_exist(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    calls = []
    monkeypatch.setattr(os, "add_dll_directory", lambda p: calls.append(p), raising=False)

    settings.ffmpeg_shared_dll_dir = tmp_path / "does-not-exist"
    configure_windows_dll_search_path(settings)

    assert calls == []


def test_registers_directory_on_windows_when_configured_and_existing(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    calls = []
    monkeypatch.setattr(os, "add_dll_directory", lambda p: calls.append(p), raising=False)

    settings.ffmpeg_shared_dll_dir = tmp_path
    configure_windows_dll_search_path(settings)

    assert calls == [str(tmp_path)]


def test_is_idempotent_only_the_first_call_does_anything(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    calls = []
    monkeypatch.setattr(os, "add_dll_directory", lambda p: calls.append(p), raising=False)

    settings.ffmpeg_shared_dll_dir = tmp_path
    configure_windows_dll_search_path(settings)
    configure_windows_dll_search_path(settings)
    configure_windows_dll_search_path(settings)

    assert calls == [str(tmp_path)]  # only the first call registered anything


def test_does_not_raise_when_add_dll_directory_itself_fails(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")

    def _boom(path):
        raise OSError("simulated failure")

    monkeypatch.setattr(os, "add_dll_directory", _boom, raising=False)

    settings.ffmpeg_shared_dll_dir = tmp_path
    configure_windows_dll_search_path(settings)  # must not raise
