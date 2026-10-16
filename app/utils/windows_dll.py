"""
Windows DLL search-path setup for TorchCodec (Section 5 — speaker diarization).

Root cause (production incident): `pyannote.audio` 4.x decodes audio via
`torchcodec` rather than `torchaudio`/`soundfile`. TorchCodec ships its own
`libtorchcodec_coreN.dll` (one per supported FFmpeg major version, N in
4..9) — verified directly against the installed `torchcodec` package's
source (`torchcodec/_internally_replaced_utils.py::load_core_libraries`) —
but that DLL only *wraps* FFmpeg's real shared libraries (avutil-*.dll,
avcodec-*.dll, avformat-*.dll, swresample-*.dll, swscale-*.dll, ...); it
does not embed them.

Since Python 3.8, the Windows DLL loader no longer implicitly searches
`PATH` or the current working directory when resolving a loaded DLL's own
dependencies — a directory must be explicitly registered with
`os.add_dll_directory()` before the load happens. Skip that, and loading
`libtorchcodec_core9.dll` fails with a misleading
"Could not find module ... (or one of its dependencies)" — Windows reports
the DLL you asked for even when the real problem is one of *its*
dependencies being unresolved.

This is made worse by a specific, easy-to-hit trap: Gyan.dev's FFmpeg
"full_build" (the common winget/direct-download choice, built with
`--enable-static`) ships only static executables (ffmpeg.exe, ffprobe.exe,
ffplay.exe) and NO shared av*/sw*.dll files at all — there is nothing in
that folder TorchCodec could ever load, no matter what directory is added
to the search path. Only the separate "full_build-**shared**" variant (or
an equivalent shared FFmpeg build) ships the required DLLs. TorchCodec's
own load-failure message agrees: "On Windows, ensure you've installed the
'full-shared' version which ships DLLs."

`configure_windows_dll_search_path()` registers the configured shared-
FFmpeg DLL directory (`Settings.ffmpeg_shared_dll_dir`) with
`os.add_dll_directory()` so that when TorchCodec's native loader later
tries to resolve `libtorchcodec_core*.dll`'s dependencies, Windows can find
them. It is a no-op on non-Windows platforms and when the setting isn't
configured (e.g. the shared DLLs are already on PATH, or diarization is
never used), so Linux/macOS behavior and existing setups are unaffected.

It must run before the first `import pyannote.audio` / `import torchcodec`
in the process (see app/pipeline/diarization.py, which calls it immediately
before that import) — app/main.py's startup also calls it eagerly so it is
already in effect before the first request, rather than only on first use.
"""

from __future__ import annotations

import os
import sys
import threading

from app.config import Settings
from app.utils.logging import get_logger

logger = get_logger("utils.windows_dll")

_lock = threading.Lock()
_configured = False


def configure_windows_dll_search_path(settings: Settings) -> None:
    """Idempotent and thread-safe: only the first call does anything (later
    calls, from any of the multiple call sites, are cheap no-ops) since
    `os.add_dll_directory()` registers process-wide, permanent state that
    never needs to be redone or undone for the life of the process. A no-op
    entirely on non-Windows platforms."""
    global _configured

    if sys.platform != "win32":
        return

    with _lock:
        if _configured:
            return
        _configured = True  # mark as attempted even below on failure/skip -
        # the outcome depends only on config that doesn't change at
        # runtime, so retrying on every call would just repeat the same
        # log message (or the same failure) for no benefit.

        dll_dir = settings.ffmpeg_shared_dll_dir
        if dll_dir is None:
            logger.info(
                "FFMPEG_SHARED_DLL_DIR is not set; skipping Windows DLL search-path setup "
                "for TorchCodec. If diarization fails with a "
                "\"libtorchcodec_core*.dll ... (or one of its dependencies)\" error, install "
                "a *shared* FFmpeg build (not a static one) and set FFMPEG_SHARED_DLL_DIR to "
                "its bin/ directory."
            )
            return

        if not dll_dir.is_dir():
            logger.warning(
                "FFMPEG_SHARED_DLL_DIR is set to '%s' but that directory does not exist; "
                "skipping Windows DLL search-path setup for TorchCodec.",
                dll_dir,
            )
            return

        try:
            os.add_dll_directory(str(dll_dir))
        except (OSError, AttributeError) as exc:
            # AttributeError: os.add_dll_directory only exists on Windows
            # (guarded above by the sys.platform check, but kept defensive
            # rather than assumed).
            logger.warning(
                "Failed to register '%s' as a Windows DLL search directory for "
                "TorchCodec: %s",
                dll_dir, exc,
            )
            return

        logger.info(
            "Registered '%s' as a Windows DLL search directory for TorchCodec.", dll_dir
        )
