#!/usr/bin/env python3
"""
Acceptance-test script for the Windows TorchCodec/FFmpeg DLL fix
(Section 5 — speaker diarization; see app/utils/windows_dll.py).

Exercises the exact same `configure_windows_dll_search_path()` call the
real FastAPI app makes at startup (app/main.py) and before the first
diarization call (app/pipeline/diarization.py), then runs the acceptance
test:

    from torchcodec.decoders import AudioDecoder
    d = AudioDecoder(<path>)
    print(d.metadata)

Run it with the SAME `.env` / environment variables the real app uses
(activate the same virtualenv, run from the project root):

    python scripts/verify_torchcodec_windows.py
    python scripts/verify_torchcodec_windows.py path\\to\\other\\file.wav

Defaults to `data/uploads/wmori.wav` if no path is given. Exits non-zero
on failure so it's script-friendly (e.g. `if ($LASTEXITCODE -ne 0) { ... }`
in PowerShell).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.utils.windows_dll import configure_windows_dll_search_path  # noqa: E402


def main() -> int:
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/uploads/wmori.wav")

    settings = get_settings()
    print(f"Platform check + FFMPEG_SHARED_DLL_DIR={settings.ffmpeg_shared_dll_dir!r}")
    configure_windows_dll_search_path(settings)

    if not audio_path.exists():
        print(f"FAIL: audio file not found: {audio_path}")
        return 2

    try:
        from torchcodec.decoders import AudioDecoder
    except Exception as exc:  # noqa: BLE001 - want the raw import failure, not a translated one
        print(f"FAIL: could not import torchcodec.decoders.AudioDecoder: {exc}")
        return 1

    try:
        decoder = AudioDecoder(str(audio_path))
        print(decoder.metadata)
    except Exception as exc:  # noqa: BLE001 - want the raw decode failure, not a translated one
        print(f"FAIL: AudioDecoder('{audio_path}') raised: {exc}")
        return 1

    print(f"PASS: TorchCodec successfully decoded '{audio_path}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
