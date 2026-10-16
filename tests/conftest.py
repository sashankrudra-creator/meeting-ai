"""
Shared pytest fixtures.

Synthetic media fixtures are generated on the fly with ffmpeg's built-in
`lavfi` source filters (sine wave tone generator, testsrc pattern
generator). This requires *only* a local ffmpeg binary — no network access
and no model downloads — which is what makes it possible to exercise the
file-validation and audio-preprocessing stages in this sandboxed
environment. It intentionally does NOT exercise transcription, diarization
or summarization, which need real speech and downloaded model weights;
those are covered by mocked/fixture-based unit tests instead (see
test_transcription.py, test_diarization.py, test_alignment.py).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.config import Settings, get_settings


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    get_settings.cache_clear()
    s = Settings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "db.sqlite3")
    s.ensure_directories()
    return s


def _run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg fixture generation failed: {proc.stderr}")


@pytest.fixture()
def valid_wav_file(tmp_path: Path) -> Path:
    """A real, valid 2-second mono sine-wave WAV file."""
    out = tmp_path / "valid_meeting.wav"
    _run_ffmpeg([
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-ac", "1", "-ar", "16000", str(out),
    ])
    return out


@pytest.fixture()
def valid_video_with_audio(tmp_path: Path) -> Path:
    """A real, valid 2-second video container with both a video and audio stream."""
    out = tmp_path / "valid_meeting.mp4"
    _run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
        "-shortest", "-c:v", "libx264", "-c:a", "aac", str(out),
    ])
    return out


@pytest.fixture()
def video_without_audio(tmp_path: Path) -> Path:
    """A valid video container with NO audio stream at all."""
    out = tmp_path / "silent_video.mp4"
    _run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=15",
        "-an", "-c:v", "libx264", str(out),
    ])
    return out


@pytest.fixture()
def corrupt_media_file(tmp_path: Path) -> Path:
    """A file with a valid extension but garbage bytes inside."""
    out = tmp_path / "corrupt.wav"
    out.write_bytes(b"this is not a real wav file" * 20)
    return out


@pytest.fixture()
def empty_file(tmp_path: Path) -> Path:
    out = tmp_path / "empty.wav"
    out.touch()
    return out


@pytest.fixture(scope="session", autouse=True)
def _require_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available in this environment")
