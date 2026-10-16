"""
FFmpeg/ffprobe wrappers: media probing and audio extraction/normalization.

Design notes:
- We shell out to the `ffprobe`/`ffmpeg` binaries directly via `subprocess`
  rather than a heavier binding, so behavior matches whatever ffmpeg the
  operator has installed (Section 19 calls for FFmpeg specifically).
- Every failure mode from Section 18 that touches media
  (corrupt file, missing audio stream, empty audio, ffmpeg failure) is
  translated here into a specific `PipelineError` subclass so the API
  never has to interpret raw ffmpeg stderr.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.utils.errors import (
    AudioExtractionError,
    CorruptMediaError,
    EmptyAudioError,
    NoAudioStreamError,
)
from app.utils.logging import get_logger

logger = get_logger("utils.audio")


@dataclass
class MediaProbe:
    has_audio: bool
    duration_seconds: float
    audio_codec: str | None
    sample_rate: int | None
    channels: int | None
    raw: dict


def probe_media(path: Path, settings: Settings) -> MediaProbe:
    """Run ffprobe against `path` and return a structured summary.

    Raises `CorruptMediaError` if ffprobe cannot parse the file at all.
    """
    if not path.exists() or path.stat().st_size == 0:
        raise CorruptMediaError(f"File '{path.name}' does not exist or is empty on disk.")

    cmd = [
        settings.ffprobe_binary,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.ffmpeg_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise CorruptMediaError(f"Timed out while probing '{path.name}'.") from exc
    except FileNotFoundError as exc:
        raise AudioExtractionError(
            f"ffprobe binary '{settings.ffprobe_binary}' was not found on this system."
        ) from exc

    if proc.returncode != 0:
        logger.warning("ffprobe failed for %s: %s", path.name, proc.stderr.strip())
        raise CorruptMediaError(
            f"Unable to read '{path.name}' as a media file. It may be corrupt or "
            "not a supported audio/video format."
        )

    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise CorruptMediaError(f"ffprobe returned unparseable output for '{path.name}'.") from exc

    streams = info.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    fmt = info.get("format", {})

    duration = 0.0
    try:
        duration = float(fmt.get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0

    if not audio_streams:
        return MediaProbe(
            has_audio=False,
            duration_seconds=duration,
            audio_codec=None,
            sample_rate=None,
            channels=None,
            raw=info,
        )

    primary = audio_streams[0]
    return MediaProbe(
        has_audio=True,
        duration_seconds=duration,
        audio_codec=primary.get("codec_name"),
        sample_rate=int(primary["sample_rate"]) if primary.get("sample_rate") else None,
        channels=primary.get("channels"),
        raw=info,
    )


def extract_and_normalize_audio(
    input_path: Path,
    output_path: Path,
    settings: Settings,
) -> Path:
    """Extract the audio stream from `input_path` (audio or video) and
    normalize it to mono/16kHz/PCM WAV at `output_path`.

    Raises:
        NoAudioStreamError: the container has no audio stream at all.
        EmptyAudioError: extraction succeeded but produced a near-silent /
            zero-length file.
        AudioExtractionError: ffmpeg itself failed.
    """
    probe = probe_media(input_path, settings)
    if not probe.has_audio:
        raise NoAudioStreamError(
            f"'{input_path.name}' does not contain an audio stream that can be transcribed."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        settings.ffmpeg_binary,
        "-y",
        "-i", str(input_path),
        "-vn",  # drop video
        "-ac", str(settings.target_channels),
        "-ar", str(settings.target_sample_rate),
        "-sample_fmt", "s16",
        "-f", "wav",
        str(output_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.ffmpeg_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioExtractionError(f"ffmpeg timed out extracting audio from '{input_path.name}'.") from exc
    except FileNotFoundError as exc:
        raise AudioExtractionError(
            f"ffmpeg binary '{settings.ffmpeg_binary}' was not found on this system."
        ) from exc

    if proc.returncode != 0 or not output_path.exists():
        logger.warning("ffmpeg failed for %s: %s", input_path.name, proc.stderr.strip()[-2000:])
        raise AudioExtractionError(
            f"ffmpeg failed to extract/normalize audio from '{input_path.name}'."
        )

    if output_path.stat().st_size <= 44:  # smaller than a bare WAV header => no audio data
        raise EmptyAudioError(
            f"Extracted audio from '{input_path.name}' contains no audio data."
        )

    return output_path


def get_audio_duration_seconds(path: Path, settings: Settings) -> float:
    probe = probe_media(path, settings)
    return probe.duration_seconds
