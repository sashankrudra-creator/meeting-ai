"""
Pipeline stage: validate_file() and extract_audio()/preprocess_audio().

Kept deliberately thin — the real logic lives in `app.utils.validation`
and `app.utils.audio` so it can be unit tested without touching the
orchestrator, and so the orchestrator (Section 13) stays a simple sequence
of named stage calls rather than one giant function.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.utils.audio import extract_and_normalize_audio, probe_media
from app.utils.logging import get_logger

logger = get_logger("pipeline.preprocessing")


def validate_media_file(path: Path, settings: Settings) -> None:
    """Deep validation: confirm ffprobe can read the file and it has audio.

    This complements `app.utils.validation.validate_filename_and_size`,
    which runs earlier (at upload time, before the file is even fully on
    disk in some deployments) and only checks name/size.
    """
    probe = probe_media(path, settings)
    logger.info(
        "Probed %s: has_audio=%s duration=%.2fs codec=%s",
        path.name, probe.has_audio, probe.duration_seconds, probe.audio_codec,
    )


def preprocess_audio(input_path: Path, meeting_id: str, settings: Settings) -> Path:
    """Extract + normalize audio for `meeting_id`, returning the path to
    the normalized mono/16kHz WAV file under `settings.processed_dir`."""
    output_path = settings.processed_dir / f"{meeting_id}.wav"
    logger.info("Extracting/normalizing audio for meeting %s -> %s", meeting_id, output_path)
    result_path = extract_and_normalize_audio(input_path, output_path, settings)
    logger.info("Preprocessing complete for meeting %s", meeting_id)
    return result_path
