"""
Cheap, fast, pre-ffmpeg validation of an uploaded file (Section 2/18).

This module only looks at the filename and byte size — it does not touch
the file's actual media content (that is `app.utils.audio.probe_media`'s
job, since it requires shelling out to ffprobe). Keeping the two separate
means a client gets an immediate 400 for an obviously-wrong extension
without waiting for ffprobe to spin up.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.utils.errors import EmptyUploadError, FileTooLargeError, UnsupportedFileExtensionError


def validate_extension(filename: str | None, settings: Settings) -> str:
    """Validate only the upload's filename/extension (Section 2/18) — cheap
    and content-independent, so it's safe to call before a single byte of
    the upload has been read off the wire. Returns the normalized
    (lowercase) extension on success. Raises a `PipelineError` subclass on
    failure."""
    if not filename or not filename.strip():
        raise UnsupportedFileExtensionError(
            "Uploaded file has no filename; cannot determine its type."
        )

    extension = Path(filename).suffix.lower()
    if extension not in settings.allowed_extensions:
        allowed = ", ".join(settings.allowed_extensions)
        raise UnsupportedFileExtensionError(
            f"Unsupported file extension '{extension}'. Allowed extensions: {allowed}."
        )
    return extension


def validate_upload_size(size_bytes: int, settings: Settings) -> None:
    """Validate a fully-known upload size against the configured limits
    (Section 2/18). Split out from `validate_filename_and_size` so the
    streaming upload endpoint (`app.api.meetings.upload_meeting`) can
    enforce the same max-size rule against a running byte count as it
    writes to disk, instead of requiring the whole file to be buffered in
    memory first just to learn its size up front."""
    if size_bytes <= 0:
        raise EmptyUploadError("Uploaded file is empty (0 bytes).")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise FileTooLargeError(
            f"Uploaded file is {size_bytes / (1024 * 1024):.1f} MB, which exceeds "
            f"the {settings.max_upload_size_mb} MB limit."
        )


def validate_filename_and_size(
    filename: str | None,
    size_bytes: int,
    settings: Settings,
) -> str:
    """Validate the upload's name and size in one call, for callers that
    already have the full size known up front (e.g. tests). Returns the
    normalized (lowercase) file extension on success. Raises a
    `PipelineError` subclass on failure.
    """
    extension = validate_extension(filename, settings)
    validate_upload_size(size_bytes, settings)
    return extension


def safe_meeting_filename(meeting_id: str, original_filename: str) -> str:
    """Build a filesystem-safe stored filename that keeps the original
    extension but can never contain path-traversal characters."""
    extension = Path(original_filename).suffix.lower()
    return f"{meeting_id}{extension}"
