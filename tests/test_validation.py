from __future__ import annotations

import pytest

from app.utils.errors import EmptyUploadError, FileTooLargeError, UnsupportedFileExtensionError
from app.utils.validation import safe_meeting_filename, validate_filename_and_size


def test_accepts_supported_extension_and_size(settings):
    ext = validate_filename_and_size("meeting.wav", 1024, settings)
    assert ext == ".wav"


def test_rejects_unsupported_extension(settings):
    with pytest.raises(UnsupportedFileExtensionError):
        validate_filename_and_size("meeting.txt", 1024, settings)


def test_rejects_no_filename(settings):
    with pytest.raises(UnsupportedFileExtensionError):
        validate_filename_and_size("", 1024, settings)
    with pytest.raises(UnsupportedFileExtensionError):
        validate_filename_and_size(None, 1024, settings)


def test_rejects_empty_upload(settings):
    with pytest.raises(EmptyUploadError):
        validate_filename_and_size("meeting.wav", 0, settings)


def test_rejects_oversized_upload(settings):
    settings.max_upload_size_mb = 1
    with pytest.raises(FileTooLargeError):
        validate_filename_and_size("meeting.wav", 2 * 1024 * 1024, settings)


def test_extension_case_insensitive(settings):
    ext = validate_filename_and_size("meeting.WAV", 1024, settings)
    assert ext == ".wav"


def test_safe_meeting_filename_strips_path_traversal():
    name = safe_meeting_filename("abc123", "../../etc/passwd.mp3")
    assert name == "abc123.mp3"
    assert "/" not in name and ".." not in name
