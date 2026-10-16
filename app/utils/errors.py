"""
Structured error taxonomy used throughout the pipeline and surfaced by the API.

Section 18 requires that the API never leak raw stack traces and instead
return a structured error body:

    {
      "status": "failed",
      "error": {
        "code": "AUDIO_EXTRACTION_FAILED",
        "message": "Unable to extract an audio stream from the uploaded file."
      }
    }

Every stage of the pipeline raises a `PipelineError` subclass with a fixed
`code`. The API layer (app/api/meetings.py) catches `PipelineError` and
converts it into that structured body; anything else is logged with full
detail internally and converted into a generic INTERNAL_ERROR response so
no stack trace ever reaches the client.
"""

from __future__ import annotations


class ErrorCode:
    UNSUPPORTED_FILE_EXTENSION = "UNSUPPORTED_FILE_EXTENSION"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    EMPTY_UPLOAD = "EMPTY_UPLOAD"
    CORRUPT_MEDIA = "CORRUPT_MEDIA"
    NO_AUDIO_STREAM = "NO_AUDIO_STREAM"
    EMPTY_AUDIO = "EMPTY_AUDIO"
    AUDIO_EXTRACTION_FAILED = "AUDIO_EXTRACTION_FAILED"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    DIARIZATION_FAILED = "DIARIZATION_FAILED"
    MODEL_LOADING_FAILED = "MODEL_LOADING_FAILED"
    GPU_OUT_OF_MEMORY = "GPU_OUT_OF_MEMORY"
    ALIGNMENT_FAILED = "ALIGNMENT_FAILED"
    STATISTICS_FAILED = "STATISTICS_FAILED"
    SUMMARIZATION_FAILED = "SUMMARIZATION_FAILED"
    INVALID_REQUEST = "INVALID_REQUEST"
    MEETING_NOT_FOUND = "MEETING_NOT_FOUND"
    MISSING_MEETING_ID = "MISSING_MEETING_ID"
    RESULT_NOT_READY = "RESULT_NOT_READY"
    STORAGE_FAILURE = "STORAGE_FAILURE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class PipelineError(Exception):
    """Base class for all errors that should be reported to API clients
    with a structured, stable error code rather than a raw traceback."""

    code: str = ErrorCode.INTERNAL_ERROR
    http_status: int = 500

    def __init__(self, message: str, *, code: str | None = None, http_status: int | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


class ValidationError(PipelineError):
    code = ErrorCode.INVALID_REQUEST
    http_status = 400


class UnsupportedFileExtensionError(ValidationError):
    code = ErrorCode.UNSUPPORTED_FILE_EXTENSION


class FileTooLargeError(ValidationError):
    code = ErrorCode.FILE_TOO_LARGE
    http_status = 413


class EmptyUploadError(ValidationError):
    code = ErrorCode.EMPTY_UPLOAD


class CorruptMediaError(PipelineError):
    code = ErrorCode.CORRUPT_MEDIA
    http_status = 422


class NoAudioStreamError(PipelineError):
    code = ErrorCode.NO_AUDIO_STREAM
    http_status = 422


class EmptyAudioError(PipelineError):
    code = ErrorCode.EMPTY_AUDIO
    http_status = 422


class AudioExtractionError(PipelineError):
    code = ErrorCode.AUDIO_EXTRACTION_FAILED
    http_status = 422


class TranscriptionError(PipelineError):
    code = ErrorCode.TRANSCRIPTION_FAILED
    http_status = 500


class DiarizationError(PipelineError):
    code = ErrorCode.DIARIZATION_FAILED
    http_status = 500


class ModelLoadingError(PipelineError):
    code = ErrorCode.MODEL_LOADING_FAILED
    http_status = 503


class GpuOutOfMemoryError(PipelineError):
    code = ErrorCode.GPU_OUT_OF_MEMORY
    http_status = 503


class AlignmentError(PipelineError):
    code = ErrorCode.ALIGNMENT_FAILED
    http_status = 500


class StatisticsError(PipelineError):
    code = ErrorCode.STATISTICS_FAILED
    http_status = 500


class SummarizationError(PipelineError):
    code = ErrorCode.SUMMARIZATION_FAILED
    http_status = 500


class MeetingNotFoundError(PipelineError):
    code = ErrorCode.MEETING_NOT_FOUND
    http_status = 404


class MissingMeetingIdError(ValidationError):
    code = ErrorCode.MISSING_MEETING_ID


class ResultNotReadyError(PipelineError):
    code = ErrorCode.RESULT_NOT_READY
    http_status = 409


class StorageError(PipelineError):
    code = ErrorCode.STORAGE_FAILURE
    http_status = 500
