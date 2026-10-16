"""
Meeting upload/status/result endpoints (Section 15).

The upload endpoint (`POST /api/v1/meetings`) intentionally does the bare
minimum synchronously — validate name/size, save the file, create a DB
row, hand off to the background worker — and returns immediately with a
meeting_id and QUEUED status (Section 16: "do not force a long meeting to
remain inside the HTTP request lifecycle").
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, UploadFile
from fastapi import File as FastAPIFile

from app.api.dependencies import get_repository, get_settings_dep, get_worker
from app.config import Settings
from app.models.schemas import (
    MeetingCreateResponse,
    MeetingResult,
    MeetingStatusResponse,
    ProcessingStatus,
    SummaryResponse,
    TranscriptResponse,
)
from app.pipeline.worker import BackgroundWorker
from app.storage.repository import MeetingRepository
from app.utils.errors import EmptyUploadError, FileTooLargeError, ResultNotReadyError, StorageError
from app.utils.logging import get_logger
from app.utils.validation import safe_meeting_filename, validate_extension

router = APIRouter(prefix="/api/v1/meetings", tags=["meetings"])
logger = get_logger("api.meetings")


@router.post("", response_model=MeetingCreateResponse, status_code=201)
async def upload_meeting(
    file: UploadFile = FastAPIFile(...),
    settings: Settings = Depends(get_settings_dep),
    repo: MeetingRepository = Depends(get_repository),
    worker: BackgroundWorker = Depends(get_worker),
) -> MeetingCreateResponse:
    # Cheap, content-independent check first (Section 2/18): an unsupported
    # extension is rejected before a single byte of the upload is read.
    validate_extension(file.filename, settings)

    meeting_id = uuid.uuid4().hex
    stored_filename = safe_meeting_filename(meeting_id, file.filename or "upload")
    stored_path = settings.uploads_dir / stored_filename

    # Stream the upload to disk in bounded chunks rather than buffering the
    # whole file in memory first (the old `await file.read()`) — a
    # multi-GB upload must never require a multi-GB allocation. The
    # configured max-size limit is enforced against the running byte count
    # as we go, so an oversized upload is aborted (and its partial file
    # removed) as soon as it crosses the limit, not after it has been
    # fully received.
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    chunk_size = settings.upload_chunk_size_bytes
    total_bytes = 0
    try:
        with stored_path.open("wb") as out_file:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise FileTooLargeError(
                        f"Uploaded file exceeds the {settings.max_upload_size_mb} MB limit; "
                        "upload aborted while streaming."
                    )
                out_file.write(chunk)
    except FileTooLargeError:
        stored_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        stored_path.unlink(missing_ok=True)
        raise StorageError(f"Failed to save uploaded file: {exc}") from exc
    finally:
        await file.close()

    if total_bytes <= 0:
        stored_path.unlink(missing_ok=True)
        raise EmptyUploadError("Uploaded file is empty (0 bytes).")

    repo.create_meeting(meeting_id, file.filename or stored_filename, stored_path)
    worker.submit(meeting_id)

    logger.info("Accepted meeting %s (%s, %d bytes)", meeting_id, file.filename, total_bytes)
    return MeetingCreateResponse(
        meeting_id=meeting_id,
        status=ProcessingStatus.QUEUED,
        filename=file.filename or stored_filename,
    )


@router.get("/{meeting_id}", response_model=MeetingStatusResponse)
def get_meeting_status(
    meeting_id: str,
    repo: MeetingRepository = Depends(get_repository),
) -> MeetingStatusResponse:
    record = repo.get_meeting(meeting_id)
    return MeetingStatusResponse(
        meeting_id=record.meeting_id,
        status=record.status,
        filename=record.filename,
        created_at=record.created_at,
        updated_at=record.updated_at,
        error=record.error,
        duration_seconds=record.duration_seconds,
        languages=record.languages or [],
    )


@router.get("/{meeting_id}/transcript", response_model=TranscriptResponse)
def get_meeting_transcript(
    meeting_id: str,
    repo: MeetingRepository = Depends(get_repository),
) -> TranscriptResponse:
    record = repo.get_meeting(meeting_id)
    transcript = repo.get_transcript(meeting_id)
    return TranscriptResponse(meeting_id=meeting_id, status=record.status, transcript=transcript)


@router.get("/{meeting_id}/summary", response_model=SummaryResponse)
def get_meeting_summary(
    meeting_id: str,
    repo: MeetingRepository = Depends(get_repository),
) -> SummaryResponse:
    record = repo.get_meeting(meeting_id)
    summary = repo.get_summary(meeting_id)
    return SummaryResponse(meeting_id=meeting_id, status=record.status, summary=summary)


@router.get("/{meeting_id}/result", response_model=MeetingResult)
def get_meeting_result(
    meeting_id: str,
    repo: MeetingRepository = Depends(get_repository),
) -> MeetingResult:
    record = repo.get_meeting(meeting_id)
    # COMPLETED_WITHOUT_SUMMARY is also a valid, retrievable outcome (C1):
    # transcript/statistics succeeded even though summarization didn't, so
    # the result (with an empty, warning-carrying summary) is available.
    if record.status not in (ProcessingStatus.COMPLETED, ProcessingStatus.COMPLETED_WITHOUT_SUMMARY):
        raise ResultNotReadyError(
            f"Meeting '{meeting_id}' is not ready yet (current status: {record.status.value})."
        )
    result = repo.get_result(meeting_id)
    if result is None:
        raise ResultNotReadyError(f"Meeting '{meeting_id}' has no stored result.")
    return result
