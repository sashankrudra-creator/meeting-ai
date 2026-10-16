"""
FastAPI application entrypoint.

    uvicorn app.main:app --reload

All shared singletons (Settings, Database, MeetingRepository,
MeetingOrchestrator, BackgroundWorker) are constructed once in the
lifespan context manager and attached to `app.state`, rather than created
per-request or as module-level globals (Section 23: "no unnecessary global
state").
"""

from __future__ import annotations

import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import health, meetings
from app.config import get_settings
from app.models.schemas import ErrorDetail, ErrorResponse
from app.pipeline.orchestrator import MeetingOrchestrator
from app.pipeline.worker import BackgroundWorker
from app.storage.database import Database
from app.storage.repository import MeetingRepository
from app.utils.errors import ErrorCode, PipelineError
from app.utils.logging import get_logger
from app.utils.windows_dll import configure_windows_dll_search_path

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_directories()
    # Register the shared-FFmpeg DLL directory (if configured) before any
    # request can reach diarization, so the FastAPI application process
    # itself is set up correctly rather than relying on the DLL search path
    # of whatever shell happened to launch uvicorn. A no-op on Linux/macOS
    # and when FFMPEG_SHARED_DLL_DIR isn't set — see app/utils/windows_dll.py.
    configure_windows_dll_search_path(settings)

    database = Database(settings.database_path)
    repository = MeetingRepository(database, settings)
    orchestrator = MeetingOrchestrator(settings, repository)
    worker = BackgroundWorker(settings, orchestrator)

    app.state.settings = settings
    app.state.database = database
    app.state.repository = repository
    app.state.worker = worker

    logger.info("%s started (env=%s)", settings.app_name, settings.environment)
    try:
        yield
    finally:
        worker.shutdown(wait=False)
        logger.info("Shutdown complete")


def register_routes_and_handlers(app: FastAPI) -> None:
    """Shared between `create_app()` and the test suite (tests/test_api.py),
    which builds its own FastAPI instance without the real lifespan (no real
    model loading, no real SQLite file at the default path) but still wants
    the exact same routes and error-handling behavior under test."""
    app.include_router(health.router)
    app.include_router(meetings.router)

    @app.exception_handler(PipelineError)
    async def pipeline_error_handler(request: Request, exc: PipelineError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=ErrorResponse(error=ErrorDetail(code=exc.code, message=exc.message)).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Section 18: a malformed request (missing field, wrong type, etc.)
        # must return the SAME structured {"status": "failed", "error": {...}}
        # contract as every other error in this API — not FastAPI/Starlette's
        # default {"detail": [...]} shape, which is a different, undocumented
        # contract that API consumers would have to special-case.
        message = "; ".join(
            f"{'.'.join(str(p) for p in err.get('loc', []))}: {err.get('msg', 'invalid value')}"
            for err in exc.errors()
        ) or "The request could not be validated."
        logger.info("Request validation failed on %s %s: %s", request.method, request.url.path, message)
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorDetail(code=ErrorCode.INVALID_REQUEST, message=message)
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Section 18: never expose raw stack traces through the API. Full
        # detail is logged internally; the client gets a generic, stable code.
        logger.error("Unhandled exception on %s %s:\n%s", request.method, request.url.path, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(code=ErrorCode.INTERNAL_ERROR, message="An unexpected internal error occurred.")
            ).model_dump(),
        )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Voice-Based Minutes of Meeting Pipeline",
        description="Local-inference-only pipeline for speaker-attributed, "
                     "multilingual meeting transcription and MoM generation.",
        version="1.0.0",
        lifespan=lifespan,
    )
    register_routes_and_handlers(app)
    return app


app = create_app()
