"""FastAPI dependency providers.

Shared singletons (settings, DB, repository, background worker) are built
once at application startup (see `app.main`'s lifespan) and stored on
`app.state`; these functions just fetch them per-request via `Depends`,
which keeps route handlers free of global state and easy to test with
`app.dependency_overrides`.
"""

from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.pipeline.worker import BackgroundWorker
from app.storage.repository import MeetingRepository


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_repository(request: Request) -> MeetingRepository:
    return request.app.state.repository


def get_worker(request: Request) -> BackgroundWorker:
    return request.app.state.worker
