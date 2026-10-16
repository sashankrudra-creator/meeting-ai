from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends

from app.api.dependencies import get_settings_dep
from app.config import Settings
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])

APP_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
def health_check(settings: Settings = Depends(get_settings_dep)) -> HealthResponse:
    details = {
        "ffmpeg_available": shutil.which(settings.ffmpeg_binary) is not None,
        "ffprobe_available": shutil.which(settings.ffprobe_binary) is not None,
        "whisper_model": settings.whisper_model_name,
        "whisper_device": settings.whisper_device,
        "diarization_model": settings.diarization_model_name,
        "summarizer_backend": settings.summarizer_backend,
        "summarizer_model": settings.summarizer_model_name,
    }
    return HealthResponse(app_name=settings.app_name, version=APP_VERSION, details=details)
