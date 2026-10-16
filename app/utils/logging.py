"""
Central logging configuration.

All modules obtain a logger via `get_logger(__name__)` rather than calling
`logging.getLogger` directly, so log format/level stays consistent and can
be reconfigured from one place (Section 23: "Proper logging").
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache

from app.config import get_settings


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import json

        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "context", None)
        if extra:
            payload["context"] = extra
        return json.dumps(payload)


@lru_cache
def _configure_root() -> None:
    settings = get_settings()
    root = logging.getLogger("voice_mom")
    root.setLevel(settings.log_level.upper())
    handler = logging.StreamHandler(stream=sys.stdout)
    if settings.log_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the shared 'voice_mom' root logger."""
    _configure_root()
    return logging.getLogger(f"voice_mom.{name}")
