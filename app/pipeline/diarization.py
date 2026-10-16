"""
Speaker diarization stage (Section 5).

    Diarizer (interface)
        |
        +-- PyannoteDiarizer

Internally, pyannote emits labels like `SPEAKER_00`, `SPEAKER_01`, ...
These are diarization-internal identifiers and are NOT guaranteed to be
assigned in a human-friendly or even consistent order across runs. This
module is also responsible for building the one, stable mapping from those
internal labels to the display names ("Person 1", "Person 2", ...) that is
used for the rest of the meeting — computed once, from the full turn list,
never per-segment (Section 5: "Do not assign speaker labels independently
for every segment").

Known limitation: pyannote/speaker-diarization-community-1 is a gated
model on the Hugging Face Hub. Using it requires (a) accepting its license
on huggingface.co and (b) an HF access token with access to it, supplied
via `HUGGINGFACE_TOKEN`/`HF_TOKEN`. This sandbox has no route to
huggingface.co at all, so this module's integration with the real model
has not been exercised in this session — only its label-mapping and
error-translation logic has (see tests/test_diarization.py).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models.schemas import DiarizationResult, DiarizationTurn
from app.utils.errors import DiarizationError, GpuOutOfMemoryError, ModelLoadingError
from app.utils.logging import get_logger
from app.utils.timeout import run_with_timeout
from app.utils.windows_dll import configure_windows_dll_search_path

logger = get_logger("pipeline.diarization")


class Diarizer(ABC):
    """Abstraction over "an engine that splits an audio file into
    per-speaker turns"."""

    @abstractmethod
    def diarize(self, audio_path: Path) -> DiarizationResult:
        raise NotImplementedError


def _is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def _extract_speaker_diarization_annotation(output: Any) -> Any:
    """Unwrap a pyannote `SpeakerDiarization` pipeline call's result into the
    `pyannote.core.Annotation` this module needs (the object `.itertracks()`
    is called on below).

    Production incident / root cause (verified against the installed
    `pyannote.audio==4.0.7` source, `pyannote/audio/pipelines/
    speaker_diarization.py`): as of pyannote.audio 4.x, calling the pipeline
    no longer returns a bare `Annotation` by default — it returns a
    `DiarizeOutput` dataclass with three fields: `speaker_diarization`
    (an `Annotation`, INCLUDING overlapping speech turns),
    `exclusive_speaker_diarization` (an `Annotation` with overlapping turns
    removed), and `speaker_embeddings`. Calling `.itertracks()` directly on
    that `DiarizeOutput` object raises `AttributeError: 'DiarizeOutput'
    object has no attribute 'itertracks'`.

    `speaker_diarization` (not `exclusive_speaker_diarization`) is the
    correct field to use here: this pipeline's overlapping turns are what
    `app/pipeline/alignment.py`/`app/pipeline/statistics.py` rely on to
    populate `overlapping_speakers` (see their module docstrings) —
    switching to the exclusive variant would silently discard that
    information.

    A plain `Annotation` (no `speaker_diarization` attribute) is still
    accepted as-is and returned unchanged: pyannote.audio's own
    `SpeakerDiarization.apply()` returns a bare `Annotation` when the
    pipeline is constructed with `legacy=True`, and older
    pyannote.audio/diarization-pipeline versions never had `DiarizeOutput`
    at all. Detected here by duck-typing (`hasattr`) rather than an
    `isinstance` check against pyannote's `DiarizeOutput` class, to avoid a
    dependency on that non-public import path.
    """
    speaker_diarization = getattr(output, "speaker_diarization", None)
    if speaker_diarization is not None:
        return speaker_diarization
    return output


class PyannoteDiarizer(Diarizer):
    """Production diarizer backed by `pyannote.audio`'s community
    speaker-diarization pipeline."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._pipeline: Any | None = None
        # Guards the lazy-load below. `process_meeting` calls run on a
        # bounded ThreadPoolExecutor (app.pipeline.worker), so more than one
        # worker thread can call `_load_pipeline()` concurrently on this
        # same, shared PyannoteDiarizer instance the first time it's used.
        # Without a lock, two threads could both see `self._pipeline is
        # None`, both pay the (multi-second, multi-GB) load cost, and race
        # to assign `self._pipeline` last.
        self._load_lock = threading.Lock()

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        with self._load_lock:
            # Re-check inside the lock (double-checked locking): another
            # thread may have finished loading while we were waiting.
            if self._pipeline is not None:
                return self._pipeline

            settings = self._settings

            # Must happen before `import pyannote.audio` (which transitively
            # imports `torchcodec`, pyannote 4.x's audio-decoding backend)
            # and, more importantly, before the actual decode call in
            # `diarize()` below — see app/utils/windows_dll.py for the full
            # root-cause explanation (a Windows-only TorchCodec/FFmpeg DLL
            # loading issue). A no-op on Linux/macOS and when unconfigured.
            configure_windows_dll_search_path(settings)

            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:  # pragma: no cover - environment issue, not logic
                raise ModelLoadingError(
                    "The 'pyannote.audio' package is not installed. Run "
                    "`pip install -r requirements.txt`."
                ) from exc

            logger.info("Loading diarization pipeline '%s'", settings.diarization_model_name)
            try:
                # Defense-in-depth against a hanging (not fast-failing)
                # network call to the Hugging Face Hub — see
                # app/utils/timeout.py and the identical, confirmed root
                # cause in app/pipeline/transcription.py. Unlike
                # faster-whisper, pyannote's `Pipeline.from_pretrained()`
                # does not expose a `local_files_only` option, so a bounded
                # timeout (rather than a cache-first attempt) is the
                # available guard here: it still turns a future hang into a
                # structured ModelLoadingError instead of an indefinite
                # stall.
                pipeline = run_with_timeout(
                    lambda: Pipeline.from_pretrained(
                        settings.diarization_model_name,
                        token=settings.huggingface_token,
                    ),
                    settings.model_load_timeout_seconds,
                    f"Loading diarization pipeline '{settings.diarization_model_name}'",
                )
                if pipeline is None:
                    raise ModelLoadingError(
                        f"pyannote returned no pipeline for '{settings.diarization_model_name}'. "
                        "Confirm the model license has been accepted and HUGGINGFACE_TOKEN is set."
                    )
                if settings.diarization_device == "cuda":
                    import torch

                    pipeline.to(torch.device("cuda"))
            except ModelLoadingError:
                raise
            except TimeoutError as exc:
                raise ModelLoadingError(
                    f"Timed out loading diarization pipeline '{settings.diarization_model_name}' "
                    f"after {settings.model_load_timeout_seconds:.0f}s: {exc} This usually means a "
                    "network call to the Hugging Face Hub is hanging rather than failing fast. "
                    "See `Settings.model_load_timeout_seconds`."
                ) from exc
            except Exception as exc:
                if _is_oom_error(exc):
                    raise GpuOutOfMemoryError(
                        f"Out of memory while loading diarization model "
                        f"'{settings.diarization_model_name}'."
                    ) from exc
                raise ModelLoadingError(
                    f"Failed to load diarization model '{settings.diarization_model_name}': {exc}"
                ) from exc

            self._pipeline = pipeline
            return self._pipeline

    def diarize(self, audio_path: Path) -> DiarizationResult:
        pipeline = self._load_pipeline()
        settings = self._settings

        kwargs: dict[str, Any] = {}
        if settings.diarization_min_speakers is not None:
            kwargs["min_speakers"] = settings.diarization_min_speakers
        if settings.diarization_max_speakers is not None:
            kwargs["max_speakers"] = settings.diarization_max_speakers

        try:
            output = pipeline(str(audio_path), **kwargs)
        except Exception as exc:
            if _is_oom_error(exc):
                raise GpuOutOfMemoryError("Out of memory while running diarization.") from exc
            raise DiarizationError(f"Diarization failed for '{audio_path.name}': {exc}") from exc

        annotation = _extract_speaker_diarization_annotation(output)

        turns: list[DiarizationTurn] = []
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(DiarizationTurn(speaker=speaker, start=float(segment.start), end=float(segment.end)))

        turns.sort(key=lambda t: t.start)
        speaker_labels = sorted({t.speaker for t in turns})
        return DiarizationResult(turns=turns, speaker_labels=speaker_labels)


def build_speaker_name_mapping(diarization_result: DiarizationResult) -> dict[str, str]:
    """Build the single, stable `SPEAKER_00 -> "Person 1"` mapping for a
    whole meeting.

    Speakers are numbered in order of first appearance (earliest turn
    start), which is deterministic and reproducible from the diarization
    output alone — calling this twice on the same `DiarizationResult`
    always yields the same mapping.
    """
    first_appearance: dict[str, float] = {}
    for turn in diarization_result.turns:
        if turn.speaker not in first_appearance or turn.start < first_appearance[turn.speaker]:
            first_appearance[turn.speaker] = turn.start

    ordered_speakers = sorted(first_appearance, key=lambda spk: first_appearance[spk])
    return {speaker: f"Person {i + 1}" for i, speaker in enumerate(ordered_speakers)}
