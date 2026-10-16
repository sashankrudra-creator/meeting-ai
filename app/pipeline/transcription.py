"""
Speech-to-text stage (Section 4).

    Transcriber (interface)
        |
        +-- FasterWhisperTranscriber

The interface exists so a different local ASR engine could be swapped in
later (Section 24) without touching the orchestrator. Only one production
implementation is provided, per the approved architecture: faster-whisper
running `large-v3` (configurable — see app/config.py).

No paid/external LLM or ASR API is used anywhere in this module. All
inference is local; only the model *weights* are fetched (once, and
cached) from the Hugging Face Hub the first time a given model name is
used, exactly as the spec allows ("Hugging Face model downloads are
acceptable where the actual inference runs locally").

Known limitation (documented, not silently swept under the rug): Whisper's
architecture detects language once from a ~30s window and decodes the
whole file in that language. For genuinely code-switched audio this is
lossy at the whole-file level, which is why this module supports an
optional, opt-in `whisper_segment_level_language_detection` mode that
re-runs a fast language-ID pass per detected speech segment. It is a
heuristic improvement, not a perfect solution — see the README's
"Known Limitations" section.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models.schemas import RawTranscriptSegment, TranscriptionResult
from app.utils.errors import GpuOutOfMemoryError, ModelLoadingError, TranscriptionError
from app.utils.logging import get_logger
from app.utils.timeout import run_with_timeout

logger = get_logger("pipeline.transcription")


class Transcriber(ABC):
    """Abstraction over "an engine that turns an audio file into
    timestamped text segments with a detected language"."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        raise NotImplementedError


def _is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message or "cudnn" in message and "memory" in message


def _is_low_confidence_segment(
    avg_logprob: float | None,
    no_speech_prob: float | None,
    settings: Settings,
) -> bool:
    """ASR quality/hallucination heuristic (Section 4/18): a segment is
    considered low-confidence if faster-whisper's own `no_speech_prob` is
    unusually high (it thought this window was probably silence/non-speech)
    or its `avg_logprob` is unusually low (it wasn't confident in the
    decoded text) — both are standard Whisper hallucination signals.
    Missing values (older faster-whisper versions, or a fake/test model)
    never count as low-confidence rather than raising."""
    if no_speech_prob is not None and no_speech_prob >= settings.asr_no_speech_prob_threshold:
        return True
    return avg_logprob is not None and avg_logprob <= settings.asr_avg_logprob_threshold


class FasterWhisperTranscriber(Transcriber):
    """Production transcriber backed by `faster-whisper`.

    The underlying `WhisperModel` is loaded lazily (first call to
    `transcribe`) and cached on the instance, since loading `large-v3` is
    expensive (seconds to tens of seconds, plus a first-time multi-GB
    download) and a single orchestrator run should only pay that cost once.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model: Any | None = None
        # Guards the lazy-load below against concurrent callers (the
        # background worker pool can run more than one job thread against
        # this same, shared Transcriber instance) — see the identical
        # comment on PyannoteDiarizer._load_lock.
        self._load_lock = threading.Lock()

    def _instantiate_whisper_model(self, *, local_files_only: bool) -> Any:
        from faster_whisper import WhisperModel

        settings = self._settings
        return WhisperModel(
            settings.whisper_model_name,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            cpu_threads=settings.whisper_cpu_threads or 0,
            num_workers=settings.whisper_num_workers,
            download_root=(
                str(settings.whisper_download_root) if settings.whisper_download_root else None
            ),
            local_files_only=local_files_only,
        )

    def _load_model_cache_first(self, settings: Settings) -> Any:
        """Load the model from the local Hugging Face cache first
        (`local_files_only=True`, which touches no network at all), only
        falling back to an online load if that fails (e.g. the model
        hasn't been downloaded yet).

        This is the fix for a real production incident: `WhisperModel(...)`
        without `local_files_only=True` always makes `huggingface_hub` do an
        online Hub check before falling back to the cache — even when the
        model is *already fully cached*. On a machine/network where that
        check hangs instead of failing fast, this silently blocked the sole
        background-worker thread forever: near-zero CPU, no exception ever
        raised, nothing logged, and the meeting stuck at TRANSCRIBING with
        no way to recover. Trying the cache first means the common case
        (a model already downloaded from a previous run) never touches the
        network at all, and a genuinely first-time download still works via
        the fallback below.
        """
        try:
            model = run_with_timeout(
                lambda: self._instantiate_whisper_model(local_files_only=True),
                settings.model_load_timeout_seconds,
                f"Loading faster-whisper model '{settings.whisper_model_name}' from local cache",
            )
        except TimeoutError:
            # local_files_only=True never touches the network, so a timeout
            # here means something else is unusually slow (e.g. disk I/O) -
            # don't mask it by silently falling back to an online load.
            raise
        except Exception as exc:
            logger.info(
                "faster-whisper model '%s' is not available in the local cache (%s); "
                "falling back to an online load (this will download the model if needed)",
                settings.whisper_model_name, exc,
            )
        else:
            logger.info(
                "Loaded faster-whisper model '%s' from local cache (no network access)",
                settings.whisper_model_name,
            )
            return model

        return run_with_timeout(
            lambda: self._instantiate_whisper_model(local_files_only=False),
            settings.model_load_timeout_seconds,
            f"Loading faster-whisper model '{settings.whisper_model_name}' online",
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:  # double-checked locking
                return self._model

            try:
                from faster_whisper import WhisperModel  # noqa: F401 - import check only
            except ImportError as exc:  # pragma: no cover - environment issue, not logic
                raise ModelLoadingError(
                    "The 'faster-whisper' package is not installed. Run "
                    "`pip install -r requirements.txt`."
                ) from exc

            settings = self._settings
            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute_type=%s)",
                settings.whisper_model_name, settings.whisper_device, settings.whisper_compute_type,
            )
            try:
                self._model = self._load_model_cache_first(settings)
            except TimeoutError as exc:
                raise ModelLoadingError(
                    f"Timed out loading faster-whisper model '{settings.whisper_model_name}' "
                    f"after {settings.model_load_timeout_seconds:.0f}s: {exc} This usually means "
                    "a network call to the Hugging Face Hub is hanging rather than failing fast "
                    "(check network connectivity), even if the model is already cached locally. "
                    "See `Settings.model_load_timeout_seconds`."
                ) from exc
            except Exception as exc:
                if _is_oom_error(exc):
                    raise GpuOutOfMemoryError(
                        f"Out of memory while loading Whisper model '{settings.whisper_model_name}'. "
                        "Try a smaller model_name or compute_type, or run on CPU."
                    ) from exc
                raise ModelLoadingError(
                    f"Failed to load faster-whisper model '{settings.whisper_model_name}': {exc}"
                ) from exc
            return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        model = self._load_model()
        settings = self._settings

        try:
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=settings.whisper_language,
                beam_size=settings.whisper_beam_size,
                vad_filter=settings.whisper_vad_filter,
            )
            raw_segments: list[RawTranscriptSegment] = []
            filtered_segment_count = 0
            flagged_segment_count = 0
            filter_mode = settings.asr_quality_filter_mode
            for seg in segments_iter:
                text = (seg.text or "").strip()
                if not text:
                    continue
                avg_logprob = getattr(seg, "avg_logprob", None)
                no_speech_prob = getattr(seg, "no_speech_prob", None)

                low_confidence = False
                if filter_mode != "off":
                    low_confidence = _is_low_confidence_segment(avg_logprob, no_speech_prob, settings)
                    if low_confidence and filter_mode == "filter":
                        filtered_segment_count += 1
                        logger.debug(
                            "Dropping low-confidence segment [%.2f-%.2f]s "
                            "(avg_logprob=%s, no_speech_prob=%s)",
                            seg.start, seg.end, avg_logprob, no_speech_prob,
                        )
                        continue
                    if low_confidence:
                        flagged_segment_count += 1

                raw_segments.append(
                    RawTranscriptSegment(
                        start=float(seg.start),
                        end=float(seg.end),
                        text=text,
                        language=info.language,
                        avg_logprob=avg_logprob,
                        no_speech_prob=no_speech_prob,
                        low_confidence=low_confidence,
                    )
                )
        except Exception as exc:
            if _is_oom_error(exc):
                raise GpuOutOfMemoryError(
                    "Out of memory while transcribing. Try a smaller model_name/compute_type."
                ) from exc
            raise TranscriptionError(f"Transcription failed for '{audio_path.name}': {exc}") from exc

        if settings.whisper_segment_level_language_detection and raw_segments:
            raw_segments = self._redetect_segment_languages(model, audio_path, raw_segments)

        if filtered_segment_count or flagged_segment_count:
            logger.info(
                "ASR quality filtering (mode=%s): %d segment(s) dropped, %d segment(s) "
                "flagged as low-confidence",
                filter_mode, filtered_segment_count, flagged_segment_count,
            )

        detected_languages = sorted({s.language for s in raw_segments if s.language} or {info.language})
        return TranscriptionResult(
            segments=raw_segments,
            detected_languages=detected_languages,
            duration_seconds=float(getattr(info, "duration", 0.0) or 0.0),
            filtered_segment_count=filtered_segment_count,
            flagged_segment_count=flagged_segment_count,
        )

    def _redetect_segment_languages(
        self,
        model: Any,
        audio_path: Path,
        segments: list[RawTranscriptSegment],
    ) -> list[RawTranscriptSegment]:
        """Best-effort per-segment language re-detection for code-switched audio.

        Loads the full waveform once, slices it per segment, and asks the
        model to identify (not re-transcribe) the language of each slice.
        Any failure for an individual segment falls back to that segment's
        already-assigned (file-level) language rather than raising —
        language tagging is an enhancement, not something that should ever
        take down the whole transcription stage.
        """
        try:
            from faster_whisper.audio import decode_audio
        except ImportError:
            logger.warning("decode_audio unavailable; skipping per-segment language detection")
            return segments

        settings = self._settings
        try:
            waveform = decode_audio(str(audio_path), sampling_rate=16000)
        except Exception as exc:
            logger.warning("Could not decode audio for per-segment language detection: %s", exc)
            return segments

        sample_rate = 16000
        for seg in segments:
            duration = seg.end - seg.start
            if duration < settings.whisper_segment_language_min_duration_seconds:
                continue
            start_sample = max(0, int(seg.start * sample_rate))
            end_sample = min(len(waveform), int(seg.end * sample_rate))
            if end_sample <= start_sample:
                continue
            clip = waveform[start_sample:end_sample]
            try:
                language, probability, _ = model.detect_language(clip)
                if probability and probability >= 0.5:
                    seg.language = language
            except Exception as exc:  # noqa: BLE001 - best-effort enhancement
                logger.debug("Per-segment language detection failed for one segment: %s", exc)
                continue
        return segments
