"""
Unit tests for app.pipeline.transcription.

IMPORTANT / honesty note: these tests stub out `FasterWhisperTranscriber._load_model`
so that the segment-building, filtering and error-translation LOGIC in this module is
verified deterministically and offline. They do NOT download or run the real
`large-v3` Whisper model — this sandbox has no route to huggingface.co (see
README "Known Limitations"), so actually loading real weights and transcribing
real English/Hindi/Odia speech has not been executed or verified in this session.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

from app.pipeline.transcription import FasterWhisperTranscriber
from app.utils.errors import GpuOutOfMemoryError, ModelLoadingError, TranscriptionError


@dataclass
class _FakeSegment:
    start: float
    end: float
    text: str
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.01


@dataclass
class _FakeInfo:
    language: str
    duration: float = 10.0


class _FakeModel:
    def __init__(self, segments, info, raise_exc: Exception | None = None):
        self._segments = segments
        self._info = info
        self._raise_exc = raise_exc

    def transcribe(self, path, language=None, beam_size=5, vad_filter=True):
        if self._raise_exc:
            raise self._raise_exc
        return iter(self._segments), self._info

    def detect_language(self, clip):
        # Alternate languages to simulate a code-switched clip.
        return "hi", 0.9, None


def _make_transcriber(settings, fake_model: _FakeModel) -> FasterWhisperTranscriber:
    t = FasterWhisperTranscriber(settings)
    t._model = fake_model  # bypass real model loading entirely
    return t


def test_transcribe_builds_segments_and_detected_language(settings, tmp_path):
    fake = _FakeModel(
        segments=[
            _FakeSegment(0.0, 2.0, "Hello everyone"),
            _FakeSegment(2.0, 4.5, "  namaste sabko  "),
        ],
        info=_FakeInfo(language="en", duration=4.5),
    )
    transcriber = _make_transcriber(settings, fake)
    result = transcriber.transcribe(tmp_path / "audio.wav")

    assert len(result.segments) == 2
    assert result.segments[0].text == "Hello everyone"
    assert result.segments[1].text == "namaste sabko"  # whitespace stripped
    assert result.detected_languages == ["en"]
    assert result.duration_seconds == 4.5


def test_transcribe_skips_empty_text_segments(settings, tmp_path):
    fake = _FakeModel(
        segments=[
            _FakeSegment(0.0, 1.0, "   "),
            _FakeSegment(1.0, 2.0, "real text"),
        ],
        info=_FakeInfo(language="en"),
    )
    transcriber = _make_transcriber(settings, fake)
    result = transcriber.transcribe(tmp_path / "audio.wav")
    assert len(result.segments) == 1
    assert result.segments[0].text == "real text"


def test_transcribe_wraps_generic_exception_as_transcription_error(settings, tmp_path):
    fake = _FakeModel(segments=[], info=_FakeInfo(language="en"), raise_exc=RuntimeError("boom"))
    transcriber = _make_transcriber(settings, fake)
    with pytest.raises(TranscriptionError):
        transcriber.transcribe(tmp_path / "audio.wav")


def test_transcribe_wraps_oom_exception_as_gpu_oom_error(settings, tmp_path):
    fake = _FakeModel(
        segments=[], info=_FakeInfo(language="en"),
        raise_exc=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
    )
    transcriber = _make_transcriber(settings, fake)
    with pytest.raises(GpuOutOfMemoryError):
        transcriber.transcribe(tmp_path / "audio.wav")


def test_segment_level_language_detection_enabled(settings, tmp_path, monkeypatch):
    settings.whisper_segment_level_language_detection = True
    settings.whisper_segment_language_min_duration_seconds = 0.1

    fake = _FakeModel(
        segments=[_FakeSegment(0.0, 2.0, "yeh deployment kal ho jayega")],
        info=_FakeInfo(language="en", duration=2.0),
    )
    transcriber = _make_transcriber(settings, fake)

    import numpy as np

    monkeypatch.setattr(
        "faster_whisper.audio.decode_audio",
        lambda path, sampling_rate=16000: np.zeros(32000, dtype="float32"),
    )

    result = transcriber.transcribe(tmp_path / "audio.wav")
    assert result.segments[0].language == "hi"  # overridden by fake detect_language()


def test_load_model_is_thread_safe_and_loads_only_once(settings, monkeypatch):
    """Regression for H3: concurrent callers of `_load_model()` (the
    background worker pool can run more than one job thread against the
    same shared Transcriber instance) must only trigger one real model
    load."""
    pytest.importorskip("faster_whisper")
    call_count = {"n": 0}
    count_lock = threading.Lock()

    class SlowFakeModel:
        def __init__(self, *args, **kwargs):
            with count_lock:
                call_count["n"] += 1
            time.sleep(0.05)  # widen the race window

    monkeypatch.setattr("faster_whisper.WhisperModel", SlowFakeModel)

    transcriber = FasterWhisperTranscriber(settings)
    threads = [threading.Thread(target=transcriber._load_model) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# Regression: production incident — WhisperModel(...) without
# local_files_only=True always attempts an online Hugging Face Hub check
# before falling back to the local cache, even when the model is already
# fully cached. On a network where that check hangs instead of failing
# fast, this left real meetings stuck at TRANSCRIBING forever (near-zero
# CPU, no exception, no logs). Fix: try local_files_only=True first (no
# network at all); fall back to an online load only if that fails; bound
# both attempts with a timeout so a future hang surfaces as a structured
# ModelLoadingError instead of hanging indefinitely.
# ---------------------------------------------------------------------------
def test_load_model_uses_local_cache_first_and_never_goes_online_when_cached(settings, monkeypatch):
    pytest.importorskip("faster_whisper")
    calls: list[bool | None] = []

    class FakeModel:
        def __init__(self, model_name, **kwargs):
            calls.append(kwargs.get("local_files_only"))

    monkeypatch.setattr("faster_whisper.WhisperModel", FakeModel)

    transcriber = FasterWhisperTranscriber(settings)
    model = transcriber._load_model()

    assert isinstance(model, FakeModel)
    assert calls == [True]  # only the local-cache attempt ran; no online fallback needed


def test_load_model_falls_back_to_online_load_when_not_cached(settings, monkeypatch):
    pytest.importorskip("faster_whisper")
    calls: list[bool | None] = []

    class FakeModel:
        def __init__(self, model_name, **kwargs):
            local_files_only = kwargs.get("local_files_only")
            calls.append(local_files_only)
            if local_files_only:
                raise FileNotFoundError("model not cached locally")

    monkeypatch.setattr("faster_whisper.WhisperModel", FakeModel)

    transcriber = FasterWhisperTranscriber(settings)
    model = transcriber._load_model()

    assert isinstance(model, FakeModel)
    assert calls == [True, False]  # cache-only attempt failed, fell back to an online load


def test_load_model_raises_model_loading_error_when_load_hangs(settings, monkeypatch):
    """The specific fix for the production hang: even if some future load
    path hangs again (network or otherwise), it must surface as a
    structured ModelLoadingError, not block forever."""
    pytest.importorskip("faster_whisper")
    settings.model_load_timeout_seconds = 0.05

    class HangingModel:
        def __init__(self, model_name, **kwargs):
            if kwargs.get("local_files_only"):
                raise FileNotFoundError("model not cached locally")
            time.sleep(1.0)  # simulate a hanging online Hugging Face Hub call

    monkeypatch.setattr("faster_whisper.WhisperModel", HangingModel)

    transcriber = FasterWhisperTranscriber(settings)
    with pytest.raises(ModelLoadingError, match="Timed out"):
        transcriber._load_model()


# ---------------------------------------------------------------------------
# Regression: H8 — configurable ASR quality/hallucination handling via
# avg_logprob/no_speech_prob. Default mode is 'flag': never silently
# deletes legitimate speech.
# ---------------------------------------------------------------------------
def test_flag_mode_marks_low_confidence_segments_without_dropping_them(settings, tmp_path):
    settings.asr_quality_filter_mode = "flag"
    settings.asr_no_speech_prob_threshold = 0.6
    settings.asr_avg_logprob_threshold = -1.0
    fake = _FakeModel(
        segments=[
            _FakeSegment(0.0, 2.0, "clear speech", avg_logprob=-0.1, no_speech_prob=0.02),
            _FakeSegment(2.0, 4.0, "likely hallucination", avg_logprob=-1.5, no_speech_prob=0.9),
        ],
        info=_FakeInfo(language="en"),
    )
    transcriber = _make_transcriber(settings, fake)
    result = transcriber.transcribe(tmp_path / "audio.wav")

    assert len(result.segments) == 2  # flag mode never drops a segment
    assert result.segments[0].low_confidence is False
    assert result.segments[1].low_confidence is True
    assert result.flagged_segment_count == 1
    assert result.filtered_segment_count == 0


def test_filter_mode_drops_low_confidence_segments_and_counts_them(settings, tmp_path):
    settings.asr_quality_filter_mode = "filter"
    fake = _FakeModel(
        segments=[
            _FakeSegment(0.0, 2.0, "clear speech", avg_logprob=-0.1, no_speech_prob=0.02),
            _FakeSegment(2.0, 4.0, "likely hallucination", avg_logprob=-1.5, no_speech_prob=0.9),
        ],
        info=_FakeInfo(language="en"),
    )
    transcriber = _make_transcriber(settings, fake)
    result = transcriber.transcribe(tmp_path / "audio.wav")

    assert len(result.segments) == 1
    assert result.segments[0].text == "clear speech"
    assert result.filtered_segment_count == 1
    assert result.flagged_segment_count == 0


def test_off_mode_preserves_original_behavior_with_no_flagging_or_filtering(settings, tmp_path):
    settings.asr_quality_filter_mode = "off"
    fake = _FakeModel(
        segments=[_FakeSegment(0.0, 2.0, "anything", avg_logprob=-5.0, no_speech_prob=0.99)],
        info=_FakeInfo(language="en"),
    )
    transcriber = _make_transcriber(settings, fake)
    result = transcriber.transcribe(tmp_path / "audio.wav")

    assert len(result.segments) == 1
    assert result.segments[0].low_confidence is False
    assert result.filtered_segment_count == 0
    assert result.flagged_segment_count == 0


def test_segment_level_language_detection_falls_back_on_short_segment(settings, tmp_path, monkeypatch):
    settings.whisper_segment_level_language_detection = True
    settings.whisper_segment_language_min_duration_seconds = 5.0  # segment below this stays as-is

    fake = _FakeModel(
        segments=[_FakeSegment(0.0, 1.0, "hi")],
        info=_FakeInfo(language="en", duration=1.0),
    )
    transcriber = _make_transcriber(settings, fake)

    import numpy as np

    monkeypatch.setattr(
        "faster_whisper.audio.decode_audio",
        lambda path, sampling_rate=16000: np.zeros(16000, dtype="float32"),
    )

    result = transcriber.transcribe(tmp_path / "audio.wav")
    assert result.segments[0].language == "en"  # unchanged: too short to re-detect
