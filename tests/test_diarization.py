"""
Unit tests for app.pipeline.diarization.

Honesty note: like test_transcription.py, these stub out the pyannote
`Pipeline` object so the turn-extraction, sorting, and speaker-name-mapping
LOGIC is verified deterministically and offline. Real diarization against
`pyannote/speaker-diarization-community-1` requires a Hugging Face token
and network access this sandbox does not have, and has not been run here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.models.schemas import DiarizationResult, DiarizationTurn
from app.pipeline.diarization import PyannoteDiarizer, build_speaker_name_mapping
from app.utils.errors import DiarizationError, GpuOutOfMemoryError, ModelLoadingError


@dataclass
class _FakeSegment:
    start: float
    end: float


class _FakeAnnotation:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=True):
        return iter(self._tracks)


class _FakePipeline:
    def __init__(self, annotation=None, raise_exc: Exception | None = None):
        self._annotation = annotation
        self._raise_exc = raise_exc

    def __call__(self, path, **kwargs):
        if self._raise_exc:
            raise self._raise_exc
        return self._annotation


def _make_diarizer(settings, fake_pipeline: _FakePipeline) -> PyannoteDiarizer:
    d = PyannoteDiarizer(settings)
    d._pipeline = fake_pipeline
    return d


def test_diarize_extracts_and_sorts_turns(settings):
    annotation = _FakeAnnotation(
        [
            (_FakeSegment(5.0, 8.0), "trackA", "SPEAKER_01"),
            (_FakeSegment(0.0, 4.5), "trackB", "SPEAKER_00"),
            (_FakeSegment(4.5, 5.0), "trackC", "SPEAKER_01"),
        ]
    )
    diarizer = _make_diarizer(settings, _FakePipeline(annotation=annotation))
    result = diarizer.diarize(Path("dummy.wav"))

    assert [t.speaker for t in result.turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]
    assert result.turns[0].start == 0.0
    assert result.speaker_labels == ["SPEAKER_00", "SPEAKER_01"]


def test_diarize_wraps_generic_exception(settings, tmp_path):
    diarizer = _make_diarizer(settings, _FakePipeline(raise_exc=RuntimeError("boom")))
    with pytest.raises(DiarizationError):
        diarizer.diarize(tmp_path / "audio.wav")


def test_diarize_wraps_oom_exception(settings, tmp_path):
    diarizer = _make_diarizer(
        settings, _FakePipeline(raise_exc=RuntimeError("CUDA out of memory"))
    )
    with pytest.raises(GpuOutOfMemoryError):
        diarizer.diarize(tmp_path / "audio.wav")


# ---------------------------------------------------------------------------
# Regression: production incident — pyannote.audio 4.x's `SpeakerDiarization`
# pipeline returns a `DiarizeOutput` dataclass (speaker_diarization /
# exclusive_speaker_diarization / speaker_embeddings), not a bare
# `pyannote.core.Annotation`, so calling `.itertracks()` directly on the
# pipeline's return value raised `AttributeError: 'DiarizeOutput' object has
# no attribute 'itertracks'` (confirmed against the real, installed
# pyannote.audio==4.0.7 source). These tests construct the REAL
# `DiarizeOutput`/`Annotation` classes (cheap dataclasses, no model weights
# needed) rather than a hand-rolled fake, so a future upstream change to
# `DiarizeOutput`'s shape would fail these tests instead of silently
# breaking only in production.
#
# `test_diarize_extracts_and_sorts_turns` above (using `_FakeAnnotation`,
# which has no `speaker_diarization` attribute) already covers the other
# supported shape: a bare `Annotation`, returned when a pipeline is
# constructed with `legacy=True` or by older pyannote.audio versions.
# ---------------------------------------------------------------------------
def test_diarize_unwraps_real_diarize_output_speaker_diarization(settings, tmp_path):
    pytest.importorskip("pyannote.audio")
    from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
    from pyannote.core import Annotation, Segment

    speaker_diarization = Annotation()
    speaker_diarization[Segment(5.0, 8.0)] = "SPEAKER_01"
    speaker_diarization[Segment(0.0, 4.5)] = "SPEAKER_00"

    output = DiarizeOutput(
        speaker_diarization=speaker_diarization,
        exclusive_speaker_diarization=Annotation(),
    )
    diarizer = _make_diarizer(settings, _FakePipeline(annotation=output))
    result = diarizer.diarize(tmp_path / "audio.wav")

    assert [t.speaker for t in result.turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert result.turns[0].start == 0.0
    assert result.speaker_labels == ["SPEAKER_00", "SPEAKER_01"]


def test_diarize_uses_speaker_diarization_not_exclusive_variant(settings, tmp_path):
    """`exclusive_speaker_diarization` has overlapping speech turns removed;
    using it instead of `speaker_diarization` would silently discard overlap
    information that app/pipeline/alignment.py's `overlapping_speakers`
    depends on."""
    pytest.importorskip("pyannote.audio")
    from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
    from pyannote.core import Annotation, Segment

    with_overlap = Annotation()
    with_overlap[Segment(0.0, 2.0)] = "SPEAKER_00"
    with_overlap[Segment(1.5, 3.0)] = "SPEAKER_01"  # overlaps the turn above

    overlap_removed = Annotation()
    overlap_removed[Segment(0.0, 1.5)] = "SPEAKER_00"

    output = DiarizeOutput(
        speaker_diarization=with_overlap,
        exclusive_speaker_diarization=overlap_removed,
    )
    diarizer = _make_diarizer(settings, _FakePipeline(annotation=output))
    result = diarizer.diarize(tmp_path / "audio.wav")

    assert len(result.turns) == 2  # from speaker_diarization, not the 1-turn exclusive variant
    assert result.speaker_labels == ["SPEAKER_00", "SPEAKER_01"]


def test_diarize_handles_empty_diarize_output(settings, tmp_path):
    """Real pyannote returns an empty `speaker_diarization` Annotation (not
    None, not an error) when no speaker is ever detected as active — must
    not raise and must produce an empty, valid result."""
    pytest.importorskip("pyannote.audio")
    from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
    from pyannote.core import Annotation

    output = DiarizeOutput(speaker_diarization=Annotation(), exclusive_speaker_diarization=Annotation())
    diarizer = _make_diarizer(settings, _FakePipeline(annotation=output))
    result = diarizer.diarize(tmp_path / "audio.wav")

    assert result.turns == []
    assert result.speaker_labels == []


def test_build_speaker_name_mapping_orders_by_first_appearance():
    result = DiarizationResult(
        turns=[
            DiarizationTurn(speaker="SPEAKER_02", start=10.0, end=12.0),
            DiarizationTurn(speaker="SPEAKER_00", start=0.0, end=2.0),
            DiarizationTurn(speaker="SPEAKER_01", start=5.0, end=6.0),
            DiarizationTurn(speaker="SPEAKER_00", start=20.0, end=22.0),
        ],
        speaker_labels=["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
    )
    mapping = build_speaker_name_mapping(result)
    assert mapping == {
        "SPEAKER_00": "Person 1",
        "SPEAKER_01": "Person 2",
        "SPEAKER_02": "Person 3",
    }


def test_build_speaker_name_mapping_is_deterministic_and_reproducible():
    result = DiarizationResult(
        turns=[
            DiarizationTurn(speaker="SPEAKER_05", start=1.0, end=2.0),
            DiarizationTurn(speaker="SPEAKER_01", start=3.0, end=4.0),
        ],
        speaker_labels=["SPEAKER_01", "SPEAKER_05"],
    )
    mapping1 = build_speaker_name_mapping(result)
    mapping2 = build_speaker_name_mapping(result)
    assert mapping1 == mapping2 == {"SPEAKER_05": "Person 1", "SPEAKER_01": "Person 2"}


def test_build_speaker_name_mapping_empty_result():
    assert build_speaker_name_mapping(DiarizationResult(turns=[], speaker_labels=[])) == {}


# ---------------------------------------------------------------------------
# Regression: C2 — pyannote.audio 4.x's Pipeline.from_pretrained() takes
# `token=`, not `use_auth_token=` (that kwarg was removed upstream and now
# raises TypeError — confirmed by actually calling the installed
# `pyannote.audio==4.0.7` in this environment). This test uses a fake
# `Pipeline` whose `from_pretrained` has that same real signature (no
# `**kwargs` catch-all), so it fails exactly the way the real library does
# if diarization.py ever regresses back to `use_auth_token=`.
# ---------------------------------------------------------------------------
class _StrictFakePipelineClass:
    """Mimics pyannote.audio==4.0.7's real `Pipeline.from_pretrained`
    signature: `token=`, no `use_auth_token`, and no **kwargs catch-all —
    an unexpected keyword raises TypeError exactly like the real library."""

    load_calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, checkpoint, revision=None, hparams_file=None, subfolder=None, token=None, cache_dir=None):
        cls.load_calls.append({"checkpoint": checkpoint, "token": token})
        return cls()


def test_load_pipeline_calls_from_pretrained_with_token_kwarg(settings, monkeypatch):
    pytest.importorskip("pyannote.audio")
    settings.huggingface_token = "fake-hf-token"
    _StrictFakePipelineClass.load_calls = []
    monkeypatch.setattr("pyannote.audio.Pipeline", _StrictFakePipelineClass)

    diarizer = PyannoteDiarizer(settings)
    pipeline = diarizer._load_pipeline()

    assert isinstance(pipeline, _StrictFakePipelineClass)
    assert _StrictFakePipelineClass.load_calls == [
        {"checkpoint": settings.diarization_model_name, "token": "fake-hf-token"}
    ]


# ---------------------------------------------------------------------------
# Regression: H3 — lazy pipeline loading must be thread-safe. Several
# background-worker threads could call `_load_pipeline()` concurrently on
# the same shared PyannoteDiarizer the first time it's used; without
# locking, more than one thread could pay the (expensive) load cost.
# ---------------------------------------------------------------------------
def test_load_pipeline_is_thread_safe_and_loads_only_once(settings, monkeypatch):
    pytest.importorskip("pyannote.audio")
    call_count = {"n": 0}
    count_lock = threading.Lock()

    class SlowFakePipelineClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            with count_lock:
                call_count["n"] += 1
            time.sleep(0.05)  # widen the race window
            return cls()

    monkeypatch.setattr("pyannote.audio.Pipeline", SlowFakePipelineClass)

    diarizer = PyannoteDiarizer(settings)
    threads = [threading.Thread(target=diarizer._load_pipeline) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# Regression: defense-in-depth against the same class of hang confirmed in
# app/pipeline/transcription.py (a hanging, not fast-failing, Hugging Face
# Hub network call). pyannote's `Pipeline.from_pretrained()` has no
# `local_files_only` option, so a bounded timeout is the guard here — any
# future hang must surface as a structured ModelLoadingError, not block the
# background worker forever.
# ---------------------------------------------------------------------------
def test_load_pipeline_raises_model_loading_error_when_load_hangs(settings, monkeypatch):
    pytest.importorskip("pyannote.audio")
    settings.model_load_timeout_seconds = 0.05

    class HangingPipelineClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            time.sleep(1.0)  # simulate a hanging online Hugging Face Hub call
            return cls()

    monkeypatch.setattr("pyannote.audio.Pipeline", HangingPipelineClass)

    diarizer = PyannoteDiarizer(settings)
    with pytest.raises(ModelLoadingError, match="Timed out"):
        diarizer._load_pipeline()
