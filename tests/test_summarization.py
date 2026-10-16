"""
Tests for app.pipeline.summarization.

Honesty note: `LocalLLMSummarizer._generate` (the actual local-LLM inference
call) is monkeypatched here so the CHUNKING, JSON-parsing, de-duplication
and hierarchical-reduce LOGIC is verified deterministically and offline.
Real generation quality from `Qwen/Qwen2.5-1.5B-Instruct` (or any other
configured model) requires downloading weights from huggingface.co, which
this sandbox cannot reach, and has not been exercised in this session.
`ExtractiveSummarizer`, by contrast, needs no model at all and IS exercised
for real below, end-to-end.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from app.models.schemas import TranscriptSegment
from app.pipeline.summarization import (
    ExtractiveSummarizer,
    LocalLLMSummarizer,
    _dedupe_preserve_order,
    _estimate_tokens,
    _extract_json_object,
    build_summarizer,
    chunk_transcript_for_summarization,
    format_chunk_as_text,
)
from app.utils.errors import ModelLoadingError, SummarizationError


def _seg(seg_id, name, text, start=None, end=None, speaker_id=None):
    start = seg_id * 1.0 if start is None else start
    end = start + 1.0 if end is None else end
    return TranscriptSegment(
        segment_id=seg_id, speaker_id=speaker_id or f"SPEAKER_{seg_id % 2:02d}",
        speaker_name=name, start=start, end=end, text=text, language="en",
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def test_chunking_empty_transcript_returns_no_chunks():
    assert chunk_transcript_for_summarization([], token_budget=100) == []


def test_chunking_respects_token_budget_and_never_splits_a_segment():
    segments = [_seg(i, "Person 1", "word " * 20) for i in range(10)]  # ~26 tokens each
    chunks = chunk_transcript_for_summarization(segments, token_budget=60, overlap_segments=0)

    assert len(chunks) > 1
    # every original segment appears exactly once across all chunks combined
    # (ignoring overlap, which is 0 here)
    all_ids = [s.segment_id for chunk in chunks for s in chunk]
    assert all_ids == [s.segment_id for s in segments]


def test_chunking_oversized_single_segment_gets_its_own_chunk():
    huge = _seg(0, "Person 1", "word " * 500)
    normal = _seg(1, "Person 2", "short reply")
    chunks = chunk_transcript_for_summarization([huge, normal], token_budget=50, overlap_segments=0)
    assert len(chunks) == 2
    assert chunks[0] == [huge]
    assert chunks[1] == [normal]


def test_chunking_overlap_repeats_trailing_segments():
    segments = [_seg(i, "Person 1", "word " * 15) for i in range(6)]
    chunks = chunk_transcript_for_summarization(segments, token_budget=25, overlap_segments=1)
    assert len(chunks) >= 2
    # the last segment of chunk N should reappear as the first segment of chunk N+1
    for i in range(len(chunks) - 1):
        assert chunks[i][-1].segment_id == chunks[i + 1][0].segment_id


def test_format_chunk_as_text_preserves_speaker_labels():
    segments = [_seg(0, "Person 1", "hello"), _seg(1, "Person 2", "hi there")]
    text = format_chunk_as_text(segments)
    assert text == "Person 1: hello\nPerson 2: hi there"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def test_extract_json_object_from_clean_json():
    result = _extract_json_object('{"key_points": ["a"], "decisions": []}')
    assert result == {"key_points": ["a"], "decisions": []}


def test_extract_json_object_from_noisy_llm_output():
    noisy = 'Sure, here is the JSON:\n{"key_points": ["a", "b"]}\nHope that helps!'
    result = _extract_json_object(noisy)
    assert result == {"key_points": ["a", "b"]}


def test_extract_json_object_returns_none_when_absent():
    assert _extract_json_object("no json here at all") is None


def test_dedupe_preserve_order_case_insensitive():
    assert _dedupe_preserve_order(["Deploy tomorrow", "deploy tomorrow", "Ship it"]) == [
        "Deploy tomorrow", "Ship it",
    ]


# ---------------------------------------------------------------------------
# ExtractiveSummarizer — runs for real, no mocking, no model download
# ---------------------------------------------------------------------------
def test_extractive_summarizer_empty_transcript(settings):
    summary = ExtractiveSummarizer(settings).summarize([])
    assert summary.overview == ""
    assert summary.key_points == []


def test_extractive_summarizer_detects_decisions_and_action_items(settings):
    transcript = [
        _seg(0, "Person 1", "We decided to launch the feature next Friday for all users"),
        _seg(1, "Person 2", "Person 2 will complete the API testing by Thursday"),
        _seg(2, "Person 1", "The weather has been pretty unpredictable lately this month"),
    ]
    summary = ExtractiveSummarizer(settings).summarize(transcript)

    assert any("launch the feature" in d for d in summary.decisions)
    assert any("API testing" in a.task for a in summary.action_items)
    assert any("weather" in k for k in summary.key_points)
    assert "2 speaker" in summary.overview


def test_build_summarizer_factory_selects_extractive(settings):
    settings.summarizer_backend = "extractive"
    assert isinstance(build_summarizer(settings), ExtractiveSummarizer)


def test_build_summarizer_factory_defaults_to_local_llm(settings):
    settings.summarizer_backend = "local_llm"
    assert isinstance(build_summarizer(settings), LocalLLMSummarizer)


# ---------------------------------------------------------------------------
# LocalLLMSummarizer — generation is mocked, hierarchical combine logic is real
# ---------------------------------------------------------------------------
def test_local_llm_summarizer_hierarchical_flow(settings, monkeypatch):
    settings.summarizer_chunk_token_budget = 10  # force multiple chunks
    settings.summarizer_chunk_overlap_segments = 0

    segments = [
        _seg(0, "Person 1", "We should deploy tomorrow morning"),
        _seg(1, "Person 2", "Agreed, Person 2 will handle the deployment"),
        _seg(2, "Person 1", "Let's also review the budget next week"),
        _seg(3, "Person 2", "I will prepare the budget report by Monday"),
    ]

    summarizer = LocalLLMSummarizer(settings)

    call_log = []

    def fake_generate(prompt: str) -> str:
        call_log.append(prompt)
        if "Transcript excerpt" in prompt:
            if "deploy" in prompt:
                return json.dumps({
                    "key_points": ["Deployment timing discussed"],
                    "decisions": ["Deploy tomorrow morning"],
                    "action_items": [{"task": "Handle deployment", "responsible": "Person 2", "deadline": None}],
                })
            return json.dumps({
                "key_points": ["Budget review discussed"],
                "decisions": [],
                "action_items": [{"task": "Prepare budget report", "responsible": "Person 2", "deadline": "Monday"}],
            })
        # reduce prompt
        return json.dumps({
            "overview": "The team discussed deployment and budget planning.",
            "key_points": ["Deployment timing discussed", "Budget review discussed"],
            "decisions": ["Deploy tomorrow morning"],
            "action_items": [
                {"task": "Handle deployment", "responsible": "Person 2", "deadline": None},
                {"task": "Prepare budget report", "responsible": "Person 2", "deadline": "Monday"},
            ],
        })

    monkeypatch.setattr(summarizer, "_generate", fake_generate)
    summary = summarizer.summarize(segments)

    assert summary.overview == "The team discussed deployment and budget planning."
    assert len(summary.key_points) == 2
    assert summary.decisions == ["Deploy tomorrow morning"]
    assert len(summary.action_items) == 2
    assert summary.action_items[1].deadline == "Monday"
    assert len(call_log) >= 3  # at least 2 chunk calls + 1 reduce call


def test_local_llm_summarizer_falls_back_gracefully_on_bad_json(settings, monkeypatch):
    segments = [_seg(0, "Person 1", "hello world this is a test")]
    summarizer = LocalLLMSummarizer(settings)
    monkeypatch.setattr(summarizer, "_generate", lambda prompt: "not json at all")

    summary = summarizer.summarize(segments)
    assert summary.key_points == []
    assert summary.decisions == []
    assert summary.action_items == []


def test_local_llm_summarizer_empty_transcript_short_circuits(settings):
    summarizer = LocalLLMSummarizer(settings)
    summary = summarizer.summarize([])
    assert summary == summary.__class__()


# ---------------------------------------------------------------------------
# Regression: H3 — lazy tokenizer/model loading must be thread-safe.
# ---------------------------------------------------------------------------
def test_load_is_thread_safe_and_loads_only_once(settings, monkeypatch):
    pytest.importorskip("transformers")
    call_count = {"n": 0}
    count_lock = threading.Lock()

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

        def encode(self, text):
            return text.split()

    class SlowFakeModel:
        def to(self, device):
            return self

        @classmethod
        def from_pretrained(cls, *a, **kw):
            with count_lock:
                call_count["n"] += 1
            time.sleep(0.05)  # widen the race window
            return cls()

    monkeypatch.setattr("transformers.AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr("transformers.AutoModelForCausalLM", SlowFakeModel)

    summarizer = LocalLLMSummarizer(settings)
    threads = [threading.Thread(target=summarizer._load) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# Regression: same fix as FasterWhisperTranscriber's cache-first loading
# (see tests/test_transcription.py) — `transformers.from_pretrained(...)`
# is tried with `local_files_only=True` first (no network at all) and only
# falls back to an online load if that fails, and either attempt is bounded
# by a timeout so a hanging Hugging Face Hub call surfaces as a structured
# ModelLoadingError instead of hanging forever.
# ---------------------------------------------------------------------------
def test_load_uses_local_cache_first_and_falls_back_when_not_cached(settings, monkeypatch):
    pytest.importorskip("transformers")
    calls: list[tuple[str, bool]] = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name, local_files_only=False):
            calls.append(("tokenizer", local_files_only))
            if local_files_only:
                raise FileNotFoundError("not cached")
            return cls()

        def encode(self, text):
            return text.split()

    class FakeModel:
        def to(self, device):
            return self

        @classmethod
        def from_pretrained(cls, name, local_files_only=False):
            calls.append(("model", local_files_only))
            return cls()

    monkeypatch.setattr("transformers.AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr("transformers.AutoModelForCausalLM", FakeModel)

    summarizer = LocalLLMSummarizer(settings)
    tokenizer, model = summarizer._load()

    assert isinstance(tokenizer, FakeTokenizer)
    assert isinstance(model, FakeModel)
    # The tokenizer's cache-only attempt failed, so the whole cache-first
    # attempt is abandoned and both tokenizer+model are retried online
    # together (never a mix of cached tokenizer + online model or vice versa).
    assert calls == [("tokenizer", True), ("tokenizer", False), ("model", False)]


def test_load_raises_model_loading_error_when_load_hangs(settings, monkeypatch):
    pytest.importorskip("transformers")
    settings.model_load_timeout_seconds = 0.05

    class HangingTokenizer:
        @classmethod
        def from_pretrained(cls, name, local_files_only=False):
            if local_files_only:
                raise FileNotFoundError("not cached")
            time.sleep(1.0)  # simulate a hanging online Hugging Face Hub call
            return cls()

    monkeypatch.setattr("transformers.AutoTokenizer", HangingTokenizer)
    monkeypatch.setattr("transformers.AutoModelForCausalLM", HangingTokenizer)

    summarizer = LocalLLMSummarizer(settings)
    with pytest.raises(ModelLoadingError, match="Timed out"):
        summarizer._load()


# ---------------------------------------------------------------------------
# Regression: production incident — `tokenizer.apply_chat_template(...,
# return_tensors="pt")` returns a `BatchEncoding` (dict-like: input_ids +
# attention_mask), not a bare tensor, because `return_dict` defaults to
# `True` in the installed `transformers` version (confirmed via
# `inspect.signature(PreTrainedTokenizerBase.apply_chat_template)` — not
# assumed). The old code passed that `BatchEncoding` straight into
# `model.generate(input_ids, ...)`; internally, `generate()` does
# `inputs_tensor.shape[0]`, and `BatchEncoding.__getattr__` turns the
# resulting `KeyError` into a bare `raise AttributeError` with NO message —
# exactly matching the real production symptom: "SUMMARIZATION_FAILED:
# Local LLM generation failed: " (nothing after the colon).
#
# These tests use a REAL `transformers` tokenizer — a tiny from-scratch BPE
# tokenizer wrapped in `PreTrainedTokenizerFast`, with a real chat template
# — so the actual, current `apply_chat_template(..., return_dict=True)`
# behavior is exercised for real rather than assumed; only the model's
# `.generate()` call (the expensive, nondeterministic part) is mocked, per
# the "mock only the model inference" requirement this is a regression for.
# ---------------------------------------------------------------------------
def _build_real_tiny_chat_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        special_tokens=["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"], vocab_size=300
    )
    corpus = ["hello world this is a meeting about the budget and action items"] * 50
    backend.train_from_iterator(corpus, trainer)

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant' }}{% endif %}"
    )
    return tokenizer


def test_generate_unwraps_batch_encoding_before_calling_model_generate(settings):
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    import torch

    tokenizer = _build_real_tiny_chat_tokenizer()
    captured: dict[str, Any] = {}

    class FakeModel:
        def to(self, device):
            return self

        def generate(self, input_ids, **kwargs):
            # This is exactly what the bug got wrong: `input_ids` used to
            # be a BatchEncoding (no .shape) instead of the real tensor.
            assert isinstance(input_ids, torch.Tensor), (
                f"expected a real tensor, got {type(input_ids)!r}"
            )
            assert isinstance(kwargs.get("attention_mask"), torch.Tensor)
            captured["input_ids"] = input_ids
            captured["kwargs"] = kwargs
            new_tokens = torch.zeros((input_ids.shape[0], 3), dtype=input_ids.dtype)
            return torch.cat([input_ids, new_tokens], dim=-1)

    summarizer = LocalLLMSummarizer(settings)
    summarizer._tokenizer = tokenizer
    summarizer._model = FakeModel()

    result = summarizer._generate("Summarize this meeting excerpt.")

    assert isinstance(result, str)
    assert captured["kwargs"]["pad_token_id"] == tokenizer.eos_token_id
    assert captured["kwargs"]["max_new_tokens"] == settings.summarizer_max_new_tokens


def test_generate_wraps_exception_with_type_name_even_when_message_is_blank(settings):
    """The real production failure was an `AttributeError` with an EMPTY
    `str(exc)`. The wrapped `SummarizationError` must still name the
    exception type so the failure is diagnosable instead of repeating
    "Local LLM generation failed: " with nothing after the colon."""
    pytest.importorskip("transformers")

    tokenizer = _build_real_tiny_chat_tokenizer()

    class BlankFailingModel:
        def to(self, device):
            return self

        def generate(self, input_ids, **kwargs):
            raise AttributeError  # exactly like the real bug: no message

    summarizer = LocalLLMSummarizer(settings)
    summarizer._tokenizer = tokenizer
    summarizer._model = BlankFailingModel()

    with pytest.raises(SummarizationError, match="AttributeError"):
        summarizer._generate("Summarize this.")


def test_generate_without_chat_template_still_passes_attention_mask(settings):
    """The non-chat-template fallback path (a tokenizer with no
    `apply_chat_template` method at all — e.g. a non-instruct-tuned
    tokenizer) must also extract and pass attention_mask, not just
    input_ids. Uses a minimal fake tokenizer (deliberately without
    `apply_chat_template`) rather than a real `transformers` tokenizer
    class, since every real one defines that method."""
    import torch

    class FakeBatchEncoding(dict):
        def to(self, device):
            return self

        def get(self, key, default=None):
            return dict.get(self, key, default)

    class FakeTokenizer:
        eos_token_id = 2

        def __call__(self, text, return_tensors=None):
            input_ids = torch.tensor([[10, 11, 12]])
            attention_mask = torch.ones_like(input_ids)
            return FakeBatchEncoding(input_ids=input_ids, attention_mask=attention_mask)

        def decode(self, ids, skip_special_tokens=True):
            return "a summary"

    captured: dict[str, Any] = {}

    class FakeModel:
        def to(self, device):
            return self

        def generate(self, input_ids, **kwargs):
            assert isinstance(input_ids, torch.Tensor)
            assert isinstance(kwargs.get("attention_mask"), torch.Tensor)
            captured["ok"] = True
            new_tokens = torch.zeros((input_ids.shape[0], 2), dtype=input_ids.dtype)
            return torch.cat([input_ids, new_tokens], dim=-1)

    summarizer = LocalLLMSummarizer(settings)
    summarizer._tokenizer = FakeTokenizer()
    summarizer._model = FakeModel()

    assert not hasattr(summarizer._tokenizer, "apply_chat_template")
    result = summarizer._generate("Summarize this.")

    assert result == "a summary"
    assert captured["ok"] is True


# ---------------------------------------------------------------------------
# Regression: H4 — tokenizer-based chunking instead of the English-centric
# word-count heuristic, with a safe fallback before the tokenizer loads.
# ---------------------------------------------------------------------------
def test_token_counter_falls_back_to_heuristic_before_tokenizer_is_loaded(settings):
    summarizer = LocalLLMSummarizer(settings)
    counter = summarizer._token_counter()
    assert counter is _estimate_tokens


def test_token_counter_uses_real_tokenizer_once_loaded(settings):
    class FakeTokenizer:
        def encode(self, text):
            # Deliberately different from the word-count heuristic (one
            # "token" per character) so this test fails if the heuristic
            # were used instead of this fake tokenizer.
            return list(text)

    summarizer = LocalLLMSummarizer(settings)
    summarizer._tokenizer = FakeTokenizer()
    summarizer._model = object()  # marks "loaded"; not used by the counter itself

    counter = summarizer._token_counter()
    assert counter is not _estimate_tokens
    assert counter("hi") == 2


def test_multilingual_chunking_uses_injected_tokenizer_for_english_hindi_odia(settings):
    """Representative English, Hindi (Devanagari) and Odia strings must all
    be chunked using the injected token-counting function (standing in for
    a real tokenizer), never split mid-segment, and never lost."""

    def char_count_tokens(text: str) -> int:
        return len(text)

    segments = [
        _seg(0, "Person 1", "We should finalize the roadmap for next quarter."),
        _seg(1, "Person 2", "हमें अगली तिमाही के लिए रोडमैप को अंतिम रूप देना चाहिए."),
        _seg(2, "Person 1", "ଆମେ ଆଗାମୀ ତ୍ରୈମାସିକ ପାଇଁ ରୋଡମ୍ୟାପକୁ ଚୂଡାନ୍ତ କରିବା ଉଚିତ."),
    ]

    chunks = chunk_transcript_for_summarization(
        segments, token_budget=60, overlap_segments=0, count_tokens=char_count_tokens,
    )

    all_ids = [s.segment_id for chunk in chunks for s in chunk]
    assert all_ids == [0, 1, 2]  # every segment present, in order, none lost
    assert len(chunks) >= 2  # a 60-char budget forces a split across these three lines
    for chunk in chunks:
        assert len(chunk) >= 1


# ---------------------------------------------------------------------------
# Regression: H5 — LLM JSON-parse failures must be surfaced as warnings,
# not silently turned into an apparently-valid empty summary.
# ---------------------------------------------------------------------------
def test_chunk_parse_failure_is_surfaced_as_a_warning(settings, monkeypatch):
    segments = [_seg(0, "Person 1", "hello world this is a test")]
    summarizer = LocalLLMSummarizer(settings)
    monkeypatch.setattr(summarizer, "_generate", lambda prompt: "not json at all")

    summary = summarizer.summarize(segments)

    assert summary.key_points == []
    assert len(summary.warnings) >= 1
    assert any("not valid JSON" in w for w in summary.warnings)


def test_reduce_parse_failure_is_surfaced_as_a_warning(settings, monkeypatch):
    settings.summarizer_chunk_token_budget = 10
    settings.summarizer_chunk_overlap_segments = 0
    segments = [
        _seg(0, "Person 1", "We should deploy tomorrow morning"),
        _seg(1, "Person 2", "Agreed, Person 2 will handle the deployment"),
    ]
    summarizer = LocalLLMSummarizer(settings)

    def fake_generate(prompt: str) -> str:
        if "Transcript excerpt" in prompt:
            return json.dumps({"key_points": ["deploy"], "decisions": [], "action_items": []})
        return "not valid json for the reduce step"

    monkeypatch.setattr(summarizer, "_generate", fake_generate)
    summary = summarizer.summarize(segments)

    # Falls back to raw concatenation rather than losing the chunk-level
    # extracts entirely...
    assert "deploy" in summary.key_points
    # ...but the failure is visible to API consumers, not silently hidden.
    assert any("reduce" in w.lower() for w in summary.warnings)


def test_successful_summary_has_no_warnings(settings, monkeypatch):
    segments = [_seg(0, "Person 1", "hello world this is a test")]
    summarizer = LocalLLMSummarizer(settings)
    monkeypatch.setattr(
        summarizer, "_generate",
        lambda prompt: json.dumps({"overview": "ok", "key_points": [], "decisions": [], "action_items": []}),
    )

    summary = summarizer.summarize(segments)
    assert summary.warnings == []


# ---------------------------------------------------------------------------
# Regression: H6 — the hierarchical reduce step must be bounded: if the
# combined chunk partials exceed the reduce token budget, they are
# recursively grouped and reduced instead of being concatenated into one
# unbounded prompt.
# ---------------------------------------------------------------------------
def test_hierarchical_reduce_recurses_when_partials_exceed_budget(settings, monkeypatch):
    settings.summarizer_chunk_token_budget = 8  # force many small chunks
    settings.summarizer_chunk_overlap_segments = 0
    settings.summarizer_reduce_token_budget = 15  # deliberately tiny: forces recursive reduce

    segments = [_seg(i, "Person 1", f"point number {i} about the roadmap") for i in range(8)]

    summarizer = LocalLLMSummarizer(settings)
    reduce_call_count = {"n": 0}

    def fake_generate(prompt: str) -> str:
        if "Transcript excerpt" in prompt:
            return json.dumps({"key_points": ["kp"], "decisions": [], "action_items": []})
        reduce_call_count["n"] += 1
        return json.dumps({"overview": "ok", "key_points": ["kp"], "decisions": [], "action_items": []})

    monkeypatch.setattr(summarizer, "_generate", fake_generate)
    summary = summarizer.summarize(segments)

    # A single, unbounded reduce call over all 8 chunk partials at once
    # would mean exactly one reduce call; recursion under a tight budget
    # must take more than one.
    assert reduce_call_count["n"] > 1
    assert summary.overview == "ok"
    assert summary.warnings == []


def test_single_chunk_meeting_skips_the_reduce_call_entirely(settings, monkeypatch):
    """When there's only one chunk, there's nothing to reduce — the single
    chunk-level extract IS the final result, so no extra LLM call is made."""
    segments = [_seg(0, "Person 1", "hello world this is a test")]
    summarizer = LocalLLMSummarizer(settings)

    generate_calls = []

    def fake_generate(prompt: str) -> str:
        generate_calls.append(prompt)
        return json.dumps({"key_points": ["kp"], "decisions": [], "action_items": []})

    monkeypatch.setattr(summarizer, "_generate", fake_generate)
    summarizer.summarize(segments)

    assert len(generate_calls) == 1  # exactly the one chunk call, no reduce call
