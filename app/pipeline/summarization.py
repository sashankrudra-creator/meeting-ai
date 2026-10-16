"""
Local MoM generation stage (Section 11/12).

    Summarizer (interface)
        |
        +-- LocalLLMSummarizer   (production: local HF `transformers` instruct model)
        +-- ExtractiveSummarizer (offline fallback: no model download, rule-based)

No paid/external LLM API is used by either implementation. `LocalLLMSummarizer`
downloads open-weight model files from the Hugging Face Hub once (configurable
via `summarizer_model_name`) and then runs 100% local inference for every
meeting. `ExtractiveSummarizer` needs no download at all; it exists so that
(a) the Summarizer abstraction is demonstrably swappable end-to-end (Section
24), and (b) the full orchestrator pipeline can be exercised offline in
environments without Hugging Face access (including this one — see the
README's "Known Limitations" section for why `LocalLLMSummarizer` itself
could not be executed against real weights in this sandbox).

Production incident (fixed): `LocalLLMSummarizer._generate()` used to call
`tokenizer.apply_chat_template(..., return_tensors="pt")` and pass its
result straight into `model.generate(input_ids, ...)`. Depending on the
installed `transformers` version, that call returns a `BatchEncoding`
(dict-like: input_ids + attention_mask), not a bare tensor — passing it
into `generate()` as-is failed deep inside `transformers.generation.utils`
with a message-LESS `AttributeError` (`BatchEncoding.__getattr__` turns a
missing-attribute `KeyError` into a bare `raise AttributeError`), which
surfaced as an unhelpful, blank "Local LLM generation failed: " error.
`_generate()` now explicitly unwraps `input_ids`/`attention_mask` from the
returned `BatchEncoding` before calling `.generate()`, and wraps any
generation exception with its type name so a blank `str(exc)` (as with
this exact `AttributeError`) never produces an undiagnosable error again.
See `tests/test_summarization.py`'s `test_generate_*` tests and
`scripts/verify_qwen_generation.py` for the real-model verification.

Long-meeting strategy (Section 12), implemented via `chunk_transcript_for_summarization`:

    Full Transcript
           |
           v
    Logical chunks (whole transcript segments only — never split mid-segment,
    so conversation boundaries are respected; each chunk stays under an
    approximate token budget)
           |
           v
    Chunk summaries (one MoM-shaped partial result per chunk)
           |
           v
    Combined intermediate summary (concatenation + de-duplication of all
    chunk-level key points / decisions / action items) — BOUNDED: if the
    combined partials would not fit in one reduce prompt within
    `summarizer_reduce_token_budget`, they are recursively grouped and
    reduced in smaller batches first, rather than concatenated into one
    unbounded prompt (see `LocalLLMSummarizer._reduce`)
           |
           v
    Final MoM (one more pass over the combined intermediate summary to
    produce a single coherent overview and a de-duplicated final list)

Anti-hallucination policy: every prompt explicitly instructs the model to
only report decisions/action items that are directly supported by the
transcript text it was given, and to use `null` for an action item's
`responsible`/`deadline` when the transcript does not explicitly name one,
rather than inventing a plausible-sounding value. This is enforced at the
prompt level (not perfectly guaranteed — no LLM's output is enforceable by
construction — which is why it's documented as a policy applied to the
model, not a hard guarantee, in the README's Known Limitations section).
"""

from __future__ import annotations

import json
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.models.schemas import ActionItem, MeetingSummary, TranscriptSegment
from app.utils.errors import ModelLoadingError, SummarizationError
from app.utils.logging import get_logger
from app.utils.timeout import run_with_timeout

logger = get_logger("pipeline.summarization")


# ---------------------------------------------------------------------------
# Chunking (Section 12) — pure, model-independent, unit-testable in isolation
# ---------------------------------------------------------------------------
def _estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~1.3 tokens/word for English;
    close enough for chunk-sizing purposes, deliberately conservative).

    This is an ENGLISH-CENTRIC heuristic: whitespace-based word counting
    badly under/over-counts languages and scripts that don't tokenize the
    way English does (Hindi/Devanagari and Odia included — see Section 10).
    It exists as the safe fallback used before a real tokenizer is
    available; `LocalLLMSummarizer` prefers the actual tokenizer of the
    configured model once it has been loaded (see `_token_counter`)."""
    words = text.split()
    return max(1, int(len(words) * 1.3))


def format_segment_line(segment: TranscriptSegment) -> str:
    return f"{segment.speaker_name}: {segment.text}"


def format_chunk_as_text(chunk: list[TranscriptSegment]) -> str:
    return "\n".join(format_segment_line(s) for s in chunk)


def chunk_transcript_for_summarization(
    transcript: list[TranscriptSegment],
    token_budget: int,
    overlap_segments: int = 0,
    count_tokens: Callable[[str], int] = _estimate_tokens,
) -> list[list[TranscriptSegment]]:
    """Split `transcript` into chunks of whole segments, each under
    `token_budget` estimated tokens, respecting conversation boundaries
    (a segment is never split across two chunks) and preserving speaker
    attribution. `overlap_segments` trailing segments from chunk N are
    repeated at the start of chunk N+1 so the model summarizing chunk N+1
    has some conversational continuity.

    `count_tokens` is injectable so callers can use an accurate,
    language-aware tokenizer instead of the default word-count heuristic
    (Section 10: Hindi/Odia text is not reliably chunked by whitespace word
    counts) — see `LocalLLMSummarizer._token_counter`. It defaults to the
    heuristic so every existing caller/test keeps its current behavior.

    A single segment that alone exceeds `token_budget` still gets its own
    chunk rather than being dropped or truncated — long individual turns
    are never silently lost.
    """
    if not transcript:
        return []

    chunks: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    current_tokens = 0

    for segment in transcript:
        seg_tokens = count_tokens(format_segment_line(segment))
        if current and current_tokens + seg_tokens > token_budget:
            chunks.append(current)
            carry_over = current[-overlap_segments:] if overlap_segments > 0 else []
            current = list(carry_over)
            current_tokens = sum(count_tokens(format_segment_line(s)) for s in current)
        current.append(segment)
        current_tokens += seg_tokens

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# JSON extraction helper — LLM output is not guaranteed to be pure JSON
# ---------------------------------------------------------------------------
def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _parse_action_items(raw_items: Any) -> list[ActionItem]:
    items: list[ActionItem] = []
    if not isinstance(raw_items, list):
        return items
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        task = raw.get("task")
        if not task:
            continue
        items.append(
            ActionItem(
                task=str(task),
                responsible=(str(raw["responsible"]) if raw.get("responsible") else None),
                deadline=(str(raw["deadline"]) if raw.get("deadline") else None),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Summarizer interface
# ---------------------------------------------------------------------------
class Summarizer(ABC):
    @abstractmethod
    def summarize(self, transcript: list[TranscriptSegment]) -> MeetingSummary:
        raise NotImplementedError


_CHUNK_PROMPT_TEMPLATE = """You are assisting with meeting minutes. Below is a portion of a \
meeting transcript. Extract ONLY information explicitly stated in this transcript excerpt. \
Do not guess, infer, or invent any names, dates, or facts that are not directly stated.

Respond with a single JSON object with exactly these keys:
{{
  "key_points": [list of short strings, important topics discussed in this excerpt],
  "decisions": [list of short strings, decisions clearly stated in this excerpt],
  "action_items": [list of objects with "task" (string), "responsible" (string or null if not \
explicitly named), "deadline" (string or null if not explicitly stated)]
}}
If nothing qualifies for a field, return an empty list for it. Output ONLY the JSON object.

Transcript excerpt:
{chunk_text}

JSON:"""

_REDUCE_PROMPT_TEMPLATE = """You are consolidating partial meeting-minutes extracts from \
different parts of the same meeting into one final result. Merge and de-duplicate the items \
below. Do not invent anything that is not already present in the input. Write a short (2-4 \
sentence) overview of the whole meeting based only on the key points and decisions given.

Respond with a single JSON object with exactly these keys:
{{
  "overview": string,
  "key_points": [list of de-duplicated short strings],
  "decisions": [list of de-duplicated short strings],
  "action_items": [list of objects with "task", "responsible" (or null), "deadline" (or null)]
}}
Output ONLY the JSON object.

Partial extracts (JSON):
{combined}

JSON:"""


class LocalLLMSummarizer(Summarizer):
    """Production summarizer: a local, open-weight instruction-tuned model
    run entirely on-device via Hugging Face `transformers`.

    The model is swappable purely through configuration
    (`settings.summarizer_model_name`) — nothing in this class or in
    `orchestrator.py` hard-codes a specific model.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        # Guards the lazy-load below against concurrent callers, same
        # rationale as PyannoteDiarizer._load_lock /
        # FasterWhisperTranscriber._load_lock.
        self._load_lock = threading.Lock()

    def _instantiate_tokenizer_and_model(self, *, local_files_only: bool) -> tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        settings = self._settings
        tokenizer = AutoTokenizer.from_pretrained(
            settings.summarizer_model_name, local_files_only=local_files_only
        )
        model = AutoModelForCausalLM.from_pretrained(
            settings.summarizer_model_name, local_files_only=local_files_only
        )
        model.to(settings.summarizer_device)
        return tokenizer, model

    def _load_cache_first(self, settings: Settings) -> tuple[Any, Any]:
        """Same cache-first-then-online-fallback fix as
        `FasterWhisperTranscriber._load_model_cache_first` (see that
        docstring for the production incident this addresses):
        `transformers.from_pretrained(..., local_files_only=True)` skips
        the network entirely and is tried first, only falling back to an
        online load (which can download the model) if that fails."""
        try:
            result = run_with_timeout(
                lambda: self._instantiate_tokenizer_and_model(local_files_only=True),
                settings.model_load_timeout_seconds,
                f"Loading summarization model '{settings.summarizer_model_name}' from local cache",
            )
        except TimeoutError:
            raise
        except Exception as exc:
            logger.info(
                "Summarization model '%s' is not available in the local cache (%s); "
                "falling back to an online load (this will download the model if needed)",
                settings.summarizer_model_name, exc,
            )
        else:
            logger.info(
                "Loaded summarization model '%s' from local cache (no network access)",
                settings.summarizer_model_name,
            )
            return result

        return run_with_timeout(
            lambda: self._instantiate_tokenizer_and_model(local_files_only=False),
            settings.model_load_timeout_seconds,
            f"Loading summarization model '{settings.summarizer_model_name}' online",
        )

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:  # double-checked locking
                return self._tokenizer, self._model

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
            except ImportError as exc:  # pragma: no cover
                raise ModelLoadingError(
                    "The 'transformers' package is not installed. Run "
                    "`pip install -r requirements.txt`."
                ) from exc

            settings = self._settings
            logger.info("Loading local summarization model '%s'", settings.summarizer_model_name)
            try:
                tokenizer, model = self._load_cache_first(settings)
            except TimeoutError as exc:
                raise ModelLoadingError(
                    f"Timed out loading summarization model '{settings.summarizer_model_name}' "
                    f"after {settings.model_load_timeout_seconds:.0f}s: {exc} This usually means "
                    "a network call to the Hugging Face Hub is hanging rather than failing fast, "
                    "even if the model is already cached locally. See "
                    "`Settings.model_load_timeout_seconds`."
                ) from exc
            except Exception as exc:
                raise ModelLoadingError(
                    f"Failed to load summarization model '{settings.summarizer_model_name}': {exc}"
                ) from exc

            self._tokenizer, self._model = tokenizer, model
            return self._tokenizer, self._model

    def _token_counter(self) -> Callable[[str], int]:
        """Token-counting function used to size chunks for this summarizer
        (Section 10/12).

        Prefers the real tokenizer of the configured summarization model —
        accurate for any language/script the model supports, including
        Hindi and Odia, unlike the English-word-count heuristic in
        `_estimate_tokens`. Only used once the tokenizer has already been
        loaded (i.e. after this instance's first successful generation
        call, here or in an earlier `summarize()` call) — chunk sizing
        never forces a model load purely to count tokens; before that has
        happened it safely falls back to the heuristic, exactly as before
        this fix."""
        tokenizer = self._tokenizer
        if tokenizer is None:
            return _estimate_tokens

        def _count(text: str) -> int:
            try:
                return len(tokenizer.encode(text))
            except Exception:  # noqa: BLE001 - token counting must never crash chunking
                return _estimate_tokens(text)

        return _count

    def _generate(self, prompt: str) -> str:
        tokenizer, model = self._load()
        settings = self._settings
        try:
            messages = [{"role": "user", "content": prompt}]
            # `return_dict=True` is passed explicitly (root-cause fix, see
            # below) rather than relied on as a default, since installed
            # `transformers` versions disagree on what `apply_chat_template`
            # returns without it.
            if hasattr(tokenizer, "apply_chat_template"):
                encoded = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                encoded = tokenizer(prompt, return_tensors="pt")
            encoded = encoded.to(settings.summarizer_device)

            # `encoded` is a `BatchEncoding` (dict-like: input_ids +
            # attention_mask), never a bare tensor — unwrap explicitly
            # rather than pass it into `model.generate()` as-is.
            #
            # Production incident / root cause (verified by reproducing
            # this exact call against a real `Qwen2ForCausalLM` +
            # `apply_chat_template` in the installed `transformers`
            # version): the previous code called
            # `tokenizer.apply_chat_template(..., return_tensors="pt")`
            # with no `return_dict` argument. Depending on the installed
            # `transformers` version, that call's default for `return_dict`
            # is `True` (confirmed via `inspect.signature` here), so it
            # already returned a `BatchEncoding`, not a plain tensor — the
            # old code's `.to(device)` still "worked" (BatchEncoding.to()
            # is a real method that moves every tensor value and returns
            # itself), but then passed that whole `BatchEncoding` straight
            # into `model.generate(input_ids, ...)` as the positional
            # `inputs` argument. Internally, `generate()` does
            # `inputs_tensor.shape[0]` on it — `BatchEncoding` has no
            # `.shape`, and `BatchEncoding.__getattr__` catches the
            # resulting `KeyError` and re-raises a bare `AttributeError`
            # with NO message, which is exactly why the persisted error was
            # "Local LLM generation failed: " with nothing after the colon.
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")

            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": settings.summarizer_max_new_tokens,
                "do_sample": settings.summarizer_temperature > 0,
                "temperature": max(settings.summarizer_temperature, 1e-5),
                "pad_token_id": getattr(tokenizer, "eos_token_id", None),
            }
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask

            output_ids = model.generate(input_ids, **generate_kwargs)
            generated = output_ids[0][input_ids.shape[-1]:]
            return tokenizer.decode(generated, skip_special_tokens=True)
        except Exception as exc:
            # `str(exc)` can be empty (see the AttributeError case above),
            # so include the exception's type name in the message that
            # propagates into the API/persisted error — otherwise a future
            # failure of this kind is silently unreportable again. Full
            # traceback always goes to the log regardless.
            logger.exception(
                "Local LLM generation failed (exception type: %s)", type(exc).__name__
            )
            raise SummarizationError(
                f"Local LLM generation failed ({type(exc).__name__}): {exc}"
            ) from exc

    def _summarize_chunk(
        self, chunk: list[TranscriptSegment], index: int, total: int, warnings: list[str]
    ) -> dict[str, Any]:
        chunk_text = format_chunk_as_text(chunk)
        prompt = _CHUNK_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
        raw_output = self._generate(prompt)
        parsed = _extract_json_object(raw_output)
        if parsed is None:
            message = (
                f"Chunk {index}/{total} summary was not valid JSON; treated as an empty extract "
                "(nothing from this part of the meeting was lost silently — see this warning)."
            )
            logger.warning(message)
            warnings.append(message)
            return {"key_points": [], "decisions": [], "action_items": []}
        return parsed

    def _reduce_group(self, partials: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
        """Run ONE reduce LLM call over a bounded group of partial extracts."""
        key_points = [str(x) for p in partials for x in p.get("key_points", []) if x]
        decisions = [str(x) for p in partials for x in p.get("decisions", []) if x]
        action_items = [item for p in partials for item in p.get("action_items", []) if isinstance(item, dict)]

        combined = json.dumps(
            {"key_points": key_points, "decisions": decisions, "action_items": action_items},
            ensure_ascii=False,
        )
        raw_final = self._generate(_REDUCE_PROMPT_TEMPLATE.format(combined=combined))
        parsed = _extract_json_object(raw_final)
        if parsed is None:
            message = (
                "A hierarchical-reduce step's output was not valid JSON; falling back to raw "
                "concatenation for that part of the meeting instead of dropping it."
            )
            logger.warning(message)
            warnings.append(message)
            return {"overview": "", "key_points": key_points, "decisions": decisions, "action_items": action_items}
        return parsed

    def _group_partials_by_budget(
        self,
        partials: list[dict[str, Any]],
        budget: int,
        count_tokens: Callable[[str], int],
    ) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_tokens = 0
        for partial in partials:
            tokens = count_tokens(json.dumps(partial, ensure_ascii=False))
            if current and current_tokens + tokens > budget:
                groups.append(current)
                current = []
                current_tokens = 0
            current.append(partial)
            current_tokens += tokens
        if current:
            groups.append(current)
        return groups

    def _reduce(
        self,
        partials: list[dict[str, Any]],
        count_tokens: Callable[[str], int],
        warnings: list[str],
    ) -> dict[str, Any]:
        """Combine partial (chunk-level) extraction dicts into one final MoM
        dict, BOUNDED (Section 12): if the combined partials would exceed
        `summarizer_reduce_token_budget` in a single reduce prompt, they are
        split into sub-groups that each fit the budget, each sub-group is
        reduced independently, and the (much smaller) set of sub-results is
        reduced again — recursing until everything fits in one final call.
        This replaces the old behavior of concatenating an arbitrarily large
        number of chunk partials into a single, unbounded reduce prompt."""
        if len(partials) == 1:
            return partials[0]

        budget = self._settings.summarizer_reduce_token_budget
        combined_text = json.dumps(partials, ensure_ascii=False)
        if count_tokens(combined_text) <= budget:
            return self._reduce_group(partials, warnings)

        groups = self._group_partials_by_budget(partials, budget, count_tokens)
        if len(groups) == len(partials):
            # Even one-partial-per-group didn't shrink anything (each
            # partial alone is already at/over budget) — reduce in fixed
            # pairs instead so recursion still makes forward progress.
            groups = [partials[i : i + 2] for i in range(0, len(partials), 2)]

        sub_results = [
            self._reduce_group(group, warnings) if len(group) > 1 else group[0] for group in groups
        ]
        return self._reduce(sub_results, count_tokens, warnings)

    def summarize(self, transcript: list[TranscriptSegment]) -> MeetingSummary:
        if not transcript:
            return MeetingSummary()

        settings = self._settings
        count_tokens = self._token_counter()
        chunks = chunk_transcript_for_summarization(
            transcript,
            token_budget=settings.summarizer_chunk_token_budget,
            overlap_segments=settings.summarizer_chunk_overlap_segments,
            count_tokens=count_tokens,
        )
        logger.info("Summarizing meeting in %d chunk(s)", len(chunks))

        warnings: list[str] = []
        partials = [
            self._summarize_chunk(chunk, i, len(chunks), warnings) for i, chunk in enumerate(chunks, start=1)
        ]

        final_parsed = self._reduce(partials, count_tokens, warnings)

        return MeetingSummary(
            overview=str(final_parsed.get("overview", "") or ""),
            key_points=_dedupe_preserve_order([str(x) for x in final_parsed.get("key_points", [])]),
            decisions=_dedupe_preserve_order([str(x) for x in final_parsed.get("decisions", [])]),
            action_items=_parse_action_items(final_parsed.get("action_items", [])),
            warnings=warnings,
        )


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)
    return result


# ---------------------------------------------------------------------------
# Offline, no-download fallback (useful for CI / low-resource / air-gapped runs)
# ---------------------------------------------------------------------------
_DECISION_MARKERS = (
    "we will", "we'll", "let's", "decided", "agreed", "we should", "we are going to",
    "going forward", "final decision",
)
_ACTION_MARKERS = (
    "will do", "will take", "will handle", "assigned to", "responsible for",
    "action item", "will complete", "will finish", "needs to", "please",
)


class ExtractiveSummarizer(Summarizer):
    """Lightweight, fully local, rule-based fallback that requires no model
    download whatsoever. It trades summary quality for zero external
    dependencies, which is why it is not the spec's primary summarizer —
    `LocalLLMSummarizer` is — but it lets the full pipeline (including this
    stage) run end-to-end in offline/CI environments, and demonstrates that
    `Summarizer` is a genuine, swappable abstraction (Section 24)."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def summarize(self, transcript: list[TranscriptSegment]) -> MeetingSummary:
        if not transcript:
            return MeetingSummary()

        key_points: list[str] = []
        decisions: list[str] = []
        action_items: list[ActionItem] = []

        for segment in transcript:
            text = segment.text.strip()
            if not text:
                continue
            lowered = text.lower()

            if any(marker in lowered for marker in _DECISION_MARKERS):
                decisions.append(text)
            elif any(marker in lowered for marker in _ACTION_MARKERS):
                action_items.append(
                    ActionItem(task=text, responsible=segment.speaker_name, deadline=None)
                )
            elif len(text.split()) >= 6:
                # A reasonably substantive sentence that isn't a decision/action
                # is treated as a candidate discussion point.
                key_points.append(text)

        overview = (
            f"Meeting with {len({s.speaker_id for s in transcript})} speaker(s) and "
            f"{len(transcript)} recorded utterance(s). "
            f"{len(decisions)} decision(s) and {len(action_items)} action item(s) were "
            "extracted using local keyword heuristics (extractive fallback — enable "
            "LocalLLMSummarizer for higher-quality abstractive summaries)."
        )

        return MeetingSummary(
            overview=overview,
            key_points=_dedupe_preserve_order(key_points)[:15],
            decisions=_dedupe_preserve_order(decisions)[:15],
            action_items=action_items[:20],
        )


def build_summarizer(settings: Settings) -> Summarizer:
    """Factory: pick the configured Summarizer implementation without the
    orchestrator ever importing a concrete class directly."""
    if settings.summarizer_backend == "extractive":
        return ExtractiveSummarizer(settings)
    return LocalLLMSummarizer(settings)
