#!/usr/bin/env python3
"""
Real-model smoke test for the `LocalLLMSummarizer._generate()` fix
(Section 11/12 — see app/pipeline/summarization.py for the root-cause
explanation of the `AttributeError` production incident this verifies).

Run it with the SAME `.env` / environment the real app uses, from the
project root, after the model has already been downloaded once (so this
never needs network access — set HF_HUB_OFFLINE=1 to prove it):

    $env:HF_HUB_OFFLINE = "1"     # PowerShell; use `export` on Linux/macOS
    python scripts/verify_qwen_generation.py

It builds a real `LocalLLMSummarizer` from your actual `Settings`, calls
its real `_load()` (loading `SUMMARIZER_MODEL_NAME`, e.g.
`Qwen/Qwen2.5-1.5B-Instruct`, from the local cache), then calls the real
`_generate()` with a short, fixed prompt. Prints PASS/FAIL and exits
non-zero on failure — this is the one test that actually exercises real
cached model weights, which this project's own CI/dev sandbox has no
network access to download, so it cannot be run there (see the README's
"Known Limitations").
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.pipeline.summarization import LocalLLMSummarizer  # noqa: E402

_TEST_PROMPT = (
    "You are assisting with meeting minutes. Below is a portion of a meeting "
    "transcript. Extract ONLY information explicitly stated in this transcript "
    "excerpt.\n\nRespond with a single JSON object with exactly these keys: "
    '{"key_points": [], "decisions": [], "action_items": []}\n\n'
    "Transcript excerpt:\nAlice: Let's ship the new pricing page by Friday.\n"
    "Bob: Sounds good, I'll handle the copy.\n\nJSON:"
)


def main() -> int:
    settings = get_settings()
    print(f"SUMMARIZER_MODEL_NAME={settings.summarizer_model_name!r}")
    print(f"SUMMARIZER_DEVICE={settings.summarizer_device!r}")
    print(f"summarizer_max_new_tokens={settings.summarizer_max_new_tokens!r}")

    summarizer = LocalLLMSummarizer(settings)

    print("Loading model + tokenizer from local cache...")
    t0 = time.monotonic()
    try:
        summarizer._load()
    except Exception as exc:  # noqa: BLE001 - want the raw failure, not a translated one
        print(f"FAIL: model loading raised {type(exc).__name__}: {exc}")
        return 1
    print(f"Loaded in {time.monotonic() - t0:.1f}s")

    print("Running a real generation call...")
    t0 = time.monotonic()
    try:
        output = summarizer._generate(_TEST_PROMPT)
    except Exception as exc:  # noqa: BLE001 - want the raw failure, not a translated one
        print(f"FAIL: generation raised {type(exc).__name__}: {exc!r}")
        return 1
    elapsed = time.monotonic() - t0

    print(f"Generated in {elapsed:.1f}s. Raw output:\n{output}")

    if not output.strip():
        print("FAIL: generation produced empty output.")
        return 1

    print("PASS: real Qwen generation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
