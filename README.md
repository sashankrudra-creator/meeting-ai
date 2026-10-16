# Voice-Based Minutes of Meeting (MoM) Pipeline

A local-inference-only pipeline that turns a recorded meeting (audio or
video) into a speaker-attributed, multilingual, timestamped transcript
plus an automatically generated Minutes of Meeting (summary, key points,
decisions, action items). **No paid or external LLM/ASR API is used
anywhere in this project** — every model runs on the machine you deploy
it on, and the pipeline has been run end-to-end, successfully, on a real
meeting recording (see "Actual Successful End-to-End Validation" below).

## 1. Objective

Given an uploaded meeting recording, the pipeline:

- transcribes speech, with built-in support for **English, Hindi and
  Odia**, including meetings where speakers switch languages
  mid-conversation ("code-switching"),
- identifies **who spoke when** (diarization) and keeps a stable
  `Person 1` / `Person 2` / ... labeling for the whole meeting,
- **aligns** the transcript to speakers using real temporal
  overlap/intersection logic (not a naive nearest-timestamp heuristic),
  and explicitly flags segments with ambiguous/overlapping speaker
  attribution,
- computes **per-speaker statistics** (speaking time, percentage, segment
  count, average/longest turn),
- generates a **Minutes of Meeting** (overview, key discussion points,
  decisions, action items with responsible/deadline where explicitly
  stated) using a **local, open-weight instruction-tuned LLM** — never a
  paid API,
- **persists** everything (SQLite metadata + filesystem transcripts/results),
- exposes all of the above through a small **FastAPI** service.

The project is organized as a sequence of small, independently testable
stages rather than one large script.

## 2. Architecture

```text
                    User / Client
                         |
                  Audio / Video Upload
                         |
                      FastAPI
                         |
                 File Validation
                         |
                     FFmpeg
                         |
              Audio Extraction/Normalization
                         |
          +--------------+--------------+
          |                             |
          v                             v
   Faster-Whisper                  Pyannote
   Speech-to-Text                 Diarization
          |                             |
          |                             |
          +-------------+---------------+
                        |
                 Temporal Alignment
                        |
               Structured Transcript
                        |
          +-------------+--------------+
          |                            |
          v                            v
 Speaker Statistics              Local MoM Engine
          |                            |
          |                    Summary / Key Points
          |                    Decisions / Actions
          |                            |
          +-------------+--------------+
                        |
                Final Structured Result
                        |
                 SQLite + Filesystem
                        |
                       API
```

Every model-facing stage is behind an interface
(`Transcriber`, `Diarizer`, `Summarizer`) so a different local model can be
swapped in later purely through configuration — see `app/config.py`.

`app/pipeline/orchestrator.py` runs, per meeting, in this order:

```text
validate_file()  ->  extract_audio()/preprocess_audio()  ->  transcribe()
   -> diarize()  ->  align_transcript_and_speakers()  ->  calculate_statistics()
   -> generate_mom()  ->  persist_result()
```

The meeting's `status` (see `ProcessingStatus` in `app/models/schemas.py`)
is updated in SQLite *before* each stage starts:

```text
QUEUED -> VALIDATING -> EXTRACTING_AUDIO -> TRANSCRIBING -> DIARIZING
   -> ALIGNING -> CALCULATING_STATISTICS -> GENERATING_SUMMARY
   -> SAVING_RESULT -> COMPLETED   (or FAILED, from any stage, with a
                                     structured error code/message)
```

`extract_audio()` and `preprocess_audio()` are implemented as **one**
ffmpeg invocation (`app/pipeline/preprocessing.py::preprocess_audio`) that
extracts the audio stream and normalizes it to mono/16kHz/PCM WAV in a
single pass — running ffmpeg twice over the same input would add I/O with
no benefit. The `EXTRACTING_AUDIO` status covers this combined step.

A long meeting is never held inside the HTTP request/response cycle:
`POST /api/v1/meetings` saves the file, creates a `QUEUED` row, hands the
meeting off to a bounded in-process `ThreadPoolExecutor`
(`app/pipeline/worker.py`), and returns immediately.

## 3. Features

- Audio and video upload with structured validation (extension, size,
  corrupt/empty-media detection).
- Automatic audio extraction/normalization via FFmpeg.
- Multilingual speech-to-text with automatic language detection
  (English/Hindi/Odia supported out of the box; not restricted to a
  hard-coded language).
- Optional per-segment language re-detection for code-switched meetings
  (`WHISPER_SEGMENT_LEVEL_LANGUAGE_DETECTION`).
- Speaker diarization with a stable, human-friendly `Person 1`/`Person 2`
  naming scheme, computed once for the whole meeting (never per-segment).
- Temporal-overlap-based alignment of transcript segments to speakers,
  including explicit flagging of segments with ambiguous/overlapping
  speaker attribution (`overlapping_speakers`).
- Per-speaker statistics: speaking time, percentage, segment count,
  average/longest turn.
- Locally generated Minutes of Meeting: overview, key points, decisions,
  and action items (with responsible/deadline where explicitly stated in
  the transcript) — via a local instruction-tuned LLM, with an
  extractive, zero-download fallback backend.
- SQLite + filesystem persistence of every meeting's transcript,
  statistics, and summary.
- A small FastAPI service exposing upload, status, transcript, summary,
  and full-result endpoints, plus a `/health` check.
- A structured error-code taxonomy (see `app/utils/errors.py`) instead of
  raw tracebacks reaching the API.
- Defensive timeouts and cache-first model loading (`app/utils/timeout.py`)
  so a stalled network call to the Hugging Face Hub fails as a clear,
  structured error instead of hanging the pipeline indefinitely.

## 4. Technology and Model Choices

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI + Pydantic | async-friendly, automatic request/response validation, built-in OpenAPI docs |
| Config | pydantic-settings | one typed `Settings` object, environment-variable driven, no hard-coded model params scattered through the codebase |
| Media | FFmpeg (via `subprocess`) | the de-facto standard for robust container/codec handling |
| ASR | faster-whisper (CTranslate2) | 4-8x faster and much lower memory than the reference `openai-whisper` implementation, fully local, first-class multilingual support |
| Diarization | pyannote.audio (`speaker-diarization-community-1`) | current, actively maintained, local, produces standard `pyannote.core.Annotation` turns |
| Summarization | Hugging Face `transformers` + a local open-weight instruct model | fully local text generation, swappable via configuration, no paid API |
| Storage | SQLite + filesystem | no separate DB server to operate for a single-node deployment |
| Background jobs | `concurrent.futures.ThreadPoolExecutor` | a bounded thread pool is enough for a single-node pipeline; avoids an unnecessary Celery/Redis/Kafka dependency |

Model choices, in detail:

- **`faster-whisper` / `large-v3`** (configurable — see `.env.example`):
  Whisper `large-v3` has the strongest documented multilingual accuracy of
  the Whisper family, including Hindi and Odia, and supports automatic
  language detection out of the box (never hard-coded to English).
  `faster-whisper` re-implements inference in CTranslate2, which is
  meaningfully faster and lighter on both CPU and GPU than the reference
  PyTorch implementation, at identical model weights/accuracy. Swap
  `WHISPER_MODEL_NAME` to `small`/`medium` on constrained hardware (see
  Section 19, Hardware Requirements, and `scripts/check_environment.py`).
  Model loading tries the local cache first (`local_files_only=True`) and
  only falls back to a network fetch if nothing is cached, with a bounded
  timeout either way, so a flaky network cannot hang the pipeline.

- **`pyannote/speaker-diarization-community-1`**: the current community
  diarization pipeline maintained by the pyannote team, distributed on the
  Hugging Face Hub. **It is gated** — see Section 9, "Model setup/download".

- **Local summarization — `Summarizer` abstraction with two implementations**:
  - `LocalLLMSummarizer` (production, default): runs a local, open-weight,
    instruction-tuned causal LM via `transformers.AutoModelForCausalLM`.
    Default model: `Qwen/Qwen2.5-1.5B-Instruct` — small enough to run on a
    CPU-only machine, Apache-2.0 licensed, and strong at following JSON-
    formatted extraction instructions for its size. Any other local
    instruct model (Llama-3.x-Instruct, Phi-3-mini, Mistral-Instruct, ...)
    can be substituted purely via `SUMMARIZER_MODEL_NAME` — nothing in
    `app/pipeline/summarization.py` or the orchestrator hard-codes a model.
  - `ExtractiveSummarizer` (offline fallback, opt-in via
    `SUMMARIZER_BACKEND=extractive`): a zero-download, keyword/heuristic
    summarizer used for CI and air-gapped smoke-testing, and to prove the
    `Summarizer` interface is genuinely swappable. It is not a substitute
    for `LocalLLMSummarizer`'s quality and is documented as such.

## 5. Setup and Execution

Requires Python 3.11+ and a working `ffmpeg`/`ffprobe` on `PATH`.

```bash
git clone <this-repo>
cd voice-mom-pipeline
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 5.1 FFmpeg Setup

- **Debian/Ubuntu**: `sudo apt-get install ffmpeg`
- **macOS (Homebrew)**: `brew install ffmpeg`
- **Windows**: download a build from https://www.gyan.dev/ffmpeg/builds/
  and add its `bin/` folder to `PATH`.
- Verify with `ffmpeg -version` and `ffprobe -version`. If your binaries
  aren't named `ffmpeg`/`ffprobe` or aren't on `PATH`, set
  `FFMPEG_BINARY`/`FFPROBE_BINARY` in `.env` to their full paths.

### 5.2 Windows-only: a *second*, shared FFmpeg build for diarization

The `ffmpeg`/`ffprobe` executables above are enough for audio
preprocessing, but `pyannote.audio` 4.x decodes audio via `torchcodec`
rather than `torchaudio`/`soundfile`, and on Windows that needs FFmpeg's
*shared library* DLLs (`avutil-*.dll`, `avcodec-*.dll`, `avformat-*.dll`,
`swresample-*.dll`, `swscale-*.dll`) — not just the `.exe` files. Gyan.dev's
default "full_build" (including what `winget install Gyan.FFmpeg` gives
you) is a **static** build that ships none of these DLLs; if you only have
that, diarization fails with:

```
Could not load libtorchcodec_core9.dll (or one of its dependencies)
```

even though the `.dll` files for TorchCodec itself are present — the
*missing* dependency is FFmpeg's own shared libraries, not TorchCodec's.

To fix it:

```powershell
# 1. Download a *shared* FFmpeg build (verified working asset, ~100MB)
New-Item -ItemType Directory -Force -Path E:\ffmpeg-shared | Out-Null
Invoke-WebRequest -Uri "https://github.com/GyanD/codexffmpeg/releases/download/8.1.2/ffmpeg-8.1.2-full_build-shared.zip" -OutFile "$env:TEMP\ffmpeg-shared.zip"
Expand-Archive -Path "$env:TEMP\ffmpeg-shared.zip" -DestinationPath E:\ffmpeg-shared -Force

# 2. Find the extracted bin/ folder (contains avutil-*.dll etc.)
Get-ChildItem -Path E:\ffmpeg-shared -Recurse -Filter "avutil-*.dll" | Select-Object -First 1 -ExpandProperty DirectoryName
```

Set the folder printed above as `FFMPEG_SHARED_DLL_DIR` in `.env`
(typically `E:\ffmpeg-shared\ffmpeg-8.1.2-full_build-shared\bin`). The app
registers it with `os.add_dll_directory()` at startup (see
`app/utils/windows_dll.py`) — this does not touch, replace, or require
removing your existing `ffmpeg.exe`/`FFMPEG_BINARY`, and does nothing on
Linux/macOS. This fix has been confirmed working end-to-end on a real
Windows machine (see Section 10, "Actual Successful End-to-End
Validation").

### 5.3 Model Setup / Download

All model weights are fetched from the Hugging Face Hub **on first use**
and cached locally (`~/.cache/huggingface` by default, or a custom
`HF_HOME`) — inference itself is always local.

1. **faster-whisper**: no account needed. The first request that reaches
   the `TRANSCRIBING` stage downloads `WHISPER_MODEL_NAME` (a few hundred
   MB to ~3GB for `large-v3`) automatically.
2. **pyannote diarization — gated, requires setup**:
   - Accept the license at
     https://huggingface.co/pyannote/speaker-diarization-community-1
     (and, if prompted, its dependency `pyannote/segmentation-3.0`).
   - Create a read-scoped access token at
     https://huggingface.co/settings/tokens.
   - Set `HUGGINGFACE_TOKEN=<your token>` in `.env`.
   - Without this, the `DIARIZING` stage fails with a structured
     `MODEL_LOADING_FAILED` error rather than a raw exception.
3. **Local summarization LLM**: no account needed for the default
   `Qwen/Qwen2.5-1.5B-Instruct`. The first request that reaches
   `GENERATING_SUMMARY` downloads it (~3GB).
4. To pre-download everything before your first real meeting (recommended
   for production), run a throwaway request against a short sample file,
   or trigger each loader directly:
   ```bash
   python -c "from app.config import get_settings; from app.pipeline.transcription import FasterWhisperTranscriber; FasterWhisperTranscriber(get_settings())._load_model()"
   python -c "from app.config import get_settings; from app.pipeline.diarization import PyannoteDiarizer; PyannoteDiarizer(get_settings())._load_pipeline()"
   python -c "from app.config import get_settings; from app.pipeline.summarization import LocalLLMSummarizer; LocalLLMSummarizer(get_settings())._load()"
   ```
   Once cached, set `HF_HUB_OFFLINE=1` to force fully offline operation.

### 5.4 Configuration

Everything is a `Settings` field in `app/config.py`, overridable via `.env`
or environment variables — nothing model/path-related is hard-coded in
business logic. See `.env.example` for the full list with defaults and
comments; the most commonly changed ones:

| Variable | Purpose |
|---|---|
| `WHISPER_MODEL_NAME`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_BEAM_SIZE` | ASR model + runtime |
| `WHISPER_LANGUAGE` | leave unset for auto-detection (required for multilingual support) |
| `WHISPER_SEGMENT_LEVEL_LANGUAGE_DETECTION` | opt-in, slower, more accurate per-segment language tagging for code-switched meetings |
| `DIARIZATION_MODEL_NAME`, `DIARIZATION_DEVICE`, `HUGGINGFACE_TOKEN` | diarization model + gated-model auth |
| `SUMMARIZER_BACKEND`, `SUMMARIZER_MODEL_NAME`, `SUMMARIZER_DEVICE` | MoM generation model |
| `SUMMARIZER_CHUNK_TOKEN_BUDGET`, `SUMMARIZER_CHUNK_OVERLAP_SEGMENTS` | hierarchical long-meeting summarization tuning |
| `ALIGNMENT_MIN_OVERLAP_RATIO` | temporal-alignment sensitivity |
| `MAX_CONCURRENT_JOBS` | background worker pool size |
| `MAX_UPLOAD_SIZE_MB`, `ALLOWED_EXTENSIONS` | upload validation |
| `MODEL_LOAD_TIMEOUT_SECONDS` | bounded timeout for model loading (defends against a hanging network call) |
| `FFMPEG_SHARED_DLL_DIR` | Windows-only; see Section 5.2 |

### 5.5 Running Locally

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

The API is then at `http://127.0.0.1:8000`, interactive docs at
`http://127.0.0.1:8000/docs`.

### 5.6 Hardware Requirements

Run `python scripts/check_environment.py` on your target machine — it
reports CPU/RAM/GPU/VRAM/disk and prints a suggested `.env` for that
hardware. General guidance:

| Profile | CPU-only | GPU |
|---|---|---|
| Minimum | 4 cores, 8GB RAM, `WHISPER_MODEL_NAME=small`, `SUMMARIZER_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct` | Any CUDA GPU with 6GB+ VRAM, `WHISPER_COMPUTE_TYPE=float16` |
| Recommended for `large-v3` + `Qwen2.5-1.5B-Instruct` | 8+ cores, 16GB+ RAM (will still be noticeably slower than GPU) | 8GB+ VRAM GPU |
| Disk | ~10GB free for model weight caches (Whisper + pyannote + summarizer) plus space for uploaded/processed audio |

The full pipeline (Whisper `large-v3` + pyannote diarization + Qwen2.5
generation) has been run end-to-end, CPU-only, on a real Windows machine
(~16 logical CPUs, ~15.7GB RAM) — see Section 10.

### 5.7 Docker

```bash
cp .env.example .env   # fill in HUGGINGFACE_TOKEN etc.
docker compose up --build
```

This builds a CPU image (see `Dockerfile`) with `ffmpeg` installed via
`apt`, mounts `./data` for persistent SQLite/transcripts/results, and
mounts a named volume for the Hugging Face model cache so weights aren't
re-downloaded on every container restart. For GPU inference, see the
commented-out `deploy.resources.reservations.devices` block in
`docker-compose.yml` (requires the NVIDIA Container Toolkit on the host)
plus setting `WHISPER_DEVICE=cuda` etc. in `.env`.

### 5.8 Optional Deployment Notes

This is intentionally a single-node service. To deploy beyond a single
Docker host:

- Put it behind a reverse proxy (nginx/Caddy) for TLS.
- Mount `data/` on persistent storage (a network volume or a managed disk)
  — SQLite plus the filesystem layout under `data/` is the entire state.
- If you outgrow a single node, the cleanest next step is swapping the
  `BackgroundWorker` (`app/pipeline/worker.py`) for a real queue (RQ,
  Celery, or a cloud queue) — the `Diarizer`/`Transcriber`/`Summarizer`
  interfaces and the orchestrator do not need to change for that.
- For GPU hosting, any machine with an NVIDIA GPU and the NVIDIA Container
  Toolkit can run the Docker image with the GPU block enabled.

## 6. API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/meetings` | Upload a meeting file (`multipart/form-data`, field `file`). Returns immediately with a `meeting_id` and `QUEUED` status. |
| `GET` | `/api/v1/meetings/{meeting_id}` | Current processing status + basic metadata. |
| `GET` | `/api/v1/meetings/{meeting_id}/transcript` | Structured, chronologically-ordered transcript (empty list until processing reaches that point). |
| `GET` | `/api/v1/meetings/{meeting_id}/summary` | Generated MoM (overview/key points/decisions/action items); `null` until `COMPLETED`. |
| `GET` | `/api/v1/meetings/{meeting_id}/result` | The complete structured result. Returns `409 RESULT_NOT_READY` until `COMPLETED`. |
| `GET` | `/health` | Liveness + environment info (ffmpeg availability, configured models). |

All error responses share one shape:

```json
{
  "status": "failed",
  "error": { "code": "AUDIO_EXTRACTION_FAILED", "message": "..." }
}
```

See `app/utils/errors.py` for the full list of structured error codes
(unsupported extension, empty upload, file too large, corrupt media, no
audio stream, empty audio, transcription/diarization/summarization
failure, model loading failure, GPU OOM, meeting not found, result not
ready, storage failure, and a generic internal-error fallback that is
used for anything unanticipated — the API never returns a raw traceback).

### 6.1 Example Request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/meetings \
  -F "file=@team_standup.wav"
```

```bash
curl http://127.0.0.1:8000/api/v1/meetings/8f14e45fceea167a5a36dedd4bea2543
```

### 6.2 Example Response

`POST /api/v1/meetings`:

```json
{
  "meeting_id": "8f14e45fceea167a5a36dedd4bea2543",
  "status": "QUEUED",
  "filename": "team_standup.wav",
  "message": "Meeting accepted for processing."
}
```

`GET /api/v1/meetings/{id}`:

```json
{
  "meeting_id": "8f14e45fceea167a5a36dedd4bea2543",
  "status": "COMPLETED",
  "filename": "team_standup.wav",
  "created_at": "2026-09-23T10:00:00+00:00",
  "updated_at": "2026-09-23T10:04:12+00:00",
  "error": null,
  "duration_seconds": 184.2,
  "languages": ["en", "hi"],
  "is_complete": true
}
```

### 6.3 Sample Output

`GET /api/v1/meetings/{id}/result` (abridged, illustrative shape):

```json
{
  "meeting_id": "8f14e45fceea167a5a36dedd4bea2543",
  "duration_seconds": 184.2,
  "languages": ["en", "hi"],
  "speakers": [
    {
      "speaker_id": "SPEAKER_00",
      "name": "Person 1",
      "speaking_time_seconds": 96.4,
      "speaking_percentage": 52.3,
      "segment_count": 18,
      "average_segment_duration_seconds": 5.4,
      "longest_segment_duration_seconds": 14.1
    },
    {
      "speaker_id": "SPEAKER_01",
      "name": "Person 2",
      "speaking_time_seconds": 87.8,
      "speaking_percentage": 47.7,
      "segment_count": 15,
      "average_segment_duration_seconds": 5.9,
      "longest_segment_duration_seconds": 11.0
    }
  ],
  "transcript": [
    {
      "segment_id": 1,
      "speaker_id": "SPEAKER_00",
      "speaker_name": "Person 1",
      "start": 0.4,
      "end": 3.1,
      "text": "Let's get started, morning everyone.",
      "language": "en",
      "overlapping_speakers": []
    }
  ],
  "summary": {
    "overview": "The team reviewed deployment status and assigned follow-up testing.",
    "key_points": ["Deployment status reviewed", "Budget review deferred to next week"],
    "decisions": ["Deploy tomorrow morning"],
    "action_items": [
      { "task": "Complete API testing", "responsible": "Person 2", "deadline": "Friday" }
    ]
  }
}
```

## 7. Testing

```bash
pip install -r requirements-dev.txt
pytest -v
ruff check .
```

As of this writing, the suite has **127 passing tests** (0 failed) across
`test_validation.py`, `test_audio.py`, `test_transcription.py`,
`test_diarization.py`, `test_alignment.py`, `test_statistics.py`,
`test_summarization.py`, `test_storage.py`, `test_orchestrator.py`,
`test_worker.py`, `test_api.py`, `test_timeout.py`, and
`test_windows_dll.py`, and **`ruff check .` reports no issues** (clean).

**What is exercised for real** (no mocking, no model download needed):
file validation; ffmpeg probing/extraction/normalization against real
synthetic audio/video generated on the fly with ffmpeg's `lavfi` sources;
the full temporal-alignment algorithm; speaker statistics;
`ExtractiveSummarizer`; SQLite + filesystem persistence; the orchestrator's
stage sequencing and error handling; the background worker's concurrency;
the model-loading timeout guard; the Windows DLL-search-path configuration
logic; and a full HTTP-level integration test
(`test_api.py::test_full_pipeline_happy_path_end_to_end`) that uploads a
real WAV file through the actual FastAPI app and checks the persisted
transcript/statistics/summary/result all the way through.

**What is verified through unit tests with the real model client stubbed
out** (`FasterWhisperTranscriber`, `PyannoteDiarizer`, `LocalLLMSummarizer`):
their segment-building, error-translation (OOM/model-loading/generic
failure -> the correct structured error code), and JSON-parsing/hierarchical-
chunking logic. The stub replaces only the underlying model object's
`transcribe`/`__call__`/`generate` call — everything around it is real code.

**What has additionally been confirmed against real model weights** (not
mocked, on a real machine, outside the automated test suite): see Section
10, "Actual Successful End-to-End Validation".

## 8. Known Limitations

Documented honestly:

- **The real end-to-end validation run (Section 10) used a single English
  meeting recording.** Hindi and Odia transcription, and code-switched
  (multi-language) meetings, are implemented and supported by the
  selected pipeline (Whisper's built-in multilingual capability plus this
  project's segment-level language-detection option — see Section 9), and
  are covered by unit tests with a stubbed ASR model, but they have **not
  yet been validated end-to-end against real Hindi or Odia audio**. Run
  the pipeline against real Hindi/Odia (and mixed-language) recordings in
  your own environment before relying on that specific claim in
  production.
- Similarly, the real end-to-end run involved 2 speakers over one ~2-minute
  recording; longer meetings, meetings with more than 2 speakers, and
  video (rather than audio) input are covered by unit/integration tests
  with synthetic fixtures, but have not all been separately re-confirmed
  against real model weights in the same run as Section 10.
- **Whisper detects language once per ~30s window**, not truly per-word.
  `WHISPER_SEGMENT_LEVEL_LANGUAGE_DETECTION=true` improves per-segment
  language tagging for code-switched meetings by re-running detection on
  each segment's own audio slice, but this is a heuristic improvement, not
  a perfect per-word language model, and it multiplies inference cost.
- **pyannote/speaker-diarization-community-1 is a gated model.** It
  requires accepting a license and an HF access token; without it,
  diarization fails with a structured `MODEL_LOADING_FAILED` error rather
  than silently falling back to anything.
- **Overlapping/cross-talk speech is attributed to exactly one dominant
  speaker per segment** (the one with the greater temporal overlap); the
  other overlapping speaker(s) are recorded on `overlapping_speakers` but
  do not get a separate contribution to speaking-time statistics. The real
  validation run in Section 10 produced 3 such ambiguous-overlap segments,
  confirming this logic is exercised in practice. See the module
  docstrings in `app/pipeline/alignment.py` and `app/pipeline/statistics.py`
  for the full rationale.
- **No evaluation metrics (WER/CER/DER) are reported** because no
  ground-truth-labeled recordings were available. None are fabricated;
  once you have labeled recordings, wire them through a small script using
  e.g. `jiwer` (WER/CER) and `pyannote.metrics` (DER) against this
  pipeline's output.
- **The background worker is in-process and not persistent**: a meeting
  stuck mid-processing when the process restarts stays at its last
  recorded status rather than being automatically retried. For production
  durability, this is the natural place to introduce a real job queue
  later — deliberately not done here to keep a single-node deployment
  simple.
- **Single-node SQLite**: fine for a "reliability over scale" deployment,
  but not intended for multi-writer/multi-node deployment as-is.
- **On Windows, pyannote 4.x's TorchCodec audio backend needs a *shared*
  FFmpeg build**, not just `ffmpeg.exe`/`ffprobe.exe` on `PATH` (Section
  5.2). This has been implemented, unit-tested, and confirmed working
  end-to-end on a real Windows machine as part of the Section 10
  validation run.

## 9. Multilingual Support

The pipeline is built to be multilingual, not English-only:

- ASR (`faster-whisper` / Whisper `large-v3`) performs automatic language
  detection by default — `WHISPER_LANGUAGE` is left unset so the model is
  never hard-coded to a single language.
- **English, Hindi, and Odia are the three languages this project
  specifically targets and are supported by the selected model and
  configuration.** Whisper `large-v3` has broad multilingual training
  coverage including Hindi and Odia.
- `WHISPER_SEGMENT_LEVEL_LANGUAGE_DETECTION=true` re-runs language
  detection on each segment's own audio slice (rather than once for the
  whole file), which is what makes code-switched meetings — where
  speakers switch between, say, English and Hindi mid-conversation —
  taggable at the segment level (`language` field on each transcript
  segment).
- Every downstream stage (diarization, alignment, statistics,
  summarization) is language-agnostic: none of them branch on the
  detected language, so multilingual support does not depend on
  special-casing any one language.

**Actual validation status**: the end-to-end run documented in Section 10
was performed on an English-language recording, and English was the
language actually detected and transcribed in that run. Hindi and Odia
transcription is **supported by the pipeline's design and configuration,
and covered by unit tests with a stubbed ASR model, but has not yet been
end-to-end validated against real Hindi or Odia speech** — this is called
out explicitly in Known Limitations (Section 8) rather than claimed as
tested.

## 10. No Paid or External LLM API Dependency

Every model used by this pipeline runs locally, on the machine you deploy
it on:

- **Speech-to-text**: local `faster-whisper` (Whisper `large-v3` by
  default), loaded from a local/cached copy of the open Whisper weights.
- **Diarization**: local `pyannote.audio` pipeline
  (`speaker-diarization-community-1`), loaded from a local/cached copy of
  the open pyannote weights.
- **Summarization / Minutes of Meeting generation**: a local, open-weight,
  instruction-tuned LLM (`Qwen/Qwen2.5-1.5B-Instruct` by default) run
  on-machine via `transformers.AutoModelForCausalLM.generate()` — **not**
  OpenAI, Anthropic, Google, or any other paid/hosted LLM API.

Model weights are fetched from the Hugging Face Hub once and cached
locally; nothing about *inference* requires network access or a paid API
key at request time, and the pipeline can be run fully offline
(`HF_HUB_OFFLINE=1`) once all three models are cached. No API keys for any
paid LLM/ASR provider are read, required, or referenced anywhere in
`app/config.py` or elsewhere in the codebase.

## 11. Actual Successful End-to-End Validation

The full pipeline — upload through Minutes-of-Meeting generation — was run
end-to-end on a real machine (Windows, CPU-only, ~16 logical CPUs,
~15.7GB RAM) against a real recorded meeting, with every model-facing
stage running its **real, local** model (no mocking, no stubbing):

| Metric | Result |
|---|---|
| Audio duration | **120.12 seconds** |
| Speakers detected | **2** |
| Transcript segments produced | **20** |
| Diarization turns produced | **27** |
| Segments with ambiguous/overlapping speaker attribution | **3** |
| Language detected | **English** |
| Transcription engine | **local Whisper** (`faster-whisper`, loaded from local cache) |
| Diarization engine | **local pyannote** (`speaker-diarization-community-1`, loaded from local cache) |
| Summarization engine | **local Qwen** (`Qwen/Qwen2.5-1.5B-Instruct`, loaded from local cache, CPU generation) |
| Final meeting status | **`COMPLETED`** |

Alongside this real-model run, the automated test suite passed in full —
**127 tests passed**, 0 failed — and static analysis is clean —
**`ruff check .` reports no issues**.

This confirms, on real hardware and with real model weights rather than
stubs: the Windows shared-FFmpeg/TorchCodec fix (Section 5.2), the
diarization output-unwrapping fix (`app/pipeline/diarization.py`), and the
Qwen generation fix (`app/pipeline/summarization.py`) all work together
correctly, end to end, and that the pipeline reaches `COMPLETED` — not
just that its individual stages pass unit tests in isolation.

As noted in Section 8 (Known Limitations), this specific run was a single
English-language, 2-speaker, ~2-minute recording; Hindi/Odia audio,
additional speaker counts, and longer recordings are supported by the
implementation but have not each individually been re-run through this
same real-model validation and should be validated in your own
environment before being relied upon for those specific cases.

---

## Project Structure

```text
voice-mom-pipeline/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, exception handlers
│   ├── config.py                 # Settings (all configurable parameters)
│   ├── api/
│   │   ├── meetings.py           # upload/status/transcript/summary/result
│   │   ├── health.py
│   │   └── dependencies.py
│   ├── pipeline/
│   │   ├── orchestrator.py       # stage sequencing + status tracking
│   │   ├── worker.py             # bounded ThreadPoolExecutor background jobs
│   │   ├── preprocessing.py      # validate_file / extract+normalize audio
│   │   ├── transcription.py      # Transcriber / FasterWhisperTranscriber
│   │   ├── diarization.py        # Diarizer / PyannoteDiarizer + name mapping
│   │   ├── alignment.py          # CRITICAL: temporal overlap alignment
│   │   ├── statistics.py         # speaker statistics
│   │   └── summarization.py      # Summarizer / LocalLLMSummarizer / ExtractiveSummarizer
│   ├── models/
│   │   └── schemas.py            # every Pydantic model in the system
│   ├── storage/
│   │   ├── database.py           # SQLite connection/schema
│   │   └── repository.py         # all persistence logic
│   └── utils/
│       ├── audio.py               # ffmpeg/ffprobe wrappers
│       ├── validation.py          # filename/size validation
│       ├── errors.py              # structured PipelineError taxonomy
│       ├── logging.py
│       ├── timeout.py             # bounded-timeout guard for model loading
│       └── windows_dll.py         # Windows shared-FFmpeg DLL search path setup
├── tests/                         # 127 tests, see "Testing" above
├── scripts/
│   ├── check_environment.py       # hardware inspection
│   ├── verify_torchcodec_windows.py
│   └── verify_qwen_generation.py
├── data/{uploads,processed,transcripts,results}/   # runtime-generated, gitignored
├── requirements.txt / requirements-dev.txt
├── Dockerfile / docker-compose.yml
├── .env.example
└── pyproject.toml                 # pytest/ruff/black/mypy config
```

## Acceptance Criteria Status (stated honestly)

| Criterion | Status |
|---|---|
| Audio upload / video upload / file validation | Implemented, tested end-to-end with real synthetic media |
| Audio extraction | Implemented, tested end-to-end with real ffmpeg |
| English transcription | Implemented and **confirmed end-to-end with real local Whisper** (Section 10) |
| Hindi/Odia transcription | Implemented and supported by the selected model/configuration; unit-tested with a stubbed model; **not yet confirmed end-to-end against real Hindi/Odia audio** |
| Multilingual/code-switching support | Implemented (segment-level language + optional re-detection); not validated against real code-switched audio |
| Multiple speakers detected, identities consistent | Implemented, tested, and **confirmed end-to-end (2 speakers, 27 diarization turns) with real local pyannote** |
| Timestamps, chronological ordering, speaker/text alignment | Implemented and thoroughly tested; **confirmed end-to-end (20 segments, 3 ambiguous overlaps) in real run** |
| Speaking duration / percentage / segment counts | Implemented, tested |
| Summary / key points / decisions / action items, generated locally | Implemented (`LocalLLMSummarizer`, no paid API) and **confirmed end-to-end with real local Qwen generation** (Section 10); `ExtractiveSummarizer` also runs and is tested for real |
| Results persisted | Implemented, tested |
| API works | Implemented, tested end-to-end |
| Invalid files handled, errors logged | Implemented, tested |
| Tests exist and pass | **127 tests, all passing** |
| Static analysis clean | **`ruff check .` reports no issues** |
| README complete | This document |
| No paid/external LLM API used | True by construction — every model is local |
| No fabricated evaluation metrics | True — no WER/CER/DER numbers are claimed; see Known Limitations |
