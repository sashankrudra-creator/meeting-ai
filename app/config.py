"""
Central application configuration.

Every model name, device, compute setting and path used anywhere in the
pipeline MUST be read from a `Settings` instance rather than hard-coded in
business logic. This is what lets the transcription model, diarization
model and summarization model be swapped later (see Section 24 of the
spec: "the system should be model-agnostic where practical").

Values are loaded from environment variables / a `.env` file (see
`.env.example`) with sensible, hardware-aware defaults for a CPU-only
development machine. Override anything via environment variables, e.g.:

    WHISPER_DEVICE=cuda WHISPER_COMPUTE_TYPE=float16 uvicorn app.main:app
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    app_name: str = "Voice-Based Minutes of Meeting Pipeline"
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    # ------------------------------------------------------------------
    # Storage (SQLite + filesystem). See app/storage and Section 17.
    # ------------------------------------------------------------------
    data_dir: Path = Field(default=Path("data"))
    uploads_subdir: str = "uploads"
    processed_subdir: str = "processed"
    transcripts_subdir: str = "transcripts"
    results_subdir: str = "results"
    database_path: Path = Field(default=Path("data/mom_pipeline.db"))

    # ------------------------------------------------------------------
    # Upload validation (Section 2 / 18)
    # ------------------------------------------------------------------
    max_upload_size_mb: int = Field(default=2048, description="Hard cap on upload size in MB")
    allowed_extensions: tuple[str, ...] = (
        ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac",
        ".mp4", ".mkv", ".mov", ".avi", ".webm",
    )
    upload_chunk_size_bytes: int = Field(
        default=1024 * 1024,
        description="Chunk size used when streaming an uploaded file to disk. Bounds peak "
                     "memory usage independently of the upload's total size — the endpoint "
                     "never buffers the whole file in RAM before validating/saving it.",
    )

    # ------------------------------------------------------------------
    # Audio preprocessing (Section 4 / FFmpeg)
    # ------------------------------------------------------------------
    ffmpeg_binary: str = Field(default="ffmpeg")
    ffprobe_binary: str = Field(default="ffprobe")
    target_sample_rate: int = Field(default=16000)
    target_channels: int = Field(default=1)  # mono
    ffmpeg_timeout_seconds: int = Field(default=3600)

    # ------------------------------------------------------------------
    # Speech recognition — faster-whisper (Section 4)
    # No paid/external LLM API is used anywhere in this pipeline.
    # ------------------------------------------------------------------
    whisper_model_name: str = Field(
        default="large-v3",
        description="faster-whisper model size/name, e.g. tiny, base, small, "
                     "medium, large-v3, or a local CTranslate2 model directory.",
    )
    whisper_device: str = Field(default="cpu", description="'cpu' or 'cuda'")
    whisper_compute_type: str = Field(
        default="int8",
        description="int8 (CPU-friendly), float16/float32 (GPU), int8_float16, etc.",
    )
    whisper_beam_size: int = Field(default=5)
    whisper_language: str | None = Field(
        default=None,
        description="Force a language (e.g. 'en'). Leave unset/None for automatic "
                     "language detection, which is required by the spec.",
    )
    whisper_vad_filter: bool = Field(default=True)
    whisper_supported_languages: tuple[str, ...] = ("en", "hi", "or")
    whisper_cpu_threads: int = Field(default=0, description="0 = let CTranslate2 decide")
    whisper_num_workers: int = Field(default=1)
    whisper_download_root: Path | None = Field(default=None)
    whisper_segment_level_language_detection: bool = Field(
        default=False,
        description="If true, re-run a lightweight per-segment language identification "
                     "pass on each ASR segment's own audio slice instead of tagging every "
                     "segment with the single language detected for the whole file. This "
                     "materially improves accuracy for code-switched meetings (Section 10) "
                     "at the cost of one extra model call per segment, so it defaults to "
                     "off for speed and is meant to be enabled for known multilingual "
                     "meetings.",
    )
    whisper_segment_language_min_duration_seconds: float = Field(
        default=1.0,
        description="Segments shorter than this are too short for reliable re-detection "
                     "and keep the file-level language instead.",
    )

    # ------------------------------------------------------------------
    # ASR quality / hallucination handling (Section 4/18)
    # ------------------------------------------------------------------
    asr_quality_filter_mode: str = Field(
        default="flag",
        description="How to handle low-confidence ASR segments, judged from faster-whisper's "
                     "own avg_logprob/no_speech_prob per segment. 'off' disables this entirely "
                     "(original behavior, no flags, no filtering). 'flag' (the default) keeps "
                     "every segment but marks low-confidence ones via "
                     "TranscriptSegment.low_confidence, so nothing is silently discarded. "
                     "'filter' drops low-confidence segments outright. Deliberately not "
                     "'filter' by default: aggressively deleting segments risks discarding "
                     "legitimate (if difficult) speech, not just hallucinations.",
    )
    asr_no_speech_prob_threshold: float = Field(
        default=0.6,
        description="A segment with no_speech_prob >= this is treated as a likely "
                     "silence/hallucination artifact (only consulted when "
                     "asr_quality_filter_mode != 'off').",
    )
    asr_avg_logprob_threshold: float = Field(
        default=-1.0,
        description="A segment with avg_logprob <= this is treated as low-confidence "
                     "(only consulted when asr_quality_filter_mode != 'off').",
    )

    # ------------------------------------------------------------------
    # Speaker diarization — pyannote (Section 5)
    # ------------------------------------------------------------------
    diarization_model_name: str = Field(default="pyannote/speaker-diarization-community-1")
    diarization_device: str = Field(default="cpu")
    huggingface_token: str | None = Field(
        default=None,
        description="Required to download the gated pyannote diarization model. "
                     "Read from HUGGINGFACE_TOKEN / HF_TOKEN env var.",
    )
    diarization_min_speakers: int | None = Field(default=None)
    diarization_max_speakers: int | None = Field(default=None)
    ffmpeg_shared_dll_dir: Path | None = Field(
        default=None,
        description="Windows only. Directory containing a *shared* FFmpeg build's DLLs "
                     "(avutil-*.dll, avcodec-*.dll, avformat-*.dll, swresample-*.dll, "
                     "swscale-*.dll, ...). pyannote.audio 4.x decodes audio via `torchcodec`, "
                     "which needs these to be discoverable via os.add_dll_directory() on "
                     "Windows (PATH alone is not enough since Python 3.8's DLL-loading "
                     "changes) — a common *static* FFmpeg build (e.g. Gyan.dev's default "
                     "'full_build', including the one winget installs) does NOT ship these "
                     "DLLs at all and cannot satisfy this requirement no matter what "
                     "directory is added. This is unrelated to FFMPEG_BINARY/FFPROBE_BINARY "
                     "above, which can keep pointing at any working ffmpeg/ffprobe "
                     "executable for audio preprocessing. Ignored on Linux/macOS.",
    )

    # ------------------------------------------------------------------
    # Temporal alignment (Section 6)
    # ------------------------------------------------------------------
    alignment_min_overlap_ratio: float = Field(
        default=0.0,
        description="Minimum fraction of an ASR segment's duration that must overlap "
                     "a diarization turn for that turn to be considered a candidate "
                     "speaker for the segment. 0.0 means any overlap counts.",
    )
    alignment_unknown_speaker_label: str = Field(default="unknown")

    # ------------------------------------------------------------------
    # Local summarization / MoM engine (Section 11 / 12)
    # ------------------------------------------------------------------
    summarizer_backend: str = Field(
        default="local_llm",
        description="Which Summarizer implementation to use. 'local_llm' is the only "
                     "production backend; 'extractive' is a lightweight offline "
                     "fallback usable with no model download for smoke-testing.",
    )
    summarizer_model_name: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="Local, open-weight instruction-tuned model run entirely on-device "
                     "via transformers. Swappable without touching business logic.",
    )
    summarizer_device: str = Field(default="cpu")
    summarizer_max_new_tokens: int = Field(default=512)
    summarizer_temperature: float = Field(default=0.2)
    summarizer_chunk_token_budget: int = Field(
        default=1500,
        description="Approximate max tokens of transcript text per hierarchical chunk.",
    )
    summarizer_chunk_overlap_segments: int = Field(
        default=2,
        description="Number of trailing segments repeated at the start of the next "
                     "chunk to preserve conversational context across chunk boundaries.",
    )
    summarizer_reduce_token_budget: int = Field(
        default=1500,
        description="Approximate max tokens of combined chunk-level partial summaries that "
                     "may be sent in a single hierarchical-reduce prompt. If the partials from "
                     "a very long meeting would exceed this, they are recursively grouped and "
                     "reduced in bounded batches instead of being concatenated into one "
                     "unbounded prompt (Section 12).",
    )

    # ------------------------------------------------------------------
    # Background processing (Section 16)
    # ------------------------------------------------------------------
    max_concurrent_jobs: int = Field(default=1, description="Simple in-process worker pool size")

    # ------------------------------------------------------------------
    # Model-loading timeout (defense-in-depth against a hanging network
    # call to the Hugging Face Hub during model loading; see
    # app/utils/timeout.py and app/pipeline/transcription.py for the
    # production incident this fixes).
    # ------------------------------------------------------------------
    model_load_timeout_seconds: float = Field(
        default=120.0,
        description="Maximum time allowed for loading any pipeline model (Whisper, "
                     "diarization, summarization) before the load is treated as failed. "
                     "Model loading is normally fast from a local cache and this exists "
                     "purely as a safety net: some environments experience a hanging "
                     "(not fast-failing) network call to the Hugging Face Hub even when "
                     "the model is already cached, which previously left a meeting stuck "
                     "at its current status forever with no error. Set to a higher value "
                     "on a slow connection when a model has not been downloaded yet.",
    )

    @field_validator("data_dir", "database_path", mode="before")
    @classmethod
    def _coerce_path(cls, v: str | Path) -> Path:
        return Path(v)

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / self.uploads_subdir

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / self.processed_subdir

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / self.transcripts_subdir

    @property
    def results_dir(self) -> Path:
        return self.data_dir / self.results_subdir

    def ensure_directories(self) -> None:
        for path in (self.uploads_dir, self.processed_dir, self.transcripts_dir, self.results_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings instance.

    Using a cached factory function (rather than a module-level singleton)
    keeps configuration testable: tests can call `get_settings.cache_clear()`
    after monkeypatching environment variables.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
