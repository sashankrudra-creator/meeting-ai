# Voice-Based Minutes of Meeting Pipeline — CPU image.
#
# For GPU inference, use an nvidia/cuda base image instead and set
# WHISPER_DEVICE=cuda / DIARIZATION_DEVICE=cuda / SUMMARIZER_DEVICE=cuda in
# your .env (see the README's "Hardware Requirements" and "Docker
# Instructions" sections).

FROM python:3.11-slim

# ffmpeg is a hard runtime dependency (Section 4) — installed via apt, not pip.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Model weights are downloaded lazily on first use and cached in the
# Hugging Face cache dir; mount a volume there in production so a container
# restart doesn't re-download multi-GB model files (see docker-compose.yml).
ENV HF_HOME=/root/.cache/huggingface

RUN mkdir -p data/uploads data/processed data/transcripts data/results

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
