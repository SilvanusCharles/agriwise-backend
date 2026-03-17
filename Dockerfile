FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# ── FIX 1: Correct model repo name (was kisan-vaani-agricultural-advisor) ──
ENV MODEL_REPO="MrCahrles00/agriadvisor-model"
ENV HF_HOME="/app/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/app/.cache/huggingface/transformers"

# Pre-download model at build time to avoid cold-start timeouts
# Uses the corrected repo name
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='MrCahrles00/agriadvisor-model', \
    local_dir='/app/model_cache', \
    local_dir_use_symlinks=False, \
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*'] \
)"

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

EXPOSE 7860

# HF Spaces expects the app on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]