---
title: Agriwise Backend
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Download model from HF Hub at build time
COPY --chown=user main.py .

ENV HF_REPO="MrCahrles00/agriadvisor-model"
ENV MODEL_PATH="/app/kisan_vani_model"

# Download model files from HF Hub
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='MrCahrles00/agriadvisor-model', local_dir='/app/kisan_vani_model')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]