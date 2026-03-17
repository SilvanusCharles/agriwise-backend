FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Download model from HF Hub at build time
COPY --chown=user main.py .

ENV HF_REPO="MrCarhles00/kisan-vaani-agricultural-advisor"
ENV MODEL_PATH="/app/kisan_vani_model"

# Download model files from HF Hub
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='MrCarhles00/kisan-vaani-agricultural-advisor', local_dir='/app/kisan_vaani_model')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]