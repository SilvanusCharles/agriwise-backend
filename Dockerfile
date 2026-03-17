FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user main.py .

ENV HF_REPO="MrCahrles00/agriadvisor-model"
ENV MODEL_PATH="/app/kisan_vaani_model"

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='MrCahrles00/agriadvisor-model', local_dir='/app/kisan_vaani_model')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]