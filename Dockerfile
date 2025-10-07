FROM python:3.11-slim

# Système : ffmpeg + toolchain pour webrtcvad
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "huggingface_hub>=0.24.0"

# Télécharge le modèle faster-whisper large-v3 AU BUILD (dans l'image)
ENV WHISPER_MODEL_DIR=/models/large-v3
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Systran/faster-whisper-large-v3', local_dir='/models/large-v3', local_dir_use_symlinks=False)"

# Code
COPY app/ /app/app/
COPY entrypoint_gcp.py /app/entrypoint_gcp.py

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/entrypoint_gcp.py"]
