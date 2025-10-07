FROM python:3.11-slim

# I/O immédiat + pip sans cache
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dépendances système :
# - ffmpeg : conversion audio
# - libsndfile1 : lecture WAV (soundfile/librosa)
# - libgomp1 : OpenMP (accélère torch/faster-whisper CPU)
# - git : utile en dépannage
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg git libsndfile1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances Python
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Code
COPY app /app

# Dossiers par défaut (montables en volumes)
RUN mkdir -p /input /output /context

# Valeurs par défaut (tu peux override au run ou via le workflow)
ENV WHISPER_MODEL_SIZE=small \
    VAD_AGGRESSIVENESS=2 \
    MAX_SPEAKERS=8 \
    MIN_SPEAKERS=1 \
    MULTILANG=true \
    PRIMARY_LANG=fr \
    SECONDARY_LANG=en \
    WHISPER_BEAM_SIZE=5 \
    WHISPER_BEST_OF=5 \
    INITIAL_PROMPT_REPEAT=1 \
    SPEAKER_SIM_THRESHOLD=0.76 \
    MAX_CHUNK_MINUTES=15 \
    CHUNK_OVERLAP_MS=3500 \
    MAX_WAV_MB=150 \
    VAD_FRAME_MS=20

# Volumes standards (facilite le run local)
VOLUME ["/input", "/output", "/context"]

# Entrée + défauts (remplaçables)
ENTRYPOINT ["python", "/app/process.py"]
CMD ["--input", "/input", "--output", "/output"]
