
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /app
COPY app /app
ENV WHISPER_MODEL_SIZE=small
ENV VAD_AGGRESSIVENESS=2
ENV MAX_SPEAKERS=8
ENV MIN_SPEAKERS=1
ENTRYPOINT ["python", "/app/process.py", "--input", "/app/../input", "--output", "/app/../output"]
