FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir \
    faster-whisper==1.0.3 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart \
    requests \
    tqdm

WORKDIR /app
COPY transcribe_api.py .

# Загружаем модель на CPU во время сборки
ARG WHISPER_MODEL=base
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('$WHISPER_MODEL', device='cpu', compute_type='int8_float32')"

EXPOSE 80

CMD ["uvicorn", "transcribe_api:app", "--host", "0.0.0.0", "--port", "80"]