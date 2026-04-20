FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/hf_cache

# ffmpeg cần thiết cho faster-whisper xử lý audio
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Cài tất cả dependencies kể cả nvidia-cublas-cu12 & nvidia-cudnn-cu12.
# Entrypoint sẽ tự set LD_LIBRARY_PATH trỏ vào site-packages/nvidia/ để
# ctranslate2 (faster-whisper) tìm đúng CUDA libs — giống cách run.sh hiện tại.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8010

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
