FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    # HF cache trỏ vào network volume (override bằng env trong compose)
    HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Cài PyTorch CUDA 12.4 trước để sentence-transformers & reranker dùng GPU torch.
# CUDA 12.4 wheels tương thích ngược với driver CUDA 12.6 trên RunPod.
RUN pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
