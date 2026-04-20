#!/usr/bin/env bash
set -e

# Tìm đường dẫn site-packages (giống logic run.sh trên máy local).
# nvidia-cublas-cu12 và nvidia-cudnn-cu12 cài vào đây qua pip.
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
CUBLAS_LIB="$SITE/nvidia/cublas/lib"
CUDNN_LIB="$SITE/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="$CUBLAS_LIB:$CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "[stt-bot] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

OLLAMA_URL="${OLLAMA_HOST:-http://ollama:11434}"

echo "[stt-bot] Waiting for Ollama at ${OLLAMA_URL}..."
for i in $(seq 1 40); do
    if curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
        echo "[stt-bot] Ollama is ready."
        break
    fi
    echo "[stt-bot] Attempt ${i}/40 — retrying in 3s..."
    sleep 3
done

if ! curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    echo "[stt-bot] ERROR: Ollama not reachable. Aborting."
    exit 1
fi

echo "[stt-bot] Starting uvicorn on :8010..."
exec python -m uvicorn main:app --host 0.0.0.0 --port 8010
