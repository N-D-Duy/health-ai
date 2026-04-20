#!/usr/bin/env bash
set -e

OLLAMA_URL="${OLLAMA_HOST:-http://ollama:11434}"
MODEL_NAME="${OLLAMA_MODEL:-chat-bot-yte}"

echo "[chat-bot] Waiting for Ollama at ${OLLAMA_URL}..."
for i in $(seq 1 40); do
    if curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
        echo "[chat-bot] Ollama is ready."
        break
    fi
    echo "[chat-bot] Attempt ${i}/40 — retrying in 3s..."
    sleep 3
done

if ! curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    echo "[chat-bot] ERROR: Ollama not reachable. Aborting."
    exit 1
fi

# Tạo model chat-bot-yte từ Modelfile nếu chưa tồn tại.
# Model được tạo qua REST API, không cần ollama CLI trong container.
if ! curl -sf "${OLLAMA_URL}/api/tags" | grep -q "\"${MODEL_NAME}\""; then
    echo "[chat-bot] Model '${MODEL_NAME}' not found — creating from Modelfile..."
    MODELFILE_CONTENT=$(cat /app/Modelfile)
    curl -sf -X POST "${OLLAMA_URL}/api/create" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"${MODEL_NAME}\", \"modelfile\": $(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$MODELFILE_CONTENT")}" \
        | tail -1
    echo "[chat-bot] Model '${MODEL_NAME}' created."
else
    echo "[chat-bot] Model '${MODEL_NAME}' already exists."
fi

echo "[chat-bot] Starting uvicorn on :8000..."
exec python -m uvicorn main:app --host 0.0.0.0 --port 8000
