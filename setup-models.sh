#!/usr/bin/env bash
# First-time setup trên RunPod:
#   1. Tạo thư mục trên network volume
#   2. Copy chroma_db từ repo lên volume
#   3. Pull LLM model qua Ollama
#   4. Tạo custom model chat-bot-yte từ Modelfile
#
# Chạy 1 lần sau khi pod khởi động:
#   bash setup-models.sh
#
# Có thể override model mặc định:
#   CHAT_LLM=qwen2.5:32b bash setup-models.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOLUME_PATH="${VOLUME_PATH:-/workspace}"
# đang test trên máy local nên để model nhẹ
CHAT_LLM="${CHAT_LLM:-qwen2.5:7b}"
CUSTOM_MODEL="${CUSTOM_MODEL:-chat-bot-yte}"

echo "=== [1/5] Tạo thư mục trên network volume: ${VOLUME_PATH} ==="
mkdir -p "${VOLUME_PATH}/ollama/models"
mkdir -p "${VOLUME_PATH}/hf_cache"
mkdir -p "${VOLUME_PATH}/chroma_db"
mkdir -p "${VOLUME_PATH}/stt_data/uploads"
mkdir -p "${VOLUME_PATH}/stt_data/store"

# Init sessions.json nếu chưa có
[ -f "${VOLUME_PATH}/stt_data/store/sessions.json" ] || echo '{"sessions":{},"transcripts":{},"extractions":{},"soap_notes":{},"reviews":{},"finalizations":{}}' > "${VOLUME_PATH}/stt_data/store/sessions.json"

echo "=== [2/5] Copy ChromaDB lên network volume ==="
CHROMA_SRC="${SCRIPT_DIR}/chat-bot/chroma_db"
if [ -d "$CHROMA_SRC" ] && [ -n "$(ls -A "$CHROMA_SRC")" ]; then
    if [ -z "$(ls -A "${VOLUME_PATH}/chroma_db")" ]; then
        cp -r "${CHROMA_SRC}/." "${VOLUME_PATH}/chroma_db/"
        echo "  Đã copy chroma_db."
    else
        echo "  chroma_db đã có dữ liệu, bỏ qua."
    fi
else
    echo "  WARN: ${CHROMA_SRC} rỗng hoặc không tồn tại — bỏ qua."
fi

echo "=== [3/5] Khởi động Ollama container ==="
docker compose up -d ollama

echo "  Đợi Ollama ready..."
for i in $(seq 1 30); do
    if docker compose exec ollama ollama list > /dev/null 2>&1; then
        echo "  Ollama ready."
        break
    fi
    echo "  Attempt ${i}/30..."
    sleep 4
done

if ! docker compose exec ollama ollama list > /dev/null 2>&1; then
    echo "ERROR: Ollama không start được."
    exit 1
fi

echo "=== [4/5] Pull model ${CHAT_LLM} ==="
docker compose exec ollama ollama pull "${CHAT_LLM}"

echo "=== [5/5] Tạo model ${CUSTOM_MODEL} từ Modelfile ==="
# Cập nhật FROM trong Modelfile để dùng đúng base model đã pull
MODELFILE_TMP=$(mktemp)
sed "s|^FROM .*|FROM ${CHAT_LLM}|" "${SCRIPT_DIR}/chat-bot/Modelfile" > "$MODELFILE_TMP"

CONTAINER_ID=$(docker compose ps -q ollama)
docker cp "$MODELFILE_TMP" "${CONTAINER_ID}:/tmp/Modelfile"
docker compose exec ollama ollama create "${CUSTOM_MODEL}" -f /tmp/Modelfile
rm -f "$MODELFILE_TMP"

echo ""
echo "=== Setup hoàn tất! ==="
echo "Chạy toàn bộ stack:"
echo "  docker compose up -d"
echo ""
echo "Logs:"
echo "  docker compose logs -f chat-bot"
echo "  docker compose logs -f stt-bot"
