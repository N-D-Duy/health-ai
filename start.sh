#!/usr/bin/env bash
# ============================================================
# start.sh — Khởi chạy chat-bot + stt-bot trực tiếp trên host
# Dùng cho RunPod (không cần Docker).
#
# Usage:
#   bash start.sh              # dùng defaults
#   LLM_MODEL=qwen2.5:14b bash start.sh
#   WHISPER_SIZE=large-v3 bash start.sh
#
# Biến env có thể override:
#   LLM_MODEL       LLM base model để pull (default: qwen2.5:7b)
#   CUSTOM_MODEL    tên model custom trong Ollama (default: chat-bot-yte)
#   WHISPER_SIZE    faster-whisper model size (default: medium)
#   CHAT_PORT       port chat-bot (default: 8000)
#   STT_PORT        port stt-bot  (default: 8010)
#   WORKSPACE       thư mục data persistent (default: /workspace)
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_ROOT="$ROOT/chat-bot"
STT_ROOT="$ROOT/stt-bot"

LLM_MODEL="${LLM_MODEL:-qwen2.5:7b}"
CUSTOM_MODEL="${CUSTOM_MODEL:-chat-bot-yte}"
WHISPER_SIZE="${WHISPER_SIZE:-medium}"
CHAT_PORT="${CHAT_PORT:-8000}"
STT_PORT="${STT_PORT:-8010}"
WORKSPACE="${WORKSPACE:-/workspace}"

PID_DIR="$ROOT/.pids"
LOG_DIR="$ROOT/.logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

# ── Màu log ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; RESET='\033[0m'
info()  { echo -e "${CYAN}[start.sh]${RESET} $*"; }
ok()    { echo -e "${GREEN}[start.sh]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[start.sh]${RESET} $*"; }
error() { echo -e "${RED}[start.sh]${RESET} $*"; exit 1; }

# ── Cleanup khi tắt script ────────────────────────────────────
cleanup() {
    info "Shutting down services..."
    for pid_file in "$PID_DIR"/*.pid; do
        [ -f "$pid_file" ] || continue
        pid=$(cat "$pid_file")
        kill "$pid" 2>/dev/null && info "Stopped PID $pid ($(basename "$pid_file" .pid))" || true
        rm -f "$pid_file"
    done
    # Không stop Ollama — để giữ model trong VRAM nếu restart service
}
trap cleanup SIGINT SIGTERM EXIT

# ============================================================
# 1. System deps
# ============================================================
info "=== [1/7] Kiểm tra system dependencies ==="

MISSING_PKGS=()
command -v ffmpeg  &>/dev/null || MISSING_PKGS+=(ffmpeg)
command -v zstd    &>/dev/null || MISSING_PKGS+=(zstd)
command -v curl    &>/dev/null || MISSING_PKGS+=(curl)
if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    info "Cài: ${MISSING_PKGS[*]}"
    apt-get update -qq && apt-get install -y -qq "${MISSING_PKGS[@]}"
fi
ok "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

# Detect Python 3.12 hoặc fallback
PYTHON=$(command -v python3.12 || command -v python3 || command -v python)
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python: $PYTHON ($PY_VER)"

# ============================================================
# 2. Ollama
# ============================================================
info "=== [2/7] Ollama ==="

if ! command -v ollama &>/dev/null; then
    info "Cài Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    export PATH="/usr/local/bin:$PATH"
fi
ok "Ollama: $(ollama --version 2>/dev/null || echo 'installed')"

# Start Ollama nếu chưa chạy
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    info "Khởi động ollama serve..."
    OLLAMA_MODELS="$WORKSPACE/ollama/models" ollama serve >> "$LOG_DIR/ollama.log" 2>&1 &
    echo $! > "$PID_DIR/ollama.pid"
    info "Đợi Ollama ready..."
    for i in $(seq 1 30); do
        curl -sf http://localhost:11434/api/tags &>/dev/null && break
        sleep 2
    done
    curl -sf http://localhost:11434/api/tags &>/dev/null || error "Ollama không start được."
    ok "Ollama ready."
else
    ok "Ollama đang chạy sẵn."
fi

# ============================================================
# 3. Pull model + tạo custom model
# ============================================================
info "=== [3/7] Models ==="

# Kiểm tra base model đã có chưa (tránh pull lại ~5GB)
if ollama list 2>/dev/null | grep -q "^${LLM_MODEL}"; then
    ok "Model ${LLM_MODEL} đã có sẵn."
else
    info "Pull ${LLM_MODEL}..."
    ollama pull "$LLM_MODEL"
fi

# Tạo custom model chat-bot-yte nếu chưa có
if ! ollama list 2>/dev/null | grep -q "^${CUSTOM_MODEL}"; then
    info "Tạo model ${CUSTOM_MODEL} từ Modelfile..."
    MODELFILE_TMP=$(mktemp)
    sed "s|^FROM .*|FROM ${LLM_MODEL}|" "$CHAT_ROOT/Modelfile" > "$MODELFILE_TMP"
    ollama create "$CUSTOM_MODEL" -f "$MODELFILE_TMP"
    rm -f "$MODELFILE_TMP"
    ok "Model ${CUSTOM_MODEL} đã tạo."
else
    ok "Model ${CUSTOM_MODEL} đã có sẵn."
fi

# ============================================================
# 4. Warmup model vào VRAM (1 lần, giữ suốt phiên)
# ============================================================
info "=== [4/7] Warmup LLM vào VRAM ==="
curl -sf -X POST http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${CUSTOM_MODEL}\", \"prompt\": \"ok\", \"stream\": false, \"options\": {\"num_predict\": 1}}" \
    > /dev/null
ok "Model ${CUSTOM_MODEL} đã load vào VRAM."

# ============================================================
# 5. Cài pip dependencies
# ============================================================
info "=== [5/7] Python dependencies ==="

# chat-bot venv
if [ ! -d "$CHAT_ROOT/.venv" ]; then
    info "Tạo venv cho chat-bot..."
    "$PYTHON" -m venv "$CHAT_ROOT/.venv"
fi
info "Cài chat-bot requirements..."
"$CHAT_ROOT/.venv/bin/pip" install --quiet --upgrade pip
# Cài torch CPU (chat-bot dùng embedding/reranker, Ollama lo LLM)
# Dùng torch+cpu để tiết kiệm VRAM cho Ollama + Whisper
"$CHAT_ROOT/.venv/bin/pip" install --quiet \
    torch --index-url https://download.pytorch.org/whl/cpu
"$CHAT_ROOT/.venv/bin/pip" install --quiet -r "$CHAT_ROOT/requirements.txt"
ok "chat-bot deps OK."

# stt-bot venv
if [ ! -d "$STT_ROOT/.venv" ]; then
    info "Tạo venv cho stt-bot..."
    "$PYTHON" -m venv "$STT_ROOT/.venv"
fi
info "Cài stt-bot requirements..."
"$STT_ROOT/.venv/bin/pip" install --quiet --upgrade pip
"$STT_ROOT/.venv/bin/pip" install --quiet -r "$STT_ROOT/requirements.txt"
ok "stt-bot deps OK."

# ============================================================
# 6. Chuẩn bị data directories + ChromaDB
# ============================================================
info "=== [6/7] Data directories ==="

CHROMA_DEST="$WORKSPACE/chroma_db"
mkdir -p "$CHROMA_DEST"
mkdir -p "$WORKSPACE/hf_cache"
mkdir -p "$WORKSPACE/stt_data/uploads"
mkdir -p "$WORKSPACE/stt_data/store"

SESSIONS_FILE="$WORKSPACE/stt_data/store/sessions.json"
if [ ! -f "$SESSIONS_FILE" ]; then
    echo '{"sessions":{},"transcripts":{},"extractions":{},"soap_notes":{},"reviews":{},"finalizations":{}}' \
        > "$SESSIONS_FILE"
    ok "sessions.json khởi tạo."
fi

# Copy ChromaDB lên workspace nếu chưa có
CHROMA_SRC="$CHAT_ROOT/chroma_db"
if [ -d "$CHROMA_SRC" ] && [ -n "$(ls -A "$CHROMA_SRC" 2>/dev/null)" ]; then
    if [ -z "$(ls -A "$CHROMA_DEST" 2>/dev/null)" ]; then
        cp -r "$CHROMA_SRC/." "$CHROMA_DEST/"
        ok "ChromaDB copied → $CHROMA_DEST"
    else
        ok "ChromaDB đã có tại $CHROMA_DEST."
    fi
fi

# ============================================================
# 7. Start services
# ============================================================
info "=== [7/7] Khởi chạy services ==="

# ── chat-bot ────────────────────────────────────────────────
STT_VENV_SITE="$STT_ROOT/.venv/lib/python${PY_VER}/site-packages"
CUBLAS_LIB="$STT_VENV_SITE/nvidia/cublas/lib"
CUDNN_LIB="$STT_VENV_SITE/nvidia/cudnn/lib"

CHAT_ENV="HF_HOME=$WORKSPACE/hf_cache \
TRANSFORMERS_CACHE=$WORKSPACE/hf_cache \
CHROMA_PATH=$CHROMA_DEST \
OLLAMA_HOST=http://localhost:11434 \
OLLAMA_MODEL=$CUSTOM_MODEL \
CUDA_VISIBLE_DEVICES="

env HF_HOME="$WORKSPACE/hf_cache" \
    TRANSFORMERS_CACHE="$WORKSPACE/hf_cache" \
    CHROMA_PATH="$CHROMA_DEST" \
    OLLAMA_HOST="http://localhost:11434" \
    OLLAMA_MODEL="$CUSTOM_MODEL" \
    CUDA_VISIBLE_DEVICES="" \
    "$CHAT_ROOT/.venv/bin/uvicorn" main:app \
        --host 0.0.0.0 --port "$CHAT_PORT" \
        --app-dir "$CHAT_ROOT" \
        >> "$LOG_DIR/chat-bot.log" 2>&1 &
echo $! > "$PID_DIR/chat-bot.pid"
ok "chat-bot PID $(cat "$PID_DIR/chat-bot.pid") → :${CHAT_PORT}  [log: .logs/chat-bot.log]"

# ── stt-bot ─────────────────────────────────────────────────
env HF_HOME="$WORKSPACE/hf_cache" \
    OLLAMA_HOST="http://localhost:11434" \
    OLLAMA_MODEL="$CUSTOM_MODEL" \
    WHISPER_MODEL_SIZE="$WHISPER_SIZE" \
    WHISPER_DEVICE="cuda" \
    WHISPER_COMPUTE_TYPE="float16" \
    UPLOAD_DIR="$WORKSPACE/stt_data/uploads" \
    STORE_FILE="$WORKSPACE/stt_data/store/sessions.json" \
    LD_LIBRARY_PATH="${CUBLAS_LIB}:${CUDNN_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$STT_ROOT/.venv/bin/uvicorn" main:app \
        --host 0.0.0.0 --port "$STT_PORT" \
        --app-dir "$STT_ROOT" \
        >> "$LOG_DIR/stt-bot.log" 2>&1 &
echo $! > "$PID_DIR/stt-bot.pid"
ok "stt-bot  PID $(cat "$PID_DIR/stt-bot.pid") → :${STT_PORT}  [log: .logs/stt-bot.log]"

# ── Chờ services ready ────────────────────────────────────────
info "Đợi services ready..."
sleep 3
for svc in "chat-bot:$CHAT_PORT/health" "stt-bot:$STT_PORT/health"; do
    name="${svc%%:*}"; endpoint="${svc#*:}"
    for i in $(seq 1 20); do
        if curl -sf "http://localhost:${endpoint}" &>/dev/null; then
            ok "$name ready → http://localhost:${endpoint%%/*}"
            break
        fi
        sleep 2
    done
done

echo ""
ok "=== Stack running ==="
echo -e "  chat-bot : http://0.0.0.0:${CHAT_PORT}"
echo -e "  stt-bot  : http://0.0.0.0:${STT_PORT}"
echo -e "  Logs     : tail -f $LOG_DIR/chat-bot.log"
echo -e "             tail -f $LOG_DIR/stt-bot.log"
echo ""
info "Ctrl+C để dừng tất cả."

# Giữ script sống để trap cleanup hoạt động
wait
