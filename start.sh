#!/usr/bin/env bash
# ============================================================
# start.sh — Khởi chạy chat-bot + stt-bot trực tiếp trên host
# Dùng cho RunPod (không cần Docker).
#
# Usage:
#   bash start.sh                                    # defaults
#   LLM_MODEL=qwen2.5:14b WHISPER_SIZE=large-v3 bash start.sh
#
# Biến env có thể override:
#   LLM_MODEL       LLM base model để pull     (default: qwen2.5:7b)
#   CUSTOM_MODEL    tên model custom Ollama     (default: chat-bot-yte)
#   WHISPER_SIZE    faster-whisper model size   (default: medium)
#   CHAT_PORT       port chat-bot               (default: 8000)
#   STT_PORT        port stt-bot                (default: 8010)
#   WORKSPACE       thư mục data persistent     (default: /workspace)
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
VPS_REGISTER_URL="${VPS_REGISTER_URL:-https://son.ndwcdy.me/internal/register}"
VPS_HEARTBEAT_URL="${VPS_HEARTBEAT_URL:-https://son.ndwcdy.me/internal/heartbeat}"
SERVICE_TOKEN="${SERVICE_TOKEN:-your_secret}"
# Thêm dòng này vào ngay sau:
[ "$SERVICE_TOKEN" = "your_secret" ] && die "SERVICE_TOKEN chưa được set!"

PID_DIR="$ROOT/.pids"
LOG_DIR="$ROOT/.logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

# ── Màu log ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; RESET='\033[0m'
info()  { echo -e "${CYAN}[start.sh]${RESET} $*"; }
ok()    { echo -e "${GREEN}[start.sh]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[start.sh]${RESET} $*"; }
die()   { echo -e "${RED}[start.sh] ERROR:${RESET} $*"; exit 1; }

# ── Cleanup khi Ctrl+C / tắt ─────────────────────────────────
cleanup() {
    echo ""
    info "Shutting down services..."
    for pid_file in "$PID_DIR"/*.pid; do
        [ -f "$pid_file" ] || continue
        pid=$(cat "$pid_file")
        name=$(basename "$pid_file" .pid)
        kill "$pid" 2>/dev/null \
            && info "Stopped $name (PID $pid)" \
            || true
        rm -f "$pid_file"
    done
}
trap cleanup SIGINT SIGTERM EXIT

# ============================================================
# 1. Detect Python TRƯỚC khi làm gì khác
# ============================================================
info "=== [1/7] System dependencies ==="

PYTHON=""
for py in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$py" &>/dev/null; then
        PYTHON=$(command -v "$py")
        break
    fi
done
[ -n "$PYTHON" ] || die "Không tìm thấy Python. Cài python3 trước."
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python: $PYTHON ($PY_VER)"

# ── Apt packages ──────────────────────────────────────────────
MISSING_PKGS=()
command -v curl   &>/dev/null || MISSING_PKGS+=(curl)
command -v git    &>/dev/null || MISSING_PKGS+=(git)
command -v ffmpeg &>/dev/null || MISSING_PKGS+=(ffmpeg)
command -v zstd   &>/dev/null || MISSING_PKGS+=(zstd)
command -v lspci  &>/dev/null || MISSING_PKGS+=(pciutils)
command -v gcc    &>/dev/null || MISSING_PKGS+=(build-essential)
# Kiểm tra python3-venv: thử tạo venv test không dùng pip
if ! "$PYTHON" -m venv --without-pip /tmp/_venv_test &>/dev/null; then
    MISSING_PKGS+=(python3-venv)
fi
rm -rf /tmp/_venv_test

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    info "Cài apt packages: ${MISSING_PKGS[*]}"
    apt-get update -qq && apt-get install -y -qq "${MISSING_PKGS[@]}"
    ok "apt OK."
else
    ok "Tất cả apt packages đã có."
fi
ok "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

# ============================================================
# 2. Ollama
# ============================================================
info "=== [2/7] Ollama ==="

if ! command -v ollama &>/dev/null; then
    info "Cài Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    export PATH="/usr/local/bin:$PATH"
fi
command -v ollama &>/dev/null || die "Ollama install thất bại."
ok "Ollama: $(ollama --version 2>/dev/null | head -1)"

# Start ollama serve nếu chưa chạy
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    info "Khởi động ollama serve..."
    OLLAMA_MODELS="$WORKSPACE/ollama/models" ollama serve \
        >> "$LOG_DIR/ollama.log" 2>&1 &
    echo $! > "$PID_DIR/ollama.pid"
    info "Đợi Ollama ready..."
    for i in $(seq 1 30); do
        curl -sf http://localhost:11434/api/tags &>/dev/null && break
        sleep 2
    done
    curl -sf http://localhost:11434/api/tags &>/dev/null \
        || die "Ollama không start được. Xem: $LOG_DIR/ollama.log"
    ok "Ollama ready."
else
    ok "Ollama đang chạy sẵn."
fi

# ============================================================
# 3. Pull model + tạo custom model
# ============================================================
info "=== [3/7] Models ==="

if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "${LLM_MODEL}"; then
    ok "Base model ${LLM_MODEL} đã có sẵn."
else
    info "Pull ${LLM_MODEL} (có thể mất vài phút)..."
    ollama pull "$LLM_MODEL"
    ok "Pull xong."
fi

if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "${CUSTOM_MODEL}"; then
    ok "Custom model ${CUSTOM_MODEL} đã có sẵn."
else
    info "Tạo model ${CUSTOM_MODEL} từ Modelfile..."
    MODELFILE_TMP=$(mktemp)
    sed "s|^FROM .*|FROM ${LLM_MODEL}|" "$CHAT_ROOT/Modelfile" > "$MODELFILE_TMP"
    ollama create "$CUSTOM_MODEL" -f "$MODELFILE_TMP"
    rm -f "$MODELFILE_TMP"
    ok "Model ${CUSTOM_MODEL} đã tạo."
fi

# ============================================================
# 4. Warmup model vào VRAM
# ============================================================
info "=== [4/7] Warmup LLM vào VRAM ==="
curl -sf -X POST http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${CUSTOM_MODEL}\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
    > /dev/null
ok "Model ${CUSTOM_MODEL} đã load vào VRAM."

# ============================================================
# 5. Python venv + pip install
# ============================================================
info "=== [5/7] Python dependencies ==="

# ── chat-bot venv ─────────────────────────────────────────────
CHAT_STAMP="$CHAT_ROOT/.venv/.install_stamp"
if [ ! -d "$CHAT_ROOT/.venv" ] || [ ! -f "$CHAT_ROOT/.venv/bin/python" ]; then
    info "Tạo venv chat-bot..."
    "$PYTHON" -m venv "$CHAT_ROOT/.venv"
fi
if [ ! -f "$CHAT_STAMP" ] || [ "$CHAT_ROOT/requirements.txt" -nt "$CHAT_STAMP" ]; then
    info "Cài chat-bot requirements..."
    "$CHAT_ROOT/.venv/bin/python" -m pip install --quiet --upgrade pip
    # Torch CUDA 12.4 — tương thích với driver CUDA 12.x trên RunPod
    "$CHAT_ROOT/.venv/bin/python" -m pip install --quiet \
        torch --index-url https://download.pytorch.org/whl/cu124
    "$CHAT_ROOT/.venv/bin/python" -m pip install --quiet \
        -r "$CHAT_ROOT/requirements.txt"
    touch "$CHAT_STAMP"
    ok "chat-bot deps OK."
else
    ok "chat-bot deps đã up-to-date (stamp)."
fi

# ── stt-bot venv ──────────────────────────────────────────────
STT_STAMP="$STT_ROOT/.venv/.install_stamp"
if [ ! -d "$STT_ROOT/.venv" ] || [ ! -f "$STT_ROOT/.venv/bin/python" ]; then
    info "Tạo venv stt-bot..."
    "$PYTHON" -m venv "$STT_ROOT/.venv"
fi
if [ ! -f "$STT_STAMP" ] || [ "$STT_ROOT/requirements.txt" -nt "$STT_STAMP" ]; then
    info "Cài stt-bot requirements..."
    "$STT_ROOT/.venv/bin/python" -m pip install --quiet --upgrade pip
    "$STT_ROOT/.venv/bin/python" -m pip install --quiet \
        -r "$STT_ROOT/requirements.txt"
    touch "$STT_STAMP"
    ok "stt-bot deps OK."
else
    ok "stt-bot deps đã up-to-date (stamp)."
fi

# ============================================================
# 6. Data directories
# ============================================================
info "=== [6/7] Data directories ==="

CHROMA_DEST="$WORKSPACE/chroma_db"
mkdir -p "$CHROMA_DEST" \
         "$WORKSPACE/hf_cache" \
         "$WORKSPACE/stt_data/uploads" \
         "$WORKSPACE/stt_data/store" \
         "$WORKSPACE/ollama/models"

SESSIONS_FILE="$WORKSPACE/stt_data/store/sessions.json"
if [ ! -f "$SESSIONS_FILE" ]; then
    printf '{"sessions":{},"transcripts":{},"extractions":{},"soap_notes":{},"reviews":{},"finalizations":{}}' \
        > "$SESSIONS_FILE"
    ok "sessions.json khởi tạo."
fi

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

# LD_LIBRARY_PATH cho faster-whisper (CUDA libs được pip install vào site-packages)
STT_SITE="$STT_ROOT/.venv/lib/python${PY_VER}/site-packages"
CUBLAS_LIB="$STT_SITE/nvidia/cublas/lib"
CUDNN_LIB="$STT_SITE/nvidia/cudnn/lib"
EXTRA_LD="${CUBLAS_LIB}:${CUDNN_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── chat-bot ──────────────────────────────────────────────────
env \
    HF_HOME="$WORKSPACE/hf_cache" \
    TRANSFORMERS_CACHE="$WORKSPACE/hf_cache" \
    CHROMA_PATH="$CHROMA_DEST" \
    OLLAMA_HOST="http://localhost:11434" \
    OLLAMA_MODEL="$CUSTOM_MODEL" \
    "$CHAT_ROOT/.venv/bin/uvicorn" main:app \
        --host 0.0.0.0 --port "$CHAT_PORT" \
        --app-dir "$CHAT_ROOT" \
        >> "$LOG_DIR/chat-bot.log" 2>&1 &
echo $! > "$PID_DIR/chat-bot.pid"
ok "chat-bot PID $(cat "$PID_DIR/chat-bot.pid") → :${CHAT_PORT}  [log: .logs/chat-bot.log]"

# ── stt-bot ───────────────────────────────────────────────────
env \
    HF_HOME="$WORKSPACE/hf_cache" \
    OLLAMA_HOST="http://localhost:11434" \
    OLLAMA_MODEL="$CUSTOM_MODEL" \
    WHISPER_MODEL_SIZE="$WHISPER_SIZE" \
    WHISPER_DEVICE="cuda" \
    WHISPER_COMPUTE_TYPE="float16" \
    UPLOAD_DIR="$WORKSPACE/stt_data/uploads" \
    STORE_FILE="$WORKSPACE/stt_data/store/sessions.json" \
    LD_LIBRARY_PATH="$EXTRA_LD" \
    "$STT_ROOT/.venv/bin/uvicorn" main:app \
        --host 0.0.0.0 --port "$STT_PORT" \
        --app-dir "$STT_ROOT" \
        >> "$LOG_DIR/stt-bot.log" 2>&1 &
echo $! > "$PID_DIR/stt-bot.pid"
ok "stt-bot  PID $(cat "$PID_DIR/stt-bot.pid") → :${STT_PORT}  [log: .logs/stt-bot.log]"

# ── Đợi cả hai healthy ────────────────────────────────────────
info "Đợi services ready..."
sleep 4
for entry in "chat-bot:${CHAT_PORT}" "stt-bot:${STT_PORT}"; do
    name="${entry%%:*}"; port="${entry##*:}"
    for i in $(seq 1 20); do
        if curl -sf "http://localhost:${port}/health" &>/dev/null; then
            ok "$name ready → http://0.0.0.0:${port}"
            break
        fi
        [ "$i" -eq 20 ] && warn "$name chưa healthy sau 40s — xem $LOG_DIR/${name}.log"
        sleep 2
    done
done

POD_ID="${RUNPOD_POD_ID:-unknown}"

CHAT_URL="https://${POD_ID}-${CHAT_PORT}.proxy.runpod.net"
STT_URL="https://${POD_ID}-${STT_PORT}.proxy.runpod.net"


register_service() {
  local name=$1
  local url=$2
  info "Register $name → $url"

  for i in {1..10}; do
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$VPS_REGISTER_URL" \
      -H "Authorization: Bearer $SERVICE_TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"service\":\"$name\",\"url\":\"$url\"}")

    if [[ "$http_code" =~ ^2 ]]; then
      ok "$name registered (HTTP $http_code)"
      return 0
    fi

    warn "Register $name failed (HTTP $http_code), retry $i/10..."
    sleep $((3 * i))
  done

  warn "Register $name failed after 10 retries"
  return 1
}

heartbeat_loop() {
  local fail_count=0
  local max_fail=5

  while true; do
    local chat_ok stt_ok
    curl -sf "http://localhost:${CHAT_PORT}/health" &>/dev/null && chat_ok=true || chat_ok=false
    curl -sf "http://localhost:${STT_PORT}/health"  &>/dev/null && stt_ok=true  || stt_ok=false

    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$VPS_HEARTBEAT_URL" \
      -H "Authorization: Bearer $SERVICE_TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"pod_id\":\"$POD_ID\",\"services\":{\"chat\":$chat_ok,\"stt\":$stt_ok}}")

    if [[ "$http_code" =~ ^2 ]]; then
      fail_count=0
    else
      fail_count=$((fail_count + 1))
      warn "Heartbeat failed (HTTP $http_code) — $fail_count/$max_fail"

      if [ "$fail_count" -ge "$max_fail" ]; then
        warn "VPS có thể đã restart, re-registering..."
        register_service "chat" "$CHAT_URL"
        register_service "stt"  "$STT_URL"
        fail_count=0
      fi
    fi

    sleep 30
  done
}

echo ""
ok "=== Stack running ==="
echo -e "  chat-bot : $CHAT_URL"
echo -e "  stt-bot  : $STT_URL"
echo -e "  Logs     : tail -f ${LOG_DIR}/chat-bot.log"
echo -e "             tail -f ${LOG_DIR}/stt-bot.log"
echo ""
info "Ctrl+C để dừng."

wait