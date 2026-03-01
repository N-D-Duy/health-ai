#!/usr/bin/env bash
# Full setup: cài Ollama (nếu chưa có) → pull qwen2.5:7b → tạo model chat-bot-yte từ Modelfile.
# Chạy từ bất kỳ đâu; script tự cd vào thư mục chat-bot.
set -e
CHAT_BOT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CHAT_BOT_ROOT"
export PATH="/usr/local/bin:/usr/bin:$PATH"

echo "== 1. Kiểm tra / cài Ollama..."
if ! command -v ollama &>/dev/null; then
  echo "Ollama chưa có. Đang cài (cần quyền sudo)..."
  curl -fsSL https://ollama.com/install.sh | sh
  export PATH="/usr/local/bin:$PATH"
  if ! command -v ollama &>/dev/null; then
    echo "Sau khi cài, chạy lại: $0"
    exit 1
  fi
fi

echo "== 2. Khởi động Ollama (nếu chưa chạy)..."
if command -v systemctl &>/dev/null && systemctl is-active ollama &>/dev/null; then
  echo "Ollama service đang chạy."
else
  (ollama serve &) 2>/dev/null || true
  echo "Đợi Ollama sẵn sàng..."
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -s http://127.0.0.1:11434/api/tags &>/dev/null; then break; fi
    sleep 1
  done
  if ! curl -s http://127.0.0.1:11434/api/tags &>/dev/null; then
    echo "Không kết nối được Ollama. Thử: sudo systemctl start ollama hoặc chạy 'ollama serve' trong terminal khác."
    exit 1
  fi
fi

echo "== 3. Pull model qwen2.5:7b..."
ollama pull qwen2.5:7b

echo "== 4. Tạo model chat-bot-yte từ Modelfile..."
ollama create chat-bot-yte -f "$CHAT_BOT_ROOT/Modelfile"

echo ""
echo "=== Xong. Tiếp theo:"
echo "  1. Chạy API:  ./scripts/run_server.sh"
echo "  2. Terminal khác test chat:  cd $CHAT_BOT_ROOT && source .venv/bin/activate && python -m scripts.test_chat \"Triệu chứng tiểu đường?\""
echo "  Hoặc test API:  curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{\"message\": \"Triệu chứng tiểu đường?\"}'"
