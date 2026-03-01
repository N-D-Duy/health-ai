#!/usr/bin/env bash
# Chạy API chat (từ thư mục chat-bot). Cần Ollama đang chạy và đã tạo model chat-bot-yte.
cd "$(dirname "$0")/.."
exec python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
