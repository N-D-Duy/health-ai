#!/usr/bin/env python3
"""
Test RAG chat: gọi chat_once (ChromaDB + Ollama) từ CLI.
Chạy từ thư mục chat-bot: python -m scripts.test_chat [câu hỏi]
Hoặc tương tác: python -m scripts.test_chat
"""
from __future__ import annotations

import argparse
import os
import sys

# Đảm bảo chạy từ repo root chat-bot
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chat import chat_once


def main() -> None:
    ap = argparse.ArgumentParser(description="Test RAG chat (ChromaDB + Ollama)")
    ap.add_argument("query", nargs="*", help="Câu hỏi (nếu bỏ trống sẽ chạy chế độ nhập liên tục)")
    args = ap.parse_args()

    if args.query:
        text = " ".join(args.query).strip()
        if not text:
            print("Bạn cần nhập câu hỏi.", file=sys.stderr)
            sys.exit(1)
        reply, _ = chat_once(text)
        print(reply)
        return

    # Chế độ tương tác
    history: list[dict] = []
    print("Chat RAG (ChromaDB + Ollama). Gõ câu hỏi và Enter; 'quit' hoặc 'exit' để thoát.\n")
    while True:
        try:
            q = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        reply, history = chat_once(q, history)
        print(f"\nBot: {reply}\n")


if __name__ == "__main__":
    main()
