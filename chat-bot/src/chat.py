from __future__ import annotations

from typing import Any, Literal, TypedDict

import ollama

from .config import (
    MAX_HISTORY_MESSAGES,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    RAG_CONTEXT_TITLE,
    SYSTEM_GUARDRAIL_VI,
    TOP_K,
)
from .retrieval import query_chromadb


Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


DISCLAIMER = "Thông tin chỉ mang tính tham khảo; bạn nên gặp bác sĩ để được chẩn đoán và điều trị."


def _build_context_block(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return ""
    lines: list[str] = [f"{RAG_CONTEXT_TITLE} (trích đoạn liên quan):"]
    for idx, ch in enumerate(chunks, start=1):
        meta = ch.get("metadata") or {}
        qtype = meta.get("qtype")
        prefix = f"[{idx}]"
        if qtype:
            prefix = f"[{idx} | {qtype}]"
        lines.append(f"{prefix} {ch.get('text','')}".strip())
    return "\n".join(lines).strip()


def _ensure_disclaimer(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return DISCLAIMER
    if "tham khảo" in t.lower() and "bác sĩ" in t.lower():
        return t
    return f"{t}\n\n{DISCLAIMER}"


def chat_once(
    user_message: str,
    history: list[ChatMessage] | None = None,
    *,
    top_k: int = TOP_K,
) -> tuple[str, list[ChatMessage]]:
    history = list(history or [])
    user_message = (user_message or "").strip()
    if not user_message:
        return "Bạn hãy nhập câu hỏi cụ thể hơn.", history

    retrieved = query_chromadb(user_message, k=top_k)
    ctx = _build_context_block(
        [{"text": c.text, "metadata": c.metadata, "id": c.id, "distance": c.distance} for c in retrieved]
    )

    system = SYSTEM_GUARDRAIL_VI
    if ctx:
        system = f"{system}\n\n{ctx}\n\nHãy ưu tiên dùng context ở trên nếu phù hợp."

    messages: list[ChatMessage] = [{"role": "system", "content": system}]

    if MAX_HISTORY_MESSAGES > 0:
        trimmed = history[-MAX_HISTORY_MESSAGES:]
        for m in trimmed:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    try:
        client = ollama.Client(host=OLLAMA_HOST)
        res = client.chat(model=OLLAMA_MODEL, messages=messages, stream=False)
        assistant = ((res or {}).get("message") or {}).get("content") or ""
        assistant = _ensure_disclaimer(assistant)
    except Exception:
        assistant = (
            "Mình chưa kết nối được Ollama để tạo câu trả lời. "
            "Bạn hãy đảm bảo Ollama đang chạy và đã `ollama create` model theo `Modelfile`, "
            f"sau đó thử lại.\n\n{DISCLAIMER}"
        )

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant})
    return assistant, history

