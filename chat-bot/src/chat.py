from __future__ import annotations

from typing import Any, Literal, TypedDict

import ollama

from .config import (
    MAX_HISTORY_MESSAGES,
    MEDICAL_CLASSIFIER_PROMPT_VI,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    QUERY_EXPANSION_ENABLED,
    QUERY_EXPANSION_N,
    QUERY_EXPANSION_PROMPT_VI,
    RAG_CONTEXT_TITLE,
    RAG_DISTANCE_OFF_TOPIC_THRESHOLD,
    RERANK_ENABLED,
    RERANK_TOP_N,
    REFUSAL_OFF_TOPIC_VI,
    RETRIEVE_CANDIDATES,
    SYSTEM_GUARDRAIL_VI,
    TOP_K,
)
from .retrieval import (
    RetrievedChunk,
    hybrid_query,
    query_chromadb,
    reciprocal_rank_fusion,
    rerank_chunks,
)


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


def _expand_query(query: str, client: ollama.Client, n: int = QUERY_EXPANSION_N) -> list[str]:
    """Dùng LLM sinh thêm n cách hỏi khác nhau cho cùng câu hỏi gốc."""
    if not QUERY_EXPANSION_ENABLED or n <= 0:
        return [query]
    prompt = QUERY_EXPANSION_PROMPT_VI.format(query=query, n=n)
    try:
        res = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            stream=False,
            options={"num_predict": 200, "temperature": 0.4},
        )
        raw = ((res or {}).get("response") or "").strip()
        # Parse: mỗi dòng là 1 câu hỏi, bỏ prefix số/ký tự thừa
        lines = [
            line.strip().lstrip("-•‣◦123456789. ")
            for line in raw.splitlines()
            if line.strip()
        ]
        expansions = [l for l in lines if len(l) > 5][:n]
        return [query] + expansions
    except Exception:
        return [query]  # fail-open: dùng query gốc nếu lỗi


def _is_medical_query(user_message: str, client: ollama.Client) -> bool:
    """Dùng LLM để phân loại câu hỏi có thuộc lĩnh vực y tế/sức khỏe không."""
    prompt = MEDICAL_CLASSIFIER_PROMPT_VI.format(query=user_message)
    try:
        res = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            stream=False,
            options={"num_predict": 10, "temperature": 0},
        )
        answer = ((res or {}).get("response") or "").strip().upper()
        # Chấp nhận "CÓ", "CO", "YES" → medical; mọi thứ còn lại → off-topic
        return answer.startswith("CÓ") or answer.startswith("CO") or answer.startswith("YES")
    except Exception:
        return True  # fail-open: nếu classify lỗi thì để qua RAG xử lý tiếp


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

    client = ollama.Client(host=OLLAMA_HOST)

    # Bước 1: LLM classifier – xác định câu hỏi có thuộc y tế/sức khỏe không.
    # Đây là cơ chế chính, không dùng keyword cứng.
    if not _is_medical_query(user_message, client):
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": REFUSAL_OFF_TOPIC_VI})
        return REFUSAL_OFF_TOPIC_VI, history

    # Bước 2a: Query Expansion – sinh thêm các cách hỏi khác nhau.
    queries = _expand_query(user_message, client)

    # Bước 2b: Hybrid Search (Semantic + BM25) cho từng query, gộp bằng RRF.
    candidates = RETRIEVE_CANDIDATES if RERANK_ENABLED else top_k
    all_lists: list[list[RetrievedChunk]] = [
        hybrid_query(q, k=candidates) for q in queries
    ]
    merged = reciprocal_rank_fusion(all_lists) if len(all_lists) > 1 else (all_lists[0] if all_lists else [])

    # Bước 2c: Reranking – chọn top-N chunk thực sự liên quan nhất.
    final_chunks = rerank_chunks(user_message, merged, top_n=RERANK_TOP_N if RERANK_ENABLED else top_k)

    # Xác định có context hợp lệ đưa vào prompt không.
    if RERANK_ENABLED and final_chunks:
        # Nếu rerank đang bật: tin tưởng score của reranker
        use_ctx = True
    else:
        # Fallback: dùng ngưỡng cosine distance gốc
        best_distance = min((c.distance for c in final_chunks if c.distance is not None), default=1.0)
        use_ctx = bool(final_chunks) and best_distance <= RAG_DISTANCE_OFF_TOPIC_THRESHOLD

    ctx = _build_context_block(
        [{"text": c.text, "metadata": c.metadata, "id": c.id, "distance": c.distance} for c in final_chunks]
    ) if use_ctx else ""

    system = SYSTEM_GUARDRAIL_VI
    if ctx:
        system = f"{system}\n\n{ctx}\n\nHãy ưu tiên dùng context ở trên nếu phù hợp."
    else:
        system = f"{system}\n\nKhông tìm thấy trích đoạn y tế phù hợp trong cơ sở tri thức. Hãy trả lời từ kiến thức y tế chung nếu câu hỏi hợp lệ."

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

