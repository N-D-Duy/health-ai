from __future__ import annotations

import os


def env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v


OLLAMA_MODEL = env("OLLAMA_MODEL", "chat-bot-yte")
OLLAMA_HOST = env("OLLAMA_HOST", "http://localhost:11434")

CHROMA_PATH = env("CHROMA_PATH", "chroma_db")
CHROMA_COLLECTION = env("CHROMA_COLLECTION", "medquad_vi")
EMBED_MODEL_NAME = env("EMBED_MODEL_NAME", "BAAI/bge-m3")

TOP_K = int(env("TOP_K", "5"))
MAX_HISTORY_MESSAGES = int(env("MAX_HISTORY_MESSAGES", "12"))
# Cosine distance: nhỏ = giống. Nếu khoảng cách tốt nhất > ngưỡng thì coi như không có ngữ cảnh y tế (từ chối strict). 0.75 = siết chặt.
RAG_DISTANCE_OFF_TOPIC_THRESHOLD = float(env("RAG_DISTANCE_OFF_TOPIC_THRESHOLD", "0.75"))

# ── Reranking ──────────────────────────────────────────────────────────────────
# CrossEncoder model dùng để score lại chunks sau khi retrieve.
RERANKER_MODEL_NAME = env("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
# Số chunks đưa vào prompt sau khi rerank (top-N tốt nhất).
RERANK_TOP_N = int(env("RERANK_TOP_N", "3"))
# Số candidates lấy từ mỗi nguồn (semantic/BM25) trước khi rerank (pool rộng hơn TOP_K).
RETRIEVE_CANDIDATES = int(env("RETRIEVE_CANDIDATES", "10"))
# Tắt/bật reranking (true = bật, false = dùng TOP_K từ kết quả gốc).
RERANK_ENABLED = env("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")

# ── Hybrid search (Semantic + BM25 + RRF) ─────────────────────────────────────
# Tắt/bật hybrid search; nếu false chỉ dùng semantic (ChromaDB).
HYBRID_SEARCH_ENABLED = env("HYBRID_SEARCH_ENABLED", "true").lower() in ("1", "true", "yes")

# ── Query Expansion ───────────────────────────────────────────────────────────
# Tắt/bật query expansion bằng LLM.
QUERY_EXPANSION_ENABLED = env("QUERY_EXPANSION_ENABLED", "true").lower() in ("1", "true", "yes")
# Số câu hỏi mở rộng sinh thêm (không tính câu gốc).
QUERY_EXPANSION_N = int(env("QUERY_EXPANSION_N", "2"))
# Prompt yêu cầu LLM sinh các cách hỏi khác nhau cho cùng nội dung.
QUERY_EXPANSION_PROMPT_VI = env(
    "QUERY_EXPANSION_PROMPT_VI",
    (
        "Hãy viết {n} cách diễn đạt khác nhau cho câu hỏi sau (tiếng Việt, ngắn gọn, "
        "không giải thích, mỗi cách trên một dòng, không đánh số, không gạch đầu dòng):\n"
        "Câu hỏi gốc: {query}"
    ),
)

RAG_CONTEXT_TITLE = env("RAG_CONTEXT_TITLE", "Context từ cơ sở tri thức")

# Câu trả lời cố định khi câu hỏi không thuộc lĩnh vực y tế. Chỉ trả về đúng câu này, không thêm gì.
REFUSAL_OFF_TOPIC_VI = env(
    "REFUSAL_OFF_TOPIC_VI",
    "Câu hỏi này không thuộc lĩnh vực y tế. Tôi chỉ hỗ trợ thông tin về sức khỏe. Bạn hãy đặt câu hỏi liên quan sức khỏe hoặc bệnh lý.",
)

# Prompt phân loại: LLM tự xác định câu hỏi có thuộc y tế/sức khỏe không (thay keyword cứng).
MEDICAL_CLASSIFIER_PROMPT_VI = env(
    "MEDICAL_CLASSIFIER_PROMPT_VI",
    (
        "Nhiệm vụ: Xác định xem câu hỏi sau có thuộc lĩnh vực y tế, sức khỏe, bệnh lý, "
        "triệu chứng, thuốc, dinh dưỡng, vệ sinh cá nhân, chăm sóc cơ thể hay y học không.\n"
        "Chỉ trả lời đúng một từ: CÓ hoặc KHÔNG. Không giải thích thêm.\n"
        "Câu hỏi: {query}"
    ),
)

# Những điều model KHÔNG được làm khi câu hỏi không về y tế (để ghi rõ trong system prompt).
OFF_TOPIC_FORBIDDEN_VI = (
    "Khi câu hỏi KHÔNG về y tế/sức khỏe, CẤM: giải thích nội dung câu hỏi; "
    "gợi ý nguồn khác; thêm bất kỳ thông tin nào ngoài câu từ chối. "
    "Chỉ được trả lời đúng một câu từ chối theo mẫu quy định."
)

SYSTEM_GUARDRAIL_VI = env(
    "SYSTEM_GUARDRAIL_VI",
    "\n".join(
        [
            "Bạn là trợ lý thông tin y tế cơ bản (không thay thế bác sĩ).",
            "Phạm vi: Chỉ trả lời câu hỏi về sức khỏe, y tế, bệnh lý (triệu chứng, nguyên nhân, cách phòng ngừa, v.v.).",
            OFF_TOPIC_FORBIDDEN_VI,
            "Không đưa chẩn đoán chắc chắn, không kê đơn/đưa liều dùng thuốc.",
            "Với câu hỏi ĐÚNG chủ đề y tế: luôn nhắc “Thông tin chỉ mang tính tham khảo; bạn nên gặp bác sĩ để được chẩn đoán và điều trị.”",
            "Trả lời tiếng Việt, rõ ràng, ngắn gọn, ưu tiên gạch đầu dòng.",
        ]
    ),
)

