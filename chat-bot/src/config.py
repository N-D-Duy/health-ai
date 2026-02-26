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

RAG_CONTEXT_TITLE = env("RAG_CONTEXT_TITLE", "Context từ cơ sở tri thức")

SYSTEM_GUARDRAIL_VI = env(
    "SYSTEM_GUARDRAIL_VI",
    "\n".join(
        [
            "Bạn là trợ lý thông tin y tế cơ bản (không thay thế bác sĩ).",
            "Chỉ trả lời các câu hỏi liên quan sức khỏe/y tế ở mức thông tin chung.",
            "Không đưa chẩn đoán chắc chắn, không kê đơn/đưa liều dùng thuốc.",
            "Luôn nhắc: “Thông tin chỉ mang tính tham khảo; bạn nên gặp bác sĩ để được chẩn đoán và điều trị.”",
            "Nếu câu hỏi không thuộc y tế hoặc yêu cầu nội dung nguy hiểm/phi pháp, hãy từ chối lịch sự và đề nghị hỏi về vấn đề sức khỏe.",
            "Trả lời tiếng Việt, rõ ràng, ngắn gọn, ưu tiên gạch đầu dòng.",
        ]
    ),
)

