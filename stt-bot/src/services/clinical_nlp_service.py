from __future__ import annotations

from src.config import MAX_INPUT_CHARS
from src.models import ClinicalExtraction
from src.services.ollama_client import OllamaClient


class ClinicalNlpService:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client

    def _build_prompt(self, transcript_text: str) -> str:
        clipped = transcript_text[:MAX_INPUT_CHARS]
        return f'''Bạn là trợ lý y khoa nội bộ cho bác sĩ. Hãy phân tích hội thoại khám bệnh tiếng Việt và trả về DUY NHẤT một JSON hợp lệ.

Yêu cầu:
- Không được bổ sung thông tin không có trong transcript.
- Nếu không rõ, đặt giá trị null hoặc mảng rỗng.
- Trường evidence_snippets chỉ trích nguyên văn các câu then chốt từ transcript.

Schema JSON bắt buộc:
{{
  "chief_complaint": "string|null",
  "onset": "string|null",
  "associated_symptoms": ["string"],
  "pertinent_negatives": ["string"],
  "short_summary": "string|null",
  "evidence_snippets": ["string"]
}}

Transcript:
"""
{clipped}
"""'''.strip()

    def extract(self, transcript_text: str) -> ClinicalExtraction:
        prompt = self._build_prompt(transcript_text)
        payload = self.ollama_client.generate_json(prompt, temperature=0.0, num_predict=900)

        return ClinicalExtraction(
            chief_complaint=payload.get("chief_complaint"),
            onset=payload.get("onset"),
            associated_symptoms=payload.get("associated_symptoms") or [],
            pertinent_negatives=payload.get("pertinent_negatives") or [],
            short_summary=payload.get("short_summary"),
            evidence_snippets=payload.get("evidence_snippets") or [],
        )
