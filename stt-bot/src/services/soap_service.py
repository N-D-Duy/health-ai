from __future__ import annotations

from datetime import datetime, timezone
import json

from src.config import MAX_INPUT_CHARS
from src.models import ClinicalExtraction, SoapDraft
from src.services.ollama_client import OllamaClient


class SoapService:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client

    def _build_prompt(self, transcript_text: str, extraction: ClinicalExtraction) -> str:
        extraction_json = json.dumps(extraction.model_dump(), ensure_ascii=False)
        clipped = transcript_text[:MAX_INPUT_CHARS]
        return f'''Bạn là trợ lý hỗ trợ viết ghi chú khám bệnh cho bác sĩ. Hãy tạo bản nháp SOAP bằng tiếng Việt.

Ràng buộc:
- Chỉ sử dụng thông tin có trong transcript và extraction.
- Nếu không rõ thì ghi "Không rõ".
- Trả về DUY NHẤT một JSON hợp lệ theo schema bên dưới.

Schema:
{{
    "subjective": "string",
    "objective": "string",
    "assessment": "string",
    "plan": "string"
}}

Clinical extraction:
{extraction_json}

Transcript:
"""
{clipped}
"""'''.strip()

    def generate(self, transcript_text: str, extraction: ClinicalExtraction) -> SoapDraft:
        prompt = self._build_prompt(transcript_text, extraction)
        payload = self.ollama_client.generate_json(prompt, temperature=0.1, num_predict=1000)

        return SoapDraft(
            subjective=payload.get("subjective") or "Khong ro",
            objective=payload.get("objective") or "Khong ro",
            assessment=payload.get("assessment") or "Khong ro",
            plan=payload.get("plan") or "Khong ro",
            generated_at=datetime.now(timezone.utc),
            model_name=self.ollama_client.model,
        )
