from __future__ import annotations

import json
import re

import ollama

from src.config import OLLAMA_HOST, OLLAMA_MODEL


class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)

    def generate_text(self, prompt: str, *, temperature: float = 0.0, num_predict: int = 800) -> str:
        res = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": num_predict},
        )
        return (res.get("response") or "").strip()

    def generate_json(self, prompt: str, *, temperature: float = 0.0, num_predict: int = 1000) -> dict:
        text = self.generate_text(prompt, temperature=temperature, num_predict=num_predict)
        # Try strict parse first, then fallback to extracting the first JSON object in text.
        try:
            return json.loads(text)
        except Exception:
            pass

        cleaned = text.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

        if "{" in cleaned and "}" in cleaned:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            snippet = cleaned[start : end + 1]
            return json.loads(snippet)

        raise ValueError("Model không trả về JSON hợp lệ")
