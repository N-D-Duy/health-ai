from __future__ import annotations

import os
from pathlib import Path


def env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


OLLAMA_HOST = env("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = env("OLLAMA_MODEL", "qwen2.5:7b")

WHISPER_MODEL_SIZE = env("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE = env("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = env("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_LANGUAGE = env("WHISPER_LANGUAGE", "vi")

UPLOAD_DIR = Path(env("UPLOAD_DIR", "data/uploads"))
STORE_FILE = Path(env("STORE_FILE", "data/store/sessions.json"))
MAX_INPUT_CHARS = int(env("MAX_INPUT_CHARS", "12000"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
