from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chat import ChatMessage, chat_once
from src.config import OLLAMA_HOST
from src.retrieval import query_chromadb

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up: load embedding model + ChromaDB khi khởi động để request đầu không bị chậm (Loading weights).
    try:
        await asyncio.to_thread(query_chromadb, "sức khỏe", 1)
        logger.info("RAG warm-up done (embedding + ChromaDB)")
    except Exception as e:
        logger.warning("RAG warm-up failed: %s", e)
    yield


app = FastAPI(title="chat-bot-yte", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


_lock = threading.Lock()
_sessions: dict[str, list[ChatMessage]] = {}
_DEFAULT_SESSION = "default"


def _get_history(session_id: str) -> list[ChatMessage]:
    with _lock:
        return list(_sessions.get(session_id, []))


def _set_history(session_id: str, history: list[ChatMessage]) -> None:
    with _lock:
        _sessions[session_id] = list(history)


@app.get("/health")
def health() -> dict[str, Any]:
    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=1.5)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {"status": "ok", "ollama_ok": ollama_ok}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = (req.session_id or _DEFAULT_SESSION).strip() or _DEFAULT_SESSION
    history = _get_history(session_id)
    response, new_history = chat_once(req.message, history)
    _set_history(session_id, new_history)
    return ChatResponse(response=response, session_id=session_id)

