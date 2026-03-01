from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Đảm bảo HF token được set TRƯỚC KHI import bất kỳ thư viện huggingface nào,
# tránh warning "unauthenticated requests" khi khởi động.
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    os.environ.setdefault("HF_TOKEN", _hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)

import chromadb
from chromadb.utils import embedding_functions

from .config import CHROMA_COLLECTION, CHROMA_PATH, EMBED_MODEL_NAME


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def query_chromadb(query: str, k: int = 5) -> list[RetrievedChunk]:
    query = (query or "").strip()
    if not query:
        return []

    collection = get_collection()
    res = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: list[RetrievedChunk] = []
    for i in range(min(len(ids), len(docs), len(metas))):
        out.append(
            RetrievedChunk(
                id=str(ids[i]),
                text=str(docs[i]),
                metadata=metas[i] or {},
                distance=float(dists[i]) if i < len(dists) and dists[i] is not None else None,
            )
        )
    return out

