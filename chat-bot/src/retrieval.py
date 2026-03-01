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

from .config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    EMBED_MODEL_NAME,
    HYBRID_SEARCH_ENABLED,
    RERANK_ENABLED,
    RERANK_TOP_N,
    RERANKER_MODEL_NAME,
    RETRIEVE_CANDIDATES,
    TOP_K,
)


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None      # cosine distance từ ChromaDB (nhỏ = gần)
    rerank_score: float | None = None  # CrossEncoder score (lớn = liên quan)


# ── Module-level lazy caches ───────────────────────────────────────────────────
_collection = None
_bm25_index = None
_bm25_corpus: list[RetrievedChunk] = []
_reranker = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_reranker():
    """Lazy-load CrossEncoder reranker (chỉ tải 1 lần)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def _get_bm25():
    """Lazy-load BM25 index xây từ toàn bộ corpus trong ChromaDB (chỉ build 1 lần)."""
    global _bm25_index, _bm25_corpus
    if _bm25_index is not None:
        return _bm25_index, _bm25_corpus

    from rank_bm25 import BM25Okapi

    collection = get_collection()
    res = collection.get(include=["documents", "metadatas"])
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []

    _bm25_corpus = [
        RetrievedChunk(
            id=str(ids[i]),
            text=str(docs[i]),
            metadata=metas[i] or {},
        )
        for i in range(min(len(ids), len(docs), len(metas)))
        if docs[i]
    ]

    # Tokenize đơn giản: lowercase + split (tiếng Việt không cần stemming)
    tokenized = [c.text.lower().split() for c in _bm25_corpus]
    _bm25_index = BM25Okapi(tokenized)
    return _bm25_index, _bm25_corpus


# ── Semantic search ────────────────────────────────────────────────────────────

def query_chromadb(query: str, k: int = TOP_K) -> list[RetrievedChunk]:
    """Semantic search thuần (embedding cosine similarity)."""
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


# ── BM25 search ────────────────────────────────────────────────────────────────

def bm25_search(query: str, k: int = RETRIEVE_CANDIDATES) -> list[RetrievedChunk]:
    """Keyword search (BM25) — tốt cho tên bệnh/thuốc cụ thể."""
    query = (query or "").strip()
    if not query:
        return []

    bm25, corpus = _get_bm25()
    if not corpus:
        return []

    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    # Chỉ trả về chunk có score > 0 (ít nhất 1 từ khớp)
    return [corpus[i] for i in top_idx if scores[i] > 0]


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Kết hợp nhiều ranked list thành 1 theo Reciprocal Rank Fusion."""
    rrf_scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            # Ưu tiên giữ chunk có distance từ semantic (thay vì chunk chỉ có BM25)
            if chunk.id not in chunks_by_id or chunk.distance is not None:
                chunks_by_id[chunk.id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
    return [chunks_by_id[i] for i in sorted_ids]


# ── Hybrid search ──────────────────────────────────────────────────────────────

def hybrid_query(query: str, k: int = RETRIEVE_CANDIDATES) -> list[RetrievedChunk]:
    """Semantic + BM25 → Reciprocal Rank Fusion."""
    semantic = query_chromadb(query, k=k)
    if not HYBRID_SEARCH_ENABLED:
        return semantic
    bm25 = bm25_search(query, k=k)
    return reciprocal_rank_fusion([semantic, bm25])


# ── Reranking ──────────────────────────────────────────────────────────────────

def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int = RERANK_TOP_N,
) -> list[RetrievedChunk]:
    """Dùng CrossEncoder score lại và chọn top-N chunk liên quan nhất."""
    if not chunks:
        return []
    if not RERANK_ENABLED:
        return chunks[:top_n]

    reranker = _get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores: list[float] = reranker.predict(pairs).tolist()

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [
        RetrievedChunk(
            id=c.id,
            text=c.text,
            metadata=c.metadata,
            distance=c.distance,
            rerank_score=float(s),
        )
        for s, c in ranked[:top_n]
    ]


