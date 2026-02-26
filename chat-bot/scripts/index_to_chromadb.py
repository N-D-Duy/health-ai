from __future__ import annotations

import argparse
import os
import re
from typing import Any, Iterable

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    out: list[str] = []
    step = max(1, max_chars - max(0, overlap))
    for start in range(0, len(text), step):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= len(text):
            break
    return out


def batched_ids(n: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for i in range(0, n, batch_size):
        yield i, min(n, i + batch_size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/medquad_vi.csv")
    ap.add_argument("--chroma-path", default=os.getenv("CHROMA_PATH", "chroma_db"))
    ap.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "medquad_vi"))
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3"))
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    # Hugging Face token (tăng limit/tốc độ tải model)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token and not os.getenv("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    df = pd.read_csv(args.in_path)
    if len(df) == 0:
        raise SystemExit("Input CSV is empty.")

    client = chromadb.PersistentClient(path=args.chroma_path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.embed_model
    )
    col = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for row_idx, row in df.iterrows():
        qtype = normalize_text(str(row.get("qtype", "")))
        q_vi = normalize_text(str(row.get("question", "")))
        a_vi = normalize_text(str(row.get("answer", "")))
        q_en = normalize_text(str(row.get("question_en", "")))
        a_en = normalize_text(str(row.get("answer_en", "")))

        question = q_vi or q_en
        answer = a_vi or a_en
        base = f"Hỏi: {question}\nĐáp: {answer}".strip()
        chunks = chunk_text(base, max_chars=args.max_chars, overlap=args.overlap)
        for ci, ch in enumerate(chunks):
            doc_id = f"medquad_{row_idx}_{ci}"
            ids.append(doc_id)
            documents.append(ch)
            metadatas.append(
                {
                    "source": "MedQuad",
                    "row": int(row_idx),
                    "chunk": int(ci),
                    "qtype": qtype,
                    "question": question,
                }
            )

    total = len(documents)
    for start, end in batched_ids(total, args.batch_size):
        col.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"Indexed {end}/{total}")

    print(f"Done. Collection='{args.collection}', path='{args.chroma_path}', docs={total}")


if __name__ == "__main__":
    main()

