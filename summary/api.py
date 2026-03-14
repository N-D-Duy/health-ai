from __future__ import annotations

import json
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from extractor import AZURE_DI_MODEL, extract_pdfs

app = FastAPI(title="summary-extractor", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": AZURE_DI_MODEL}


@app.post("/extract")
async def extract(files: list[UploadFile] = File(...)) -> dict:
    """
    Nhận 1 hoặc nhiều file PDF, trả về JSON kết quả extract từ Azure Document Intelligence.
    """
    for f in files:
        if f.content_type not in ("application/pdf", "application/octet-stream") and \
           not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"'{f.filename}' không phải file PDF.")

    # Lưu tạm các file upload ra disk để extractor đọc
    tmp_paths: list[Path] = []
    try:
        for upload in files:
            suffix = Path(upload.filename or "file").suffix or ".pdf"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(await upload.read())
            tmp.flush()
            tmp.close()
            tmp_paths.append(Path(tmp.name))

        docs = extract_pdfs(tmp_paths)

        # Gắn lại tên file gốc (thay tên tmp)
        results = []
        for doc, upload in zip(docs, files):
            d = doc.to_dict()
            d["file"] = upload.filename or doc.file
            results.append(d)

        return {"documents": results, "count": len(results)}

    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)
