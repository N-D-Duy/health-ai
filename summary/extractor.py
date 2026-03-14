from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential


# ── Config ─────────────────────────────────────────────────────────────────────
# Azure Document Intelligence (formerly Form Recognizer)
AZURE_DI_ENDPOINT = os.getenv("AZURE_DI_ENDPOINT", "")  # https://<resource>.cognitiveservices.azure.com/
AZURE_DI_KEY = os.getenv("AZURE_DI_KEY", "")
# prebuilt-layout: text + tables + selection marks (tốt nhất cho hồ sơ y tế)
# prebuilt-read: chỉ text (nhanh hơn, nhẹ hơn)
AZURE_DI_MODEL = os.getenv("AZURE_DI_MODEL", "prebuilt-layout")


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class TableCell:
    row: int
    col: int
    row_span: int
    col_span: int
    content: str
    kind: str  # "content" | "columnHeader" | "rowHeader"


@dataclass
class ExtractedTable:
    row_count: int
    col_count: int
    cells: list[TableCell] = field(default_factory=list)

    def to_rows(self) -> list[list[str]]:
        """Chuyển cells thành list[row][col] để dễ đọc."""
        grid: list[list[str]] = [[""] * self.col_count for _ in range(self.row_count)]
        for c in self.cells:
            if c.row < self.row_count and c.col < self.col_count:
                grid[c.row][c.col] = c.content
        return grid


@dataclass
class ExtractedDocument:
    file: str                                          # tên file gốc
    model_id: str                                      # model Azure dùng
    pages: list[str] = field(default_factory=list)    # text theo từng trang
    tables: list[ExtractedTable] = field(default_factory=list)
    key_value_pairs: dict[str, str] = field(default_factory=dict)
    full_text: str = ""                                # toàn bộ text ghép lại

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "model_id": self.model_id,
            "page_count": len(self.pages),
            "pages": self.pages,
            "tables": [
                {"row_count": t.row_count, "col_count": t.col_count, "rows": t.to_rows()}
                for t in self.tables
            ],
            "key_value_pairs": self.key_value_pairs,
            "full_text": self.full_text,
        }


# ── Client factory ─────────────────────────────────────────────────────────────

def get_client(endpoint: str = AZURE_DI_ENDPOINT, key: str = AZURE_DI_KEY) -> DocumentIntelligenceClient:
    if not endpoint or not key:
        raise ValueError(
            "AZURE_DI_ENDPOINT and AZURE_DI_KEY must be set "
            "(via environment variables or .env file)."
        )
    return DocumentIntelligenceClient(
        endpoint=endpoint.rstrip("/"),
        credential=AzureKeyCredential(key),
    )


# ── Core extraction ────────────────────────────────────────────────────────────

def extract_pdf(
    pdf_path: str | Path,
    *,
    client: DocumentIntelligenceClient | None = None,
    model_id: str = AZURE_DI_MODEL,
) -> ExtractedDocument:
    """
    Extract text, tables và key-value pairs từ 1 file PDF bằng Azure Document Intelligence.

    Args:
        pdf_path: Đường dẫn local tới file PDF.
        client:   DocumentIntelligenceClient — nếu None sẽ tự tạo từ env vars.
        model_id: Model Azure dùng (mặc định: prebuilt-layout).

    Returns:
        ExtractedDocument chứa toàn bộ nội dung đã parse.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if client is None:
        client = get_client()

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Gửi file bytes lên Azure Document Intelligence
    poller = client.begin_analyze_document(
        model_id=model_id,
        body=AnalyzeDocumentRequest(bytes_source=pdf_bytes),
        content_type="application/json",
    )
    result = poller.result()

    doc = ExtractedDocument(file=pdf_path.name, model_id=model_id)

    # ── Pages → text theo từng trang ──────────────────────────────────────────
    if result.pages:
        for page in result.pages:
            page_lines = []
            if page.lines:
                for line in page.lines:
                    page_lines.append(line.content)
            doc.pages.append("\n".join(page_lines))

    # ── Tables ────────────────────────────────────────────────────────────────
    if result.tables:
        for tbl in result.tables:
            ext_table = ExtractedTable(
                row_count=tbl.row_count,
                col_count=tbl.column_count,
            )
            if tbl.cells:
                for cell in tbl.cells:
                    ext_table.cells.append(
                        TableCell(
                            row=cell.row_index,
                            col=cell.column_index,
                            row_span=cell.row_span or 1,
                            col_span=cell.column_span or 1,
                            content=(cell.content or "").strip(),
                            kind=str(cell.kind) if cell.kind else "content",
                        )
                    )
            doc.tables.append(ext_table)

    # ── Key-value pairs ───────────────────────────────────────────────────────
    if result.key_value_pairs:
        for kv in result.key_value_pairs:
            if kv.key and kv.value:
                key_text = (kv.key.content or "").strip()
                val_text = (kv.value.content or "").strip()
                if key_text:
                    doc.key_value_pairs[key_text] = val_text

    # ── Full text ─────────────────────────────────────────────────────────────
    doc.full_text = "\n\n--- Trang {} ---\n".join(
        [""] + [str(i + 1) for i in range(len(doc.pages))]
    )
    doc.full_text = "\n\n".join(
        f"--- Trang {i + 1} ---\n{text}" for i, text in enumerate(doc.pages)
    )

    return doc


def extract_pdfs(
    pdf_paths: list[str | Path],
    *,
    model_id: str = AZURE_DI_MODEL,
    endpoint: str = AZURE_DI_ENDPOINT,
    key: str = AZURE_DI_KEY,
) -> list[ExtractedDocument]:
    """
    Extract nhiều file PDF — dùng chung 1 client để tái sử dụng connection.
    """
    client = get_client(endpoint=endpoint, key=key)
    results: list[ExtractedDocument] = []
    for path in pdf_paths:
        print(f"  Extracting: {Path(path).name} ...", flush=True)
        doc = extract_pdf(path, client=client, model_id=model_id)
        results.append(doc)
        print(f"  ✓ {doc.file}  ({len(doc.pages)} trang, {len(doc.tables)} bảng, {len(doc.key_value_pairs)} key-value pairs)")
    return results
