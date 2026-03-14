from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

from extractor import extract_pdfs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract dữ liệu từ PDF hồ sơ bệnh án bằng Azure Document Intelligence."
    )
    ap.add_argument(
        "pdfs",
        nargs="+",
        metavar="PDF",
        help="Đường dẫn tới 1 hoặc nhiều file PDF.",
    )
    ap.add_argument(
        "--model",
        default="prebuilt-layout",
        help="Model Azure Document Intelligence (mặc định: prebuilt-layout).",
    )
    ap.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help="Lưu kết quả JSON ra file (mặc định: in ra stdout).",
    )
    ap.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="In JSON đẹp (indent=2). Mặc định: bật.",
    )
    args = ap.parse_args()

    # Kiểm tra file tồn tại
    missing = [p for p in args.pdfs if not Path(p).exists()]
    if missing:
        print(f"[error] Không tìm thấy file: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Đang xử lý {len(args.pdfs)} file PDF với model '{args.model}'...")
    docs = extract_pdfs(args.pdfs, model_id=args.model)

    output = [doc.to_dict() for doc in docs]
    json_str = json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None)

    if args.out:
        Path(args.out).write_text(json_str, encoding="utf-8")
        print(f"\n[✓] Kết quả đã lưu vào: {args.out}")
    else:
        print("\n" + json_str)


if __name__ == "__main__":
    main()
