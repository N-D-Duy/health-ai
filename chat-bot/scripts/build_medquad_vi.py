from __future__ import annotations

import argparse
import os
import random
import re
import time
from typing import Iterable

import pandas as pd
import requests
from datasets import load_dataset
from dotenv import load_dotenv


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def batched(it: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(it), n):
        yield it[i : i + n]


def translate_texts(texts: list[str], *, provider: str, target_lang: str = "vi") -> list[str]:
    provider = (provider or "none").strip().lower()
    if provider in ("none", "off", "false"):
        return texts

    if provider == "azure":
        # Azure Translator Text API v3
        base_url = os.getenv("TRANSLATE_URL", "").rstrip("/")
        if not base_url:
            raise ValueError("TRANSLATE_URL is required for translate provider 'azure'")

        # Hai kiểu endpoint phổ biến:
        # - Global: https://api.cognitive.microsofttranslator.com
        #   -> dùng path /translate?api-version=3.0&from=en&to=vi
        # - Custom (resource): https://<name>.cognitiveservices.azure.com
        #   -> dùng path /translator/text/v3.0/translate?from=en&to=vi
        if "cognitiveservices.azure.com" in base_url and "translator/text" not in base_url:
            url = f"{base_url}/translator/text/v3.0/translate?from=en&to={target_lang}"
        else:
            # Giữ nguyên nếu bạn tự đưa đúng path/params vào TRANSLATE_URL
            if "/translate" in base_url:
                url = base_url
            else:
                url = f"{base_url}/translate"
            if "api-version=" not in url:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}api-version=3.0&from=en&to={target_lang}"

        key = os.getenv("KEY", "").strip()
        if not key:
            raise ValueError("KEY (Azure Translator subscription key) is required for provider 'azure'")

        region = os.getenv("AZURE_REGION", "").strip()

        headers = {
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/json",
        }
        if region:
            headers["Ocp-Apim-Subscription-Region"] = region

        out: list[str] = []
        for batch in batched(texts, 20):
            body = [{"Text": t} for t in batch]
            translated_batch: list[str] = []
            for attempt in range(4):
                try:
                    r = requests.post(url, headers=headers, json=body, timeout=20)
                    r.raise_for_status()
                    data = r.json() or []
                    for item, orig in zip(data, batch):
                        # expected: [{"translations":[{"text": "...", "to": "vi"}], ...}, ...]
                        try:
                            txt = item["translations"][0]["text"]
                        except Exception:
                            txt = orig
                        translated_batch.append(normalize_text(txt))
                    break
                except Exception:
                    if attempt == 3:
                        translated_batch.extend(batch)
                    else:
                        time.sleep(0.75 * (attempt + 1))
            out.extend(translated_batch)
        return out

    if provider in ("libretranslate", "http"):
        # Example LibreTranslate-compatible endpoint:
        #   POST $TRANSLATE_URL  json={"q": "...", "source": "en", "target": "vi", "format": "text"}
        # Response: {"translatedText": "..."}
        url = os.getenv("TRANSLATE_URL", "").strip()
        if not url:
            raise ValueError("TRANSLATE_URL is required for translate provider 'http/libretranslate'")

        out: list[str] = []
        for batch in batched(texts, 10):
            for text in batch:
                payload = {
                    "q": text,
                    "source": "en",
                    "target": target_lang,
                    "format": "text",
                }
                translated = None
                for attempt in range(4):
                    try:
                        r = requests.post(url, json=payload, timeout=20)
                        r.raise_for_status()
                        translated = (r.json() or {}).get("translatedText")
                        break
                    except Exception:
                        if attempt == 3:
                            translated = text
                        else:
                            time.sleep(0.75 * (attempt + 1))
                out.append(normalize_text(translated or text))
        return out

    raise ValueError(f"Unsupported translate provider: {provider}")


def main() -> None:
    # Load .env if present so TRANSLATE_URL / KEY are available
    load_dotenv(override=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/medquad_vi.csv")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--translate-provider", default=os.getenv("TRANSLATE_PROVIDER", "none"))
    args = ap.parse_args()

    random.seed(args.seed)

    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    allowed = {"symptoms"}
    ds = ds.filter(lambda r: r["qtype"] in allowed)

    rows = list(ds)
    random.shuffle(rows)
    rows = rows[: max(1, args.limit)]

    questions_en = [normalize_text(r["Question"]) for r in rows]
    answers_en = [normalize_text(r["Answer"]) for r in rows]
    qtypes = [normalize_text(r["qtype"]) for r in rows]

    questions_vi = translate_texts(questions_en, provider=args.translate_provider)
    answers_vi = translate_texts(answers_en, provider=args.translate_provider)

    df = pd.DataFrame(
        {
            "qtype": qtypes,
            "question_en": questions_en,
            "answer_en": answers_en,
            "question": questions_vi,
            "answer": answers_vi,
        }
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()

