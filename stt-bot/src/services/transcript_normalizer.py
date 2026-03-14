from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    # Keep prototype normalization simple to avoid accidentally changing medical terms.
    cleaned = text.replace("\t", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_lines(lines: list[str]) -> list[str]:
    return [normalize_text(line) for line in lines if normalize_text(line)]
