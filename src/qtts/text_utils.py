from __future__ import annotations

import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


def split_text(text: str, max_chars: int = 240) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(sentence) <= max_chars:
            current = sentence
            continue

        for idx in range(0, len(sentence), max_chars):
            chunks.append(sentence[idx : idx + max_chars])

    if current:
        chunks.append(current)

    return chunks
