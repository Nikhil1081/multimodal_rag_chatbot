from __future__ import annotations

import re
from typing import Iterable, List


_whitespace_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = _whitespace_re.sub(" ", text)
    return text.strip()


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """Simple, robust chunker.

    - Splits on paragraph-ish boundaries first
    - Falls back to char windows
    """

    text = normalize_text(text)
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    def flush_buf():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paras:
        if not buf:
            buf = p
            continue
        if len(buf) + 1 + len(p) <= max_chars:
            buf = f"{buf}\n{p}"
        else:
            flush_buf()
            buf = p

    flush_buf()

    # Ensure no chunk is too large
    final: list[str] = []
    for ch in chunks:
        if len(ch) <= max_chars:
            final.append(ch)
            continue
        start = 0
        while start < len(ch):
            end = min(len(ch), start + max_chars)
            window = ch[start:end].strip()
            if window:
                final.append(window)
            if end == len(ch):
                break
            start = max(0, end - overlap)

    return final
