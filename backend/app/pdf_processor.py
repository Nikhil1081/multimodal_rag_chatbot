from __future__ import annotations

from dataclasses import dataclass
from typing import List

from io import BytesIO

from pypdf import PdfReader


@dataclass
class PageText:
    page: int
    text: str


def extract_pdf_text_by_page(pdf_bytes: bytes) -> List[PageText]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(PageText(page=i + 1, text=text))
    return pages
