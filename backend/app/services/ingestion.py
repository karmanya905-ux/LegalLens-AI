from pathlib import Path
from pypdf import PdfReader


def extract_pdf_pages(path: str | Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page": idx, "text": text.strip()})
    return pages


def chunk_pages(pages: list[dict], chunk_size: int = 1000, overlap: int = 160) -> list[dict]:
    """Simple page-aware chunker; each chunk retains its source page number."""
    chunks = []
    for page in pages:
        text = page["text"]
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append({"text": piece, "page": page["page"]})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks
