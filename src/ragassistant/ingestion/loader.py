"""Document loaders for PDF, Markdown, and plain text files."""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from ragassistant.schemas.document import DocumentPage

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".markdown"}


def _clean_text(raw_text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r"\r\n|\r", "\n", raw_text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(file_path: Path) -> list[DocumentPage]:
    """Extract page-level text from PDF files using pypdf."""
    reader = PdfReader(str(file_path))
    pages: list[DocumentPage] = []

    for idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = _clean_text(raw_text)
        if cleaned:
            pages.append(
                DocumentPage(
                    doc_name=file_path.name,
                    page_number=idx,
                    text=cleaned,
                    metadata={"source": str(file_path), "total_pages": len(reader.pages)},
                )
            )

    return pages


def load_markdown_or_text(file_path: Path) -> list[DocumentPage]:
    """Load Markdown or text files, splitting by top-level section headers or treating as page 1."""
    content = file_path.read_text(encoding="utf-8")
    cleaned = _clean_text(content)

    if not cleaned:
        return []

    # If the markdown document has distinct top-level H1/H2 sections, partition into logical sections
    sections = re.split(r"\n(?=##?\s+)", cleaned)
    sections = [s.strip() for s in sections if s.strip()]

    pages: list[DocumentPage] = []
    if len(sections) > 1:
        for idx, sec in enumerate(sections, start=1):
            pages.append(
                DocumentPage(
                    doc_name=file_path.name,
                    page_number=idx,
                    text=sec,
                    metadata={"source": str(file_path), "section_index": idx},
                )
            )
    else:
        pages.append(
            DocumentPage(
                doc_name=file_path.name,
                page_number=1,
                text=cleaned,
                metadata={"source": str(file_path)},
            )
        )

    return pages


def load_document(file_path: str | Path) -> list[DocumentPage]:
    """Dispatcher to load supported document formats into DocumentPage instances."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        return load_pdf(path)

    return load_markdown_or_text(path)
