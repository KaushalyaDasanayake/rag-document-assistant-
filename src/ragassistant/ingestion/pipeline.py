"""End-to-end ingestion pipeline for single files and directories."""

from __future__ import annotations

import logging
from pathlib import Path

from ragassistant.ingestion.chunker import chunk_document_pages
from ragassistant.ingestion.loader import load_document
from ragassistant.retrieval.vector_store import VectorStore
from ragassistant.schemas.document import IngestionStats

logger = logging.getLogger(__name__)


def ingest_file(
    file_path: str | Path,
    vector_store: VectorStore,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> IngestionStats:
    """Load, chunk, and index a single document into the vector store."""
    path = Path(file_path)
    pages = load_document(path)

    if not pages:
        return IngestionStats(
            doc_name=path.name,
            total_pages=0,
            total_chunks=0,
            total_characters=0,
        )

    chunks = chunk_document_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    total_characters = sum(len(p.text) for p in pages)

    if chunks:
        vector_store.add_chunks(chunks)

    stats = IngestionStats(
        doc_name=path.name,
        total_pages=len(pages),
        total_chunks=len(chunks),
        total_characters=total_characters,
    )
    logger.info("Ingested %s: %d pages, %d chunks", path.name, stats.total_pages, stats.total_chunks)
    return stats


def ingest_directory(
    directory_path: str | Path,
    vector_store: VectorStore,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[IngestionStats]:
    """Ingest all supported documents (pdf, md, txt) from a directory."""
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")

    supported_suffixes = {".pdf", ".md", ".txt"}
    files = sorted([f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in supported_suffixes])

    results: list[IngestionStats] = []
    for file in files:
        stats = ingest_file(
            file_path=file,
            vector_store=vector_store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        results.append(stats)

    return results
