"""Unit tests for document and chunk schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragassistant.schemas.document import (
    DocumentChunk,
    DocumentPage,
    IngestionStats,
    RetrievedChunk,
)


def test_document_page_valid():
    page = DocumentPage(
        doc_name="handbook.md",
        page_number=1,
        text="Sample policy text",
        metadata={"author": "HR"},
    )
    assert page.doc_name == "handbook.md"
    assert page.page_number == 1
    assert page.text == "Sample policy text"


def test_document_page_extra_forbid():
    with pytest.raises(ValidationError):
        DocumentPage(
            doc_name="test.md",
            page_number=1,
            text="hello",
            unknown_field=123,
        )


def test_document_chunk_valid():
    chunk = DocumentChunk(
        chunk_id="handbook_p1_c0",
        doc_name="handbook.md",
        page_number=1,
        chunk_index=0,
        text="A semantic slice of policy text",
        token_count=7,
    )
    assert chunk.chunk_id == "handbook_p1_c0"
    assert chunk.token_count == 7


def test_ingestion_stats():
    stats = IngestionStats(
        doc_name="spec.pdf",
        total_pages=5,
        total_chunks=22,
        total_characters=8400,
    )
    assert stats.total_chunks == 22


def test_retrieved_chunk_valid():
    chunk = DocumentChunk(
        chunk_id="doc_p1_c0",
        doc_name="doc.txt",
        page_number=1,
        chunk_index=0,
        text="Retrieved content",
        token_count=2,
    )
    retrieved = RetrievedChunk(chunk=chunk, score=0.88)
    assert retrieved.chunk.chunk_id == "doc_p1_c0"
    assert retrieved.score == 0.88
