"""Unit tests for recursive document chunker."""

from __future__ import annotations

import pytest

from ragassistant.ingestion.chunker import chunk_document_pages
from ragassistant.schemas.document import DocumentPage


def test_chunk_single_page_under_limit():
    page = DocumentPage(
        doc_name="doc1.txt",
        page_number=1,
        text="A brief single sentence document.",
    )
    chunks = chunk_document_pages([page], chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].doc_name == "doc1.txt"
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_id == "doc1_txt_p1_c0"
    assert chunks[0].text == "A brief single sentence document."
    assert chunks[0].token_count > 0


def test_chunk_multiple_pages_indexing():
    page1 = DocumentPage(doc_name="guide.md", page_number=1, text="First section content.")
    page2 = DocumentPage(doc_name="guide.md", page_number=2, text="Second section content.")

    chunks = chunk_document_pages([page1, page2], chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_number == 1
    assert chunks[1].chunk_index == 1
    assert chunks[1].page_number == 2


def test_chunk_splits_long_text():
    long_text = "\n\n".join([f"Paragraph {i}: This is descriptive content for testing chunking." for i in range(20)])
    page = DocumentPage(doc_name="policy.pdf", page_number=3, text=long_text)

    chunks = chunk_document_pages([page], chunk_size=150, chunk_overlap=30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.doc_name == "policy.pdf"
        assert chunk.page_number == 3
        assert len(chunk.text) > 0


def test_chunker_validation_errors():
    page = DocumentPage(doc_name="test.txt", page_number=1, text="Hello world")

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_document_pages([page], chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        chunk_document_pages([page], chunk_size=100, chunk_overlap=150)
