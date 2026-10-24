"""Unit tests for ChromaDB VectorStore."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ragassistant.retrieval.vector_store import VectorStore
from ragassistant.schemas.document import DocumentChunk


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="handbook_p1_c0",
            doc_name="handbook.md",
            page_number=1,
            chunk_index=0,
            text="Standard working hours are 9am to 5pm Monday through Friday.",
            token_count=12,
        ),
        DocumentChunk(
            chunk_id="handbook_p2_c1",
            doc_name="handbook.md",
            page_number=2,
            chunk_index=1,
            text="Employees receive 20 days of paid annual leave per year.",
            token_count=10,
        ),
    ]


def test_vector_store_empty_query():
    store = VectorStore(collection_name="empty_test_coll")
    assert store.count() == 0
    results = store.query(query_embedding=[0.1] * 10)
    assert results == []


def test_vector_store_non_positive_top_k(sample_chunks: list[DocumentChunk]):
    store = VectorStore(collection_name="topk_coll")
    store.add_chunks(sample_chunks, embeddings=[[0.1, 0.2], [0.3, 0.4]])
    assert store.query(query_embedding=[0.1, 0.2], top_k=0) == []
    assert store.query(query_embedding=[0.1, 0.2], top_k=-1) == []


def test_vector_store_add_and_query_precomputed(sample_chunks: list[DocumentChunk]):
    store = VectorStore(collection_name="precomputed_coll")
    fake_embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    count = store.add_chunks(sample_chunks, embeddings=fake_embeddings)
    assert count == 2
    assert store.count() == 2

    # Query with vector identical to first chunk
    results = store.query(query_embedding=[1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "handbook_p1_c0"
    assert results[0].chunk.page_number == 1
    assert results[0].score >= 0.99


def test_vector_store_query_with_embedding_manager(sample_chunks: list[DocumentChunk]):
    mock_embed = MagicMock()
    mock_embed.embed_texts.return_value = [[0.5, 0.5], [0.1, 0.9]]
    mock_embed.embed_query.return_value = [0.5, 0.5]

    store = VectorStore(collection_name="manager_coll", embedding_manager=mock_embed)
    store.add_chunks(sample_chunks)

    results = store.query(query="working hours", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "handbook_p1_c0"
    mock_embed.embed_query.assert_called_once_with("working hours")


def test_vector_store_clear(sample_chunks: list[DocumentChunk]):
    store = VectorStore(collection_name="clear_coll")
    store.add_chunks(sample_chunks, embeddings=[[0.1, 0.2], [0.3, 0.4]])
    assert store.count() == 2

    store.clear()
    assert store.count() == 0


def test_vector_store_persistence(tmp_path: Path, sample_chunks: list[DocumentChunk]):
    db_dir = tmp_path / "chroma_test"

    store1 = VectorStore(persist_directory=db_dir, collection_name="persistent_coll")
    store1.add_chunks(sample_chunks, embeddings=[[0.2, 0.4], [0.6, 0.8]])
    assert store1.count() == 2

    # Reopen same persistent directory
    store2 = VectorStore(persist_directory=db_dir, collection_name="persistent_coll")
    assert store2.count() == 2


def test_vector_store_delete_document_and_retention(sample_chunks: list[DocumentChunk]):
    store = VectorStore(collection_name="retention_coll")
    store.add_chunks(sample_chunks, embeddings=[[0.1, 0.2], [0.3, 0.4]])
    assert store.count() == 2
    assert store.list_documents() == ["handbook.md"]

    deleted = store.delete_document("handbook.md")
    assert deleted == 2
    assert store.count() == 0
    assert store.list_documents() == []

    # Deleting non-existent doc returns 0
    assert store.delete_document("nonexistent.pdf") == 0

    # Cleanup retention
    store.add_chunks(sample_chunks, embeddings=[[0.1, 0.2], [0.3, 0.4]])
    assert store.cleanup_retention() == 2
    assert store.count() == 0
