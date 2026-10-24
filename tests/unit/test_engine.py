"""Unit tests for unified RetrievalEngine."""

from __future__ import annotations

from unittest.mock import MagicMock

from ragassistant.retrieval.engine import RetrievalEngine
from ragassistant.schemas.document import DocumentChunk, RetrievedChunk


def _mock_chunk(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        doc_name="doc.md",
        page_number=1,
        chunk_index=0,
        text=text,
        token_count=10,
    )
    return RetrievedChunk(chunk=chunk, score=score)


def test_retrieval_engine_empty_query():
    mock_store = MagicMock()
    engine = RetrievalEngine(vector_store=mock_store)
    assert engine.retrieve("   ") == []
    mock_store.query.assert_not_called()


def test_retrieval_engine_biencoder_only():
    mock_store = MagicMock()
    c1 = _mock_chunk("c1", "Chunk 1 text", 0.9)
    c2 = _mock_chunk("c2", "Chunk 2 text", 0.8)
    c3 = _mock_chunk("c3", "Chunk 3 text", 0.7)
    mock_store.query.return_value = [c1, c2, c3]

    engine = RetrievalEngine(vector_store=mock_store, reranker=None, top_k_candidates=10, top_k_reranked=2)
    results = engine.retrieve("query", use_reranker=False)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c1"
    assert results[1].chunk.chunk_id == "c2"


def test_retrieval_engine_with_reranker():
    mock_store = MagicMock()
    c1 = _mock_chunk("c1", "Chunk 1 text", 0.9)
    c2 = _mock_chunk("c2", "Chunk 2 text", 0.8)
    mock_store.query.return_value = [c1, c2]

    mock_reranker = MagicMock()
    # Reranker reverses the order
    reranked_c2 = _mock_chunk("c2", "Chunk 2 text", 0.95)
    reranked_c1 = _mock_chunk("c1", "Chunk 1 text", 0.40)
    mock_reranker.rerank.return_value = [reranked_c2, reranked_c1]

    engine = RetrievalEngine(vector_store=mock_store, reranker=mock_reranker, top_k_candidates=10, top_k_reranked=2)
    results = engine.retrieve("query", use_reranker=True)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c2"
    mock_reranker.rerank.assert_called_once_with(query="query", candidates=[c1, c2], top_k=2)


def test_retrieval_engine_non_positive_top_k():
    mock_store = MagicMock()
    mock_store.query.return_value = [_mock_chunk("c1", "Text", 0.9)]
    engine = RetrievalEngine(vector_store=mock_store)

    assert engine.retrieve("query", top_k=0) == []
    assert engine.retrieve("query", top_k=-1) == []
