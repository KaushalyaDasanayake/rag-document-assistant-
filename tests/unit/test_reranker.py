"""Unit tests for CrossEncoderReranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from ragassistant.retrieval.reranker import CrossEncoderReranker
from ragassistant.schemas.document import DocumentChunk, RetrievedChunk


def _make_candidate(chunk_id: str, text: str, initial_score: float = 0.5) -> RetrievedChunk:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        doc_name="doc.txt",
        page_number=1,
        chunk_index=0,
        text=text,
        token_count=10,
    )
    return RetrievedChunk(chunk=chunk, score=initial_score)


def test_reranker_lazy_initialization():
    reranker = CrossEncoderReranker(model_name="mock-reranker", device="cpu")
    assert reranker._model is None


def test_reranker_empty_candidates():
    reranker = CrossEncoderReranker(model_name="mock-reranker", device="cpu")
    assert reranker.rerank("query text", []) == []


@patch("ragassistant.retrieval.reranker.CrossEncoder")
def test_reranker_rescoring_and_ordering(mock_ce_class):
    mock_instance = MagicMock()
    # First chunk gets low score, second chunk gets high score
    mock_instance.predict.return_value = np.array([-2.0, 3.0], dtype=np.float32)
    mock_ce_class.return_value = mock_instance

    reranker = CrossEncoderReranker(model_name="mock-reranker", device="cpu", apply_sigmoid=True)
    c1 = _make_candidate("c1", "Irrelevant text about flowers", 0.9)
    c2 = _make_candidate("c2", "Relevant policy text about hours", 0.6)

    reranked = reranker.rerank("working hours", [c1, c2])

    assert len(reranked) == 2
    # c2 must be ranked first now because its score was 3.0 vs -2.0
    assert reranked[0].chunk.chunk_id == "c2"
    assert reranked[1].chunk.chunk_id == "c1"
    assert reranked[0].score > reranked[1].score
    assert 0.0 <= reranked[0].score <= 1.0


@patch("ragassistant.retrieval.reranker.CrossEncoder")
def test_reranker_top_k_clipping(mock_ce_class):
    mock_instance = MagicMock()
    mock_instance.predict.return_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mock_ce_class.return_value = mock_instance

    reranker = CrossEncoderReranker(model_name="mock-reranker", device="cpu")
    candidates = [
        _make_candidate("c1", "Text 1"),
        _make_candidate("c2", "Text 2"),
        _make_candidate("c3", "Text 3"),
    ]

    top_2 = reranker.rerank("query", candidates, top_k=2)
    assert len(top_2) == 2

    assert reranker.rerank("query", candidates, top_k=0) == []
    assert reranker.rerank("query", candidates, top_k=-1) == []
