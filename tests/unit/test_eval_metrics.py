"""Unit tests for evaluation metrics computation."""

from __future__ import annotations

from unittest.mock import MagicMock

from ragassistant.retrieval.evaluation import (
    evaluate_adversarial,
    evaluate_mode,
    is_evidence_in_chunk,
)
from ragassistant.schemas.document import DocumentChunk, RetrievedChunk


def _chunk(chunk_id: str, text: str, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            doc_name="test.md",
            page_number=1,
            chunk_index=0,
            text=text,
            token_count=10,
        ),
        score=score,
    )


def test_is_evidence_in_chunk():
    assert is_evidence_in_chunk(["10:00 AM to 3:00 PM"], "Core hours are 10:00 AM to 3:00 PM.")
    assert is_evidence_in_chunk(["Case InSensitive"], "This is case insensitive text.")
    assert not is_evidence_in_chunk(["Not present"], "Other text here.")


def test_evaluate_mode_metrics():
    mock_engine = MagicMock()
    # Query 1: hit at rank 1
    # Query 2: hit at rank 2
    mock_engine.retrieve.side_effect = [
        [_chunk("c1", "Target evidence here", 0.9), _chunk("c2", "Irrelevant", 0.3)],
        [_chunk("c3", "Irrelevant", 0.8), _chunk("c4", "Target evidence here", 0.7)],
    ]

    samples = [
        {"id": "q1", "question": "Question 1?", "expected_evidence": ["Target evidence"]},
        {"id": "q2", "question": "Question 2?", "expected_evidence": ["Target evidence"]},
    ]

    metrics = evaluate_mode(mock_engine, samples, use_reranker=False, top_k=2)

    assert metrics["recall@3"] == 1.0
    assert metrics["recall@5"] == 1.0
    # MRR: (1/1 + 1/2) / 2 = 1.5 / 2 = 0.75
    assert metrics["mrr"] == 0.75


def test_evaluate_adversarial():
    mock_engine = MagicMock()
    # First query top score 0.2 (below threshold 0.35 -> refused)
    # Second query top score 0.5 (above threshold 0.35 -> not refused)
    mock_engine.retrieve.side_effect = [
        [_chunk("c1", "Vague text", 0.20)],
        [_chunk("c2", "Confusing match", 0.50)],
    ]

    adversarial_samples = [
        {"id": "q21", "question": "Fake policy 1?"},
        {"id": "q22", "question": "Fake policy 2?"},
    ]

    result = evaluate_adversarial(mock_engine, adversarial_samples, threshold=0.35)
    assert result["refusal_rate"] == 0.5
    assert result["total_adversarial_queries"] == 2
