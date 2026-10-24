"""Retrieval package containing embedding management and vector store integration."""

from __future__ import annotations

from ragassistant.retrieval.embeddings import EmbeddingManager, resolve_device
from ragassistant.retrieval.engine import RetrievalEngine
from ragassistant.retrieval.evaluation import (
    evaluate_adversarial,
    evaluate_mode,
    is_evidence_in_chunk,
)
from ragassistant.retrieval.reranker import CrossEncoderReranker
from ragassistant.retrieval.vector_store import VectorStore

__all__ = [
    "EmbeddingManager",
    "VectorStore",
    "CrossEncoderReranker",
    "RetrievalEngine",
    "is_evidence_in_chunk",
    "evaluate_mode",
    "evaluate_adversarial",
    "resolve_device",
]
