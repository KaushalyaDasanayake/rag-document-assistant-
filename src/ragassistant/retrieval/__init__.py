"""Retrieval package containing embedding management and vector store integration."""

from __future__ import annotations

from ragassistant.retrieval.embeddings import EmbeddingManager, resolve_device
from ragassistant.retrieval.engine import RetrievalEngine
from ragassistant.retrieval.reranker import CrossEncoderReranker
from ragassistant.retrieval.vector_store import VectorStore

__all__ = [
    "EmbeddingManager",
    "VectorStore",
    "CrossEncoderReranker",
    "RetrievalEngine",
    "resolve_device",
]
