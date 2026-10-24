"""Retrieval engine coordinating candidate bi-encoder search and cross-encoder reranking."""

from __future__ import annotations

import logging
from typing import Any

from ragassistant.config import load_config
from ragassistant.retrieval.embeddings import EmbeddingManager
from ragassistant.retrieval.reranker import CrossEncoderReranker
from ragassistant.retrieval.vector_store import VectorStore
from ragassistant.schemas.document import RetrievedChunk

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Unified retrieval pipeline performing dense similarity search and optional reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        reranker: CrossEncoderReranker | None = None,
        top_k_candidates: int = 10,
        top_k_reranked: int = 3,
        relevance_threshold: float = 0.35,
    ) -> None:
        self.vector_store = vector_store
        self.reranker = reranker
        self.top_k_candidates = top_k_candidates
        self.top_k_reranked = top_k_reranked
        self.relevance_threshold = relevance_threshold

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> RetrievalEngine:
        """Construct RetrievalEngine from configuration dictionary or base.yaml."""
        cfg = config or load_config()
        retrieval_cfg = cfg.get("retrieval", {})
        storage_cfg = cfg.get("storage", {})

        device = retrieval_cfg.get("device", "auto")
        embedding_model = retrieval_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
        reranker_model = retrieval_cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        db_dir = storage_cfg.get("chroma_db_dir", "data/chroma_db")

        embed_manager = EmbeddingManager(model_name=embedding_model, device=device)
        vector_store = VectorStore(persist_directory=db_dir, embedding_manager=embed_manager)
        reranker = CrossEncoderReranker(model_name=reranker_model, device=device)

        return cls(
            vector_store=vector_store,
            reranker=reranker,
            top_k_candidates=int(retrieval_cfg.get("top_k_candidates", 10)),
            top_k_reranked=int(retrieval_cfg.get("top_k_reranked", 3)),
            relevance_threshold=float(retrieval_cfg.get("relevance_threshold", 0.35)),
        )

    def retrieve(
        self,
        query: str,
        use_reranker: bool = True,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve most relevant chunks for a query with optional reranking."""
        if not query.strip():
            return []

        target_k = top_k if top_k is not None else self.top_k_reranked
        if target_k <= 0:
            return []

        candidate_k = max(self.top_k_candidates, target_k)
        candidates = self.vector_store.query(query=query, top_k=candidate_k)
        if not candidates:
            return []

        if use_reranker and self.reranker is not None:
            return self.reranker.rerank(query=query, candidates=candidates, top_k=target_k)

        return candidates[:target_k]
