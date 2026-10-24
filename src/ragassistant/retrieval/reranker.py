"""Cross-encoder reranking module to rescore candidate chunks."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from ragassistant.retrieval.embeddings import resolve_device
from ragassistant.schemas.document import RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranker using a Cross-Encoder transformer model to rescore (query, chunk) pairs."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
        apply_sigmoid: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.apply_sigmoid = apply_sigmoid
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy load CrossEncoder model."""
        if self._model is None:
            logger.info("Loading CrossEncoder %s on %s", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Rescore candidate chunks against query and return descending ranked chunks."""
        if not candidates:
            return []

        pairs = [[query, candidate.chunk.text] for candidate in candidates]
        raw_scores: Any = self.model.predict(pairs, show_progress_bar=False)

        reranked: list[RetrievedChunk] = []
        for candidate, raw_score in zip(candidates, raw_scores, strict=False):
            val = float(raw_score)
            if self.apply_sigmoid:
                # Sigmoid to normalize logits to [0.0, 1.0] with clipping to prevent overflow
                clipped_val = float(np.clip(val, -50.0, 50.0))
                score = float(1.0 / (1.0 + np.exp(-clipped_val)))
            else:
                score = val

            reranked.append(RetrievedChunk(chunk=candidate.chunk, score=score))

        reranked.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            if top_k <= 0:
                return []
            return reranked[:top_k]

        return reranked
