"""Embedding model manager with GPU/CPU resolution and batch encoding."""

from __future__ import annotations

import logging
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def resolve_device(device_preference: str = "auto") -> str:
    """Resolve target execution device ('cuda' or 'cpu') based on hardware availability."""
    normalized = device_preference.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU")
            return "cpu"
        return normalized
    return "cpu"


class EmbeddingManager:
    """Manages dense vector embeddings using SentenceTransformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "auto",
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.normalize_embeddings = normalize_embeddings
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            logger.info("Loading embedding model %s on device: %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        """Return dimensionality of the embedding vectors."""
        if self._dimension is None:
            # Trigger lazy load
            _ = self.model
        assert self._dimension is not None
        return self._dimension

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Encode a batch of text chunks into normalized float vectors."""
        if not texts:
            return []

        embeddings: Any = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode a single search query."""
        results = self.embed_texts([query])
        return results[0]
