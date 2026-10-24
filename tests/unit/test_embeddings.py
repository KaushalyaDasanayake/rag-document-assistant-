"""Unit tests for embedding manager and device resolver."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ragassistant.retrieval.embeddings import EmbeddingManager, resolve_device


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto():
    device = resolve_device("auto")
    assert device in ("cuda", "cpu")


@patch("ragassistant.retrieval.embeddings.torch.cuda.is_available", return_value=False)
def test_resolve_device_cuda_fallback(mock_cuda):
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("cuda:0") == "cpu"


@patch("ragassistant.retrieval.embeddings.torch.cuda.is_available", return_value=True)
def test_resolve_device_cuda_available(mock_cuda):
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") == "cuda"


def test_embedding_manager_lazy_initialization():
    manager = EmbeddingManager(model_name="mock-model", device="cpu")
    assert manager._model is None


@patch("ragassistant.retrieval.embeddings.SentenceTransformer")
def test_embedding_manager_encode_flow(mock_st_class):
    mock_instance = MagicMock()
    mock_instance.get_embedding_dimension.return_value = 384
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_instance.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    mock_st_class.return_value = mock_instance

    manager = EmbeddingManager(model_name="test-model", device="cpu")
    assert manager.dimension == 384

    texts = ["First chunk", "Second chunk"]
    vectors = manager.embed_texts(texts)

    assert len(vectors) == 2
    assert pytest.approx(vectors[0][0]) == 0.1
    mock_instance.encode.assert_called_once()


@patch("ragassistant.retrieval.embeddings.SentenceTransformer")
def test_embedding_manager_embed_query(mock_st_class):
    mock_instance = MagicMock()
    mock_instance.encode.return_value = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
    mock_st_class.return_value = mock_instance

    manager = EmbeddingManager(model_name="test-model", device="cpu")
    vector = manager.embed_query("What is the policy?")
    assert len(vector) == 3
    assert pytest.approx(vector[0]) == 0.9


def test_embedding_manager_empty_list():
    manager = EmbeddingManager(model_name="test-model", device="cpu")
    assert manager.embed_texts([]) == []
