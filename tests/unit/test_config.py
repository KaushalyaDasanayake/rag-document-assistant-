"""Unit tests for configuration loading."""

from __future__ import annotations

from ragassistant.config import load_config


def test_load_default_config():
    cfg = load_config()
    assert "retrieval" in cfg
    assert "generation" in cfg
    assert "storage" in cfg
    assert cfg["retrieval"]["chunk_size"] > 0
    assert cfg["retrieval"]["chunk_overlap"] >= 0
