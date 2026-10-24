"""Ingestion package initialization."""

from __future__ import annotations

from ragassistant.ingestion.loader import load_document, load_markdown_or_text, load_pdf

__all__ = ["load_document", "load_pdf", "load_markdown_or_text"]
