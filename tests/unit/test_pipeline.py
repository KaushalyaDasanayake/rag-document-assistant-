"""Unit tests for ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from ragassistant.ingestion.pipeline import ingest_directory, ingest_file


def test_ingest_file(tmp_path: Path):
    doc_path = tmp_path / "test_doc.md"
    doc_path.write_text("# Title\n\nContent paragraph 1.\n\nContent paragraph 2.", encoding="utf-8")

    mock_store = MagicMock()
    stats = ingest_file(doc_path, vector_store=mock_store, chunk_size=200, chunk_overlap=30)

    assert stats.doc_name == "test_doc.md"
    assert stats.total_pages >= 1
    assert stats.total_chunks >= 1
    assert stats.total_characters > 0
    mock_store.add_chunks.assert_called_once()


def test_ingest_directory(tmp_path: Path):
    (tmp_path / "doc1.md").write_text("Markdown document content.", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("Text document content.", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("col1,col2\nval1,val2", encoding="utf-8")

    mock_store = MagicMock()
    stats_list = ingest_directory(tmp_path, vector_store=mock_store)

    assert len(stats_list) == 2
    assert {s.doc_name for s in stats_list} == {"doc1.md", "doc2.txt"}
    assert mock_store.add_chunks.call_count == 2
