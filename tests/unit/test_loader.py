"""Unit tests for document ingestion loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragassistant.ingestion.loader import load_document


def test_load_sample_markdown_doc():
    sample_path = Path("data/sample_docs/employee_handbook.md")
    pages = load_document(sample_path)

    assert len(pages) >= 4, f"Expected at least 4 sections/pages, got {len(pages)}"
    assert pages[0].doc_name == "employee_handbook.md"
    assert pages[0].page_number == 1
    assert "Acme Technologies" in pages[0].text


def test_load_plain_text(tmp_path: Path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Meeting notes from Monday morning.\nAction items included.", encoding="utf-8")

    pages = load_document(txt_file)
    assert len(pages) == 1
    assert pages[0].doc_name == "notes.txt"
    assert "Meeting notes" in pages[0].text


def test_unsupported_file_extension(tmp_path: Path):
    invalid_file = tmp_path / "data.csv"
    invalid_file.write_text("a,b,c\n1,2,3", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_document(invalid_file)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_document("non_existent_file.pdf")
