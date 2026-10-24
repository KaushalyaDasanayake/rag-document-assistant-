"""Unit tests for prompt templates and context formatting."""

from __future__ import annotations

from ragassistant.generation.prompts import (
    build_system_prompt,
    build_user_prompt,
    format_context_passages,
)
from ragassistant.schemas.document import DocumentChunk, RetrievedChunk
from ragassistant.schemas.grounding import STANDARD_REFUSAL


def test_format_context_passages_empty():
    assert format_context_passages([]) == "No relevant context passages found."


def test_format_context_passages():
    chunk = DocumentChunk(
        chunk_id="doc1_p3_c0",
        doc_name="policy.md",
        page_number=3,
        chunk_index=0,
        text="Vacation days cannot exceed 20 days per year.",
        token_count=8,
    )
    retrieved = RetrievedChunk(chunk=chunk, score=0.9123)

    formatted = format_context_passages([retrieved])
    assert "[Passage 1]" in formatted
    assert "Document: policy.md" in formatted
    assert "Page: 3" in formatted
    assert "Relevance Score: 0.9123" in formatted
    assert "Vacation days cannot exceed 20 days per year." in formatted


def test_build_system_prompt():
    prompt = build_system_prompt()
    assert "ONLY USE PROVIDED CONTEXT" in prompt
    assert STANDARD_REFUSAL in prompt
    assert "VERBATIM CITATIONS" in prompt
    assert "RESPONSE FORMAT" in prompt


def test_build_user_prompt():
    prompt = build_user_prompt("What is the leave policy?", "Context text")
    assert "Context text" in prompt
    assert "What is the leave policy?" in prompt
