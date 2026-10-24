"""Unit tests for AnswerGenerator and citation quote verification."""

from __future__ import annotations

from unittest.mock import MagicMock

from ragassistant.generation.generator import AnswerGenerator, verify_citation_quotes
from ragassistant.generation.llm import DeterministicFallbackClient
from ragassistant.schemas.document import DocumentChunk, RetrievedChunk
from ragassistant.schemas.grounding import STANDARD_REFUSAL, Citation


def _retrieved_chunk(chunk_id: str, doc_name: str, page: int, text: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            doc_name=doc_name,
            page_number=page,
            chunk_index=0,
            text=text,
            token_count=10,
        ),
        score=score,
    )


def test_verify_citation_quotes_valid_and_invalid():
    passages = [
        _retrieved_chunk("c1", "handbook.md", 2, "Standard core hours are 10:00 AM to 3:00 PM in local time."),
    ]

    valid_cit = Citation(
        doc_name="handbook.md",
        page=2,
        quote="10:00 AM to 3:00 PM",
    )
    hallucinated_cit = Citation(
        doc_name="handbook.md",
        page=2,
        quote="Employees must wear formal suits",  # Not in passage
    )
    wrong_page_cit = Citation(
        doc_name="handbook.md",
        page=5,
        quote="10:00 AM to 3:00 PM",  # Page mismatch
    )

    verified = verify_citation_quotes([valid_cit, hallucinated_cit, wrong_page_cit], passages)
    assert len(verified) == 1
    assert verified[0].quote == "10:00 AM to 3:00 PM"


def test_generator_refuses_when_candidates_empty():
    mock_engine = MagicMock()
    mock_engine.retrieve.return_value = []
    mock_llm = MagicMock()

    gen = AnswerGenerator(retrieval_engine=mock_engine, llm_client=mock_llm)
    res = gen.generate("Any query")

    assert res.refused is True
    assert res.answer == STANDARD_REFUSAL
    mock_llm.generate.assert_not_called()


def test_generator_refuses_when_below_threshold():
    mock_engine = MagicMock()
    mock_engine.relevance_threshold = 0.35
    # Best candidate score is 0.20 < 0.35
    mock_engine.retrieve.return_value = [
        _retrieved_chunk("c1", "doc.md", 1, "Vaguely related sentence", score=0.20)
    ]
    mock_llm = MagicMock()

    gen = AnswerGenerator(retrieval_engine=mock_engine, llm_client=mock_llm)
    res = gen.generate("How to fly a rocket?")

    assert res.refused is True
    assert res.answer == STANDARD_REFUSAL
    mock_llm.generate.assert_not_called()


def test_generator_grounded_success():
    mock_engine = MagicMock()
    mock_engine.relevance_threshold = 0.35
    mock_engine.retrieve.return_value = [
        _retrieved_chunk("c1", "handbook.md", 2, "Standard core hours are 10:00 AM to 3:00 PM.", score=0.95)
    ]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = """{
        "answer": "Core hours are 10:00 AM to 3:00 PM.",
        "citations": [{"doc_name": "handbook.md", "page": 2, "quote": "10:00 AM to 3:00 PM"}],
        "refused": false,
        "confidence": 0.95
    }"""

    gen = AnswerGenerator(retrieval_engine=mock_engine, llm_client=mock_llm)
    res = gen.generate("What are core hours?")

    assert res.refused is False
    assert len(res.citations) == 1
    assert res.citations[0].quote == "10:00 AM to 3:00 PM"
    assert res.confidence == 0.95


def test_generator_refuses_when_citations_are_hallucinated():
    mock_engine = MagicMock()
    mock_engine.relevance_threshold = 0.35
    mock_engine.retrieve.return_value = [
        _retrieved_chunk("c1", "handbook.md", 2, "Standard core hours are 10:00 AM to 3:00 PM.", score=0.95)
    ]

    mock_llm = MagicMock()
    # Model hallucinated quote that doesn't exist in source text
    mock_llm.generate.return_value = """{
        "answer": "Office hours are from 7am to 11pm.",
        "citations": [{"doc_name": "handbook.md", "page": 2, "quote": "7am to 11pm"}],
        "refused": false,
        "confidence": 0.95
    }"""

    gen = AnswerGenerator(retrieval_engine=mock_engine, llm_client=mock_llm)
    res = gen.generate("What are office hours?")

    # Enforces refusal because citation was hallucinated
    assert res.refused is True
    assert res.answer == STANDARD_REFUSAL
    assert res.citations == []


def test_generator_with_deterministic_fallback():
    mock_engine = MagicMock()
    mock_engine.relevance_threshold = 0.35
    mock_engine.retrieve.return_value = [
        _retrieved_chunk("c1", "handbook.md", 2, "Acme employees must provide two weeks written notice.", score=0.95)
    ]

    gen = AnswerGenerator(retrieval_engine=mock_engine, llm_client=DeterministicFallbackClient())
    res = gen.generate("How many weeks notice must employees provide?")

    assert res.refused is False
    assert len(res.citations) == 1
    assert res.citations[0].doc_name == "handbook.md"
