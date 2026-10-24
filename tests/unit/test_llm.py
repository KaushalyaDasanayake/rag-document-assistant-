"""Unit tests for unified LLM client interface."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ragassistant.generation.llm import (
    DeterministicFallbackClient,
    GroqClient,
    OpenAIClient,
    get_llm_client,
)
from ragassistant.schemas.grounding import STANDARD_REFUSAL


def test_deterministic_fallback_canned_response():
    client = DeterministicFallbackClient(canned_responses={"hours": {"answer": "9 to 5", "refused": False, "confidence": 1.0, "citations": []}})
    resp = client.generate(system_prompt="", user_prompt="User Question: What are the hours?")
    data = json.loads(resp)
    assert data["answer"] == "9 to 5"
    assert data["refused"] is False


def test_deterministic_fallback_extracts_evidence():
    user_prompt = (
        "Context Passages:\n"
        "[Passage 1]\n"
        "Document: handbook.md\n"
        "Page: 2\n"
        "Content:\n"
        "Acme employees enjoy standard core hours between 10am and 3pm daily.\n\n"
        "User Question: What are the standard core hours at Acme?"
    )
    client = DeterministicFallbackClient()
    resp = client.generate(system_prompt="", user_prompt=user_prompt)
    data = json.loads(resp)

    assert data["refused"] is False
    assert len(data["citations"]) == 1
    assert data["citations"][0]["doc_name"] == "handbook.md"
    assert data["citations"][0]["page"] == 2
    assert "10am and 3pm" in data["citations"][0]["quote"]


def test_deterministic_fallback_refuses_unanswerable():
    user_prompt = (
        "Context Passages:\n"
        "[Passage 1]\n"
        "Document: handbook.md\n"
        "Page: 1\n"
        "Content:\n"
        "This handbook outlines general workplace principles.\n\n"
        "User Question: What is the gym reimbursement amount in Tokyo?"
    )
    client = DeterministicFallbackClient()
    resp = client.generate(system_prompt="", user_prompt=user_prompt)
    data = json.loads(resp)

    assert data["refused"] is True
    assert data["answer"] == STANDARD_REFUSAL
    assert data["citations"] == []


def test_get_llm_client_fallback_when_keys_missing():
    # Neither GROQ_API_KEY nor OPENAI_API_KEY is provided
    client_groq = get_llm_client(provider="groq", api_key=None, fallback_on_missing_key=True)
    assert isinstance(client_groq, DeterministicFallbackClient)

    client_openai = get_llm_client(provider="openai", api_key=None, fallback_on_missing_key=True)
    assert isinstance(client_openai, DeterministicFallbackClient)

    client_mock = get_llm_client(provider="mock")
    assert isinstance(client_mock, DeterministicFallbackClient)


@patch("groq.Groq")
def test_groq_client_invocation(mock_groq_cls):
    mock_instance = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"answer": "Real Groq answer", "refused": False, "confidence": 0.95, "citations": []})
    mock_instance.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    mock_groq_cls.return_value = mock_instance

    client = GroqClient(model="llama-3.1-8b-instant", api_key="gsk_test")
    resp = client.generate("system prompt", "user prompt")
    data = json.loads(resp)
    assert data["answer"] == "Real Groq answer"
    mock_instance.chat.completions.create.assert_called_once()


@patch("openai.OpenAI")
def test_openai_client_invocation(mock_openai_cls):
    mock_instance = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"answer": "Real OpenAI answer", "refused": False, "confidence": 0.98, "citations": []})
    mock_instance.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    mock_openai_cls.return_value = mock_instance

    client = OpenAIClient(model="gpt-4o-mini", api_key="sk-test")
    resp = client.generate("system prompt", "user prompt")
    data = json.loads(resp)
    assert data["answer"] == "Real OpenAI answer"
    mock_instance.chat.completions.create.assert_called_once()
