"""Unified LLM client interface supporting Groq, OpenAI, and deterministic fallback."""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from ragassistant.schemas.grounding import STANDARD_REFUSAL

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract interface for LLM completion providers."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """Execute chat completion and return the raw response string."""
        pass


class GroqClient(BaseLLMClient):
    """LLM client for Groq API."""

    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str | None = None) -> None:
        from groq import Groq

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError("GROQ_API_KEY environment variable or parameter is required for GroqClient")

        self.model = model
        self.client = Groq(api_key=resolved_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or "{}"
        return content


class OpenAIClient(BaseLLMClient):
    """LLM client for OpenAI API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        from openai import OpenAI

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY environment variable or parameter is required for OpenAIClient")

        self.model = model
        self.client = OpenAI(api_key=resolved_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or "{}"
        return content


class DeterministicFallbackClient(BaseLLMClient):
    """Offline deterministic LLM client for testing and zero-API-key fallback."""

    def __init__(self, canned_responses: dict[str, Any] | None = None) -> None:
        self.canned_responses = canned_responses or {}

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        # Check canned responses by query match
        for key, resp in self.canned_responses.items():
            if key.lower() in user_prompt.lower():
                return json.dumps(resp)

        # Parse passages and user question from prompt
        question_match = re.search(r"User Question:\s*(.+)", user_prompt, re.IGNORECASE)
        question = question_match.group(1).strip() if question_match else ""

        # Extract passages: [Passage 1]\nDocument: ...\nPage: ...\nContent:\n...
        passage_blocks = re.findall(
            r"Document:\s*([^\n]+)\s*\nPage:\s*(\d+)\s*\n(?:Relevance Score:[^\n]+\n)?Content:\n([\s\S]*?)(?=\n\n\[Passage|\n\nUser Question:|\Z)",
            user_prompt,
        )

        if not passage_blocks:
            return json.dumps(
                {
                    "answer": STANDARD_REFUSAL,
                    "citations": [],
                    "refused": True,
                    "confidence": 0.0,
                }
            )

        # Keyword overlap search between question and passage sentences
        q_words = {w.lower() for w in re.findall(r"\w+", question) if len(w) > 3}
        best_sentence = ""
        best_doc = ""
        best_page = 1
        best_overlap = 0

        for doc_name, page_str, content in passage_blocks:
            sentences = re.split(r"(?<=[.!?])\s+", content.strip())
            for sent in sentences:
                s_words = {w.lower() for w in re.findall(r"\w+", sent)}
                overlap = len(q_words.intersection(s_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = sent.strip()
                    best_doc = doc_name.strip()
                    best_page = int(page_str)

        # If keyword overlap is too low (< 2 matching key terms), refuse
        if best_overlap < 2 or not best_sentence:
            return json.dumps(
                {
                    "answer": STANDARD_REFUSAL,
                    "citations": [],
                    "refused": True,
                    "confidence": 0.0,
                }
            )

        return json.dumps(
            {
                "answer": f"According to {best_doc}: {best_sentence}",
                "citations": [
                    {
                        "doc_name": best_doc,
                        "page": best_page,
                        "quote": best_sentence,
                    }
                ],
                "refused": False,
                "confidence": 0.90,
            }
        )


def get_llm_client(
    provider: str = "groq",
    model: str | None = None,
    api_key: str | None = None,
    fallback_on_missing_key: bool = True,
) -> BaseLLMClient:
    """Factory creating appropriate LLM client with seamless deterministic fallback."""
    norm_provider = provider.strip().lower()

    if norm_provider in ("mock", "deterministic", "fallback"):
        return DeterministicFallbackClient()

    if norm_provider == "groq":
        try:
            return GroqClient(model=model or "llama-3.1-8b-instant", api_key=api_key)
        except ValueError as err:
            if fallback_on_missing_key:
                logger.warning("%s. Falling back to DeterministicFallbackClient.", err)
                return DeterministicFallbackClient()
            raise

    if norm_provider == "openai":
        try:
            return OpenAIClient(model=model or "gpt-4o-mini", api_key=api_key)
        except ValueError as err:
            if fallback_on_missing_key:
                logger.warning("%s. Falling back to DeterministicFallbackClient.", err)
                return DeterministicFallbackClient()
            raise

    logger.warning("Unknown provider '%s'. Using DeterministicFallbackClient.", provider)
    return DeterministicFallbackClient()
