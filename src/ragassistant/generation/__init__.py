"""Generation package containing prompt engineering, LLM interfaces, and grounded answering."""

from __future__ import annotations

from ragassistant.generation.generator import (
    AnswerGenerator,
    verify_citation_quotes,
)
from ragassistant.generation.llm import (
    BaseLLMClient,
    DeterministicFallbackClient,
    GroqClient,
    OpenAIClient,
    get_llm_client,
)
from ragassistant.generation.prompts import (
    build_system_prompt,
    build_user_prompt,
    format_context_passages,
)

__all__ = [
    "format_context_passages",
    "build_system_prompt",
    "build_user_prompt",
    "BaseLLMClient",
    "GroqClient",
    "OpenAIClient",
    "DeterministicFallbackClient",
    "get_llm_client",
    "AnswerGenerator",
    "verify_citation_quotes",
]
