"""Prompt templates and context formatting enforcing strict factual grounding."""

from __future__ import annotations

from ragassistant.schemas.document import RetrievedChunk
from ragassistant.schemas.grounding import STANDARD_REFUSAL


def format_context_passages(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved document chunks into numbered context blocks with provenance headers."""
    if not chunks:
        return "No relevant context passages found."

    blocks: list[str] = []
    for idx, item in enumerate(chunks, start=1):
        c = item.chunk
        block = (
            f"[Passage {idx}]\n"
            f"Document: {c.doc_name}\n"
            f"Page: {c.page_number}\n"
            f"Relevance Score: {item.score:.4f}\n"
            f"Content:\n{c.text}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_system_prompt() -> str:
    """Construct the grounding system prompt with strict refusal and citation rules."""
    return f"""You are an enterprise document assistant that provides strictly grounded answers.

CRITICAL OPERATING RULES:
1. ONLY USE PROVIDED CONTEXT: Answer the query using ONLY the facts explicitly stated in the provided context passages. Never extrapolate, speculate, or introduce external knowledge.
2. REFUSAL MANDATE: If the provided passages do not contain explicit, verifiable evidence to answer the question completely, you MUST refuse:
   - Set "refused": true
   - Set "answer": "{STANDARD_REFUSAL}"
   - Set "citations": []
   - Set "confidence": 0.0
3. VERBATIM CITATIONS: Every claim made in an unrefused answer must be supported by at least one citation.
   - "doc_name": exact document filename matching the passage header
   - "page": exact integer page number from the passage header
   - "quote": an exact, verbatim substring quoted directly from the passage content. NEVER paraphrase or alter quotes.
4. CONFIDENCE: Set confidence between 0.0 and 1.0 reflecting how completely and unambiguously the context supports the answer.

RESPONSE FORMAT:
You must respond ONLY with a valid JSON object matching this schema:
{{
  "answer": "string",
  "citations": [
    {{
      "doc_name": "string",
      "page": 1,
      "quote": "verbatim substring from passage"
    }}
  ],
  "refused": false,
  "confidence": 0.95
}}
Do not include any text, markdown backticks, or commentary outside the JSON object."""


def build_user_prompt(query: str, context_passages: str) -> str:
    """Construct the user prompt combining the user question and formatted context."""
    return f"""Context Passages:
{context_passages}

User Question: {query}

Provide a grounded JSON response following all system rules."""
