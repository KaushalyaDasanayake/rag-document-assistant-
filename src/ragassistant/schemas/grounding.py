"""Schemas for grounded answers, exact citations, and refusal mechanisms."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

STANDARD_REFUSAL = "I cannot find evidence in the provided documents regarding this query."


class Citation(BaseModel):
    """Verbatim quote citation referencing source document name and page number."""

    model_config = ConfigDict(extra="forbid")

    doc_name: str = Field(..., description="Source document filename")
    page: int = Field(..., ge=1, description="1-indexed source document page number")
    quote: str = Field(..., min_length=1, description="Verbatim text quote from the source chunk")


class GroundedAnswer(BaseModel):
    """Structured response enforcing factual grounding, citations, and refusal."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., description="Grounded natural language answer or standard refusal")
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of exact citations supporting the answer claims",
    )
    refused: bool = Field(
        default=False,
        description="True if the query cannot be answered solely from the provided context",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )
