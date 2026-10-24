"""Document, chunk, and ingestion schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentPage(BaseModel):
    """Raw extracted text from a single page or file section."""

    model_config = ConfigDict(extra="forbid")

    doc_name: str = Field(..., min_length=1, description="Original filename")
    page_number: int = Field(..., ge=1, description="1-indexed page or section number")
    text: str = Field(..., min_length=1, description="Extracted raw text content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags")


class DocumentChunk(BaseModel):
    """A semantic chunk of text prepared for vector indexing and citations."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., description="Unique chunk identifier: e.g. doc_name_p1_c0")
    doc_name: str = Field(..., description="Source document filename")
    page_number: int = Field(..., ge=1, description="Source page number")
    chunk_index: int = Field(..., ge=0, description="Sequential index within the document")
    text: str = Field(..., min_length=1, description="Chunk text content")
    token_count: int = Field(default=0, ge=0, description="Estimated token count")


class IngestionStats(BaseModel):
    """Summary statistics for an ingestion job."""

    model_config = ConfigDict(extra="forbid")

    doc_name: str
    total_pages: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    total_characters: int = Field(..., ge=0)


class RetrievedChunk(BaseModel):
    """A retrieved chunk paired with its relevance score."""

    model_config = ConfigDict(extra="forbid")

    chunk: DocumentChunk
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
