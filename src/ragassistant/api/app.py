"""FastAPI application providing grounded query, retention, and observability endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ragassistant.generation.generator import AnswerGenerator
from ragassistant.observability.tracer import Tracer, TraceRecord, get_tracer
from ragassistant.retrieval.vector_store import VectorStore
from ragassistant.schemas.grounding import Citation

app = FastAPI(
    title="RAG Document Assistant API",
    description="Enterprise RAG assistant with grounded citations, refusal mechanisms, and token cost accounting.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    """Payload for asking questions to the document assistant."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Question to ask")
    use_reranker: bool = Field(default=True, description="Whether to apply Cross-Encoder reranking")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of context chunks to use")


class QueryResponse(BaseModel):
    """Grounded answer bundled with provenance citations and execution trace."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation]
    refused: bool
    confidence: float
    trace: TraceRecord


class DeleteDocumentResponse(BaseModel):
    """Confirmation payload for data retention deletion."""

    model_config = ConfigDict(extra="forbid")

    doc_name: str
    deleted_chunks: int
    message: str


class CleanupResponse(BaseModel):
    """Confirmation payload for index retention wipe."""

    model_config = ConfigDict(extra="forbid")

    purged_chunks: int
    message: str


# Dependency singletons
_GENERATOR: AnswerGenerator | None = None


def get_generator() -> AnswerGenerator:
    """Dependency provider for AnswerGenerator singleton."""
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = AnswerGenerator.from_config()
    return _GENERATOR


def get_store() -> VectorStore:
    """Dependency provider for VectorStore."""
    return get_generator().retrieval_engine.vector_store


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "ragassistant"}


@app.post("/v1/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
def query_documents(
    payload: QueryRequest,
    generator: AnswerGenerator = Depends(get_generator),
) -> QueryResponse:
    """Query documents and receive a grounded answer with citations and execution telemetry."""
    answer, trace = generator.generate_with_trace(
        query=payload.query,
        use_reranker=payload.use_reranker,
        top_k=payload.top_k,
    )
    return QueryResponse(
        answer=answer.answer,
        citations=answer.citations,
        refused=answer.refused,
        confidence=answer.confidence,
        trace=trace,
    )


@app.get("/v1/documents", tags=["Data Retention"])
def list_documents(store: VectorStore = Depends(get_store)) -> dict[str, list[str]]:
    """List all document names currently indexed in the vector store."""
    return {"documents": store.list_documents()}


@app.delete("/v1/documents/{doc_name}", response_model=DeleteDocumentResponse, tags=["Data Retention"])
def delete_document(
    doc_name: str,
    store: VectorStore = Depends(get_store),
) -> DeleteDocumentResponse:
    """Delete all indexed chunks for a document under corporate retention policies."""
    deleted_count = store.delete_document(doc_name)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found in index")

    return DeleteDocumentResponse(
        doc_name=doc_name,
        deleted_chunks=deleted_count,
        message=f"Successfully deleted {deleted_count} chunks for document '{doc_name}'.",
    )


@app.post("/v1/documents/cleanup", response_model=CleanupResponse, tags=["Data Retention"])
def cleanup_retention(store: VectorStore = Depends(get_store)) -> CleanupResponse:
    """Purge all index entries in compliance with data retention lifecycle rules."""
    purged_count = store.cleanup_retention()
    return CleanupResponse(
        purged_chunks=purged_count,
        message=f"Data retention wipe complete. Purged {purged_count} chunks from index.",
    )


@app.get("/v1/metrics", tags=["Observability & Cost Accounting"])
def get_metrics(tracer: Tracer = Depends(get_tracer)) -> dict[str, Any]:
    """Retrieve latency metrics, token consumption, and cost estimates per 1k queries."""
    return tracer.get_metrics()


@app.get("/v1/traces", response_model=list[TraceRecord], tags=["Observability & Cost Accounting"])
def get_traces(limit: int = 50, tracer: Tracer = Depends(get_tracer)) -> list[TraceRecord]:
    """Retrieve recent query trace records."""
    return tracer.get_traces(limit=limit)
