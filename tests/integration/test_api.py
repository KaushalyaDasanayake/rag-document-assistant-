"""Integration tests for FastAPI endpoints: query, data retention, and observability."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from ragassistant.api.app import app, get_generator, get_store
from ragassistant.generation.generator import AnswerGenerator
from ragassistant.observability.tracer import Tracer, get_tracer
from ragassistant.retrieval.vector_store import VectorStore
from ragassistant.schemas.document import DocumentChunk, RetrievedChunk


@pytest.fixture
def mock_dependencies():
    """Setup mock generator, store, and isolated tracer for API testing."""
    test_tracer = Tracer()
    mock_store = MagicMock(spec=VectorStore)
    mock_engine = MagicMock()
    mock_engine.vector_store = mock_store
    mock_engine.relevance_threshold = 0.35

    mock_llm = MagicMock()
    generator = AnswerGenerator(
        retrieval_engine=mock_engine,
        llm_client=mock_llm,
        tracer=test_tracer,
        model_name="llama-3.1-8b-instant",
    )

    # Override FastAPI dependencies
    app.dependency_overrides[get_generator] = lambda: generator
    app.dependency_overrides[get_store] = lambda: mock_store
    app.dependency_overrides[get_tracer] = lambda: test_tracer

    yield {
        "generator": generator,
        "store": mock_store,
        "engine": mock_engine,
        "llm": mock_llm,
        "tracer": test_tracer,
    }

    app.dependency_overrides.clear()


@pytest.fixture
def client(mock_dependencies):
    return TestClient(app)


def test_health_check(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query_endpoint(client: TestClient, mock_dependencies):
    mock_deps = mock_dependencies
    mock_deps["engine"].retrieve.return_value = [
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id="handbook_p2_c0",
                doc_name="handbook.md",
                page_number=2,
                chunk_index=0,
                text="Standard core hours are 10:00 AM to 3:00 PM.",
                token_count=10,
            ),
            score=0.92,
        )
    ]
    mock_deps["llm"].generate.return_value = """{
        "answer": "Core hours are 10:00 AM to 3:00 PM.",
        "citations": [{"doc_name": "handbook.md", "page": 2, "quote": "10:00 AM to 3:00 PM"}],
        "refused": false,
        "confidence": 0.95
    }"""

    resp = client.post(
        "/v1/query",
        json={"query": "What are core hours?", "use_reranker": True, "top_k": 3},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"] == "Core hours are 10:00 AM to 3:00 PM."
    assert data["refused"] is False
    assert len(data["citations"]) == 1
    assert data["citations"][0]["quote"] == "10:00 AM to 3:00 PM"
    assert "trace" in data
    trace = data["trace"]
    assert trace["retrieval_latency_ms"] >= 0.0
    assert trace["generation_latency_ms"] >= 0.0
    assert trace["total_tokens"] > 0
    assert trace["estimated_cost_usd"] > 0.0


def test_list_and_delete_documents(client: TestClient, mock_dependencies):
    mock_store = mock_dependencies["store"]
    mock_store.list_documents.return_value = ["handbook.md", "cloud.pdf"]
    mock_store.delete_document.side_effect = lambda name: 5 if name == "handbook.md" else 0

    # List documents
    resp_list = client.get("/v1/documents")
    assert resp_list.status_code == 200
    assert resp_list.json()["documents"] == ["handbook.md", "cloud.pdf"]

    # Delete existing document
    resp_del = client.delete("/v1/documents/handbook.md")
    assert resp_del.status_code == 200
    assert resp_del.json()["deleted_chunks"] == 5

    # Delete non-existent document
    resp_404 = client.delete("/v1/documents/missing.md")
    assert resp_404.status_code == 404


def test_cleanup_retention_endpoint(client: TestClient, mock_dependencies):
    mock_store = mock_dependencies["store"]
    mock_store.cleanup_retention.return_value = 23

    resp = client.post("/v1/documents/cleanup")
    assert resp.status_code == 200
    assert resp.json()["purged_chunks"] == 23


def test_metrics_and_traces_endpoints(client: TestClient, mock_dependencies):
    # Simulate a query first so tracer has data
    tracer = mock_dependencies["tracer"]
    with tracer.trace("Sample query", model="llama-3.1-8b-instant") as ctx:
        ctx.record_retrieval(latency_ms=15.0)
        ctx.record_generation(latency_ms=50.0, input_tokens=100, output_tokens=25, refused=False)

    resp_metrics = client.get("/v1/metrics")
    assert resp_metrics.status_code == 200
    metrics = resp_metrics.json()
    assert metrics["total_queries"] == 1
    assert metrics["avg_retrieval_latency_ms"] == 15.0
    assert metrics["avg_generation_latency_ms"] == 50.0
    assert metrics["total_tokens"] == 125
    assert metrics["total_cost_usd"] > 0.0

    resp_traces = client.get("/v1/traces")
    assert resp_traces.status_code == 200
    traces = resp_traces.json()
    assert len(traces) == 1
    assert traces[0]["query"] == "Sample query"
