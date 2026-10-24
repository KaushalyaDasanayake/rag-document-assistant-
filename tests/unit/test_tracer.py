"""Unit tests for Tracer, CostEstimator, and TraceRecord."""

from __future__ import annotations

import time

from ragassistant.observability.tracer import (
    CostEstimator,
    Tracer,
    TraceRecord,
    estimate_tokens,
)


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") >= 1
    long_text = "word " * 100
    assert estimate_tokens(long_text) >= 50


def test_cost_estimator():
    cost, per_1k = CostEstimator.calculate(
        model="llama-3.1-8b-instant",
        input_tokens=100_000,
        output_tokens=100_000,
    )
    # 100k input @ $0.05/1M = $0.005, 100k output @ $0.08/1M = $0.008 -> total = $0.013
    assert abs(cost - 0.013) < 1e-4
    assert abs(per_1k - 13.0) < 1e-2


def test_tracer_context_and_metrics():
    tracer = Tracer()
    tracer.clear()

    with tracer.trace("What are the hours?", model="llama-3.1-8b-instant") as ctx:
        time.sleep(0.01)
        ctx.record_retrieval(latency_ms=12.5, rerank_ms=5.0)
        ctx.record_generation(
            latency_ms=45.0,
            input_tokens=150,
            output_tokens=30,
            refused=False,
            num_citations=1,
        )

    records = tracer.get_traces()
    assert len(records) == 1
    rec = records[0]
    assert rec.query == "What are the hours?"
    assert rec.retrieval_latency_ms == 12.5
    assert rec.rerank_latency_ms == 5.0
    assert rec.generation_latency_ms == 45.0
    assert rec.total_latency_ms >= 10.0
    assert rec.input_tokens == 150
    assert rec.output_tokens == 30
    assert rec.total_tokens == 180
    assert rec.refused is False
    assert rec.num_citations == 1

    metrics = tracer.get_metrics()
    assert metrics["total_queries"] == 1
    assert metrics["refusal_rate"] == 0.0
    assert metrics["avg_retrieval_latency_ms"] == 12.5
    assert metrics["avg_generation_latency_ms"] == 45.0
    assert metrics["total_tokens"] == 180
    assert metrics["total_cost_usd"] > 0.0


def test_tracer_refusal_accounting():
    tracer = Tracer()
    tracer.clear()

    with tracer.trace("Adversarial query", model="llama-3.1-8b-instant") as ctx:
        ctx.record_retrieval(latency_ms=10.0)
        ctx.record_generation(latency_ms=5.0, input_tokens=50, output_tokens=10, refused=True)

    metrics = tracer.get_metrics()
    assert metrics["total_queries"] == 1
    assert metrics["refusal_rate"] == 1.0


def test_trace_record_extra_forbid():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TraceRecord(
            trace_id="1",
            timestamp="now",
            query="test",
            retrieval_latency_ms=1.0,
            generation_latency_ms=1.0,
            total_latency_ms=2.0,
            input_tokens=10,
            output_tokens=10,
            total_tokens=20,
            estimated_cost_usd=0.01,
            cost_per_1k_queries_usd=10.0,
            model="test",
            unsupported_field=123,
        )
