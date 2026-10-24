"""Observability, latency profiling, and token cost accounting."""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Model pricing rates in USD per 1,000,000 tokens
MODEL_PRICING: dict[str, dict[str, float]] = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "deterministic": {"input": 0.0, "output": 0.0},
    "mock": {"input": 0.0, "output": 0.0},
    "default": {"input": 0.10, "output": 0.20},
}


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string (~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class CostEstimator:
    """Calculates financial cost and extrapolated per-1,000 queries pricing."""

    @classmethod
    def calculate(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> tuple[float, float]:
        """Return (cost_usd, cost_per_1k_queries_usd)."""
        pricing = MODEL_PRICING.get(model.lower(), MODEL_PRICING["default"])
        input_cost = (input_tokens / 1_000_000.0) * pricing["input"]
        output_cost = (output_tokens / 1_000_000.0) * pricing["output"]
        total_cost = round(input_cost + output_cost, 8)
        cost_per_1k = round(total_cost * 1_000.0, 6)
        return total_cost, cost_per_1k


class TraceRecord(BaseModel):
    """Complete observability record for a single RAG query execution."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(..., description="Unique UUID for query trace")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    query: str = Field(..., description="User query text")
    retrieval_latency_ms: float = Field(..., ge=0.0, description="Dense retrieval latency")
    rerank_latency_ms: float = Field(default=0.0, ge=0.0, description="Reranking latency")
    generation_latency_ms: float = Field(..., ge=0.0, description="LLM generation latency")
    total_latency_ms: float = Field(..., ge=0.0, description="End-to-end pipeline latency")
    input_tokens: int = Field(..., ge=0, description="Input token count")
    output_tokens: int = Field(..., ge=0, description="Output token count")
    total_tokens: int = Field(..., ge=0, description="Total tokens consumed")
    estimated_cost_usd: float = Field(..., ge=0.0, description="Estimated query cost in USD")
    cost_per_1k_queries_usd: float = Field(..., ge=0.0, description="Cost extrapolated per 1,000 queries")
    model: str = Field(..., description="Generation model name")
    refused: bool = Field(default=False, description="Whether query was refused")
    num_citations: int = Field(default=0, ge=0, description="Count of verified citations")


class TraceContext:
    """Active tracing session accumulator."""

    def __init__(self, trace_id: str, query: str, model: str) -> None:
        self.trace_id = trace_id
        self.query = query
        self.model = model
        self.start_time = time.perf_counter()
        self.retrieval_latency_ms = 0.0
        self.rerank_latency_ms = 0.0
        self.generation_latency_ms = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.refused = False
        self.num_citations = 0

    def record_retrieval(self, latency_ms: float, rerank_ms: float = 0.0) -> None:
        """Record retrieval and optional rerank latencies."""
        self.retrieval_latency_ms = round(latency_ms, 2)
        self.rerank_latency_ms = round(rerank_ms, 2)

    def record_generation(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        refused: bool = False,
        num_citations: int = 0,
    ) -> None:
        """Record generation latency, token counts, and grounding outcome."""
        self.generation_latency_ms = round(latency_ms, 2)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.refused = refused
        self.num_citations = num_citations

    def build_record(self) -> TraceRecord:
        """Assemble finalized immutable TraceRecord."""
        total_latency = round((time.perf_counter() - self.start_time) * 1000.0, 2)
        total_tokens = self.input_tokens + self.output_tokens
        cost, cost_per_1k = CostEstimator.calculate(
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

        return TraceRecord(
            trace_id=self.trace_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=self.query,
            retrieval_latency_ms=self.retrieval_latency_ms,
            rerank_latency_ms=self.rerank_latency_ms,
            generation_latency_ms=self.generation_latency_ms,
            total_latency_ms=total_latency,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            cost_per_1k_queries_usd=cost_per_1k,
            model=self.model,
            refused=self.refused,
            num_citations=self.num_citations,
        )


class Tracer:
    """Thread-safe ring-buffer tracer managing metrics aggregation."""

    def __init__(self, max_records: int = 1000) -> None:
        self.records: deque[TraceRecord] = deque(maxlen=max_records)

    @contextmanager
    def trace(self, query: str, model: str = "llama-3.1-8b-instant") -> Generator[TraceContext, None, None]:
        """Context manager tracking end-to-end query execution."""
        ctx = TraceContext(trace_id=f"tr_{uuid.uuid4().hex[:12]}", query=query, model=model)
        try:
            yield ctx
        finally:
            record = ctx.build_record()
            self.records.append(record)

    def add_record(self, record: TraceRecord) -> None:
        """Manually append a pre-built TraceRecord."""
        self.records.append(record)

    def get_traces(self, limit: int = 50) -> list[TraceRecord]:
        """Return the most recent traces in reverse chronological order."""
        return list(reversed(self.records))[:limit]

    def get_metrics(self) -> dict[str, Any]:
        """Calculate system-wide aggregate telemetry, token accounting, and cost per 1k queries."""
        count = len(self.records)
        if count == 0:
            return {
                "total_queries": 0,
                "refusal_rate": 0.0,
                "avg_total_latency_ms": 0.0,
                "avg_retrieval_latency_ms": 0.0,
                "avg_generation_latency_ms": 0.0,
                "p95_total_latency_ms": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_cost_per_query_usd": 0.0,
                "cost_per_1k_queries_usd": 0.0,
            }

        refusals = sum(1 for r in self.records if r.refused)
        retrieval_lats = [r.retrieval_latency_ms for r in self.records]
        generation_lats = [r.generation_latency_ms for r in self.records]
        total_lats = sorted(r.total_latency_ms for r in self.records)

        p95_idx = max(0, math.ceil(0.95 * count) - 1)
        p95_lat = total_lats[p95_idx]

        in_tokens = sum(r.input_tokens for r in self.records)
        out_tokens = sum(r.output_tokens for r in self.records)
        tot_cost = sum(r.estimated_cost_usd for r in self.records)

        avg_cost = tot_cost / count
        cost_per_1k = avg_cost * 1_000.0

        return {
            "total_queries": count,
            "refusal_rate": round(refusals / count, 4),
            "avg_total_latency_ms": round(sum(total_lats) / count, 2),
            "avg_retrieval_latency_ms": round(sum(retrieval_lats) / count, 2),
            "avg_generation_latency_ms": round(sum(generation_lats) / count, 2),
            "p95_total_latency_ms": round(p95_lat, 2),
            "total_input_tokens": in_tokens,
            "total_output_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
            "total_cost_usd": round(tot_cost, 6),
            "avg_cost_per_query_usd": round(avg_cost, 6),
            "cost_per_1k_queries_usd": round(cost_per_1k, 4),
        }

    def clear(self) -> None:
        """Clear recorded telemetry."""
        self.records.clear()


# Default singleton instance
_GLOBAL_TRACER: Tracer | None = None


def get_tracer() -> Tracer:
    """Retrieve process-wide global Tracer instance."""
    global _GLOBAL_TRACER
    if _GLOBAL_TRACER is None:
        _GLOBAL_TRACER = Tracer()
    return _GLOBAL_TRACER
