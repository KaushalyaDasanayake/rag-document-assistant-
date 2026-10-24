"""Grounded answer generation coordinator with quote verification and refusal logic."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from ragassistant.config import load_config
from ragassistant.generation.llm import BaseLLMClient, get_llm_client
from ragassistant.generation.prompts import (
    build_system_prompt,
    build_user_prompt,
    format_context_passages,
)
from ragassistant.observability.tracer import (
    Tracer,
    TraceRecord,
    estimate_tokens,
    get_tracer,
)
from ragassistant.retrieval.engine import RetrievalEngine
from ragassistant.schemas.document import RetrievedChunk
from ragassistant.schemas.grounding import (
    STANDARD_REFUSAL,
    Citation,
    GroundedAnswer,
)

logger = logging.getLogger(__name__)


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Extract valid JSON dictionary from LLM response text, stripping markdown backticks if present."""
    text = raw_text.strip()
    # Remove markdown code fences if LLM wrapped JSON in ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback regex search for { ... }
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse valid JSON from response: {raw_text[:200]}") from None


def verify_citation_quotes(
    citations: list[Citation],
    passages: list[RetrievedChunk],
) -> list[Citation]:
    """Verify that every citation quote is a genuine verbatim substring of the source document."""
    verified: list[Citation] = []

    for cit in citations:
        cit_quote_norm = " ".join(cit.quote.lower().split())
        if not cit_quote_norm:
            continue
        found_in_source = False

        for item in passages:
            c = item.chunk
            if c.doc_name.lower() == cit.doc_name.lower() and c.page_number == cit.page:
                chunk_norm = " ".join(c.text.lower().split())
                if cit_quote_norm in chunk_norm:
                    found_in_source = True
                    break

        if found_in_source:
            verified.append(cit)
        else:
            logger.warning(
                "Discarding unverified hallucinated citation quote: '%s' in %s p.%d",
                cit.quote[:50],
                cit.doc_name,
                cit.page,
            )

    return verified


class AnswerGenerator:
    """Coordinates retrieval, prompt assembly, LLM execution, and citation verification."""

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        llm_client: BaseLLMClient,
        tracer: Tracer | None = None,
        model_name: str = "llama-3.1-8b-instant",
    ) -> None:
        self.retrieval_engine = retrieval_engine
        self.llm_client = llm_client
        self.tracer = tracer or get_tracer()
        model_attr = getattr(llm_client, "model", None)
        self.model_name = model_attr if isinstance(model_attr, str) else str(model_name)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> AnswerGenerator:
        """Instantiate AnswerGenerator from configuration dictionary."""
        cfg = config or load_config()
        gen_cfg = cfg.get("generation", {})

        engine = RetrievalEngine.from_config(cfg)
        provider = gen_cfg.get("provider", "groq")
        model = gen_cfg.get("model", "llama-3.1-8b-instant")
        client = get_llm_client(provider=provider, model=model, fallback_on_missing_key=True)

        return cls(retrieval_engine=engine, llm_client=client, model_name=model)

    def generate(
        self,
        query: str,
        use_reranker: bool = True,
        top_k: int | None = None,
    ) -> GroundedAnswer:
        """Generate a grounded answer with verified citations and automatic refusal."""
        answer, _ = self.generate_with_trace(
            query=query,
            use_reranker=use_reranker,
            top_k=top_k,
        )
        return answer

    def generate_with_trace(
        self,
        query: str,
        use_reranker: bool = True,
        top_k: int | None = None,
    ) -> tuple[GroundedAnswer, TraceRecord]:
        """Generate a grounded answer and return execution telemetry."""
        clean_query = query.strip()
        with self.tracer.trace(query=clean_query or "empty", model=self.model_name) as ctx:
            if not clean_query:
                ans = GroundedAnswer(
                    answer=STANDARD_REFUSAL,
                    citations=[],
                    refused=True,
                    confidence=0.0,
                )
                ctx.record_retrieval(latency_ms=0.0)
                ctx.record_generation(latency_ms=0.0, input_tokens=0, output_tokens=0, refused=True)
                return ans, ctx.build_record()

            # 1. Retrieve candidate passages with latency instrumentation
            t_retrieval_start = time.perf_counter()
            candidates = self.retrieval_engine.retrieve(
                query=clean_query,
                use_reranker=use_reranker,
                top_k=top_k,
            )
            retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000.0
            ctx.record_retrieval(latency_ms=retrieval_ms)

            # 2. Refuse if no chunks found
            if not candidates:
                ans = GroundedAnswer(
                    answer=STANDARD_REFUSAL,
                    citations=[],
                    refused=True,
                    confidence=0.0,
                )
                ctx.record_generation(latency_ms=0.0, input_tokens=0, output_tokens=0, refused=True)
                return ans, ctx.build_record()

            # 3. Refuse if top candidate relevance score is below refusal threshold
            max_score = max(c.score for c in candidates)
            if max_score < self.retrieval_engine.relevance_threshold:
                logger.info(
                    "Top retrieval score (%.4f) below threshold (%.4f); refusing",
                    max_score,
                    self.retrieval_engine.relevance_threshold,
                )
                ans = GroundedAnswer(
                    answer=STANDARD_REFUSAL,
                    citations=[],
                    refused=True,
                    confidence=0.0,
                )
                ctx.record_generation(latency_ms=0.0, input_tokens=0, output_tokens=0, refused=True)
                return ans, ctx.build_record()

            # 4. Format context and execute LLM with token & latency accounting
            context_str = format_context_passages(candidates)
            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(clean_query, context_str)
            in_tokens = estimate_tokens(system_prompt + "\n" + user_prompt)

            t_gen_start = time.perf_counter()
            try:
                raw_response = self.llm_client.generate(system_prompt, user_prompt)
                gen_ms = (time.perf_counter() - t_gen_start) * 1000.0
                out_tokens = estimate_tokens(raw_response)
                payload = _extract_json_payload(raw_response)
                grounded = GroundedAnswer.model_validate(payload)
            except Exception as err:
                gen_ms = (time.perf_counter() - t_gen_start) * 1000.0
                logger.error("Failed to generate or validate grounded answer: %s", err)
                ans = GroundedAnswer(
                    answer=STANDARD_REFUSAL,
                    citations=[],
                    refused=True,
                    confidence=0.0,
                )
                ctx.record_generation(
                    latency_ms=gen_ms,
                    input_tokens=in_tokens,
                    output_tokens=0,
                    refused=True,
                )
                return ans, ctx.build_record()

            # 5. If model decided to refuse, return clean refusal
            if grounded.refused:
                ans = GroundedAnswer(
                    answer=grounded.answer or STANDARD_REFUSAL,
                    citations=[],
                    refused=True,
                    confidence=0.0,
                )
                ctx.record_generation(
                    latency_ms=gen_ms,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    refused=True,
                )
                return ans, ctx.build_record()

            # 6. Verify verbatim quotes against retrieved source passages
            valid_citations = verify_citation_quotes(grounded.citations, candidates)

            # Unrefused answers must have at least one genuine verified citation
            if not valid_citations:
                logger.warning("Unrefused answer contains no valid citations. Enforcing refusal.")
                ans = GroundedAnswer(
                    answer=STANDARD_REFUSAL,
                    citations=[],
                    refused=True,
                    confidence=0.0,
                )
                ctx.record_generation(
                    latency_ms=gen_ms,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    refused=True,
                )
                return ans, ctx.build_record()

            # Adjust confidence if some citations were dropped
            final_confidence = grounded.confidence
            if len(valid_citations) < len(grounded.citations):
                drop_ratio = len(valid_citations) / len(grounded.citations)
                final_confidence = round(grounded.confidence * drop_ratio, 4)

            ans = GroundedAnswer(
                answer=grounded.answer,
                citations=valid_citations,
                refused=False,
                confidence=final_confidence,
            )
            ctx.record_generation(
                latency_ms=gen_ms,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                refused=False,
                num_citations=len(valid_citations),
            )
            return ans, ctx.build_record()
