"""Evaluation metrics and scoring utilities for retrieval systems."""

from __future__ import annotations

from typing import Any

from ragassistant.retrieval.engine import RetrievalEngine


def is_evidence_in_chunk(evidence_list: list[str], chunk_text: str) -> bool:
    """Check if any expected evidence snippet is found in the chunk text."""
    normalized_chunk = " ".join(chunk_text.lower().split())
    for ev in evidence_list:
        normalized_ev = " ".join(ev.lower().split())
        if normalized_ev and normalized_ev in normalized_chunk:
            return True
    return False


def evaluate_mode(
    engine: RetrievalEngine,
    samples: list[dict[str, Any]],
    use_reranker: bool,
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate retrieval performance in either baseline or reranked mode."""
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks: list[float] = []
    item_details: list[dict[str, Any]] = []

    for item in samples:
        qid = item["id"]
        question = item["question"]
        expected_ev = item.get("expected_evidence", [])

        results = engine.retrieve(query=question, use_reranker=use_reranker, top_k=top_k)

        first_hit_rank = 0
        hit_chunk_id = None

        for rank, res in enumerate(results, start=1):
            if is_evidence_in_chunk(expected_ev, res.chunk.text):
                first_hit_rank = rank
                hit_chunk_id = res.chunk.chunk_id
                break

        if first_hit_rank > 0:
            reciprocal_rank = 1.0 / first_hit_rank
            if first_hit_rank <= 3:
                hits_at_3 += 1
            if first_hit_rank <= 5:
                hits_at_5 += 1
        else:
            reciprocal_rank = 0.0

        reciprocal_ranks.append(reciprocal_rank)

        item_details.append(
            {
                "id": qid,
                "question": question,
                "first_hit_rank": first_hit_rank,
                "hit_chunk_id": hit_chunk_id,
                "reciprocal_rank": round(reciprocal_rank, 4),
                "top_retrieved": [
                    {
                        "chunk_id": r.chunk.chunk_id,
                        "doc_name": r.chunk.doc_name,
                        "page_number": r.chunk.page_number,
                        "score": round(r.score, 4),
                    }
                    for r in results
                ],
            }
        )

    n = len(samples)
    recall_3 = hits_at_3 / n if n > 0 else 0.0
    recall_5 = hits_at_5 / n if n > 0 else 0.0
    mrr = sum(reciprocal_ranks) / n if n > 0 else 0.0

    return {
        "recall@3": round(recall_3, 4),
        "recall@5": round(recall_5, 4),
        "mrr": round(mrr, 4),
        "hits_at_3": hits_at_3,
        "hits_at_5": hits_at_5,
        "total_eval_queries": n,
        "query_details": item_details,
    }


def evaluate_adversarial(
    engine: RetrievalEngine,
    adversarial_samples: list[dict[str, Any]],
    threshold: float = 0.35,
) -> dict[str, Any]:
    """Evaluate how adversarial queries score against relevance refusal threshold."""
    refused_count = 0
    details = []

    for item in adversarial_samples:
        qid = item["id"]
        question = item["question"]
        results = engine.retrieve(query=question, use_reranker=True, top_k=3)

        top_score = results[0].score if results else 0.0
        is_below_threshold = top_score < threshold
        if is_below_threshold:
            refused_count += 1

        details.append(
            {
                "id": qid,
                "question": question,
                "top_score": round(top_score, 4),
                "would_refuse": is_below_threshold,
            }
        )

    n = len(adversarial_samples)
    refusal_rate = refused_count / n if n > 0 else 0.0

    return {
        "refusal_rate": round(refusal_rate, 4),
        "refusal_threshold": threshold,
        "total_adversarial_queries": n,
        "query_details": details,
    }
