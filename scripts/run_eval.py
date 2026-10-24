#!/usr/bin/env python3
"""Evaluation script for Milestone 2 retrieval metrics (Recall@3, Recall@5, MRR)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ragassistant.config import load_config
from ragassistant.ingestion.pipeline import ingest_directory
from ragassistant.retrieval.engine import RetrievalEngine
from ragassistant.retrieval.evaluation import (
    evaluate_adversarial,
    evaluate_mode,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG Retrieval Engine against golden QA dataset.")
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=Path("eval/golden_qa.jsonl"),
        help="Path to frozen golden QA dataset",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("eval/metrics.json"),
        help="Path to save evaluation metrics JSON",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindexing sample docs before evaluation",
    )
    args = parser.parse_args()

    cfg = load_config()
    engine = RetrievalEngine.from_config(cfg)

    # Ingest docs if collection is empty or reindex is requested
    docs_dir = Path(cfg.get("storage", {}).get("sample_docs_dir", "data/sample_docs"))
    if args.reindex or engine.vector_store.count() == 0:
        logger.info("Indexing sample documents from %s...", docs_dir)
        engine.vector_store.clear()
        stats = ingest_directory(docs_dir, engine.vector_store)
        logger.info("Indexed %d documents, total %d chunks.", len(stats), engine.vector_store.count())

    logger.info("Loading golden dataset from %s...", args.golden_path)
    all_data = [
        json.loads(line)
        for line in args.golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    factual_samples = [s for s in all_data if s.get("expected_evidence")]
    adversarial_samples = [s for s in all_data if s.get("should_refuse")]

    logger.info("Evaluating %d factual questions and %d adversarial questions...", len(factual_samples), len(adversarial_samples))

    logger.info("Running Baseline (Bi-Encoder similarity search)...")
    baseline_metrics = evaluate_mode(engine, factual_samples, use_reranker=False, top_k=5)

    logger.info("Running Reranked (Bi-Encoder + CrossEncoder)...")
    reranked_metrics = evaluate_mode(engine, factual_samples, use_reranker=True, top_k=5)

    adversarial_metrics = evaluate_adversarial(
        engine,
        adversarial_samples,
        threshold=engine.relevance_threshold,
    )

    metrics_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "embedding_model": engine.vector_store.embedding_manager.model_name if engine.vector_store.embedding_manager else "unknown",
        "reranker_model": engine.reranker.model_name if engine.reranker else "none",
        "total_eval_samples": len(factual_samples),
        "baseline_biencoder": {
            "recall@3": baseline_metrics["recall@3"],
            "recall@5": baseline_metrics["recall@5"],
            "mrr": baseline_metrics["mrr"],
        },
        "reranked_crossencoder": {
            "recall@3": reranked_metrics["recall@3"],
            "recall@5": reranked_metrics["recall@5"],
            "mrr": reranked_metrics["mrr"],
        },
        "adversarial_guard": {
            "refusal_rate": adversarial_metrics["refusal_rate"],
            "threshold": adversarial_metrics["refusal_threshold"],
        },
        "details": {
            "baseline": baseline_metrics["query_details"],
            "reranked": reranked_metrics["query_details"],
            "adversarial": adversarial_metrics["query_details"],
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.info("Saved metrics to %s", args.output_path)

    # Print summary table
    print("\n" + "=" * 62)
    print("      MILESTONE 2: RETRIEVAL & RERANKING EVALUATION RESULTS")
    print("=" * 62)
    print(f"{'Metric':<25} | {'Baseline (Bi-Encoder)':<15} | {'Reranked (Cross-Encoder)':<15}")
    print("-" * 62)
    print(f"{'Recall@3':<25} | {baseline_metrics['recall@3']:<15.2%} | {reranked_metrics['recall@3']:<15.2%}")
    print(f"{'Recall@5':<25} | {baseline_metrics['recall@5']:<15.2%} | {reranked_metrics['recall@5']:<15.2%}")
    print(f"{'MRR (Mean Reciprocal Rank)':<25} | {baseline_metrics['mrr']:<15.4f} | {reranked_metrics['mrr']:<15.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
