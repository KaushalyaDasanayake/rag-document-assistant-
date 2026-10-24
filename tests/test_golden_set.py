"""Tests for Milestone 0: evaluation dataset integrity and project contracts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "eval" / "golden_qa.jsonl"
SAMPLE_DOCS_DIR = ROOT / "data" / "sample_docs"


def test_golden_qa_file_exists():
    assert GOLDEN_PATH.exists(), f"Golden QA file missing at {GOLDEN_PATH}"


def test_sample_docs_exist():
    assert SAMPLE_DOCS_DIR.exists()
    docs = list(SAMPLE_DOCS_DIR.glob("*.md")) + list(SAMPLE_DOCS_DIR.glob("*.pdf"))
    assert len(docs) >= 3, f"Expected at least 3 sample documents, found {len(docs)}"


def test_golden_qa_composition():
    """Verify golden dataset contains required 25 items with balanced query types."""
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 25, f"Expected 25 golden items, got {len(records)}"

    straightforward = [r for r in records if r["type"] == "straightforward"]
    multihop = [r for r in records if r["type"] == "multihop"]
    adversarial = [r for r in records if r["type"] == "adversarial"]

    assert len(straightforward) == 15, f"Expected 15 straightforward, got {len(straightforward)}"
    assert len(multihop) == 5, f"Expected 5 multihop, got {len(multihop)}"
    assert len(adversarial) == 5, f"Expected 5 adversarial refusals, got {len(adversarial)}"


def test_golden_qa_schema():
    """Validate contract fields on all golden QA rows."""
    required_keys = {
        "id",
        "type",
        "question",
        "expected_doc",
        "expected_evidence",
        "ground_truth_answer",
        "should_refuse",
    }

    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            record = json.loads(line)
            assert required_keys.issubset(
                record.keys()
            ), f"Row {idx} missing keys: {required_keys - record.keys()}"
            assert isinstance(record["question"], str) and record["question"].strip()
            assert isinstance(record["ground_truth_answer"], str)
            assert isinstance(record["should_refuse"], bool)

            if record["should_refuse"]:
                assert record["type"] == "adversarial"
                assert len(record["expected_evidence"]) == 0
            else:
                assert len(record["expected_evidence"]) > 0
