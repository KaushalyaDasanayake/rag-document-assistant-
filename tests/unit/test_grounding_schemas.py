"""Unit tests for GroundedAnswer and Citation schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragassistant.schemas.grounding import STANDARD_REFUSAL, Citation, GroundedAnswer


def test_valid_grounded_answer():
    citation = Citation(
        doc_name="employee_handbook.md",
        page=2,
        quote="10:00 AM to 3:00 PM in the employee's local time zone",
    )
    answer = GroundedAnswer(
        answer="Standard core working hours are 10:00 AM to 3:00 PM.",
        citations=[citation],
        refused=False,
        confidence=0.98,
    )

    assert answer.refused is False
    assert len(answer.citations) == 1
    assert answer.citations[0].page == 2
    assert answer.confidence == 0.98


def test_valid_refusal_answer():
    answer = GroundedAnswer(
        answer=STANDARD_REFUSAL,
        citations=[],
        refused=True,
        confidence=0.0,
    )
    assert answer.refused is True
    assert answer.citations == []
    assert answer.confidence == 0.0


def test_confidence_boundary_validation():
    with pytest.raises(ValidationError):
        GroundedAnswer(
            answer="Test",
            citations=[],
            refused=False,
            confidence=1.5,  # > 1.0 invalid
        )

    with pytest.raises(ValidationError):
        GroundedAnswer(
            answer="Test",
            citations=[],
            refused=False,
            confidence=-0.1,  # < 0.0 invalid
        )


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        Citation(
            doc_name="doc.md",
            page=1,
            quote="Quote",
            unknown_attr=123,
        )
