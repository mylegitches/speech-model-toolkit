from __future__ import annotations

from dataclasses import dataclass

from app.agents.memory import needs_drift_nudge, render_recent, safe_word_matches


@dataclass
class Row:
    speaker: str
    text: str


def test_safe_word_handles_case_and_punctuation() -> None:
    assert safe_word_matches("Okay, THAT'S A WRAP!", "that's a wrap")
    assert not safe_word_matches("the wrapper is here", "wrap")


def test_render_recent_is_bounded() -> None:
    rows = [Row("A", "one"), Row("B", "two"), Row("C", "three")]
    assert render_recent(rows, 2) == "B: two\nC: three"


def test_drift_nudge_detects_repeated_openings() -> None:
    rows = [Row("A", "I think we should try this") for _ in range(4)]
    assert needs_drift_nudge(rows)
