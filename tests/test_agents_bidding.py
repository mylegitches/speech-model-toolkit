from __future__ import annotations

import random

from app.agents.bidding import BidCandidate, choose_speaker, score_bid, was_directly_addressed


def test_direct_address_is_case_insensitive_and_word_bounded() -> None:
    assert was_directly_addressed("HAL", "What do you think, hal?")
    assert not was_directly_addressed("Al", "That is always true")


def test_score_applies_eagerness_and_recency() -> None:
    eager = BidCandidate(1, "A", 5, 1.0, 8)
    recent = BidCandidate(2, "B", 5, 0.0, 1)
    assert score_bid(eager) > score_bid(recent)


def test_choose_speaker_is_deterministic_with_seed() -> None:
    candidates = [
        BidCandidate(1, "A", 3, 0.2, 5),
        BidCandidate(2, "B", 8, 0.2, 5),
    ]
    assert choose_speaker(candidates, rng=random.Random(5)).character_id == 2
