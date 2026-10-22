from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class BidCandidate:
    character_id: int
    name: str
    raw_bid: float
    eagerness: float
    turns_since_spoke: int
    directly_addressed: bool = False
    score: float = 0.0


def was_directly_addressed(name: str, latest_text: str) -> bool:
    return bool(re.search(rf"\b{re.escape(name)}\b", latest_text, flags=re.IGNORECASE))


def score_bid(
    candidate: BidCandidate,
    *,
    eagerness_weight: float = 2.0,
    direct_address_bonus: float = 3.0,
    jitter: float = 0.0,
) -> float:
    recency_penalty = max(0.0, 2.5 - candidate.turns_since_spoke * 0.75)
    candidate.score = (
        max(0.0, min(10.0, candidate.raw_bid))
        + max(0.0, min(1.0, candidate.eagerness)) * eagerness_weight
        + max(-0.5, min(0.5, jitter))
        - recency_penalty
        + (direct_address_bonus if candidate.directly_addressed else 0.0)
    )
    return candidate.score


def choose_speaker(
    candidates: Iterable[BidCandidate],
    *,
    eagerness_weight: float = 2.0,
    direct_address_bonus: float = 3.0,
    rng: random.Random | None = None,
) -> BidCandidate:
    rng = rng or random.Random()
    scored = list(candidates)
    if not scored:
        raise ValueError("No eligible speakers")
    for candidate in scored:
        score_bid(
            candidate,
            eagerness_weight=eagerness_weight,
            direct_address_bonus=direct_address_bonus,
            jitter=rng.uniform(-0.35, 0.35),
        )
    return max(scored, key=lambda item: item.score)
