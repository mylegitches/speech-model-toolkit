from __future__ import annotations

import re
from collections import Counter
from typing import Protocol


class TranscriptLine(Protocol):
    speaker: str
    text: str


def render_recent(lines: list[TranscriptLine], limit: int = 30) -> str:
    return "\n".join(f"{line.speaker}: {line.text}" for line in lines[-limit:])


def needs_drift_nudge(lines: list[TranscriptLine], window: int = 6) -> bool:
    recent = lines[-window:]
    if len(recent) < 4:
        return False
    openings: list[str] = []
    for line in recent:
        words = re.findall(r"[a-z']+", line.text.casefold())[:4]
        if words:
            openings.append(" ".join(words))
    counts = Counter(openings)
    return bool(counts and counts.most_common(1)[0][1] >= 3)


def safe_word_matches(text: str, safe_word: str | None) -> bool:
    if not safe_word or not safe_word.strip():
        return False
    normalized_text = " ".join(re.findall(r"[\w']+", text.casefold()))
    normalized_safe = " ".join(re.findall(r"[\w']+", safe_word.casefold()))
    return bool(normalized_safe) and bool(
        re.search(rf"(?:^|\s){re.escape(normalized_safe)}(?:$|\s)", normalized_text)
    )
