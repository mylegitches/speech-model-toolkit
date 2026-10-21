"""Pauses in spoken text: punctuation pauses you can set, and <break time="1s"/> tags.

The text becomes an ordered list of events, ("speech", text) and ("pause", ms). Piper
speaks each piece of speech on its own and the silences go in between, so it works with
any Piper voice without touching the model. That also means it only applies where this
app speaks (Test Lab, Voices → Speak): Hermes or Home Assistant run Piper themselves.

  - Break tags always work: <break time="1s"/>, <break time="250ms"/>, <break time="2.5s" />.
    They're never spoken. A malformed tag stays text (only the angle brackets go).
  - Punctuation pauses (only when enabled) are extra silence on top of Piper's own:
    after . ? ! … (and ...), and at paragraph breaks. Runs like ?! or !!! give one pause,
    the longest of them. Decimals (3.14), abbreviations (Dr. e.g.) and URL-like text
    don't count as a sentence end.
  - Commas, semicolons and colons (experimental, 0 = off): the sentence is split there, so
    Piper speaks each part as its own sentence and the voice drops at the split.
"""

import re
from typing import Any, Dict, List, Tuple

Event = Tuple[str, Any]  # ("speech", str) or ("pause", int ms)

# (key, label, default ms, slider max ms)
MARKS = [
    ("period", "Period  .", 350, 2000),
    ("question", "Question mark  ?", 450, 2000),
    ("exclamation", "Exclamation mark  !", 400, 2000),
    ("ellipsis", "Ellipsis  …  or  ...", 950, 3000),
    ("paragraph", "Paragraph break (empty line)", 650, 3000),
    ("comma", "Comma  ,  (experimental)", 0, 1000),
    ("semicolon", "Semicolon  ;  (experimental)", 0, 1500),
    ("colon", "Colon  :  (experimental)", 0, 1500),
]
DEFAULTS: Dict[str, Any] = {"enabled": False, **{key: ms for key, _label, ms, _max in MARKS}}
MAX_PAUSE_MS = 10000      # one pause, tag or punctuation
MAX_TOTAL_PAUSE_MS = 60000  # all inserted silence in one request
MAX_TAGS = 100

_BREAK = re.compile(r"""<\s*break\s+time\s*=\s*["']?\s*(\d+(?:\.\d+)?)\s*(ms|s)?\s*["']?\s*/?\s*>""", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^<>]{0,200}>")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "vs", "etc", "e.g", "i.e", "mt", "no", "fig", "inc", "ltd", "co"}
# A punctuation run that ends a piece: followed by whitespace, a closing quote/bracket or the end
_MARK_RUN = re.compile(r"(\.{3,}|…|[.?!]+|[,;:])(?=[\s\"'”’)\]]|$)")


def clean_settings(values: Dict[str, Any]) -> Dict[str, Any]:
    """Settings from the page, clamped to sensible values."""
    out = dict(DEFAULTS)
    for key, _label, _ms, top in MARKS:
        try:
            out[key] = max(0, min(top, int(float(values.get(key, out[key])))))
        except (TypeError, ValueError):
            pass
    out["enabled"] = bool(values.get("enabled", out["enabled"]))
    return out


def _mark_kind(run: str) -> List[str]:
    if run.startswith("..") or run == "…":
        return ["ellipsis"]
    kinds = []
    if "?" in run:
        kinds.append("question")
    if "!" in run:
        kinds.append("exclamation")
    if not kinds:
        kinds = {".": ["period"], ",": ["comma"], ";": ["semicolon"], ":": ["colon"]}.get(run[0], [])
    return kinds


def _is_abbreviation(text: str, end: int) -> bool:
    """The word just before a full stop at `end` is an abbreviation (Dr. e.g.)."""
    word = re.search(r"([A-Za-z.]+)$", text[:end])
    return bool(word) and word.group(1).lower().rstrip(".") in _ABBREVIATIONS


def _split_speech(text: str, cfg: Dict[str, Any]) -> List[Event]:
    """Punctuation pauses inside a stretch of text with no tags."""
    events: List[Event] = []
    paragraphs = re.split(r"\n\s*\n", text)
    for p_index, paragraph in enumerate(paragraphs):
        if p_index and cfg["paragraph"] > 0:
            events.append(("pause", cfg["paragraph"]))
        start = 0
        for m in _MARK_RUN.finditer(paragraph):
            kinds = _mark_kind(m.group(1))
            ms = max((cfg[k] for k in kinds), default=0)
            if ms <= 0:
                continue
            if m.group(1) == "." and _is_abbreviation(paragraph, m.start()):
                continue
            piece = paragraph[start:m.end()]
            if re.search(r"\w", piece):
                events.append(("speech", " ".join(piece.split())))
                events.append(("pause", ms))
                start = m.end()
        rest = paragraph[start:]
        if re.search(r"\w", rest):
            events.append(("speech", " ".join(rest.split())))
        elif rest.strip() and events and events[-1][0] == "speech":
            events[-1] = ("speech", events[-1][1] + " " + rest.strip())
    return events


def parse(text: str, settings: Dict[str, Any]) -> List[Event]:
    """Text with break tags (and, when enabled, punctuation pauses) as speech and pause events."""
    cfg = clean_settings(settings or {})
    events: List[Event] = []
    pos = 0
    tags = 0
    for m in _BREAK.finditer(text):
        tags += 1
        if tags > MAX_TAGS:
            break
        events += _speech_events(text[pos:m.start()], cfg)
        value = float(m.group(1))
        ms = value if (m.group(2) or "s").lower() == "ms" else value * 1000
        events.append(("pause", int(max(0, min(MAX_PAUSE_MS, ms)))))
        pos = m.end()
    events += _speech_events(text[pos:], cfg)
    return _tidy(events)


def _speech_events(text: str, cfg: Dict[str, Any]) -> List[Event]:
    text = _ANY_TAG.sub(lambda m: m.group(0)[1:-1], text)  # other <...>: their words stay, never markup
    if not re.search(r"\w", text):
        return []
    if not cfg["enabled"]:
        return [("speech", " ".join(text.split()))]
    return _split_speech(text, cfg)


def _tidy(events: List[Event]) -> List[Event]:
    """Adjacent automatic/explicit pauses add up (an explicit break adds its own silence),
    no pause at the very start or end, total silence capped."""
    out: List[Event] = []
    total = 0
    for kind, value in events:
        if kind == "pause":
            value = int(max(0, min(MAX_PAUSE_MS, value, MAX_TOTAL_PAUSE_MS - total)))
            if value <= 0:
                continue
            total += value
            if out and out[-1][0] == "pause":
                out[-1] = ("pause", out[-1][1] + value)
                continue
        if kind == "speech" and not value:
            continue
        out.append((kind, value))
    while out and out[-1][0] == "pause":
        out.pop()
    while out and out[0][0] == "pause":
        out.pop(0)
    return out
