"""People found across a voice's diarized takes, recognised from file to file.

voices/<name>/speakers.json:
  {"people": [{"id": "p1", "name": "Person 1", "centroid": [...], "seconds": 123.4,
               "takes": ["<take id>", ...], "sample": {"take": id, "index": n} | null,
               "lines": [{"text", "file"}, ...],   # a few of their lines, for AI naming
               "ai": {"name", "actor", "confidence", "reason", "lines", "at"} | absent}],
   "ai": {"state", "error", "at", "identified", "cast"},   # last AI identification run
   "target": "p1" | null,   # "This is the voice": preselected in every take
   "notSame": [["p1", "p4"], ...]}   # suggestions the user said are different people

Each take is diarized on its own (diarize.py). Afterwards every speaker in the
take is compared with the people already known, using the mean ECAPA voice
fingerprint: a close match is the same person (and refines their fingerprint),
anything else is a new person. So a series of videos with recurring people gets
the same names on the same people in every file.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .voices import Voice

_LOGGER = logging.getLogger(__name__)

SAME_PERSON = 0.5
"""Fingerprint similarity above which a take's speaker is a known person.

Fingerprints of one person from different recordings typically score 0.6-0.9,
different people below 0.35."""

MAX_LINES = 16
"""Lines kept per person (the longest from each file) for identification."""

MAYBE_SAME = 0.4
"""Similarity above which two people are suggested as possibly the same.

Diarization within one file can split a person (shouting vs. whispering, a
phone call); those halves never merge automatically because a file's speakers
are kept apart, so similar pairs are offered to the user instead."""

_lock = threading.Lock()


def _path(voice: Voice) -> Path:
    return voice.root / "speakers.json"


def load(voice: Voice) -> Dict[str, Any]:
    try:
        data = json.loads(_path(voice).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.setdefault("people", [])
    data.setdefault("target", None)
    data.setdefault("notSame", [])
    return data


def _save(voice: Voice, data: Dict[str, Any]) -> None:
    tmp = _path(voice).with_name("speakers.json.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(_path(voice))


def _suggestions(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """For each person, others who might be the same person (best first)."""
    people = [p for p in data["people"] if p.get("centroid")]
    dismissed = {frozenset(pair) for pair in data["notSame"]}
    out: Dict[str, List[Dict[str, Any]]] = {p["id"]: [] for p in data["people"]}
    if len(people) < 2:
        return out
    matrix = np.stack([_unit(p["centroid"]) for p in people])
    scores = matrix @ matrix.T
    for i, a in enumerate(people):
        for j, b in enumerate(people):
            if i == j or frozenset((a["id"], b["id"])) in dismissed:
                continue
            same_character = ai_character(a) and ai_character(a) == ai_character(b)
            if scores[i, j] >= MAYBE_SAME or same_character:
                hint = {"id": b["id"], "score": round(float(scores[i, j]), 3)}
                if same_character:
                    hint["reason"] = f"AI thinks both are {a['ai']['name']}"
                out[a["id"]].append(hint)
        # AI-backed suggestions first, then by voice similarity
        out[a["id"]].sort(key=lambda s: (-("reason" in s), -s["score"]))
    return out


def ai_character(person: Dict[str, Any]) -> str:
    """The character the AI named with at least medium confidence (normalised), or ""."""
    ai = person.get("ai") or {}
    name = (ai.get("name") or "").strip().lower()
    if not name or name in ("mixed", "unknown") or ai.get("confidence") not in ("high", "medium"):
        return ""
    return " ".join(name.split())


def public(voice: Voice) -> Dict[str, Any]:
    """The people without their fingerprints (what the page needs)."""
    data = load(voice)
    similar = _suggestions(data)
    return {
        "target": data["target"],
        "ai": data.get("ai"),
        "people": [
            {**{k: v for k, v in p.items() if k != "centroid"}, "similar": similar.get(p["id"], [])}
            for p in data["people"]
        ],
    }


def not_same(voice: Voice, person_id: str, other_id: str) -> Dict[str, Any]:
    """Stop suggesting that two people might be the same."""
    with _lock:
        data = load(voice)
        ids = {p["id"] for p in data["people"]}
        if person_id not in ids or other_id not in ids:
            raise KeyError("No such person")
        pair = sorted([person_id, other_id])
        if pair not in data["notSame"]:
            data["notSame"].append(pair)
        _save(voice, data)
    return public(voice)


def _unit(vector: List[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _add_lines(person: Dict[str, Any], take: Dict[str, Any], speaker_id: int) -> None:
    """Remember the speaker's longest lines in this take (newest files last)."""
    texts = sorted(
        {s["text"].strip() for s in take.get("segments", []) if s.get("speaker") == speaker_id and s.get("text")},
        key=len, reverse=True,
    )[:4]
    name = take.get("name") or ""
    lines = person.setdefault("lines", [])
    lines.extend({"text": t[:240], "file": name} for t in texts)
    del lines[:-MAX_LINES]


def link(voice: Voice, take_id: str, take: Dict[str, Any]) -> None:
    """Match the take's speakers to known people (adds "person" to each speaker)."""
    speakers = take.get("speakers") or []
    if not speakers:
        return
    with _lock:
        data = load(voice)
        people = data["people"]
        known = [_unit(p["centroid"]) for p in people]
        # Best pairs first; each person at most once per take
        pairs = sorted(
            ((float(_unit(s["centroid"]) @ k), si, pi)
             for si, s in enumerate(speakers) if s.get("centroid")
             for pi, k in enumerate(known)),
            reverse=True,
        )
        taken_speakers, taken_people = set(), set()
        for score, si, pi in pairs:
            if score < SAME_PERSON or si in taken_speakers or pi in taken_people:
                continue
            taken_speakers.add(si)
            taken_people.add(pi)
            person, speaker = people[pi], speakers[si]
            # Refine the fingerprint, weighted by how much each has spoken
            w_old, w_new = person["seconds"], speaker["seconds"]
            mixed = _unit(person["centroid"]) * w_old + _unit(speaker["centroid"]) * w_new
            person["centroid"] = [round(float(v), 5) for v in _unit(mixed.tolist())]
            person["seconds"] = round(w_old + w_new, 1)
            if take_id not in person["takes"]:
                person["takes"].append(take_id)
            if not person.get("sample") and speaker["samples"]:
                person["sample"] = {"take": take_id, "index": speaker["samples"][0]}
            _add_lines(person, take, speaker["id"])
            speaker.update(person=person["id"], personScore=round(score, 3), newPerson=False)

        for si, speaker in enumerate(speakers):
            if si in taken_speakers or not speaker.get("centroid"):
                continue
            number = max([int(p["id"][1:]) for p in people] + [0]) + 1
            person = {
                "id": f"p{number}", "name": f"Person {number}",
                "centroid": speaker["centroid"], "seconds": speaker["seconds"],
                "takes": [take_id],
                "sample": {"take": take_id, "index": speaker["samples"][0]} if speaker["samples"] else None,
            }
            _add_lines(person, take, speaker["id"])
            people.append(person)
            speaker.update(person=person["id"], personScore=None, newPerson=True)

        _save(voice, data)
    matched = []
    for s in speakers:
        if s.get("person"):
            how = "new" if s["newPerson"] else f"{s['personScore']:.2f}"
            matched.append(f"{s['person']} ({how})")
    _LOGGER.info("Take %s/%s: speakers matched to people: %s", voice.name, take_id, ", ".join(matched))


def update(voice: Voice, person_id: str, name: Optional[str] = None, target: Optional[bool] = None) -> Dict[str, Any]:
    with _lock:
        data = load(voice)
        person = next((p for p in data["people"] if p["id"] == person_id), None)
        if person is None:
            raise KeyError(f"No person {person_id}")
        if name is not None:
            name = " ".join(str(name).split())[:40]
            if not name:
                raise ValueError("Give the person a name")
            person["name"] = name
        if target is not None:
            data["target"] = person_id if target else (None if data["target"] == person_id else data["target"])
        _save(voice, data)
    return public(voice)


def merge(voice: Voice, person_id: str, into_id: str, retag: Any) -> Dict[str, Any]:
    """Merge one person into another (the same person was found twice).

    retag(take_id, old_person_id, new_person_id) rewrites the takes."""
    if person_id == into_id:
        raise ValueError("Pick a different person to merge with")
    with _lock:
        data = load(voice)
        source = next((p for p in data["people"] if p["id"] == person_id), None)
        target = next((p for p in data["people"] if p["id"] == into_id), None)
        if source is None or target is None:
            raise KeyError("No such person")
        mixed = _unit(source["centroid"]) * source["seconds"] + _unit(target["centroid"]) * target["seconds"]
        target["centroid"] = [round(float(v), 5) for v in _unit(mixed.tolist())]
        target["seconds"] = round(source["seconds"] + target["seconds"], 1)
        target["takes"] = list(dict.fromkeys(target["takes"] + source["takes"]))
        target["sample"] = target.get("sample") or source.get("sample")
        target["lines"] = (target.get("lines", []) + source.get("lines", []))[-MAX_LINES:]
        if not target.get("ai") and source.get("ai"):
            target["ai"] = source["ai"]
        data["people"] = [p for p in data["people"] if p["id"] != person_id]
        if data["target"] == person_id:
            data["target"] = into_id
        # Pairs involving the merged person now apply to the one it went into
        pairs = {tuple(sorted(into_id if x == person_id else x for x in pair)) for pair in data["notSame"]}
        data["notSame"] = [list(p) for p in pairs if p[0] != p[1]]
        _save(voice, data)
    for take_id in source["takes"]:
        retag(take_id, person_id, into_id)
    _LOGGER.info("Voice %s: merged %s into %s", voice.name, person_id, into_id)
    return public(voice)


def forget_take(voice: Voice, take_id: str) -> None:
    """A take was saved or discarded: its samples can't be played any more."""
    with _lock:
        data = load(voice)
        changed = False
        for person in data["people"]:
            sample = person.get("sample")
            if sample and sample.get("take") == take_id:
                person["sample"] = None
                changed = True
        if changed:
            _save(voice, data)
