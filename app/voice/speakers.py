"""People found across a voice's diarized takes, recognised from file to file.

voices/<name>/speakers.json:
  {"people": [{"id": "p1", "name": "Person 1", "centroid": [...], "seconds": 123.4,
               "takes": ["<take id>", ...], "sample": {"take": id, "index": n} | null}],
   "target": "p1" | null}   # "This is the voice": preselected in every take

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
    return data


def _save(voice: Voice, data: Dict[str, Any]) -> None:
    tmp = _path(voice).with_name("speakers.json.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(_path(voice))


def public(voice: Voice) -> Dict[str, Any]:
    """The people without their fingerprints (what the page needs)."""
    data = load(voice)
    return {
        "target": data["target"],
        "people": [{k: v for k, v in p.items() if k != "centroid"} for p in data["people"]],
    }


def _unit(vector: List[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


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
        data["people"] = [p for p in data["people"] if p["id"] != person_id]
        if data["target"] == person_id:
            data["target"] = into_id
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
