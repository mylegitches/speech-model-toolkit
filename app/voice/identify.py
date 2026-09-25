"""Name the people found in imported files with the active AI connection.

For each person (speakers.py) the AI gets a few of their transcribed lines and
the files they appear in, plus - when the file names carry an IMDb ID
("tt0141842") or sit in a show's folder ("The Sopranos/Season 1/...") - the
show's cast from TVmaze (free, no key; TVmaze links shows to their IMDb IDs).
It answers who each person is, with a confidence. Then:

  - every card shows the AI's answer, with "Use this name"
  - two cards the AI names as the same character are suggested for merging
  - with "autoName", untouched "Person N" cards get the name when the AI is sure
  - with "autoMerge", cards the AI is sure are one character are merged when
    their voices are also at least somewhat alike (so a wrong guess can't merge
    two clearly different voices)

Runs after each diarized file (when enabled in Settings) and on demand.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import numpy as np

from ..settings import providers
from ..settings import store as settings_store
from . import speakers
from .voices import Voice

_LOGGER = logging.getLogger(__name__)

MAX_PEOPLE = 60          # per request, those with the most speech first
LINES_PER_PERSON = 8
AUTO_MERGE_MIN_VOICE = 0.25
"""Voice similarity two cards need before the AI's "same character" merges them."""

TVMAZE = "https://api.tvmaze.com"
_IMDB_ID = re.compile(r"\btt\d{7,9}\b")
_DEFAULT_NAME = re.compile(r"^Person \d+$")
# Folder names that say nothing about the show (never search TVmaze for these)
_GENERIC_FOLDER = re.compile(
    r"^(season|series|staffel|saison|temporada|s)\s*\d+$|^(disc|disk|dvd|cd)\s*\d*$|^(extras?|specials?|"
    r"downloads?|videos?|movies?|films?|tv|tv shows?|shows?|media|imports?|clips?|new folder)$",
    re.IGNORECASE,
)

SYSTEM = (
    "You identify who is speaking in transcribed dialogue from TV series and films. "
    "Each numbered person is one voice that was grouped automatically, so a person may occasionally "
    "include a few lines from someone else. Use the cast list, the file names, the dialogue and your "
    "knowledge (and web search if you have it). Answer with JSON only, no other text."
)

_running: Set[str] = set()
_again: Set[str] = set()


def status(voice: Voice) -> Dict[str, Any]:
    conn = settings_store.active_connection()
    settings = settings_store.load()["identify"]
    return {
        "enabled": settings["enabled"],
        "connected": bool(conn),
        "provider": providers.get_provider(conn["provider"]).label if conn else None,
        "webSearch": providers.web_search_support(conn) if settings["webSearch"] else "",
        "running": voice.name in _running,
    }


# ---- Cast lookup (TVmaze) --------------------------------------------------------


def _show_hints(files: List[str]) -> Tuple[List[str], List[str]]:
    """IMDb IDs and show names (top-level folder) from file paths."""
    ids = list(dict.fromkeys(m for f in files for m in _IMDB_ID.findall(f)))
    names = []
    for f in files:
        # The first folder that looks like a show's name: "The Sopranos/Season 1/..."
        for folder in f.split("/")[:-1]:
            folder = folder.strip()
            if folder and not _GENERIC_FOLDER.match(folder):
                names.append(folder)
                break
    names = list(dict.fromkeys(names))
    return ids, names


async def cast_lookup(files: List[str]) -> Tuple[str, List[str]]:
    """("Show title", ["Character (Actor)", ...]) from TVmaze, or ("", [])."""
    ids, names = _show_hints(files)
    if not ids and not names:
        return "", []
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        show = None
        for imdb in ids[:3]:
            r = await client.get(f"{TVMAZE}/lookup/shows", params={"imdb": imdb})
            if r.status_code == 200:
                show = r.json()
                break
        for name in names[:3] if show is None else []:
            r = await client.get(f"{TVMAZE}/singlesearch/shows", params={"q": name})
            if r.status_code == 200:
                show = r.json()
                break
        if not show:
            return "", []
        cast: List[str] = []
        r = await client.get(f"{TVMAZE}/shows/{show['id']}/cast")
        if r.status_code == 200:
            for entry in r.json():
                cast.append(f"{entry['character']['name']} ({entry['person']['name']})")
        # Recurring and guest characters, from every episode
        r = await client.get(f"{TVMAZE}/shows/{show['id']}/episodes", params={"embed": "guestcast"})
        if r.status_code == 200:
            for episode in r.json():
                for entry in (episode.get("_embedded") or {}).get("guestcast", []):
                    cast.append(f"{entry['character']['name']} ({entry['person']['name']})")
        cast = list(dict.fromkeys(cast))[:400]
        _LOGGER.info("TVmaze: %s, %d characters", show.get("name"), len(cast))
        return f"{show.get('name')} ({(show.get('premiered') or '')[:4]})", cast


# ---- Asking the AI ------------------------------------------------------------------


def _prompt(people: List[Dict[str, Any]], named: List[Dict[str, Any]], files: List[str],
            show: str, cast: List[str]) -> str:
    parts = []
    if files:
        parts.append("Files these people were found in (up to 40):\n" + "\n".join(f"- {f}" for f in files[:40]))
    if cast:
        parts.append(f"Cast of {show} (from TVmaze, character (actor); found from the file names, "
                     "so ignore it if it clearly doesn't fit the dialogue):\n" + "; ".join(cast))
    if named:
        parts.append("Already named by the user, trust these:\n" + "\n".join(
            f'- {p["id"]} = {p["name"]}' for p in named))
    blocks = []
    for p in people:
        lines = "\n".join(
            f'  - "{line["text"]}"' + (f' ({line["file"]})' if line.get("file") else "")
            for line in p.get("lines", [])[-LINES_PER_PERSON:]
        )
        blocks.append(f'{p["id"]} ({round(p["seconds"] / 60, 1)} min of speech, in {len(p["takes"])} files):\n{lines}')
    parts.append("People to identify:\n\n" + "\n\n".join(blocks))
    parts.append(
        'For each person above, who is speaking? Reply with JSON only:\n'
        '{"people": [{"id": "p3", "name": "Character name", "actor": "Actor name or null", '
        '"confidence": "high|medium|low", "reason": "one short sentence"}]}\n'
        'Use name "mixed" if their lines clearly come from several characters, and null if you '
        'cannot tell. Give the same name to every person who is the same character.'
    )
    return "\n\n".join(parts)


def _parse(reply: str) -> List[Dict[str, Any]]:
    text = reply.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise providers.ProviderError("The AI's answer wasn't JSON. Try another model.")
    try:
        data = json.loads(text[start:end + 1])
    except ValueError as err:
        raise providers.ProviderError(f"The AI's answer wasn't valid JSON ({err}). Try another model.") from err
    return [p for p in data.get("people", []) if isinstance(p, dict) and p.get("id")]


def _set_status(voice: Voice, **values: Any) -> None:
    with speakers._lock:
        data = speakers.load(voice)
        data["ai"] = {**(data.get("ai") or {}), **values, "at": time.time()}
        speakers._save(voice, data)


async def identify(voice: Voice, everyone: bool = False) -> Dict[str, Any]:
    """Ask the AI about the people not identified yet (or everyone)."""
    settings = settings_store.load()
    options = settings["identify"]
    conn = settings_store.active_connection(settings)
    if conn is None:
        raise RuntimeError("Add an AI connection in Settings → AI connections first")

    data = speakers.load(voice)
    candidates = [
        p for p in data["people"]
        if p.get("lines") and (everyone or not p.get("ai") or p["ai"].get("lines") != len(p["lines"]))
    ]
    candidates = sorted(candidates, key=lambda p: -p["seconds"])[:MAX_PEOPLE]
    if not candidates:
        _set_status(voice, state="done", error=None, identified=0)
        return speakers.public(voice)

    files = list(dict.fromkeys(line["file"] for p in candidates for line in p.get("lines", []) if line.get("file")))
    show, cast = "", []
    if options["castLookup"]:
        try:
            show, cast = await cast_lookup(files)
        except (httpx.HTTPError, ValueError, KeyError) as err:
            _LOGGER.warning("TVmaze cast lookup failed: %s", err)
    named = [p for p in data["people"] if not _DEFAULT_NAME.match(p["name"])]

    _set_status(voice, state="running", error=None, cast=show or None)
    started = time.monotonic()
    reply = await providers.chat(
        conn,
        [{"role": "user", "content": _prompt(candidates, named, files, show, cast)}],
        system=SYSTEM,
        temperature=0.2,
        max_tokens=min(8000, 400 + 120 * len(candidates)),
        web_search=options["webSearch"],
    )
    answers = {a["id"]: a for a in _parse(reply)}

    renamed = 0
    with speakers._lock:
        data = speakers.load(voice)
        for person in data["people"]:
            answer = answers.get(person["id"])
            if not answer:
                continue
            name = (answer.get("name") or "").strip() or None
            person["ai"] = {
                "name": name,
                "actor": (answer.get("actor") or "").strip() or None,
                "confidence": answer.get("confidence") if answer.get("confidence") in ("high", "medium", "low") else "low",
                "reason": str(answer.get("reason") or "")[:300],
                "lines": len(person.get("lines", [])),
                "at": time.time(),
            }
            if (options["autoName"] and name and name.lower() not in ("mixed", "unknown")
                    and person["ai"]["confidence"] == "high" and _DEFAULT_NAME.match(person["name"])):
                person["name"] = name[:40]
                renamed += 1
        data["ai"] = {"state": "done", "error": None, "at": time.time(), "cast": show or None,
                      "identified": len(answers)}
        speakers._save(voice, data)

    merged = _auto_merge(voice) if options.get("autoMerge") else 0
    _LOGGER.info("AI identification for %s: %d of %d people answered, %d renamed, %d merged, in %.0fs%s",
                 voice.name, len(answers), len(candidates), renamed, merged, time.monotonic() - started,
                 f" (cast: {show})" if show else "")
    return speakers.public(voice)


def _auto_merge(voice: Voice) -> int:
    """Merge cards the AI is sure are the same character, if their voices are alike enough."""
    from . import freeform  # imports speakers too

    merged = 0
    while True:
        data = speakers.load(voice)
        dismissed = {frozenset(pair) for pair in data["notSame"]}
        by_character: Dict[str, List[Dict[str, Any]]] = {}
        for person in data["people"]:
            ai = person.get("ai") or {}
            if ai.get("confidence") == "high" and speakers.ai_character(person):
                by_character.setdefault(speakers.ai_character(person), []).append(person)
        pair = None
        for group in by_character.values():
            group.sort(key=lambda p: -p["seconds"])
            main = group[0]
            for other in group[1:]:
                score = float(speakers._unit(main["centroid"]) @ speakers._unit(other["centroid"]))
                if score >= AUTO_MERGE_MIN_VOICE and frozenset((main["id"], other["id"])) not in dismissed:
                    pair = (other, main)
                    break
            if pair:
                break
        if not pair:
            return merged
        other, main = pair
        speakers.merge(voice, other["id"], main["id"],
                       lambda take_id, old, new: freeform.retag_person(voice, take_id, old, new))
        merged += 1


# ---- Running in the background ------------------------------------------------------


def schedule(voice: Voice, everyone: bool = False) -> None:
    """Identify in the background; if one is already running, run once more after it."""
    if voice.name in _running:
        _again.add(voice.name)
        return
    _running.add(voice.name)

    async def run() -> None:
        full = everyone
        try:
            again = True
            while again:
                _again.discard(voice.name)
                try:
                    await identify(voice, full)
                except Exception as err:  # shown in the People panel
                    if isinstance(err, (providers.ProviderError, RuntimeError)):
                        _LOGGER.warning("AI identification for %s failed: %s", voice.name, err)
                    else:
                        _LOGGER.exception("AI identification for %s failed", voice.name)
                    _set_status(voice, state="error", error=str(err))
                    break
                again = voice.name in _again
                full = False
        finally:
            _running.discard(voice.name)

    asyncio.create_task(run())


def after_take(voice: Voice) -> None:
    """Called when a diarized file is done: identify new people if enabled."""
    settings = settings_store.load()
    if settings["identify"]["enabled"] and settings_store.active_connection(settings):
        schedule(voice)
