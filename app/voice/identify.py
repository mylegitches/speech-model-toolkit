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

MAX_PEOPLE = 600         # per run, those with the most speech first
BATCH = 25               # people per AI request
REQUEST_TIMEOUT = 300    # seconds; writing answers for a whole group takes a while
LINES_PER_PERSON = 8
MAX_TOKENS = 8000        # reply budget (reasoning + search + the JSON); most models allow 8k
RETRY_MAX_TOKENS = 16000  # once more with this when the model ran out before answering
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
    "include a few lines from someone else. Use the cast list, the file names and metadata, the "
    "dialogue (names people call each other) and your knowledge. Answer with JSON only, no other text."
)
SEARCH_HINT = (
    "You can search the web: search a few of the most distinctive lines word for word, in quotes "
    "(episode transcripts, subtitle and quote sites), to find which show and episode they are from "
    "and which character says them, and check it against the file names, metadata, timestamps and cast. "
    "The lines are machine transcriptions, so a word may be slightly off: if an exact search finds "
    "nothing, try a shorter part of the line."
)
_NAME_WORD = re.compile(r"(?<=[a-z,] )[A-Z][a-z]+")

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


_EPISODE_NAME = re.compile(r"^(.+?)[\s._-]*(?:s\d{1,2}[\s._-]?e\d{1,3}|\d{1,2}x\d{2})", re.IGNORECASE)


def _show_from_title(text: str) -> str:
    """ "The.Sopranos.S01E01.720p" -> "The Sopranos" ("" without an episode number)."""
    m = _EPISODE_NAME.match(text)
    if not m:
        return ""
    name = re.sub(r"[._]+", " ", m.group(1)).strip(" -([")
    name = re.sub(r"\s*\(?(19|20)\d\d\)?$", "", name).strip()
    return "" if _GENERIC_FOLDER.match(name) else name


def _show_hints(files: List[str], tags: Optional[Dict[str, Dict[str, str]]] = None) -> Tuple[List[str], List[str]]:
    """IMDb IDs and show names from file metadata, folders and file names."""
    tags = tags or {}
    file_tags = [tags.get(f) or {} for f in files]
    ids = list(dict.fromkeys(
        m for f, t in zip(files, file_tags) for text in (f, *t.values()) for m in _IMDB_ID.findall(text)
    ))
    names = []
    for t in file_tags:
        # Metadata first: a "show" tag, or a title like "The Sopranos - S01E01 - Pilot"
        for value in (t.get("show"), t.get("series"), _show_from_title(t.get("title") or "")):
            if value and not _GENERIC_FOLDER.match(value):
                names.append(value)
    for f in files:
        # The first folder that looks like a show's name: "The Sopranos/Season 1/..."
        for folder in f.split("/")[:-1]:
            folder = folder.strip()
            if folder and not _GENERIC_FOLDER.match(folder):
                names.append(folder)
                break
    for f in files:
        # Or the file name before the episode number: "The.Sopranos.S01E01.720p.mkv"
        name = _show_from_title(f.split("/")[-1])
        if name:
            names.append(name)
    names = list(dict.fromkeys(names))
    return ids, names


async def cast_lookup(files: List[str], tags: Optional[Dict[str, Dict[str, str]]] = None,
                      known_show: Optional[str] = None) -> Tuple[str, List[str]]:
    """("Show title", ["Character (Actor)", ...]) from TVmaze, or ("", [])."""
    ids, names = _show_hints(files, tags)
    if known_show:  # the show the AI recognised last time
        names = [re.sub(r"\s*\((19|20)\d\d\)$", "", known_show)] + [n for n in names if n != known_show]
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


def _clock(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _distinctive(lines: List[Dict[str, Any]], count: int = LINES_PER_PERSON) -> List[Dict[str, Any]]:
    """The lines most worth searching: long, with names in them, from different files."""
    def score(line: Dict[str, Any]) -> float:
        text = line["text"]
        return min(len(text), 120) + 25 * len(_NAME_WORD.findall(text)) - (40 if len(text.split()) < 5 else 0)

    unique: Dict[str, Dict[str, Any]] = {}
    for line in lines:  # the same line in several files: once is enough
        unique.setdefault(re.sub(r"\W+", " ", line["text"].lower()).strip(), line)
    ranked = sorted(unique.values(), key=score, reverse=True)
    picked, seen_files = [], set()
    for line in ranked:  # one per file first, then the rest
        if line.get("file") not in seen_files:
            picked.append(line)
            seen_files.add(line.get("file"))
    picked += [line for line in ranked if line not in picked]
    return picked[:count]


def _prompt(people: List[Dict[str, Any]], named: List[Dict[str, Any]], files: List[str],
            show: str, cast: List[str], tags: Optional[Dict[str, Dict[str, str]]] = None,
            web_search: bool = False) -> str:
    tags = tags or {}
    parts = []
    if files:
        rows = []
        for f in files[:40]:
            meta = "; ".join(f"{k}: {v}" for k, v in (tags.get(f) or {}).items())
            rows.append(f"- {f}" + (f"  [metadata: {meta}]" if meta else ""))
        parts.append("Files these people were found in (up to 40):\n" + "\n".join(rows))
    if cast:
        parts.append(f"Cast of {show} (from TVmaze, character (actor); found from the file names or metadata, "
                     "so ignore it if it clearly doesn't fit the dialogue):\n" + "; ".join(cast))
    if named:
        parts.append("Already named (by the user: trust these; marked AI: earlier answers, "
                     "use the same name for the same character):\n" + "\n".join(
            f'- {p["id"]} = {p["name"]}' for p in named))
    blocks = []
    for p in people:
        lines = "\n".join(
            f'  - "{line["text"]}"'
            + (f' ({line["file"]}' + (f' at {_clock(line.get("at"))}' if line.get("at") is not None else "") + ")"
               if line.get("file") else "")
            for line in _distinctive(p.get("lines", []))
        )
        blocks.append(f'{p["id"]} ({round(p["seconds"] / 60, 1)} min of speech, in {len(p["takes"])} files):\n{lines}')
    parts.append("People to identify:\n\n" + "\n\n".join(blocks))
    if web_search:
        parts.append(SEARCH_HINT)
    parts.append(
        'For each person above, who is speaking? Reply with JSON only:\n'
        '{"show": "Show or film title (year), or null", '
        '"people": [{"id": "p3", "name": "Character name", "actor": "Actor name or null", '
        '"confidence": "high|medium|low", "reason": "one short sentence"}]}\n'
        'Use name "mixed" if their lines clearly come from several characters, and null if you '
        'cannot tell. Give the same name to every person who is the same character.'
    )
    return "\n\n".join(parts)


def _parse(reply: str) -> List[Dict[str, Any]]:
    return _parse_reply(reply)[1]


def _parse_reply(reply: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """(show the AI recognised or None, answers per person)."""
    text = reply.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise providers.ProviderError("The AI's answer wasn't JSON. Try another model.")
    try:
        data = json.loads(text[start:end + 1])
    except ValueError as err:
        raise providers.ProviderError(f"The AI's answer wasn't valid JSON ({err}). Try another model.") from err
    show = data.get("show")
    show = " ".join(show.split())[:120] if isinstance(show, str) and show.strip().lower() not in ("", "null", "unknown") else None
    return show, [p for p in data.get("people", []) if isinstance(p, dict) and p.get("id")]


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

    await asyncio.to_thread(backfill, voice)
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
    tags = data.get("files") or {}
    known_show = (data.get("ai") or {}).get("show")
    show, cast = "", []
    if options["castLookup"]:
        show, cast = await _safe_cast_lookup(files, tags, known_show)
    searching = bool(options["webSearch"] and providers.web_search_support(conn))

    async def ask(batch: List[Dict[str, Any]], named: List[Dict[str, Any]], show: str,
                  cast: List[str]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        batch_files = list(dict.fromkeys(
            line["file"] for p in batch for line in p.get("lines", []) if line.get("file")))
        prompt = [{"role": "user", "content": _prompt(batch, named, batch_files, show, cast, tags, searching)}]
        # Room for reasoning models and web search results, not only the JSON
        budget = min(MAX_TOKENS, 3000 + 150 * len(batch))
        try:
            reply = await providers.chat(conn, prompt, system=SYSTEM, temperature=0.2, max_tokens=budget,
                                         web_search=options["webSearch"], timeout=REQUEST_TIMEOUT)
        except providers.EmptyAnswer as err:
            if not err.out_of_tokens or budget >= RETRY_MAX_TOKENS:
                raise
            _LOGGER.info("AI ran out of tokens (%d) before answering; retrying with %d", budget, RETRY_MAX_TOKENS)
            try:
                reply = await providers.chat(conn, prompt, system=SYSTEM, temperature=0.2,
                                             max_tokens=RETRY_MAX_TOKENS, web_search=options["webSearch"],
                                             timeout=REQUEST_TIMEOUT)
            except providers.ProviderError as retry_err:
                if isinstance(retry_err, providers.EmptyAnswer):
                    raise
                raise err from retry_err  # e.g. the model doesn't allow that many tokens
        return _parse_reply(reply)

    def known_names(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Names so far (the user's, and the AI's from earlier groups), for consistent answers."""
        people = speakers.load(voice)["people"]
        asking = {p["id"] for p in batch}
        named = [{"id": p["id"], "name": p["name"]} for p in people if not _DEFAULT_NAME.match(p["name"])]
        named += [{"id": p["id"], "name": f'{p["ai"]["name"]} (AI, {p["ai"]["confidence"]})'}
                  for p in people if _DEFAULT_NAME.match(p["name"]) and (p.get("ai") or {}).get("name")
                  and p["id"] not in asking]
        return named[:300]

    batches = [candidates[i:i + BATCH] for i in range(0, len(candidates), BATCH)]
    _set_status(voice, state="running", error=None, cast=show or None,
                detail=f"group 1 of {len(batches)}" if len(batches) > 1 else None)
    started = time.monotonic()
    answered = renamed = 0
    recognised = None
    for n, batch in enumerate(batches):
        if n:
            _set_status(voice, state="running", detail=f"group {n + 1} of {len(batches)}")
        named = known_names(batch)
        found_show, answer_list = await ask(batch, named, show, cast)
        recognised = recognised or found_show
        if found_show and not cast and options["castLookup"]:
            # The AI recognised the show from the dialogue: get its cast and ask again with it
            show, cast = await _safe_cast_lookup([], {}, found_show)
            if cast:
                _set_status(voice, state="running", cast=show)
                _LOGGER.info("AI recognised %s from the dialogue; asking again with its cast", show)
                found_show, answer_list = await ask(batch, named, show, cast)
        answers = {a["id"]: a for a in answer_list}
        renamed += _store_answers(voice, answers, options)
        answered += len(answers)

    with speakers._lock:
        data = speakers.load(voice)
        data["ai"] = {"state": "done", "error": None, "at": time.time(), "cast": show or None,
                      "identified": answered, "show": show or recognised or known_show, "detail": None}
        speakers._save(voice, data)

    merged = _auto_merge(voice) if options.get("autoMerge") else 0
    _LOGGER.info("AI identification for %s: %d of %d people answered in %d group(s), %d renamed, %d merged, in %.0fs%s",
                 voice.name, answered, len(candidates), len(batches), renamed, merged,
                 time.monotonic() - started, f" (cast: {show})" if show else "")
    return speakers.public(voice)


def _store_answers(voice: Voice, answers: Dict[str, Dict[str, Any]], options: Dict[str, Any]) -> int:
    """Save the AI's answers on the people (right away, so each group shows up); returns how many were renamed."""
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
        speakers._save(voice, data)
    return renamed


def backfill(voice: Voice) -> int:
    """Give people found before lines were kept (older imports) their lines and file metadata.

    Uses the takes still waiting for review (saved takes are gone); runs once per voice.
    """
    from . import freeform  # imports speakers too

    if speakers.load(voice).get("backfilled"):
        return 0
    found: Dict[str, List[Tuple[str, Dict[str, Any], int]]] = {}
    tags: Dict[str, Dict[str, str]] = {}
    takes_dir = freeform._takes_dir(voice)
    for take_dir in sorted(takes_dir.iterdir()) if takes_dir.is_dir() else []:
        if not (take_dir / "take.json").is_file() or take_dir.name in freeform._live:
            continue
        try:
            take = freeform._read(take_dir)
        except (OSError, ValueError):
            continue
        name = take.get("name") or ""
        for speaker in take.get("speakers") or []:
            if speaker.get("person"):
                found.setdefault(speaker["person"], []).append((take_dir.name, take, speaker["id"]))
        source = take_dir / (take.get("source") or "")
        if name and take.get("speakers") and "tags" not in take and source.is_file():
            tags[name] = freeform.file_tags(source)

    added = 0
    with speakers._lock:
        data = speakers.load(voice)
        for person in data["people"]:
            have = {line.get("file") for line in person.get("lines", [])}
            for _take_id, take, speaker_id in found.get(person["id"], []):
                if (take.get("name") or "") not in have:
                    speakers._add_lines(person, take, speaker_id)
                    added += 1
        files = data.setdefault("files", {})
        for name, value in tags.items():
            if value:
                files.setdefault(name, value)
        data["backfilled"] = True
        speakers._save(voice, data)
    if added:
        _LOGGER.info("AI identification for %s: added lines from %d earlier files", voice.name, added)
    return added


async def _safe_cast_lookup(files: List[str], tags: Dict[str, Dict[str, str]],
                            known_show: Optional[str]) -> Tuple[str, List[str]]:
    try:
        return await cast_lookup(files, tags, known_show)
    except (httpx.HTTPError, ValueError, KeyError) as err:
        _LOGGER.warning("TVmaze cast lookup failed: %s", err)
        return "", []


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
