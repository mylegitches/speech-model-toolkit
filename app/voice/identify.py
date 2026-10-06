"""Who says each line: the AI reads imported files' transcripts like a script.

For every imported file with speaker detection, the active AI connection gets
the transcript in order, in parts of CHUNK lines ("L12 [12:41] B: ..."), with
the voice groups speaker detection found (A, B, C...), what earlier files tell
about those voices, the show's cast from TVmaze (found from an IMDb ID
"tt0141842", the show's folder or file names, or file metadata; free, no key)
and, where the provider can, web search to look lines up. It answers who says
each line, with a confidence. Then:

  - a line is "confirmed" when the AI is sure and its voice group agrees (most
    of that group's lines are the same character); other lines the AI gives to
    a character, and unclear lines in a voice group that is mostly one
    character, are "possible" for the user to check
  - characters() lists every character with their clips; character_clips()
    returns one character's confirmed and possible clips across all files
  - the people cards (speakers.py) get the character most of their lines
    belong to; with "autoName" untouched "Person N" cards are renamed, with
    "autoMerge" cards that are clearly one character are merged when their
    voices are also somewhat alike

Runs after each diarized file (when enabled in Settings) and on demand.
"""

import asyncio
import json
import logging
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from ..settings import providers
from ..settings import store as settings_store
from . import speakers
from .voices import Voice

_LOGGER = logging.getLogger(__name__)

CHUNK = 120              # transcript lines per AI request
CONTEXT = 8              # earlier lines (with their answers) repeated for continuity
REQUEST_TIMEOUT = 300    # seconds per request
MAX_TOKENS = 8000        # reply budget (reasoning + search + the JSON); most models allow 8k
RETRY_MAX_TOKENS = 16000  # once more with this when the model ran out before answering
AUTO_MERGE_MIN_VOICE = 0.25
"""Voice similarity two cards need before being one character merges them."""
GROUP_MAJORITY = 0.5     # share of a voice group's lines a character needs to "own" the group

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
    "You attribute the lines of a TV or film transcript to the characters who speak them. "
    "The transcript is machine-made: words may be slightly off, and the voice groups (A, B, C...) come "
    "from automatic speaker detection, which is usually right but sometimes mixes two people or splits one. "
    "Use who is addressed and named, who replies to whom, the voice groups, the cast list and your "
    "knowledge of the episode. Answer with JSON only, no other text."
)
SEARCH_HINT = (
    "You can search the web: look up a few distinctive lines word for word, in quotes (episode "
    "transcripts, subtitle and quote sites), to confirm who says them. If an exact search finds nothing, "
    "try a shorter part of the line."
)

_running: Set[str] = set()
_again: Set[str] = set()


def status(voice: Voice) -> Dict[str, Any]:
    conn = settings_store.active_connection()
    settings = settings_store.load()["identify"]
    return {
        "enabled": True,  # Character clone always identifies; it only needs an AI connection
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


# ---- The transcript, as the AI sees it ------------------------------------------------


def _clock(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _letters(take: Dict[str, Any]) -> Dict[int, str]:
    """Voice groups as letters, the one that talks most first: {speaker id: "A"}."""
    order = sorted(take.get("speakers") or [], key=lambda s: -(s.get("seconds") or 0))
    names = {}
    for n, speaker in enumerate(order):
        names[speaker["id"]] = chr(65 + n) if n < 26 else f"Z{n - 25}"
    return names


def _line(n: int, segment: Dict[str, Any], letters: Dict[int, str]) -> str:
    who = letters.get(segment.get("speaker"), "?")
    return f"L{n + 1} [{_clock(segment.get('start'))}] {who}: {segment['text'].strip()}"


def _group_hints(take: Dict[str, Any], letters: Dict[int, str], people: Dict[str, Dict[str, Any]]) -> List[str]:
    """What earlier files say about each voice group (through the people cards)."""
    hints = []
    counts = Counter(s.get("speaker") for s in take["segments"] if s.get("text"))
    for speaker in sorted(take.get("speakers") or [], key=lambda s: letters.get(s["id"], "?")):
        letter = letters.get(speaker["id"], "?")
        person = people.get(speaker.get("person") or "")
        note = ""
        if person and not _DEFAULT_NAME.match(person["name"]):
            note = f" = {person['name']} (named by the user)"
        elif person and (person.get("ai") or {}).get("name") and person["ai"].get("confidence") in ("high", "medium"):
            note = f": in other files mostly {person['ai']['name']}"
        hints.append(f"{letter} ({counts.get(speaker['id'], 0)} lines){note}")
    return hints


def _prompt(file: str, tags: Dict[str, str], show: str, cast: List[str], hints: List[str],
            context: List[str], lines: List[str], searching: bool) -> str:
    parts = []
    meta = "; ".join(f"{k}: {v}" for k, v in tags.items())
    parts.append(f"File: {file}" + (f"  [metadata: {meta}]" if meta else ""))
    if cast:
        parts.append(f"Cast of {show} (from TVmaze, character (actor); found from the file names or metadata, "
                     "so ignore it if it clearly doesn't fit the dialogue):\n" + "; ".join(cast))
    if hints:
        parts.append("Voice groups in this file:\n" + "\n".join(f"- {h}" for h in hints))
    if context:
        parts.append("Just before (already attributed, for context):\n" + "\n".join(context))
    parts.append("Transcript:\n" + "\n".join(lines))
    if searching:
        parts.append(SEARCH_HINT)
    parts.append(
        'Who says each transcript line? Reply with JSON only, one entry per line, in order:\n'
        '{"show": "Show or film title (year), or null", '
        '"lines": [["L1", "Character name", "high"], ["L2", null, "low"]]}\n'
        'Confidence is high, medium or low. Use the cast\'s character names exactly, and null when you '
        'cannot tell who speaks.'
    )
    return "\n\n".join(parts)


_TRIPLE = re.compile(r'\[\s*"L(\d+)"\s*,\s*(null|"((?:[^"\\]|\\.)*)")\s*,\s*"?(high|medium|low)"?\s*\]', re.IGNORECASE)


class NotTheJSON(providers.ProviderError):
    """The AI answered, but not with the JSON asked for (prose, another layout...)."""


_LINE_ID = re.compile(r"L?(\d+)", re.IGNORECASE)
_PROSE = re.compile(r'^[\s*\-•]*\**"?L(\d+)"?\**\s*[:=\-–|]\s*([^\n(|]+?)\s*(?:[(|,\-–]\s*(high|medium|low)\s*\)?)?\s*$',
                    re.IGNORECASE | re.MULTILINE)


def _first(entry: Dict[str, Any], *keys: str) -> Any:
    return next((entry[k] for k in keys if entry.get(k) is not None), None)


def _entry(key: Any, value: Any) -> Optional[Tuple[int, Optional[str], str]]:
    """One answer in any of the layouts models use: ["L1", "Tony", "high"],
    {"id": "L1", "speaker": "Tony", "confidence": "high"}, or "L1": "Tony" / {...}."""
    if isinstance(value, dict):
        key = _first(value, "id", "line", "l", "number") if key is None else key
        name = _first(value, "name", "character", "speaker", "who", "char")
        conf = _first(value, "confidence", "conf", "certainty", "c")
    elif isinstance(value, (list, tuple)):
        if key is None:
            if len(value) < 2:
                return None
            key, value = value[0], value[1:]
        name = value[0] if value else None
        conf = value[1] if len(value) > 1 else None
    else:
        name, conf = value, None
    m = _LINE_ID.fullmatch(str(key).strip()) if key is not None else None
    if not m:
        return None
    name = (name.strip() if isinstance(name, str)
            and name.strip().lower() not in ("", "null", "none", "unknown", "?", "unclear") else None)
    conf = str(conf or "low").strip().lower()
    return int(m.group(1)), name, conf if conf in ("high", "medium", "low") else "low"


def _parse(reply: str) -> Tuple[Optional[str], List[Tuple[int, Optional[str], str]]]:
    """(show or None, [(line number, character or None, confidence)])."""
    text = reply.strip()
    data: Any = None
    # The JSON may be wrapped in prose or a code block: take the outermost object or list
    for opening, closing in (("{", "}"), ("[", "]")):
        start, end = text.find(opening), text.rfind(closing)
        if 0 <= start < end:
            try:
                data = json.loads(text[start:end + 1])
                break
            except ValueError:
                data = None
    show = None
    answers: List[Tuple[int, Optional[str], str]] = []
    if isinstance(data, dict):
        raw = data.get("show")
        if isinstance(raw, str) and raw.strip().lower() not in ("", "null", "unknown"):
            show = " ".join(raw.split())[:120]
        lines = data.get("lines", data.get("attributions", data.get("answers")))
        if lines is None:  # {"L1": "Tony", "L2": ...} with no wrapper
            lines = {k: v for k, v in data.items() if _LINE_ID.fullmatch(str(k))}
        data = lines
    if isinstance(data, dict):
        answers = [a for a in (_entry(k, v) for k, v in data.items()) if a]
    elif isinstance(data, list):
        answers = [a for a in (_entry(None, v) for v in data) if a]
    if answers:
        return show, answers
    # Broken JSON (an unescaped quote...): keep every entry that can still be read
    for m in _TRIPLE.finditer(text):
        name = None if m.group(2).lower() == "null" else m.group(3)
        answers.append((int(m.group(1)), name, m.group(4).lower()))
    if not answers:
        # Or plain lines: "L12: Tony Soprano (high)", "- L13 - unknown"
        for m in _PROSE.finditer(text):
            name = m.group(2).strip(' "*')
            name = None if name.lower() in ("null", "none", "unknown", "?", "unclear") else name
            answers.append((int(m.group(1)), name, (m.group(3) or "low").lower()))
    if not answers:
        raise NotTheJSON("The AI's answer wasn't the JSON asked for. Try another model.")
    _LOGGER.info("The AI's JSON was broken; recovered %d line answers", len(answers))
    return show, answers


def _canonical(name: Optional[str], cast_names: List[str]) -> Optional[str]:
    """The cast's spelling of a name the AI gave ("tony" -> "Tony Soprano" when that's the only Tony)."""
    name = " ".join((name or "").split())
    if not name or name.lower() in ("null", "none", "unknown", "?", "mixed"):
        return None
    lower = {c.lower(): c for c in cast_names}
    if name.lower() in lower:
        return lower[name.lower()]
    first = [c for c in cast_names if c.lower().split()[0] == name.lower()]
    if len(first) == 1:
        return first[0]
    return name[:60]


# ---- Takes (freeform.py) ------------------------------------------------------------


def _takes(voice: Voice) -> List[Tuple[str, Dict[str, Any]]]:
    """Character clone files that are done: [(take id, take)], by file name.
    (Files the AI already read before the Character clone tab existed count too.)"""
    from . import freeform  # freeform imports this module

    found = []
    root = freeform._takes_dir(voice)
    for take_dir in sorted(root.iterdir()) if root.is_dir() else []:
        if not (take_dir / "take.json").is_file() or take_dir.name in freeform._live:
            continue
        try:
            take = freeform._read(take_dir)
        except (OSError, ValueError):
            continue
        if (take.get("state") == "done" and take.get("speakers") and take.get("segments")
                and (take.get("clone") or take.get("attributed"))):
            found.append((take_dir.name, take))
    return sorted(found, key=lambda t: (t[1].get("name") or t[0]).lower())


def _confirm(take: Dict[str, Any]) -> None:
    """Mark lines where the AI is sure and the voice group agrees; remember each group's character."""
    counts: Dict[int, Counter] = defaultdict(Counter)
    for segment in take["segments"]:
        if segment.get("character") and segment.get("conf") in ("high", "medium"):
            counts[segment.get("speaker")][segment["character"]] += 1
    owners: Dict[str, str] = {}
    for speaker, count in counts.items():
        character, n = count.most_common(1)[0]
        if n / sum(count.values()) >= GROUP_MAJORITY:
            owners[str(speaker)] = character
    take["groupCharacters"] = owners
    for segment in take["segments"]:
        segment["confirmed"] = bool(
            segment.get("character") and segment.get("conf") == "high"
            and owners.get(str(segment.get("speaker"))) == segment["character"]
        )


def _store(voice: Voice, take_id: str, answers: Dict[int, Tuple[Optional[str], str]], complete: bool, model: str) -> None:
    """Write the answers into take.json (re-read first: the page may have changed it meanwhile)."""
    from . import freeform

    take_dir = freeform._takes_dir(voice) / take_id
    if take_id in freeform._live or not (take_dir / "take.json").is_file():
        return
    take = freeform._read(take_dir)
    for index, (character, conf) in answers.items():
        if 0 <= index < len(take["segments"]):
            take["segments"][index]["character"] = character
            take["segments"][index]["conf"] = conf
    _confirm(take)
    if complete:
        take["attributed"] = time.time()
        take["attributedWith"] = model
    freeform._write(take_dir, take)


# ---- Asking the AI ------------------------------------------------------------------


def _set_status(voice: Voice, **values: Any) -> None:
    with speakers._lock:
        data = speakers.load(voice)
        data["ai"] = {**(data.get("ai") or {}), **values, "at": time.time()}
        speakers._save(voice, data)


async def _ask(conn: Dict[str, Any], options: Dict[str, Any], prompt: str, lines: int
               ) -> Tuple[Optional[str], List[Tuple[int, Optional[str], str]]]:
    messages = [{"role": "user", "content": prompt}]
    # Room for reasoning models and web search results, not only the JSON
    budget = min(MAX_TOKENS, 1500 + 30 * lines)
    kwargs = dict(system=SYSTEM, temperature=0.2, web_search=options["webSearch"], timeout=REQUEST_TIMEOUT,
                  think=False, json_mode=True)
    try:
        reply = await providers.chat(conn, messages, max_tokens=budget, **kwargs)
    except providers.EmptyAnswer as err:
        if not err.out_of_tokens or budget >= RETRY_MAX_TOKENS:
            raise
        _LOGGER.info("AI ran out of tokens (%d) before answering; retrying with %d", budget, RETRY_MAX_TOKENS)
        try:
            reply = await providers.chat(conn, messages, max_tokens=RETRY_MAX_TOKENS, **kwargs)
        except providers.ProviderError as retry_err:
            if isinstance(retry_err, providers.EmptyAnswer):
                raise
            raise err from retry_err  # e.g. the model doesn't allow that many tokens
    try:
        return _parse(reply)
    except NotTheJSON:
        # Usually after a web search (JSON mode is off then): have it restate its answer as the JSON,
        # this time in JSON mode and without searching
        _LOGGER.warning("The AI's answer wasn't the JSON asked for (%s…); asking it to reformat",
                        " ".join(reply.split())[:300])
        again = messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "Reply again with only the JSON in the format asked for: "
                                        '{"show": ..., "lines": [["L1", "Character name", "high"], ...]}, '
                                        "one entry per transcript line, and nothing else."},
        ]
        reply = await providers.chat(conn, again, max_tokens=budget,
                                     **{**kwargs, "web_search": False, "json_mode": True})
        return _parse(reply)


async def _attribute(voice: Voice, take_id: str, take: Dict[str, Any], ctx: Dict[str, Any],
                     progress: Any) -> Optional[str]:
    """Ask who says each line of one file, part by part; returns the show the AI named, if any."""
    segments = take["segments"]
    letters = _letters(take)
    order = [i for i, s in enumerate(segments) if (s.get("text") or "").strip()]
    chunks = [order[k:k + CHUNK] for k in range(0, len(order), CHUNK)]
    hints = _group_hints(take, letters, ctx["people"])
    file = take.get("name") or take_id
    answers: Dict[int, Tuple[Optional[str], str]] = {}
    found_show = None
    complete = False
    try:
        for n, chunk in enumerate(chunks):
            progress(n + 1, len(chunks))
            before = [i for i in order if i < chunk[0]][-CONTEXT:]
            context = [
                _line(i, segments[i], letters) + f"  -> {answers.get(i, (None,))[0] or '?'}" for i in before
            ]
            prompt = _prompt(file, ctx["tags"].get(file) or take.get("tags") or {}, ctx["show"], ctx["cast"],
                             hints, context, [_line(i, segments[i], letters) for i in chunk], ctx["searching"])
            show, got = await _ask(ctx["conn"], ctx["options"], prompt, len(chunk))
            found_show = found_show or show
            wanted = set(chunk)
            for number, name, conf in got:
                if number - 1 in wanted:
                    answers[number - 1] = (_canonical(name, ctx["cast_names"]), conf)
        complete = True
    finally:
        if answers or complete:
            _store(voice, take_id, answers, complete, ctx["conn"].get("model", ""))
    return found_show


def _name_cards(voice: Voice, options: Dict[str, Any], actors: Dict[str, str]) -> int:
    """Each people card gets the character most of its lines belong to; returns how many were renamed."""
    counts: Dict[str, Counter] = defaultdict(Counter)
    for _take_id, take in _takes(voice):
        people = {s["id"]: s.get("person") for s in take.get("speakers") or []}
        for segment in take["segments"]:
            person = people.get(segment.get("speaker"))
            if person and segment.get("character") and segment.get("conf") in ("high", "medium"):
                counts[person][segment["character"]] += 1
    renamed = 0
    with speakers._lock:
        data = speakers.load(voice)
        for person in data["people"]:
            count = counts.get(person["id"])
            if not count:
                continue
            name, n = count.most_common(1)[0]
            total = sum(count.values())
            share = n / total
            confidence = "high" if share >= 0.75 and n >= 5 else "medium" if share >= 0.5 and n >= 2 else "low"
            person["ai"] = {
                "name": name, "actor": actors.get(name), "confidence": confidence,
                "reason": f"{n} of their {total} identified lines are {name}",
                "lines": len(person.get("lines", [])), "at": time.time(),
            }
            if options["autoName"] and confidence == "high" and _DEFAULT_NAME.match(person["name"]):
                person["name"] = name[:40]
                renamed += 1
        speakers._save(voice, data)
    return renamed


async def identify(voice: Voice, everyone: bool = False) -> Dict[str, Any]:
    """Attribute the lines of the files not done yet (or all of them) and name the cards."""
    settings = settings_store.load()
    options = settings["identify"]
    conn = settings_store.active_connection(settings)
    if conn is None:
        raise RuntimeError("Add an AI connection in Settings → AI connections first")

    takes = _takes(voice)
    pending = takes if everyone else [(i, t) for i, t in takes if not t.get("attributed")]
    data = speakers.load(voice)
    known_show = (data.get("ai") or {}).get("show")
    tags = {**(data.get("files") or {}), **{t.get("name"): t["tags"] for _i, t in takes if t.get("tags") and t.get("name")}}
    show, cast = "", []
    if pending and options["castLookup"]:
        show, cast = await _safe_cast_lookup([t.get("name") or i for i, t in takes], tags, known_show)
    ctx = {
        "conn": conn, "options": options, "tags": tags, "show": show, "cast": cast,
        "cast_names": [c.rsplit(" (", 1)[0] for c in cast],
        "people": {p["id"]: p for p in data["people"]},
        "searching": bool(options["webSearch"] and providers.web_search_support(conn)),
    }

    _set_status(voice, state="running", error=None, cast=show or None,
                detail=f"file 1 of {len(pending)}" if pending else None)
    started = time.monotonic()
    failed: List[str] = []
    recognised = None
    for n, (take_id, take) in enumerate(pending):
        def progress(part: int, parts: int, n: int = n, take_id: str = take_id) -> None:
            _set_status(voice, state="running",
                        detail=f"file {n + 1} of {len(pending)}" + (f" · part {part} of {parts}" if parts > 1 else ""),
                        take=take_id, part=part, parts=parts)  # which file the AI is reading (the Files table)

        try:
            found = await _attribute(voice, take_id, take, ctx, progress)
        except providers.ProviderError as err:
            _LOGGER.warning("AI identification for %s, file %s failed: %s", voice.name, take.get("name") or take_id, err)
            failed.append(str(err))
            continue
        recognised = recognised or found
        if found and not ctx["cast"] and options["castLookup"]:
            # The AI recognised the show from the dialogue: use its cast from the next file on
            ctx["show"], ctx["cast"] = await _safe_cast_lookup([], {}, found)
            ctx["cast_names"] = [c.rsplit(" (", 1)[0] for c in ctx["cast"]]
            if ctx["cast"]:
                _LOGGER.info("AI recognised %s from the dialogue; using its cast", ctx["show"])
                _set_status(voice, state="running", cast=ctx["show"])

    actors = dict(c[:-1].rsplit(" (", 1) for c in ctx["cast"] if c.endswith(")") and " (" in c)
    renamed = _name_cards(voice, options, actors)
    merged = _auto_merge(voice) if options.get("autoMerge") else 0
    with speakers._lock:
        data = speakers.load(voice)
        data["ai"] = {"state": "done", "error": None, "at": time.time(), "cast": ctx["show"] or None,
                      "identified": len(pending) - len(failed), "show": ctx["show"] or recognised or known_show,
                      "detail": None, "take": None}
        if failed:
            data["ai"].update(state="error" if len(failed) == len(pending) else "done",
                              error=f"{len(failed)} of {len(pending)} files failed: {failed[-1]}")
        speakers._save(voice, data)
    _LOGGER.info("AI identification for %s: %d of %d files attributed, %d cards renamed, %d merged, in %.0fs%s",
                 voice.name, len(pending) - len(failed), len(pending), renamed, merged,
                 time.monotonic() - started, f" (cast: {ctx['show']})" if ctx["show"] else "")
    return speakers.public(voice)


# ---- Characters and their clips ------------------------------------------------------


def characters(voice: Voice) -> List[Dict[str, Any]]:
    """Every character the AI found: confirmed and possible clips, speech time, files."""
    stats: Dict[str, Dict[str, Any]] = {}

    def entry(name: str) -> Dict[str, Any]:
        return stats.setdefault(name, {"name": name, "confirmed": 0, "possible": 0, "seconds": 0.0,
                                       "saved": 0, "files": set()})

    for take_id, take in _takes(voice):
        owners = take.get("groupCharacters") or {}
        for segment in take["segments"]:
            character = segment.get("character")
            owner = owners.get(str(segment.get("speaker")))
            if character:
                e = entry(character)
                e["files"].add(take_id)
                if segment.get("confirmed"):
                    e["confirmed"] += 1
                    e["seconds"] += segment["end"] - segment["start"]
                else:
                    e["possible"] += 1
                e["saved"] += bool(segment.get("saved"))
            elif owner and "character" in segment:
                entry(owner)["possible"] += 1
    out = [{**e, "seconds": round(e["seconds"], 1), "files": len(e["files"])} for e in stats.values()]
    return sorted(out, key=lambda e: (-e["seconds"], -e["possible"]))


def character_clips(voice: Voice, name: str) -> Dict[str, Any]:
    """One character's clips across all files: confirmed (AI sure, voice agrees) and possible.
    Files in the order they were added, oldest first (take ids are upload timestamps)."""
    confirmed, possible = [], []
    for take_id, take in sorted(_takes(voice), key=lambda t: (t[1].get("created") or 0, t[0])):
        owners = take.get("groupCharacters") or {}
        for index, segment in enumerate(take["segments"]):
            character = segment.get("character")
            owner = owners.get(str(segment.get("speaker")))
            if character != name and not (character is None and "character" in segment and owner == name):
                continue
            clip = {
                "take": take_id, "file": take.get("name") or take_id, "index": index,
                "text": segment.get("savedText") or segment["text"],
                "start": segment["start"], "end": segment["end"], "conf": segment.get("conf"),
                "saved": bool(segment.get("saved")), "denoise": take.get("denoise") or "off",
                "trimmed": bool(segment.get("trims")), "trims": len(segment.get("trims") or []),
            }
            if segment.get("confirmed"):
                confirmed.append(clip)
                continue
            if character is None:
                clip["reason"] = f"the AI couldn't tell; the voice is mostly {name} in this file"
            elif owner and owner != name:
                clip["reason"] = f"the AI says {name}, but this voice is mostly {owner} in this file"
            elif not owner:
                clip["reason"] = f"the AI says {name} ({segment.get('conf')}); this voice is mixed in this file"
            else:
                clip["reason"] = f"the AI is {segment.get('conf') or 'not'} sure"
            possible.append(clip)
    return {"name": name, "confirmed": confirmed, "possible": possible}


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
    """Called when a Character clone file is done: work out who says each line."""
    if settings_store.active_connection():
        schedule(voice)
