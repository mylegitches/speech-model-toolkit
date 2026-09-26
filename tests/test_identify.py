"""AI naming of the people found across files (app/voice/identify.py)."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.settings import providers
from app.voice import identify, speakers

RNG = np.random.default_rng(1)
TONY, CARMELA, PAULIE = (RNG.normal(size=192) for _ in range(3))


def voice_of(base, noise):
    v = base / np.linalg.norm(base) + RNG.normal(size=192) * noise / np.sqrt(192)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def voice(tmp_path, monkeypatch):
    v = SimpleNamespace(root=tmp_path, name="series")
    people = [
        # Tony split into two cards: a normal and a very different-sounding take (shouting)
        {"id": "p1", "name": "Person 1", "centroid": voice_of(TONY, 0.3), "seconds": 300.0, "takes": ["t1", "t2"],
         "sample": None, "lines": [{"text": "I'm the boss here.", "file": "The Sopranos/Season 1/S01E01 tt0141842.mkv"}]},
        {"id": "p2", "name": "Person 2", "centroid": voice_of(TONY, 1.2), "seconds": 40.0, "takes": ["t3"],
         "sample": None, "lines": [{"text": "Get the hell out of my house!", "file": "The Sopranos/Season 1/S01E05.mkv"}]},
        {"id": "p3", "name": "Carmela", "centroid": voice_of(CARMELA, 0.3), "seconds": 200.0, "takes": ["t1"],
         "sample": None, "lines": [{"text": "Anthony, dinner is ready.", "file": "The Sopranos/Season 1/S01E01 tt0141842.mkv"}]},
        # The AI will (wrongly) call Paulie "Tony Soprano": voices differ, so no auto-merge
        {"id": "p4", "name": "Person 4", "centroid": voice_of(PAULIE, 0.3), "seconds": 90.0, "takes": ["t2"],
         "sample": None, "lines": [{"text": "Hey, T, what's the matter?", "file": "The Sopranos/Season 1/S01E02.mkv"}]},
    ]
    speakers._save(v, {"people": people, "target": None, "notSame": []})

    settings = {"identify": {"enabled": True, "webSearch": True, "castLookup": True, "autoName": True, "autoMerge": True}}
    monkeypatch.setattr(identify.settings_store, "load", lambda: settings)
    monkeypatch.setattr(identify.settings_store, "active_connection",
                        lambda s=None: {"provider": "openrouter", "model": "some/model", "name": "OR"})
    return v


def test_parse_accepts_code_fences_and_rejects_prose():
    reply = 'Here you go:\n```json\n{"people": [{"id": "p1", "name": "Tony Soprano"}]}\n```'
    assert identify._parse(reply) == [{"id": "p1", "name": "Tony Soprano"}]
    with pytest.raises(providers.ProviderError):
        identify._parse("I think p1 is Tony.")


def test_show_hints_from_paths():
    ids, names = identify._show_hints(["The Sopranos/Season 1/S01E01 tt0141842.mkv", "clip.mp4"])
    assert ids == ["tt0141842"] and names == ["The Sopranos"]


def test_identify_names_and_auto_merges(voice, monkeypatch):
    seen = {}

    async def fake_cast(files, *args):
        seen["files"] = files
        return "The Sopranos (1999)", ["Tony Soprano (James Gandolfini)", "Carmela Soprano (Edie Falco)"]

    async def fake_chat(conn, messages, **kwargs):
        seen["prompt"] = messages[0]["content"]
        seen["web_search"] = kwargs.get("web_search")
        return ('{"people": ['
                '{"id": "p1", "name": "Tony Soprano", "actor": "James Gandolfini", "confidence": "high", "reason": "boss"},'
                '{"id": "p2", "name": "Tony Soprano", "actor": "James Gandolfini", "confidence": "high", "reason": "his house"},'
                '{"id": "p3", "name": "Carmela Soprano", "confidence": "high", "reason": "calls him Anthony"},'
                '{"id": "p4", "name": "Tony Soprano", "confidence": "high", "reason": "wrong guess"}]}')

    monkeypatch.setattr(identify, "cast_lookup", fake_cast)
    monkeypatch.setattr(identify.providers, "chat", fake_chat)
    monkeypatch.setattr("app.voice.freeform.retag_person", lambda *a: None)

    result = asyncio.run(identify.identify(voice))
    people = {p["id"]: p for p in result["people"]}

    # The prompt had the cast, the lines, and the user's own names as hints
    assert "Tony Soprano (James Gandolfini)" in seen["prompt"]
    assert "Get the hell out of my house!" in seen["prompt"]
    assert "p3 = Carmela" in seen["prompt"] and seen["web_search"] is True

    # p2 (same character, voice still alike enough) merged into p1; p4 (different voice) not
    assert "p2" not in people and "p4" in people
    assert people["p1"]["name"] == "Tony Soprano"           # auto-named
    assert people["p3"]["name"] == "Carmela"                # the user's name is kept
    assert people["p4"]["name"] == "Tony Soprano"           # named, but kept as its own card...
    assert any(h.get("reason") for h in people["p4"]["similar"])  # ...and offered as a merge
    assert result["ai"]["state"] == "done" and result["ai"]["cast"] == "The Sopranos (1999)"


def test_auto_merge_off_only_suggests(voice, monkeypatch):
    identify.settings_store.load()["identify"]["autoMerge"] = False

    async def fake_chat(conn, messages, **kwargs):
        return ('{"people": [{"id": "p1", "name": "Tony Soprano", "confidence": "high"},'
                '{"id": "p2", "name": "Tony Soprano", "confidence": "medium"}]}')

    async def no_cast(files, *args):
        return "", []

    monkeypatch.setattr(identify, "cast_lookup", no_cast)
    monkeypatch.setattr(identify.providers, "chat", fake_chat)
    people = {p["id"]: p for p in asyncio.run(identify.identify(voice))["people"]}
    assert "p2" in people
    assert people["p1"]["similar"][0] == {**people["p1"]["similar"][0], "id": "p2", "reason": "AI thinks both are Tony Soprano"}


def test_generic_folders_are_not_show_names():
    _, names = identify._show_hints(["Season 1/ep1.mp4", "Downloads/Disc 2/x.mkv", "Videos/The Wire/S01/e1.mkv"])
    assert names == ["The Wire"]


def test_show_name_from_episode_file_names():
    _, names = identify._show_hints([
        "Season 01/The.Sopranos.S01E01.720p.BluRay.mkv", "Season 01/The Sopranos (1999) - S01E02 - 46 Long.mkv",
        "Season 1/S01E03 - Denial.mkv", "Season 1/sopranos_1x04.avi",
    ])
    assert names == ["The Sopranos", "sopranos"]


def test_show_from_file_metadata():
    ids, names = identify._show_hints(
        ["Season 01/S01E01.mkv", "Season 01/S01E02.mkv"],
        {"Season 01/S01E01.mkv": {"title": "The Sopranos - S01E01 - Pilot"},
         "Season 01/S01E02.mkv": {"show": "The Sopranos", "comment": "imdb tt0705276"}},
    )
    assert names == ["The Sopranos"] and ids == ["tt0705276"]


def test_distinctive_lines_prefer_names_and_spread_over_files():
    lines = [{"text": "Yeah.", "file": "a"}, {"text": "I told you already, it is not my problem anymore.", "file": "a"},
             {"text": "Go ask Paulie and Christopher about it tonight.", "file": "a"},
             {"text": "What do you want from me here?", "file": "b"}]
    picked = identify._distinctive(lines, 2)
    assert picked[0]["text"].startswith("Go ask Paulie") and picked[1]["file"] == "b"


def test_recognised_show_gets_cast_and_second_pass(voice, monkeypatch):
    prompts, lookups = [], []

    async def fake_cast(files, tags=None, known_show=None):
        lookups.append(known_show)
        return ("The Sopranos (1999)", ["Tony Soprano (James Gandolfini)"]) if known_show else ("", [])

    async def fake_chat(conn, messages, **kwargs):
        prompts.append(messages[0]["content"])
        if len(prompts) == 1:
            return '{"show": "The Sopranos (1999)", "people": [{"id": "p1", "name": "Tony Soprano", "confidence": "medium"}]}'
        return '{"show": "The Sopranos (1999)", "people": [{"id": "p1", "name": "Tony Soprano", "confidence": "high"}]}'

    monkeypatch.setattr(identify, "cast_lookup", fake_cast)
    monkeypatch.setattr(identify.providers, "chat", fake_chat)
    monkeypatch.setattr("app.voice.freeform.retag_person", lambda *a: None)
    result = asyncio.run(identify.identify(voice))

    assert len(prompts) == 2 and "Tony Soprano (James Gandolfini)" in prompts[1]
    assert "word for word" in prompts[0]  # openrouter supports web search: quote-search hint
    assert result["ai"]["show"] == "The Sopranos (1999)"
    assert {p["id"]: p for p in result["people"]}["p1"]["ai"]["confidence"] == "high"


def test_backfill_lines_from_takes_waiting_for_review(voice):
    take_dir = voice.root / "freeform" / "t9"
    take_dir.mkdir(parents=True)
    (take_dir / "take.json").write_text(json.dumps({
        "name": "Season 01/S01E09.mkv", "source": "S01E09.mkv",
        "segments": [{"start": 61.0, "end": 63.0, "text": "Where's Meadow tonight?", "speaker": 0}],
        "speakers": [{"id": 0, "person": "p3"}],
    }), encoding="utf-8")
    assert identify.backfill(voice) == 1
    lines = {p["id"]: p for p in speakers.load(voice)["people"]}["p3"]["lines"]
    assert lines[-1] == {"text": "Where's Meadow tonight?", "file": "Season 01/S01E09.mkv", "at": 61.0}
    assert identify.backfill(voice) == 0  # once per voice


def test_retries_with_more_tokens_when_the_model_ran_out(voice, monkeypatch):
    budgets = []

    async def fake_chat(conn, messages, **kwargs):
        budgets.append(kwargs["max_tokens"])
        if len(budgets) == 1:
            raise providers.EmptyAnswer("ran out", out_of_tokens=True)
        return '{"people": [{"id": "p1", "name": "Tony Soprano", "confidence": "medium"}]}'

    async def no_cast(*args):
        return "", []

    monkeypatch.setattr(identify, "cast_lookup", no_cast)
    monkeypatch.setattr(identify.providers, "chat", fake_chat)
    people = {p["id"]: p for p in asyncio.run(identify.identify(voice))["people"]}
    assert budgets == [min(identify.MAX_TOKENS, 3000 + 150 * 4), identify.RETRY_MAX_TOKENS]
    assert people["p1"]["ai"]["name"] == "Tony Soprano"
