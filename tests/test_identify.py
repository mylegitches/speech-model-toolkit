"""AI attribution of every transcript line (app/voice/identify.py)."""

import asyncio
import json
import re

import numpy as np
import pytest

from app.settings import providers
from app.voice import freeform, identify, speakers
from app.voice.voices import Voice

RNG = np.random.default_rng(1)
TONY, CARMELA = (RNG.normal(size=192).tolist() for _ in range(2))

# Who really says each line (the fake AI answers from this)
WHO = {
    "Anthony, dinner is ready.": "Carmela Soprano",
    "I'm the boss here.": "Tony Soprano",
    "Get the hell out of my house!": "Tony Soprano",
    "Hey T, what's the matter?": "Paulie Gualtieri",
    "Where's Meadow tonight?": "Carmela Soprano",
    "Somebody pass the gabagool.": None,
}


def seg(start, text, speaker):
    return {"start": start, "end": start + 2.0, "text": text, "speaker": speaker}


@pytest.fixture
def voice(tmp_path, monkeypatch):
    v = Voice(name="series", language="en-US", espeak_voice="en-us", root=tmp_path)
    takes = {
        "t1": {"name": "The Sopranos/Season 1/S01E01 tt0141842.mkv", "segments": [
            seg(0, "I'm the boss here.", 0), seg(3, "Anthony, dinner is ready.", 1),
            seg(6, "Get the hell out of my house!", 0), seg(9, "Hey T, what's the matter?", 0),
            seg(12, "Somebody pass the gabagool.", 0),
        ], "speakers": [{"id": 0, "person": "p1", "seconds": 8}, {"id": 1, "person": "p2", "seconds": 2}]},
        "t2": {"name": "The Sopranos/Season 1/S01E02.mkv", "segments": [
            seg(0, "Where's Meadow tonight?", 0), seg(3, "I'm the boss here.", 1),
        ], "speakers": [{"id": 0, "person": "p2", "seconds": 2}, {"id": 1, "person": "p1", "seconds": 2}]},
    }
    for take_id, take in takes.items():
        (tmp_path / "freeform" / take_id).mkdir(parents=True)
        freeform._write(tmp_path / "freeform" / take_id, {"state": "done", "denoise": "off", **take})
    people = [
        {"id": "p1", "name": "Person 1", "centroid": TONY, "seconds": 10.0, "takes": ["t1", "t2"], "sample": None},
        {"id": "p2", "name": "Person 2", "centroid": CARMELA, "seconds": 4.0, "takes": ["t1", "t2"], "sample": None},
    ]
    speakers._save(v, {"people": people, "target": None, "notSame": []})

    settings = {"identify": {"enabled": True, "webSearch": True, "castLookup": True, "autoName": True, "autoMerge": False}}
    monkeypatch.setattr(identify.settings_store, "load", lambda: settings)
    monkeypatch.setattr(identify.settings_store, "active_connection",
                        lambda s=None: {"provider": "openrouter", "model": "some/model", "name": "OR"})

    async def fake_cast(files, *args):
        return "The Sopranos (1999)", ["Tony Soprano (James Gandolfini)", "Carmela Soprano (Edie Falco)",
                                       "Paulie Gualtieri (Tony Sirico)"]

    monkeypatch.setattr(identify, "cast_lookup", fake_cast)
    return v


def script_ai(prompts, fail_on=None):
    """A fake AI that knows the script: answers every Ln line of the transcript part."""
    async def chat(conn, messages, **kwargs):
        prompt = messages[0]["content"]
        prompts.append(prompt)
        if fail_on and fail_on in prompt:
            raise providers.ProviderError("the AI is down")
        transcript = prompt.split("Transcript:\n", 1)[1]
        lines = []
        for m in re.finditer(r"^L(\d+) \[[^\]]*\] \w+: (.+)$", transcript, re.MULTILINE):
            who = WHO.get(m.group(2))
            lines.append([f"L{m.group(1)}", "paulie" if who == "Paulie Gualtieri" else who,
                          "high" if who else "low"])
        return json.dumps({"show": "The Sopranos (1999)", "lines": lines})
    return chat


def take(voice, take_id):
    return freeform._read(voice.root / "freeform" / take_id)


def test_every_line_is_attributed_and_checked_against_the_voice(voice, monkeypatch):
    prompts = []
    monkeypatch.setattr(identify.providers, "chat", script_ai(prompts))
    asyncio.run(identify.identify(voice))

    t1 = take(voice, "t1")["segments"]
    assert [s["character"] for s in t1] == ["Tony Soprano", "Carmela Soprano", "Tony Soprano",
                                            "Paulie Gualtieri", None]  # "paulie" -> the cast's name
    # Voice group A is mostly Tony: his lines are confirmed, Paulie's line in that group isn't
    assert [s["confirmed"] for s in t1] == [True, True, True, False, False]
    assert take(voice, "t1")["attributed"]

    # The prompt was a script: file, cast, voice groups, lines in order
    assert "S01E01 tt0141842.mkv" in prompts[0] and "Tony Soprano (James Gandolfini)" in prompts[0]
    assert "L1 [0:00] A: I'm the boss here." in prompts[0] and "- A (4 lines)" in prompts[0]
    assert "search" in prompts[0]  # OpenRouter can search the web

    # Cards are named after the character most of their lines are
    people = {p["id"]: p for p in speakers.load(voice)["people"]}
    assert people["p1"]["ai"]["name"] == "Tony Soprano" and people["p2"]["ai"]["name"] == "Carmela Soprano"


def test_characters_and_their_clips(voice, monkeypatch):
    monkeypatch.setattr(identify.providers, "chat", script_ai([]))
    asyncio.run(identify.identify(voice))

    chars = {c["name"]: c for c in identify.characters(voice)}
    assert chars["Tony Soprano"]["confirmed"] == 3 and chars["Tony Soprano"]["files"] == 2
    assert chars["Paulie Gualtieri"] == {**chars["Paulie Gualtieri"], "confirmed": 0, "possible": 1}

    tony = identify.character_clips(voice, "Tony Soprano")
    assert [(c["take"], c["index"]) for c in tony["confirmed"]] == [("t1", 0), ("t1", 2), ("t2", 1)]
    # The line the AI couldn't place, in a voice that's mostly Tony, is offered as possible
    assert [(c["take"], c["index"]) for c in tony["possible"]] == [("t1", 4)]
    assert "couldn't tell" in tony["possible"][0]["reason"]
    paulie = identify.character_clips(voice, "Paulie Gualtieri")
    assert "mostly Tony Soprano" in paulie["possible"][0]["reason"]


def test_long_files_go_in_parts_with_context(voice, monkeypatch):
    monkeypatch.setattr(identify, "CHUNK", 2)
    prompts = []
    monkeypatch.setattr(identify.providers, "chat", script_ai(prompts))
    asyncio.run(identify.identify(voice))
    assert len(prompts) == 4  # t1: 3 parts, t2: 1
    assert "Just before (already attributed" in prompts[1]
    assert "L2 [0:03] B: Anthony, dinner is ready.  -> Carmela Soprano" in prompts[1]


def test_a_failed_file_does_not_stop_the_others(voice, monkeypatch):
    monkeypatch.setattr(identify.providers, "chat", script_ai([], fail_on="S01E02"))
    result = asyncio.run(identify.identify(voice))
    assert take(voice, "t1").get("attributed") and not take(voice, "t2").get("attributed")
    assert "1 of 2 files failed" in result["ai"]["error"]
    # Next run only asks about the file that failed
    prompts = []
    monkeypatch.setattr(identify.providers, "chat", script_ai(prompts))
    asyncio.run(identify.identify(voice))
    assert len(prompts) == 1 and "S01E02" in prompts[0]


def test_retries_with_more_tokens_when_the_model_ran_out(voice, monkeypatch):
    budgets = []
    real = script_ai([])

    async def chat(conn, messages, **kwargs):
        budgets.append(kwargs["max_tokens"])
        if len(budgets) == 1:
            raise providers.EmptyAnswer("ran out", out_of_tokens=True)
        return await real(conn, messages, **kwargs)

    monkeypatch.setattr(identify.providers, "chat", chat)
    asyncio.run(identify.identify(voice))
    assert budgets[:2] == [1500 + 30 * 5, identify.RETRY_MAX_TOKENS]
    assert take(voice, "t1").get("attributed")


def test_parse_json_and_broken_json():
    show, lines = identify._parse('{"show": "X (1999)", "lines": [["L1", "Tony", "high"], ["L2", null, "low"]]}')
    assert show == "X (1999)" and lines == [(1, "Tony", "high"), (2, None, "low")]
    broken = '{"lines": [["L1", "Tony "T" Soprano", "high"], ["L2", "Carmela Soprano", "medium"]]}'
    assert identify._parse(broken)[1] == [(2, "Carmela Soprano", "medium")]
    with pytest.raises(providers.ProviderError):
        identify._parse("I think it's Tony.")


def test_canonical_names():
    cast = ["Tony Soprano", "Tony Blundetto", "Carmela Soprano"]
    assert identify._canonical("carmela", cast) == "Carmela Soprano"
    assert identify._canonical("TONY SOPRANO", cast) == "Tony Soprano"
    assert identify._canonical("Tony", cast) == "Tony"  # two Tonys: not guessed
    assert identify._canonical("unknown", cast) is None


def test_saving_clips_keeps_the_files(voice, monkeypatch):
    monkeypatch.setattr(freeform, "clip_wav", lambda v, t, i, level=None: b"RIFF" + b"\0" * 2000)
    saved = freeform.save_clips(voice, [{"take": "t1", "index": 0, "text": "I'm the boss here."},
                                        {"take": "t2", "index": 1, "text": "I'm the boss here."}])
    assert saved == 2 and voice.num_recorded() == 2
    assert take(voice, "t1")["segments"][0]["saved"] and not take(voice, "t1")["segments"][1].get("saved")


def test_show_hints_from_paths():
    ids, names = identify._show_hints(["The Sopranos/Season 1/S01E01 tt0141842.mkv", "clip.mp4"])
    assert ids == ["tt0141842"] and names == ["The Sopranos"]


def test_generic_folders_are_not_show_names():
    _, names = identify._show_hints(["Season 1/ep1.mp4", "Downloads/Disc 2/x.mkv", "Videos/The Wire/S01/e1.mkv"])
    assert names == ["The Wire"]


def test_show_from_file_metadata_and_names():
    ids, names = identify._show_hints(
        ["Season 01/S01E01.mkv", "Season 01/The.Sopranos.S01E02.720p.mkv"],
        {"Season 01/S01E01.mkv": {"title": "The Sopranos - S01E01 - Pilot", "comment": "imdb tt0705276"}},
    )
    assert names == ["The Sopranos"] and ids == ["tt0705276"]
