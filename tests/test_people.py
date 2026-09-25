"""People are recognised from file to file (app/voice/speakers.py)."""

from types import SimpleNamespace

import numpy as np
import pytest

from app.voice import speakers

RNG = np.random.default_rng(0)
ALICE, BOB, CAROL = (RNG.normal(size=192) for _ in range(3))


def fingerprint(base, noise=0.35):
    """The same person in another recording: close to their base vector."""
    v = base / np.linalg.norm(base) + RNG.normal(size=192) * noise / np.sqrt(192)
    return (v / np.linalg.norm(v)).tolist()


def take(*people_seconds):
    return {"speakers": [
        {"id": i, "centroid": fingerprint(base), "seconds": sec, "samples": [i]}
        for i, (base, sec) in enumerate(people_seconds)
    ]}


@pytest.fixture
def voice(tmp_path):
    return SimpleNamespace(root=tmp_path, name="v")


def test_same_people_across_files_and_new_ones(voice):
    first = take((ALICE, 60), (BOB, 40))
    speakers.link(voice, "t1", first)
    alice, bob = (s["person"] for s in first["speakers"])
    assert alice != bob and all(s["newPerson"] for s in first["speakers"])

    # Next episode: Bob speaks first, Carol is new
    second = take((BOB, 30), (CAROL, 20), (ALICE, 10))
    speakers.link(voice, "t2", second)
    got = [s["person"] for s in second["speakers"]]
    assert got[0] == bob and got[2] == alice
    assert got[1] not in (alice, bob) and second["speakers"][1]["newPerson"]

    people = {p["id"]: p for p in speakers.public(voice)["people"]}
    assert len(people) == 3
    assert people[alice]["takes"] == ["t1", "t2"] and people[alice]["seconds"] == 70
    assert "centroid" not in people[alice]  # fingerprints stay on the server


def test_one_person_per_speaker_in_a_take(voice):
    speakers.link(voice, "t1", take((ALICE, 60)))
    # Two speakers in a take both resembling Alice: only the closer one is her
    twin = take((ALICE, 30), (ALICE, 20))
    speakers.link(voice, "t2", twin)
    assert len({s["person"] for s in twin["speakers"]}) == 2


def test_rename_target_and_merge(voice):
    t = take((ALICE, 60), (BOB, 40))
    speakers.link(voice, "t1", t)
    alice, bob = (s["person"] for s in t["speakers"])
    speakers.update(voice, alice, name="  Alice   Smith ", target=True)
    data = speakers.public(voice)
    assert data["target"] == alice
    assert next(p for p in data["people"] if p["id"] == alice)["name"] == "Alice Smith"

    retagged = []
    speakers.merge(voice, alice, bob, lambda *args: retagged.append(args))
    data = speakers.public(voice)
    assert [p["id"] for p in data["people"]] == [bob] and data["target"] == bob
    assert retagged == [("t1", alice, bob)]

    with pytest.raises(ValueError):
        speakers.update(voice, bob, name="   ")
    with pytest.raises(KeyError):
        speakers.update(voice, "p99", name="x")
