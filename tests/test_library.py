"""Media library: browsing stays inside the library, picks expand to media files, a
take links to the file (never copies or changes it), copies of a dataset keep the link."""

import os

import pytest

from app.voice import dataset, library
from app.voice.voices import VoiceStore


@pytest.fixture
def media(tmp_path, monkeypatch):
    tv = tmp_path / "tv"
    season = tv / "The Sopranos" / "Season 1"
    season.mkdir(parents=True)
    for name in ("S01E10 A Hit Is a Hit.mkv", "S01E02 46 Long.mkv", "S01E01 Pilot.mkv", "notes.txt", ".hidden.mkv"):
        (season / name).write_bytes(b"x" * 2000)
    (tv / "The Sopranos" / "Season 2").mkdir()
    (tv / "The Sopranos" / "Season 2" / "S02E01.avi").write_bytes(b"y" * 3000)
    outside = tmp_path / "secret.mkv"
    outside.write_bytes(b"z" * 2000)
    (season / "escape.mkv").symlink_to(outside)  # a link out of the library
    monkeypatch.setitem(library.ROOTS, "tv", ("TV", tv))
    monkeypatch.setitem(library.ROOTS, "movies", ("Movies", tmp_path / "not-mounted"))
    store = VoiceStore(tmp_path / "voices")
    return store.create("v", "en_US", "male"), store, tv


def test_status(media):
    status = {lib["id"]: lib["mounted"] for lib in library.status()}
    assert status == {"tv": True, "movies": False}


def test_stays_inside_the_library(media):
    voice, _, _ = media
    for bad in ("..", "../..", "The Sopranos/../../", "/etc"):
        with pytest.raises(KeyError):
            library.browse(voice, "tv", bad) if bad != "/etc" else library.source_for("tv", "../secret.mkv")
    with pytest.raises(KeyError):
        library.browse(voice, "music", "")


def test_browse_sorts_episodes_and_hides_the_rest(media):
    voice, _, _ = media
    root = library.browse(voice, "tv", "")
    assert [f["name"] for f in root["folders"]] == ["The Sopranos"] and root["path"] == ""
    season = library.browse(voice, "tv", "The Sopranos/Season 1")
    names = [f["name"] for f in season["files"]]
    # natural order, media only, no hidden files, no links out of the library
    assert names == ["S01E01 Pilot.mkv", "S01E02 46 Long.mkv", "S01E10 A Hit Is a Hit.mkv"]


def test_expand_and_link_without_copying(media):
    voice, store, tv = media
    files = library.expand(voice, "tv", ["The Sopranos"])
    assert [f["path"] for f in files] == [
        "The Sopranos/Season 1/S01E01 Pilot.mkv", "The Sopranos/Season 1/S01E02 46 Long.mkv",
        "The Sopranos/Season 1/S01E10 A Hit Is a Hit.mkv", "The Sopranos/Season 2/S02E01.avi"]
    assert not any(f["added"] for f in files)

    # What the add endpoint does: the take's source is a link to the library file
    real = library.source_for("tv", files[0]["path"])
    take = voice.root / "freeform" / "t1"
    take.mkdir(parents=True)
    os.symlink(real, take / "source.mkv")
    assert library.expand(voice, "tv", [files[0]["path"]])[0]["added"]
    assert library.browse(voice, "tv", "The Sopranos/Season 1")["files"][0]["added"]

    # Copying the dataset links to the library file too: never a copy of the video
    copy = store.create("fork", "en_US", "male")
    dataset.copy_dataset(voice, copy)
    linked = copy.root / "freeform" / "t1" / "source.mkv"
    assert linked.is_symlink() and os.path.realpath(linked) == str(real)

    # Removing a take removes the link, never the media file
    import shutil
    shutil.rmtree(take)
    assert real.read_bytes() == b"x" * 2000
