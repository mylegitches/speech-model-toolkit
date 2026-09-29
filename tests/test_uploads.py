"""Resumable uploads: pieces in order, a repeated or missing piece, finishing."""

import pytest

from app.voice import uploads
from app.voice.voices import Voice


def _voice(tmp_path):
    return Voice(name="v", language="en_US", espeak_voice="en-us", root=tmp_path)


def test_pieces_resume_and_finish(tmp_path):
    voice = _voice(tmp_path)
    data = bytes(range(256)) * 40
    upload = uploads.create(voice, "episode.mkv", len(data))
    assert uploads.append(voice, upload["id"], 0, data[:4000]) == 4000
    # The same piece again (its answer was lost): the right offset comes back
    with pytest.raises(ValueError) as err:
        uploads.append(voice, upload["id"], 0, data[:4000])
    assert err.value.args[0] == 4000
    # Resuming later: the server says how much it has
    assert uploads.status(voice, upload["id"])["received"] == 4000
    with pytest.raises(ValueError):  # not all there yet
        uploads.finish(voice, upload["id"])
    uploads.append(voice, upload["id"], 4000, data[4000:])
    with pytest.raises(OverflowError):
        uploads.append(voice, upload["id"], len(data), b"x")
    part = uploads.finish(voice, upload["id"])
    assert part.read_bytes() == data
    with pytest.raises(KeyError):  # finished: gone
        uploads.status(voice, upload["id"])


def test_bad_ids(tmp_path):
    voice = _voice(tmp_path)
    for bad in ("../x", "", "0" * 32):
        with pytest.raises(KeyError):
            uploads.status(voice, bad)
