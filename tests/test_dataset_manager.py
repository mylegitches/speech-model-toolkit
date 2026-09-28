"""Clips in a dataset: list, correct, delete (app/voice/dataset.py)."""

import io
import wave

import pytest

from app.voice import dataset, freeform
from app.voice.voices import Voice


def wav_bytes(seconds=0.5, rate=22050):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(seconds * rate))
    return buffer.getvalue()


@pytest.fixture
def voice(tmp_path):
    v = Voice(name="tony", language="en-US", espeak_voice="en-us", root=tmp_path)
    v.save_recording("general", "0001", "Hello there.", wav_bytes(), ".wav")
    v.save_recording("freeform", "t1_0003", "I'm the boss here.", wav_bytes(1.0), ".wav")
    take_dir = tmp_path / "freeform" / "t1"
    take_dir.mkdir(parents=True)
    freeform._write(take_dir, {"state": "done", "segments": [
        {"start": 0, "end": 1, "text": f"line {i}", "saved": i == 3} for i in range(4)]})
    return v


def take(voice):
    return freeform._read(voice.root / "freeform" / "t1")


def test_list_clips(voice):
    clips = dataset.list_clips(voice)
    assert [(c["id"], c["text"], c["seconds"]) for c in clips] == [
        ("freeform/t1_0003", "I'm the boss here.", 1.0), ("general/0001", "Hello there.", 0.5)]


def test_correcting_a_clip_updates_its_file_too(voice):
    assert dataset.set_text(voice, "freeform/t1_0003", " I'm  the boss! ") == "I'm the boss!"
    assert (voice.recordings_dir / "freeform" / "t1_0003.txt").read_text(encoding="utf-8") == "I'm the boss!"
    assert take(voice)["segments"][3]["savedText"] == "I'm the boss!"
    with pytest.raises(ValueError):
        dataset.set_text(voice, "general/0001", "   ")
    with pytest.raises(KeyError):
        dataset.set_text(voice, "../etc/0001", "x")


def test_deleting_clips(voice):
    assert dataset.delete_clips(voice, ["freeform/t1_0003", "general/nope"]) == 1
    assert voice.num_recorded() == 1
    assert take(voice)["segments"][3]["saved"] is False  # the review shows it as not saved


def test_delete_dataset_keeps_models(voice):
    (voice.root / "exports" / "epoch_10").mkdir(parents=True)
    dataset.delete_dataset(voice)
    assert voice.num_recorded() == 0 and not (voice.root / "freeform").exists()
    assert (voice.root / "exports" / "epoch_10").is_dir()
