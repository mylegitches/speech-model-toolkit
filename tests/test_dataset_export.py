"""Exporting a voice's dataset as an LJSpeech zip, and importing it again."""

import io
import shutil
import subprocess
import wave
import zipfile

import pytest

from app.voice.voices import Voice


def wav_bytes(seconds=0.5, rate=22050):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(seconds * rate))
    return buffer.getvalue()


def make_voice(root, name="me"):
    root.mkdir()
    return Voice(name=name, language="en-US", espeak_voice="en-us", root=root)


def test_export_then_import_round_trip(tmp_path):
    voice = make_voice(tmp_path / "a")
    voice.save_recording("general", "0001", "Hello there.", wav_bytes(), ".wav")
    voice.save_recording("freeform", "take-3", "Pipes | and\nnew lines.", wav_bytes(), ".wav")
    (voice.recordings_dir / "general" / "0002.txt").write_text("No audio for this one.", encoding="utf-8")

    dest = tmp_path / "out.zip"
    assert voice.export_zip(dest) == 2
    with zipfile.ZipFile(dest) as archive:
        names = set(archive.namelist())
        metadata = archive.read("metadata.csv").decode()
    assert {"metadata.csv", "wavs/general_0001.wav", "wavs/freeform_take-3.wav"} <= names
    assert "general_0001|Hello there." in metadata
    assert "Pipes | and new lines." in metadata  # newline flattened

    other = make_voice(tmp_path / "b", "other")
    assert other.import_zip(dest.read_bytes()) == 2
    assert other.num_recorded() == 2


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_non_wav_clips_are_converted(tmp_path):
    voice = make_voice(tmp_path / "a")
    src = tmp_path / "clip.ogg"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=d=0.5", str(src)], check=True)
    voice.save_recording("general", "0001", "Converted.", src.read_bytes(), ".ogg")
    dest = tmp_path / "out.zip"
    voice.export_zip(dest)
    with zipfile.ZipFile(dest) as archive, wave.open(io.BytesIO(archive.read("wavs/general_0001.wav"))) as w:
        assert w.getframerate() == 22050 and w.getnchannels() == 1


def test_empty_dataset_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="no clips"):
        make_voice(tmp_path / "a").export_zip(tmp_path / "out.zip")
