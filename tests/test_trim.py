"""Trimming a review clip: the times move, words cut off leave the text, undo restores both."""

import json

import pytest

from app.voice import freeform
from app.voice.freeform import words_still_heard
from app.voice.voices import VoiceStore


def test_words_cut_off_the_front_or_back_leave_the_text():
    # A laugh and "Yeah," cut off the front: Whisper now hears the rest
    assert words_still_heard("Yeah, I know what you did.", "I know what you did", True, False) == "I know what you did."
    # The other character's "Wait-" at the end is gone
    assert words_still_heard("Get in the car. Wait-", "Get in the car.", False, True) == "Get in the car."
    # Only the trimmed side changes: a mishearing at the other end doesn't cost a word
    assert words_still_heard("Well hello there", "hello dare", True, False) == "Hello there"
    # Corrections the person made stay as written
    assert words_still_heard("Um, Chucky's back", "chuckys back", True, False) == "Chucky's back"
    # Nothing heard: leave the words alone rather than wipe them
    assert words_still_heard("Hi there", "", True, False) == "Hi there"


@pytest.fixture
def take(tmp_path, monkeypatch):
    voice = VoiceStore(tmp_path).create("v", "en-US", "male")
    take_dir = voice.root / "freeform" / "t1"
    take_dir.mkdir(parents=True)
    segments = [{"start": 1.0, "end": 3.0, "text": "Yeah, I know what you did."}]
    (take_dir / "take.json").write_text(json.dumps({"state": "done", "segments": segments}))
    monkeypatch.setattr(freeform, "_hear", lambda *a: "I know what you did")
    monkeypatch.setattr(freeform, "clip_wav", lambda *a, **k: b"RIFF")
    return voice, take_dir


def test_trim_and_undo(take):
    voice, take_dir = take
    out = freeform.trim_clip(voice, "t1", 0, front=0.25)
    assert (out["start"], out["end"], out["trimmed"], out["text"]) == (1.25, 3.0, True, "I know what you did.")
    out = freeform.trim_clip(voice, "t1", 0, back=0.25)
    assert (out["start"], out["end"]) == (1.25, 2.75)
    out = freeform.trim_clip(voice, "t1", 0, reset=True)
    assert (out["start"], out["end"], out["trimmed"], out["text"]) == (1.0, 3.0, False, "Yeah, I know what you did.")
    seg = json.loads((take_dir / "take.json").read_text())["segments"][0]
    assert "trims" not in seg


def test_undo_the_last_trim_one_step_at_a_time(take, monkeypatch):
    voice, _ = take
    freeform.trim_clip(voice, "t1", 0, front=0.25)            # "Yeah," goes
    monkeypatch.setattr(freeform, "_hear", lambda *a: "I know what you")
    out = freeform.trim_clip(voice, "t1", 0, back=0.25)       # "did." goes too: too much
    assert (out["end"], out["trims"], out["text"]) == (2.75, 2, "I know what you")
    out = freeform.trim_clip(voice, "t1", 0, undo=True)       # just the last one back
    assert (out["start"], out["end"], out["trims"], out["text"]) == (1.25, 3.0, 1, "I know what you did.")
    out = freeform.trim_clip(voice, "t1", 0, undo=True)       # and the first
    assert (out["start"], out["end"], out["trimmed"], out["text"]) == (1.0, 3.0, False, "Yeah, I know what you did.")
    with pytest.raises(ValueError):                           # nothing left to undo
        freeform.trim_clip(voice, "t1", 0, undo=True)


def test_trim_keeps_some_of_the_clip(take):
    voice, _ = take
    with pytest.raises(ValueError):
        freeform.trim_clip(voice, "t1", 0, front=1.0, back=0.75)


def test_trimming_a_saved_clip_redoes_it_in_the_dataset(take, monkeypatch):
    voice, take_dir = take
    data = json.loads((take_dir / "take.json").read_text())
    data["segments"][0].update(saved=True, savedText="Yeah, I know what you did.")
    (take_dir / "take.json").write_text(json.dumps(data))
    written = {}
    monkeypatch.setattr(voice, "save_recording", lambda group, stem, text, audio, ext: written.update(stem=stem, text=text))
    freeform.trim_clip(voice, "t1", 0, front=0.25)
    assert written == {"stem": "t1_0000", "text": "I know what you did."}
    seg = json.loads((take_dir / "take.json").read_text())["segments"][0]
    assert seg["savedText"] == "I know what you did." and seg["start"] == 1.25


def _audio(take_dir, seconds=6.0, rate=22050):
    import wave
    import numpy as np
    t = np.arange(int(seconds * rate)) / rate
    pcm = (0.3 * np.sin(2 * np.pi * 200 * t) * (t > 1.0) * (t < 3.0) * 32767).astype(np.int16)
    with wave.open(str(take_dir / "audio.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())


def test_waveform_view_around_the_clip(take):
    voice, take_dir = take
    _audio(take_dir)
    view = freeform.clip_view(voice, "t1", 0, bins=100)
    assert view["viewStart"] == 0.0 and 4.4 <= view["viewEnd"] <= 4.5   # 1.5 s either side, within the file
    assert len(view["peaks"]) == 100 and max(view["peaks"]) == 1.0
    assert (view["start"], view["end"], view["firstStart"], view["firstEnd"]) == (1.0, 3.0, 1.0, 3.0)


def test_drag_to_an_exact_cut_and_back_out_again(take, monkeypatch):
    voice, take_dir = take
    _audio(take_dir)
    out = freeform.trim_clip(voice, "t1", 0, start=1.3, end=2.9)        # dragged in: "Yeah," goes
    assert (out["start"], out["end"], out["text"]) == (1.3, 2.9, "I know what you did.")
    monkeypatch.setattr(freeform, "_hear", lambda *a: "Yeah, I know what you did")
    out = freeform.trim_clip(voice, "t1", 0, start=1.0, end=2.9)        # dragged back out
    assert out["text"] == "Yeah, I know what you did."                  # the words of the first cut
    out = freeform.trim_clip(voice, "t1", 0, start=-5, end=99)          # never past the margin
    assert (out["start"], out["end"]) == (0.0, 4.5)
    assert freeform.trim_clip(voice, "t1", 0, reset=True)["start"] == 1.0


def test_cropping_only_cuts_the_cleaned_audio(tmp_path):
    """The noise reduction runs over the whole file once: a later start cuts the same sound,
    it doesn't filter again from the new edge (which gated the first syllables)."""
    import shutil
    import wave
    import numpy as np
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    voice = VoiceStore(tmp_path).create("v", "en-US", "male")
    take_dir = voice.root / "freeform" / "t1"
    take_dir.mkdir(parents=True)
    rate = freeform.CLIP_RATE
    t = np.arange(int(5 * rate)) / rate
    rng = np.random.default_rng(0)
    audio = 0.02 * rng.normal(size=t.size) + 0.3 * np.sin(2 * np.pi * 180 * t) * (t > 1.2) * (t < 3.5)
    with wave.open(str(take_dir / "audio.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    segments = [{"start": 1.0, "end": 3.0, "text": "a"}, {"start": 1.5, "end": 3.0, "text": "a"}]
    (take_dir / "take.json").write_text(json.dumps({"state": "done", "denoise": "light", "segments": segments}))

    def pcm(index):
        import io
        with wave.open(io.BytesIO(freeform.clip_wav(voice, "t1", index)), "rb") as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    full, cropped = pcm(0), pcm(1)
    assert (take_dir / "audio.light.flac").exists()
    offset = int(0.5 * rate)
    assert np.array_equal(cropped, full[offset: offset + len(cropped)])


def test_words_heard_drops_only_words_that_are_gone():
    from app.voice.freeform import words_heard
    # someone else's "Hey!" in the middle was silenced
    assert words_heard("I told you, hey! you never listen.", "I told you you never listen") == "I told you, you never listen."
    # a mishearing ("lissen") isn't a cut: the word stays as written
    assert words_heard("You never listen.", "You never lissen") == "You never listen."
    assert words_heard("Hi there", "") == "Hi there"


def test_silencing_a_stretch(take, monkeypatch):
    import io
    import wave
    import numpy as np
    voice, take_dir = take
    _audio(take_dir)                       # a tone from 1.0 s to 3.0 s
    monkeypatch.undo()                     # the fixture fakes clip cutting: use the real one here
    monkeypatch.setattr(freeform, "_hear", lambda *a: "I know what you did")
    data = json.loads((take_dir / "take.json").read_text())
    data["denoise"] = "off"
    (take_dir / "take.json").write_text(json.dumps(data))

    def pcm():
        with wave.open(io.BytesIO(freeform.clip_wav(voice, "t1", 0)), "rb") as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), w.getframerate()

    out = freeform.trim_clip(voice, "t1", 0, silence=[1.5, 2.0])
    assert out["silence"] == [[1.5, 2.0]] and out["trims"] == 1
    assert out["text"] == "I know what you did."      # "Yeah," no longer heard
    audio, rate = pcm()
    inside = audio[int(0.55 * rate): int(0.95 * rate)]   # 1.55-1.95 s of the file
    outside = audio[int(0.1 * rate): int(0.4 * rate)]    # 1.1-1.4 s
    assert np.abs(inside).max() == 0 and np.abs(outside).max() > 5000
    freeform.trim_clip(voice, "t1", 0, undo=True)
    audio, rate = pcm()
    assert np.abs(audio[int(0.55 * rate): int(0.95 * rate)]).max() > 5000   # sound back
    with pytest.raises(ValueError):
        freeform.trim_clip(voice, "t1", 0, silence=[1.5, 1.505])           # too short to mean it
