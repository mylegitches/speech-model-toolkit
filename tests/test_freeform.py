"""How a freeform take is split into training clips."""

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
from app.voice.freeform import MAX_CLIP, MIN_CLIP, split_words  # noqa: E402


def words(*items):
    """items: (word, start, end)"""
    return [SimpleNamespace(word=w, start=s, end=e) for w, s, e in items]


def test_splits_at_sentence_ends():
    ws = words((" Hello", 0.0, 0.5), (" there,", 0.6, 1.0), (" my", 1.1, 1.3), (" friend.", 1.4, 2.2),
               (" How", 2.5, 2.8), (" are", 2.9, 3.1), (" you", 3.2, 3.4), (" doing", 3.5, 3.9), (" today?", 4.0, 4.8))
    clips = split_words(ws, 5.0)
    assert [c["text"] for c in clips] == ["Hello there, my friend.", "How are you doing today?"]
    # Padded, but never overlapping the neighbour
    assert clips[0]["end"] <= clips[1]["start"]


def test_long_pause_ends_a_clip_even_mid_sentence():
    ws = words((" one", 0.0, 0.6), (" two", 0.7, 1.4), (" three", 2.5, 3.2), (" four", 3.3, 4.0))
    assert [c["text"] for c in split_words(ws, 4.5)] == ["one two", "three four"]


def test_never_longer_than_max_and_drops_tiny_clips():
    ws = words(*[(f" w{i}", i * 0.5, i * 0.5 + 0.45) for i in range(60)])  # 30 s, no pauses
    clips = split_words(ws, 30.0)
    assert all(c["end"] - c["start"] <= MAX_CLIP + 0.6 for c in clips)
    assert all(c["end"] - c["start"] >= MIN_CLIP for c in clips)
    assert " ".join(c["text"] for c in clips).split()[0] == "w0"


def test_short_utterance_dropped():
    assert split_words(words((" Hm.", 0.0, 0.3)), 0.5) == []


def test_turns_split_at_whisper_segments():
    ws = words((" Where", 0.0, 0.4), (" are", 0.45, 0.7), (" you", 0.75, 1.0), (" going", 1.05, 1.6),
               (" Home,", 1.8, 2.3), (" I", 2.35, 2.5), (" think", 2.55, 3.1))
    ws[4].segment_start = True  # a new Whisper segment: the other person answers
    assert len(split_words(ws, 3.5)) == 1          # no pause, no sentence end: one clip
    assert [c["text"] for c in split_words(ws, 3.5, turns=True)] == ["Where are you going", "Home, I think"]


def track(index, language="", title="", default=False, channels=2):
    return {"index": index, "language": language, "title": title, "default": default,
            "channels": channels, "surround": channels >= 5}


def test_recommended_track_prefers_voice_language_over_commentary():
    from app.voice.freeform import recommended_track
    tracks = [track(0, "spa", default=True, channels=6), track(1, "eng", "Director's commentary"),
              track(2, "eng", channels=6), track(3, "fre")]
    assert recommended_track(tracks, "en-US") == 2
    assert recommended_track(tracks, "es-ES") == 0
    assert recommended_track(tracks, "fr-FR") == 3
    # Unknown language: the file's default track
    assert recommended_track([track(0), track(1, default=True)], "de-DE") == 1


def sound(duration, voiced, rate=16000):
    """Silence (with faint hiss) except the (start, end) stretches, which are loud."""
    import numpy as np
    audio = np.random.default_rng(0).normal(0, 0.002, int(duration * rate)).astype(np.float32)
    t = np.arange(len(audio)) / rate
    for start, end in voiced:
        span = (t >= start) & (t < end)
        audio[span] += 0.3 * np.sin(2 * np.pi * 180 * t[span])
    return audio


def test_cut_follows_the_sound_not_whispers_early_word_end():
    from app.voice.freeform import loudness
    # "...that's it." really ends at 2.9 s, but Whisper says 2.5; the next line starts at 3.3
    ws = words((" Well,", 0.2, 0.6), (" that's", 0.7, 1.4), (" it.", 1.5, 2.5),
               (" Who", 3.3, 3.6), (" are", 3.7, 3.9), (" you?", 4.0, 4.6))
    audio = sound(5.0, [(0.2, 2.9), (3.3, 4.6)])
    clips = split_words(ws, 5.0, db=loudness(audio, 16000))
    assert [c["text"] for c in clips] == ["Well, that's it.", "Who are you?"]
    assert clips[0]["end"] >= 2.9          # the end of "it" stays in its clip
    assert clips[1]["start"] >= 2.9        # and doesn't spill into the next one
    assert clips[0]["end"] <= clips[1]["start"]
    # Without the sound, the old cut lands at 2.5 + 0.3 = 2.8: the tail of the word is lost
    old = split_words(ws, 5.0)
    assert old[0]["end"] < 2.9


def test_no_split_at_a_sentence_end_without_a_real_pause():
    from app.voice.freeform import loudness
    # Whisper puts a full stop and a 0.15 s gap, but the voice runs straight on
    ws = words((" I", 0.0, 0.3), (" know", 0.4, 1.0), (" that.", 1.1, 2.2),
               (" Really", 2.35, 2.9), (" do.", 3.0, 3.6))
    audio = sound(4.0, [(0.0, 3.6)])
    clips = split_words(ws, 4.0, db=loudness(audio, 16000))
    assert len(clips) == 1 and clips[0]["text"] == "I know that. Really do."
