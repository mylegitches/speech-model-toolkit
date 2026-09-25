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
