"""Clip quality measurements on synthetic audio."""

import numpy as np

from app.voice import quality

SR = 22050


def tone(seconds, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_level_and_snr():
    quiet_noise = np.random.default_rng(0).normal(0, 0.001, SR).astype(np.float32)
    clip = np.concatenate([quiet_noise, tone(1.0), quiet_noise])
    m = quality.level(clip, SR)
    assert -15 < m["level_db"] < -10          # a 0.3 sine is about -13 dBFS
    assert m["snr_db"] > 40 and m["clipped"] == 0.0
    loud = quality.level(np.clip(tone(1.0, amp=2.0), -1, 1), SR)
    assert loud["clipped"] > 0.1


def test_edges_find_a_cut_word():
    silence = np.zeros(int(0.3 * SR), dtype=np.float32)
    clean = np.concatenate([silence, tone(1.0), silence])
    cut = np.concatenate([tone(1.0), silence])     # starts mid-sound
    assert quality.edges(clean, SR)["edge_start_db"] < -40
    assert quality.edges(cut, SR)["edge_start_db"] > -3


def test_holes_from_noise_removal():
    raw = tone(2.0)
    heard = raw.copy()
    heard[: len(heard) // 2] *= 0.01               # half the voice wiped out (-40 dB)
    assert quality.holes(raw, raw, SR)["holes"] == 0.0
    assert 0.4 < quality.holes(heard, raw, SR)["holes"] < 0.6


def test_words_match():
    assert quality.words_match("Hi, I'm Chucky!", "hi i'm chucky") == 1.0
    assert quality.words_match("wanna play", "something else entirely") < 0.3


def test_robust_centroid_ignores_outliers():
    rng = np.random.default_rng(1)
    voice = rng.normal(size=16); voice /= np.linalg.norm(voice)
    clips = voice + rng.normal(0, 0.2, (40, 16))
    others = rng.normal(size=(15, 16))                # mislabelled clips
    vectors = np.vstack([clips, others])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    centre = quality.robust_centroid(vectors)
    assert centre @ voice > 0.95
