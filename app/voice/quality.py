"""Clip quality checks: the measurements behind an automatically built, clean dataset.

Each check looks at one way a clip from a film or TV episode goes wrong:

  speaker      the clip's voice fingerprint is far from the character's own voice
  consistency  the first and second half sound like different people (overlap, a cut)
  words        Whisper, run again on the clip alone, doesn't hear the same words, or
               isn't sure of some of them
  level        too quiet next to its own background (low signal-to-noise), or clipped
  edges        speech running into the very start or end (a word cut off)
  holes        noise removal wiped out parts of the voice ("in and out")
  quality      a speech-quality model (torchaudio SQUIM: STOI, PESQ, SI-SDR estimates)

All of these are measurements only; quality_report.py compares them with the clips a
person kept or left out, to pick thresholds.
"""

import difflib
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

FRAME = 0.02  # seconds


def frame_db(audio: np.ndarray, sr: int) -> np.ndarray:
    n = max(1, int(sr * FRAME))
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    return 10 * np.log10((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-10)


def level(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Speech level (loud frames), background (quiet frames), their difference, clipping."""
    db = frame_db(audio, sr)
    if not len(db):
        return {"level_db": -100.0, "floor_db": -100.0, "snr_db": 0.0, "clipped": 0.0}
    speech, floor = np.percentile(db, 90), np.percentile(db, 10)
    return {
        "level_db": float(speech), "floor_db": float(floor), "snr_db": float(speech - floor),
        "clipped": float((np.abs(audio) > 0.98).mean()),
    }


def edges(audio: np.ndarray, sr: int, edge: float = 0.04) -> Dict[str, float]:
    """How loud the first and last 40 ms are next to the speech: high = cut mid-sound."""
    db = frame_db(audio, sr)
    if len(db) < 6:
        return {"edge_start_db": 0.0, "edge_end_db": 0.0}
    speech = np.percentile(db, 90)
    k = max(1, int(edge / FRAME))
    return {"edge_start_db": float(db[:k].max() - speech), "edge_end_db": float(db[-k:].max() - speech)}


def holes(heard: np.ndarray, raw: np.ndarray, sr: int) -> Dict[str, float]:
    """Noise removal damage: share of frames with sound where it took out 12 dB or more."""
    a, b = frame_db(heard, sr), frame_db(raw, sr)
    m = min(len(a), len(b))
    if not m:
        return {"holes": 0.0, "removed_db": 0.0}
    a, b = a[:m], b[:m]
    sound = b > b.max() - 30
    change = a[sound] - b[sound]
    return {"holes": float((change < -12).mean()), "removed_db": float(-np.median(change))}


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def words_match(expected: str, heard: str) -> float:
    """1.0 = the same words, 0 = nothing in common."""
    a, b = _words(expected), _words(heard)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def recheck_words(model: Any, audio16k: np.ndarray, text: str, language: Optional[str]) -> Dict[str, float]:
    """Transcribe the clip on its own; compare with its text and keep Whisper's confidence."""
    segments, _ = model.transcribe(audio16k, language=language, beam_size=5, word_timestamps=True,
                                   condition_on_previous_text=False, vad_filter=False)
    heard, probs = [], []
    for segment in segments:
        heard.append(segment.text)
        probs += [w.probability for w in (segment.words or [])]
    return {
        "words_match": words_match(text, " ".join(heard)),
        "word_prob_min": float(min(probs)) if probs else 0.0,
        "word_prob_mean": float(np.mean(probs)) if probs else 0.0,
    }


def consistency(embed: Callable[[np.ndarray], np.ndarray], audio16k: np.ndarray) -> Optional[float]:
    """Voice similarity of the clip's two halves (None if too short to tell)."""
    if len(audio16k) < 16000 * 2.0:
        return None
    half = len(audio16k) // 2
    return float(embed(audio16k[:half]) @ embed(audio16k[half:]))


def robust_centroid(vectors: np.ndarray, keep: float = 0.7, rounds: int = 3) -> np.ndarray:
    """The character's voice: the mean of their clips, re-taken over the most typical ones
    (so mislabelled clips don't pull it off)."""
    centre = vectors.mean(axis=0)
    for _ in range(rounds):
        centre /= np.linalg.norm(centre) + 1e-9
        sims = vectors @ centre
        best = sims >= np.quantile(sims, 1 - keep)
        centre = vectors[best].mean(axis=0)
    return centre / (np.linalg.norm(centre) + 1e-9)
