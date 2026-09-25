"""Who is speaking in each clip of a take (speaker diarization).

Run as a subprocess, like speaker_match.py:

    python -m app.voice.diarize < request.json

Request: {"wav": 22050 Hz take audio, "segments": [[start, end], ...],
          "recordings": [paths of this voice's recordings], "cache_dir": dir}
Prints "LOG <text>" progress lines, then "RESULT {json}":
  {"labels": [speaker id per segment, -1 = too little speech to tell],
   "speakers": [{"id", "seconds", "clips", "samples": [segment indexes],
                 "similarity": to the voice's recordings or null}, ...]}  (most speech first)

Each clip gets an ECAPA speaker embedding; clips are grouped by average-linkage
clustering on cosine distance. Clips are already split at pauses and Whisper
segment boundaries, so almost all of them hold one speaker.
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np

from .speaker_match import load_embedder, log, voice_embedding

SAME_SPEAKER = 0.6
"""Cosine distance below which clips are grouped (similarity above 0.4)."""

MIN_SPEAKER_SECONDS = 8.0
"""Groups with less speech than this are too small to train on or to trust."""

MAX_EMBED_SECONDS = 10.0


def main() -> None:
    request = json.load(sys.stdin)
    cache_dir = Path(request["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    segments = request["segments"]

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.signal import resample_poly

    embed = load_embedder(cache_dir)

    vectors = []
    with wave.open(request["wav"], "rb") as source:
        rate = source.getframerate()
        for i, (start, end) in enumerate(segments):
            source.setpos(min(int(start * rate), source.getnframes()))
            frames = source.readframes(int(min(end - start, MAX_EMBED_SECONDS) * rate))
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            vectors.append(embed(resample_poly(audio, 320, 441).astype(np.float32)))
            if (i + 1) % 25 == 0 or i + 1 == len(segments):
                log(f"Finding speakers… {i + 1}/{len(segments)} clips")
    vectors = np.stack(vectors)

    if len(vectors) == 1:
        raw_labels = np.array([1])
    else:
        raw_labels = fcluster(
            linkage(vectors, method="average", metric="cosine"), t=SAME_SPEAKER, criterion="distance"
        )

    durations = np.array([end - start for start, end in segments])
    groups = []
    for label in set(raw_labels.tolist()):
        members = np.flatnonzero(raw_labels == label)
        seconds = float(durations[members].sum())
        if seconds >= MIN_SPEAKER_SECONDS:
            groups.append((seconds, members))
    groups.sort(key=lambda g: -g[0])

    mine = voice_embedding(request.get("recordings") or [], embed) if request.get("recordings") else None

    labels = [-1] * len(segments)
    speakers = []
    for speaker_id, (seconds, members) in enumerate(groups):
        for i in members:
            labels[int(i)] = speaker_id
        centroid = vectors[members].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        # Samples: the most typical clips of a comfortable length
        typical = sorted(members, key=lambda i: -float(vectors[i] @ centroid))
        good = [int(i) for i in typical if 2.0 <= durations[i] <= 8.0] or [int(i) for i in typical]
        speakers.append({
            "id": speaker_id,
            "seconds": round(seconds, 1),
            "clips": int(len(members)),
            "samples": good[:3],
            "similarity": None if mine is None else round(float(centroid @ mine), 3),
        })

    log(f"Found {len(speakers)} speaker(s)")
    print("RESULT " + json.dumps({"labels": labels, "speakers": speakers}), flush=True)


if __name__ == "__main__":
    main()
