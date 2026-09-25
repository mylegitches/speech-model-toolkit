"""Rank pretrained Piper voices by how much they sound like your recordings.

Run as a subprocess (it loads PyTorch + SpeechBrain, which the web server
shouldn't hold in memory):

    python -m app.voice.speaker_match < request.json

Request: {"recordings": [paths], "candidates": [{"url", "sample"}], "cache_dir": dir}
Prints progress lines as "LOG <text>", then one line "RESULT {json}" with
{"scores": {checkpoint url: cosine similarity}, "used": number of recordings}.

Speaker embeddings come from SpeechBrain's ECAPA-TDNN (VoxCeleb). Your voice is
the mean embedding of your recordings; each candidate is the embedding of its
official sample clip from piper-samples.
"""

import json
import subprocess
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

SAMPLE_RATE = 16000
MIN_SECONDS = 1.0
MAX_SECONDS = 12.0
MODEL = "speechbrain/spkrec-ecapa-voxceleb"


def log(line: str) -> None:
    print("LOG " + line, flush=True)


def decode(path: str) -> Optional[np.ndarray]:
    """Any audio file -> 16 kHz mono float32 (via ffmpeg)."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    # Trim leading/trailing near-silence so it doesn't dilute the embedding
    loud = np.flatnonzero(np.abs(audio) > 0.02)
    if loud.size:
        audio = audio[max(0, loud[0] - 1600) : loud[-1] + 1600]
    if audio.size < MIN_SECONDS * SAMPLE_RATE:
        return None
    return audio[: int(MAX_SECONDS * SAMPLE_RATE)]


def load_embedder(cache_dir: Path) -> Callable[[np.ndarray], np.ndarray]:
    """16 kHz float audio -> unit-length speaker embedding (ECAPA, CPU)."""
    warnings.filterwarnings("ignore")  # torch/speechbrain deprecation noise
    import torch
    from speechbrain.pretrained import EncoderClassifier

    log("Loading the speaker recognition model…")
    encoder = EncoderClassifier.from_hparams(
        source=MODEL,
        savedir=str(cache_dir / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"},
    )

    def embed(audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            vector = encoder.encode_batch(torch.from_numpy(audio)[None, :]).squeeze().numpy()
        return vector / (np.linalg.norm(vector) + 1e-9)

    return embed


def voice_embedding(paths: List[str], embed: Callable[[np.ndarray], np.ndarray]) -> Optional[np.ndarray]:
    """Mean embedding of a voice's recordings (None if none could be read)."""
    vectors = [embed(audio) for audio in (decode(p) for p in paths) if audio is not None]
    if not vectors:
        return None
    log(f"Analysed {len(vectors)} recordings")
    mean = np.mean(vectors, axis=0)
    return mean / (np.linalg.norm(mean) + 1e-9)


def main() -> None:
    request = json.load(sys.stdin)
    cache_dir = Path(request["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    embed = load_embedder(cache_dir)

    # Your voice
    mine = voice_embedding(request["recordings"], embed)
    if mine is None:
        raise SystemExit("None of the recordings could be read")

    # Candidates (embeddings cached by sample URL)
    cache_file = cache_dir / "sample-embeddings.json"
    try:
        cached = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        cached = {}

    scores = {}
    for candidate in request["candidates"]:
        sample = candidate["sample"]
        if sample not in cached:
            clip = cache_dir / "samples" / sample.split("/samples/", 1)[-1].replace("/", "_")
            clip.parent.mkdir(parents=True, exist_ok=True)
            if not clip.exists():
                try:
                    urllib.request.urlretrieve(sample, clip)
                except OSError as err:
                    log(f"  no sample for {candidate['url']}: {err}")
                    continue
            audio = decode(str(clip))
            if audio is None:
                continue
            cached[sample] = embed(audio).tolist()
        score = float(np.dot(mine, np.asarray(cached[sample])))
        scores[candidate["url"]] = round(score, 4)

    cache_file.write_text(json.dumps(cached))
    print("RESULT " + json.dumps({"scores": scores, "used": len(vectors)}), flush=True)


if __name__ == "__main__":
    main()
