"""Measure every clip of a character in the reviewed files of a dataset (read only).

    python -m scripts.quality_report <voice> [character] [--out /data/quality]

Writes one JSON line per clip to <out>/<voice>.jsonl (resumes where it stopped), with
the person's decision as the label: saved = kept; confirmed but not saved in a file they
reviewed = left out. quality_eval.py turns that into a report.
"""

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly

from app.lab import stt
from app.voice import freeform, quality
from app.voice.main import store
from app.voice.speaker_match import load_embedder


def clip(voice, take_id, index, denoise):
    audio, sr = sf.read(io.BytesIO(freeform.clip_wav(voice, take_id, index, denoise)), dtype="float32")
    return (audio.mean(axis=1) if audio.ndim > 1 else audio), sr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("voice")
    parser.add_argument("character", nargs="?")
    parser.add_argument("--out", default="/data/quality")
    parser.add_argument("--all", action="store_true", help="every clip of the character, not only confirmed or kept ones")
    parser.add_argument("--threads", type=int, default=0, help="limit CPU threads (0 = all)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    voice = store.get(args.voice)
    takes = []
    for take_dir in sorted((voice.root / "freeform").iterdir()):
        try:
            take = json.loads((take_dir / "take.json").read_text())
        except (OSError, ValueError):
            continue
        if take.get("state") == "done" and any(s.get("saved") for s in take.get("segments", [])):
            takes.append((take_dir.name, take))  # reviewed: has clips the person chose
    character = args.character or Counter(
        s.get("character") for _, t in takes for s in t["segments"] if s.get("saved")).most_common(1)[0][0]
    print(f"{voice.name}: {len(takes)} reviewed files, character {character}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{voice.name}.jsonl"
    # Measured before, with the noise removal the file still uses (else measured again)
    denoise = {tid: t.get("denoise") for tid, t in takes}
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if denoise.get(row["take"]) == row.get("denoise"):
                done.add((row["take"], row["index"]))

    todo = [(tid, t, i, s) for tid, t in takes for i, s in enumerate(t["segments"])
            if s.get("character") == character and (args.all or s.get("confirmed") or s.get("saved"))
            and (tid, i) not in done]
    print(f"{len(todo)} clips to measure ({len(done)} done before)", flush=True)

    embed = load_embedder(Path("/data/voice/speaker-match"))
    whisper = {}
    squim = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval()

    with path.open("a") as sink:
        for n, (take_id, take, index, seg) in enumerate(todo, 1):
            try:
                heard, sr = clip(voice, take_id, index, None)      # with the file's noise removal
                raw, _ = clip(voice, take_id, index, "off")         # without
            except Exception as err:  # noqa: BLE001 - a broken clip is skipped, not fatal
                print(f"skip {take_id}/{index}: {err}", flush=True)
                continue
            heard16 = resample_poly(heard, 16000, sr).astype(np.float32)
            raw16 = resample_poly(raw, 16000, sr).astype(np.float32)
            text = seg.get("savedText") or seg.get("text", "")
            model_id = take.get("model") or "small.en"
            if model_id not in whisper:
                if args.threads:
                    from faster_whisper import WhisperModel
                    whisper[model_id] = WhisperModel(model_id, device="cpu", compute_type="int8",
                                                     cpu_threads=args.threads, download_root=str(stt.MODELS_DIR))
                else:
                    whisper[model_id] = stt._load(model_id)
            row = {
                "take": take_id, "index": index, "file": take.get("name", ""), "text": text,
                "saved": bool(seg.get("saved")), "confirmed": bool(seg.get("confirmed")),
                "conf": seg.get("conf"), "denoise": take.get("denoise"), "seconds": len(heard) / sr,
                **quality.level(heard, sr), **quality.edges(heard, sr), **quality.holes(heard, raw, sr),
                **quality.recheck_words(whisper[model_id], heard16, text,
                                        None if model_id.endswith(".en") else voice.language.split("-")[0]),
                "vector": embed(raw16).round(5).tolist(),
                "halves": quality.consistency(embed, raw16),
            }
            with torch.no_grad():
                stoi, pesq, sisdr = squim(torch.from_numpy(heard16)[None])
            row.update(stoi=float(stoi), pesq=float(pesq), sisdr=float(sisdr))
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if n % 50 == 0 or n == len(todo):
                print(f"{n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
