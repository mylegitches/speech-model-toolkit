"""Auto-detect the pretrained voice that sounds most like a voice's recordings.

The comparison itself runs in speaker_match.py (a subprocess); this module
picks the candidates and recordings, and caches the ranking per voice in
voices/<name>/voice-match.json.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .training import catalog_groups, load_checkpoint_catalog
from ..errors import explain_failure
from .voices import AUDIO_EXTENSIONS, Voice

_LOGGER = logging.getLogger(__name__)

MIN_RECORDINGS = 5
"""Fewer than this and a match isn't reliable (training falls back to the default pick)."""

MAX_RECORDINGS = 24
"""Enough to average out one-off takes; more adds time, not accuracy."""

_REPO_DIR = Path(__file__).resolve().parents[2]
_CATALOG = load_checkpoint_catalog()


def candidates(voice: Voice) -> List[Dict[str, Any]]:
    """Pretrained voices for this voice's language (any-language list if none)."""
    for group in catalog_groups(voice.language, voice.espeak_voice) + ["generic"]:
        entries = [e for e in _CATALOG.get(group, []) if e.sample_url]
        if entries:
            return [
                {"url": e.url, "sample": e.sample_url, "name": e.name, "gender": e.gender}
                for e in entries
            ]
    return []


def recordings(voice: Voice) -> List[Path]:
    """Up to MAX_RECORDINGS recordings, spread over everything recorded."""
    paths = sorted(
        p for p in voice.recordings_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if len(paths) <= MAX_RECORDINGS:
        return paths
    step = len(paths) / MAX_RECORDINGS
    return [paths[int(i * step)] for i in range(MAX_RECORDINGS)]


def _cache_file(voice: Voice) -> Path:
    return voice.root / "voice-match.json"


def cached(voice: Voice) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_cache_file(voice).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


async def find(
    voice: Voice,
    cache_dir: Path,
    log: Callable[[str], None] = lambda _line: None,
) -> Dict[str, Any]:
    """Rank this voice's candidates; returns {"results": [...best first], "used", "recordings", "at"}."""
    options = candidates(voice)
    if not options:
        raise RuntimeError("No pretrained voices to compare for this language")

    paths = recordings(voice)
    if len(paths) < MIN_RECORDINGS:
        raise RuntimeError(f"Record at least {MIN_RECORDINGS} sentences first")

    _LOGGER.info("Auto-detect for %s: %d recordings vs %d starting voices", voice.name, len(paths), len(options))
    started = time.monotonic()
    request = {
        "recordings": [str(p) for p in paths],
        "candidates": options,
        "cache_dir": str(cache_dir),
    }
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "app.voice.speaker_match",
        cwd=str(_REPO_DIR),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps(request).encode())
    proc.stdin.close()

    result = None
    tail: List[str] = []
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT ") :])
        elif line.startswith("LOG "):
            log(line[len("LOG ") :])
        elif line:
            tail = (tail + [line])[-5:]  # library output: only shown if it fails
    await proc.wait()
    if proc.returncode != 0 or result is None:
        _LOGGER.error("Auto-detect for %s failed (exit %s): %s", voice.name, proc.returncode, " | ".join(tail))
        raise RuntimeError(explain_failure("Comparing voices", proc.returncode, tail))

    results = sorted(
        (
            {**{k: o[k] for k in ("url", "name", "gender", "sample")}, "score": result["scores"][o["url"]]}
            for o in options
            if o["url"] in result["scores"]
        ),
        key=lambda r: r["score"],
        reverse=True,
    )
    if not results:
        raise RuntimeError("Could not load any sample clips to compare with")

    match = {
        "results": results,
        "used": result["used"],
        "recordings": voice.num_recorded(),
        "at": time.time(),
    }
    _cache_file(voice).write_text(json.dumps(match, indent=2), encoding="utf-8")
    _LOGGER.info("Auto-detect for %s in %.0fs: best %s (%.2f), next %s", voice.name, time.monotonic() - started,
                 results[0]["name"], results[0]["score"],
                 f"{results[1]['name']} ({results[1]['score']:.2f})" if len(results) > 1 else "none")
    return match
