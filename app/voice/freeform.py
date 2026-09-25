"""Freeform recording: one long take -> Whisper -> sentence-sized clips to review.

voices/<name>/freeform/<take id>/
  source.<ext>   what was recorded or uploaded (audio or video)
  audio.wav      22050 Hz mono copy the clips are cut from
  take.json      {"state", "detail", "error", "duration", "created", "denoise",
                  "segments": [{"start", "end", "text", "speaker"?}], "speakers"?}

Piper trains on short clips (about 1-15 s) with exact transcripts, so the take
is split at sentence ends and pauses using Whisper's word timestamps. With
"diarize" (movies, interviews, podcasts) every clip is also assigned to a
speaker so only one person's clips are kept. Nothing becomes a recording until
it is reviewed and saved.
"""

import asyncio
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..errors import explain_failure, friendly_ffmpeg_error
from ..lab import stt
from ..settings import store as settings_store
from .voices import Voice

_LOGGER = logging.getLogger(__name__)

CLIP_RATE = 22050
MIN_CLIP = 1.0        # seconds; shorter clips are dropped
SOFT_MAX_CLIP = 10.0  # past this, cut at the next short pause
MAX_CLIP = 15.0       # never longer
PAUSE = 0.7           # a pause this long always ends a clip
SHORT_PAUSE = 0.25
MAX_UPLOAD_BYTES = 8 * 2**30
GROUP = "freeform"
_REPO_DIR = Path(__file__).resolve().parents[2]

# Noise reduction, applied per clip when it is played or saved (the original
# audio is kept). A TTS voice learns whatever is in its training audio, so
# "light" only removes steady noise; "strong" (RNNoise, a speech denoiser)
# also removes changing noise like music, traffic and crowds.
RNNOISE_MODEL = Path(os.environ.get("RNNOISE_MODEL", "/opt/models/sh.rnnn"))
DENOISE_FILTERS = {
    "light": "highpass=f=80,afftdn=nr=24:nf=-30",
    "strong": f"aresample=48000,highpass=f=80,arnndn=m={RNNOISE_MODEL},aresample={CLIP_RATE}",
}
DENOISE_LEVELS = ("off", "light", "strong")
_CONTEXT = 0.5  # seconds of audio around a clip so the filters settle before it

_live: Dict[str, Dict[str, Any]] = {}  # take id -> take being processed


def _takes_dir(voice: Voice) -> Path:
    return voice.root / "freeform"


def _take_dir(voice: Voice, take_id: str) -> Path:
    path = _takes_dir(voice) / take_id
    if not take_id.replace("-", "").isalnum() or not (path / "take.json").is_file():
        raise KeyError(f"No take {take_id}")
    return path


def _read(take_dir: Path) -> Dict[str, Any]:
    return _live.get(take_dir.name) or json.loads((take_dir / "take.json").read_text(encoding="utf-8"))


def _write(take_dir: Path, take: Dict[str, Any]) -> None:
    tmp = take_dir / "take.json.tmp"
    tmp.write_text(json.dumps(take, indent=1), encoding="utf-8")
    tmp.replace(take_dir / "take.json")


def _level(value: Any) -> str:
    if value is True:
        return "light"
    if value in (False, None, ""):
        return "off"
    if value not in DENOISE_LEVELS:
        raise ValueError(f"Unknown noise reduction: {value}")
    return value


def _public(take_id: str, take: Dict[str, Any]) -> Dict[str, Any]:
    if take["state"] == "running" and take_id not in _live:
        # The server restarted while this take was being processed
        take = {**take, "state": "error", "error": "Interrupted; upload or record it again"}
    return {"id": take_id, **take, "denoise": _level(take.get("denoise"))}


def list_takes(voice: Voice) -> List[Dict[str, Any]]:
    takes = []
    if _takes_dir(voice).is_dir():
        for take_dir in sorted(_takes_dir(voice).iterdir()):
            if (take_dir / "take.json").is_file():
                takes.append(_public(take_dir.name, _read(take_dir)))
    return takes


def get_take(voice: Voice, take_id: str) -> Dict[str, Any]:
    return _public(take_id, _read(_take_dir(voice, take_id)))


# ---- Splitting -------------------------------------------------------------------


def split_words(words: List[Any], duration: float, turns: bool = False) -> List[Dict[str, Any]]:
    """Group Whisper words (with .start/.end/.word) into clips.

    A clip ends at a sentence end once it is 2 s long, at any pause >= PAUSE,
    at a short pause once it is SOFT_MAX_CLIP long, and never exceeds MAX_CLIP.
    With turns=True it also ends wherever Whisper started a new segment
    (usually a change of speaker), words marked with .segment_start.
    """
    clips: List[List[Any]] = []
    current: List[Any] = []
    for word in words:
        if current:
            gap = word.start - current[-1].end
            length = current[-1].end - current[0].start
            sentence_end = current[-1].word.strip()[-1:] in ".!?…"
            if (
                gap >= PAUSE
                or (sentence_end and length >= 2.0 and gap >= 0.1)
                or (length >= SOFT_MAX_CLIP and gap >= SHORT_PAUSE)
                or word.end - current[0].start > MAX_CLIP
                or (turns and getattr(word, "segment_start", False))
            ):
                clips.append(current)
                current = []
        current.append(word)
    if current:
        clips.append(current)

    segments = []
    for i, clip in enumerate(clips):
        # Pad around the words (Whisper timings are approximate), but only up
        # to the middle of the pause so neighbouring clips never overlap
        prev_end = clips[i - 1][-1].end if i > 0 else 0.0
        next_start = clips[i + 1][0].start if i + 1 < len(clips) else duration
        start = max((prev_end + clip[0].start) / 2 if i > 0 else 0.0, clip[0].start - 0.2, 0.0)
        end = min((clip[-1].end + next_start) / 2 if i + 1 < len(clips) else duration, clip[-1].end + 0.3, duration)
        text = "".join(w.word for w in clip).strip()
        if end - start >= MIN_CLIP and text:
            segments.append({"start": round(start, 2), "end": round(end, 2), "text": text})
    return segments


# ---- Processing ------------------------------------------------------------------


def _ffmpeg(args: List[str]) -> bytes:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")
        _LOGGER.warning("ffmpeg failed (%s): %s", proc.returncode, stderr.strip()[-500:])
        raise RuntimeError(friendly_ffmpeg_error(stderr))
    return proc.stdout


class _Word:
    __slots__ = ("start", "end", "word", "segment_start")

    def __init__(self, word: Any, segment_start: bool) -> None:
        self.start, self.end, self.word = word.start, word.end, word.word
        self.segment_start = segment_start


def _transcribe(take_dir: Path, source: Path, model_id: str, language: str, turns: bool,
                progress: Callable[[str], None]) -> Dict[str, Any]:
    progress("Extracting audio…")
    wav = take_dir / "audio.wav"
    _ffmpeg(["-i", str(source), "-vn", "-ac", "1", "-ar", str(CLIP_RATE), "-c:a", "pcm_s16le", str(wav)])
    pcm = _ffmpeg(["-i", str(wav), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"])
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio) / 16000
    if duration < MIN_CLIP:
        raise RuntimeError("The recording is empty or too short")

    progress(f"Loading Whisper {model_id}…" if stt.is_cached(model_id) else f"Downloading Whisper {model_id} (first use)…")
    try:
        model = stt._load(model_id)
    except Exception as err:
        raise RuntimeError(explain_failure(f"Loading the Whisper model {model_id}", None, [str(err)])) from err
    lang = None if model_id.endswith(".en") else (language or None)
    segments, _info = model.transcribe(
        audio,
        language=lang,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,  # skip silence and non-speech
        condition_on_previous_text=False,
    )
    words: List[_Word] = []
    for segment in segments:  # a generator: transcribes as it goes
        for i, word in enumerate(segment.words or []):
            words.append(_Word(word, i == 0))
        progress(f"Transcribing… {min(99, int(segment.end / duration * 100))}%")
    return {"duration": round(duration, 1), "segments": split_words(words, duration, turns)}


async def _diarize(voice: Voice, take_dir: Path, take: Dict[str, Any], progress: Callable[[str], None]) -> None:
    from . import matching

    request = {
        "wav": str(take_dir / "audio.wav"),
        "segments": [[s["start"], s["end"]] for s in take["segments"]],
        "recordings": [str(p) for p in matching.recordings(voice)] if voice.num_recorded() >= matching.MIN_RECORDINGS else [],
        "cache_dir": str(voice.root.parent.parent / "speaker-match"),
    }
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "app.voice.diarize", cwd=str(_REPO_DIR),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps(request).encode())
    proc.stdin.close()
    result, tail = None, []
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
        elif line.startswith("LOG "):
            progress(line[len("LOG "):])
        elif line:
            tail = (tail + [line])[-5:]
    await proc.wait()
    if proc.returncode != 0 or result is None:
        _LOGGER.error("Speaker detection failed (exit %s): %s", proc.returncode, " | ".join(tail))
        raise RuntimeError(explain_failure("Finding speakers", proc.returncode, tail))

    for segment, label in zip(take["segments"], result["labels"]):
        segment["speaker"] = label
    take["speakers"] = result["speakers"]


def _model_for(voice: Voice) -> str:
    """The Whisper model from Settings, multilingual if the voice isn't English."""
    model_id = settings_store.load()["speech"]["sttModel"]
    if not voice.language.lower().startswith("en") and model_id.endswith(".en"):
        model_id = model_id[: -len(".en")]
    return model_id


def new_take(voice: Voice, filename: str) -> Path:
    """Create a take folder; returns the path to write the source file to."""
    take_id = time.strftime("%Y%m%d-%H%M%S")
    while (_takes_dir(voice) / take_id).exists():
        take_id += "x"
    take_dir = _takes_dir(voice) / take_id
    take_dir.mkdir(parents=True)
    suffix = Path(filename or "").suffix.lower()
    return take_dir / f"source{suffix if suffix and len(suffix) <= 6 else '.webm'}"


def start(voice: Voice, source: Path, denoise: Any = "light", diarize: bool = False) -> Dict[str, Any]:
    """Transcribe (and optionally diarize) the source in the background."""
    take_dir = source.parent
    take_id = take_dir.name
    size = source.stat().st_size
    if size > MAX_UPLOAD_BYTES or size < 1000:
        shutil.rmtree(take_dir, ignore_errors=True)
        raise ValueError("The file is empty" if size < 1000 else "The file is larger than 8 GB")
    if not stt.available():
        shutil.rmtree(take_dir, ignore_errors=True)
        raise RuntimeError("Speech recognition (faster-whisper) is not installed")

    model_id = _model_for(voice)
    take: Dict[str, Any] = {
        "state": "running", "detail": "Starting…", "error": None, "created": time.time(),
        "duration": None, "segments": [], "model": model_id,
        "denoise": _level(denoise), "diarize": bool(diarize),
    }
    _write(take_dir, take)
    _live[take_id] = take

    def progress(line: str) -> None:
        take["detail"] = line

    _LOGGER.info("Freeform take %s/%s started: %s (%.1f MB), whisper=%s, noise=%s, diarize=%s",
                 voice.name, take_id, source.name, size / 2**20, model_id, take["denoise"], bool(diarize))
    started = time.monotonic()

    async def run() -> None:
        try:
            result = await asyncio.to_thread(
                _transcribe, take_dir, source, model_id,
                voice.language.split("-")[0].lower(), bool(diarize), progress,
            )
            take.update(result)
            if diarize and take["segments"]:
                progress("Finding speakers…")
                await _diarize(voice, take_dir, take, progress)
            take.update(state="done", detail="")
            _LOGGER.info("Freeform take %s/%s done in %.0fs: %.0fs of audio, %d clips%s",
                         voice.name, take_id, time.monotonic() - started, take["duration"] or 0,
                         len(take["segments"]),
                         f", {len(take['speakers'])} speakers" if take.get("speakers") is not None else "")
        except Exception as err:  # shown on the page
            if isinstance(err, RuntimeError):
                _LOGGER.error("Freeform take %s/%s failed: %s", voice.name, take_id, err)
            else:
                _LOGGER.exception("Freeform take %s/%s failed", voice.name, take_id)
                err = RuntimeError(f"Unexpected error while processing: {err}")
            take.update(state="error", error=str(err))
        finally:
            if take_dir.exists():
                _write(take_dir, take)
            _live.pop(take_id, None)

    asyncio.create_task(run())
    return _public(take_id, take)


# ---- Review ---------------------------------------------------------------------------


def clip_wav(voice: Voice, take_id: str, index: int, level: Optional[str] = None) -> bytes:
    """One clip as WAV, with the take's noise reduction (or the given level)."""
    take_dir = _take_dir(voice, take_id)
    take = _read(take_dir)
    if not 0 <= index < len(take["segments"]):
        raise KeyError(f"No clip {index}")
    segment = take["segments"][index]
    level = _level(level if level is not None else take.get("denoise"))
    if level == "strong" and not RNNOISE_MODEL.exists():
        level = "light"  # RNNoise model missing (outside the Docker image)
    source = take_dir / "audio.wav"

    if level == "off":
        with wave.open(str(source), "rb") as src:
            rate = src.getframerate()
            src.setpos(min(int(segment["start"] * rate), src.getnframes()))
            frames = src.readframes(int((segment["end"] - segment["start"]) * rate))
        pcm = np.frombuffer(frames, dtype=np.int16)
    else:
        # Filter the clip with some context around it, then cut the context off
        before = min(_CONTEXT, segment["start"])
        length = segment["end"] - segment["start"]
        out = _ffmpeg([
            "-ss", f"{segment['start'] - before:.3f}", "-t", f"{length + before + _CONTEXT:.3f}",
            "-i", str(source), "-af", DENOISE_FILTERS[level],
            "-ac", "1", "-ar", str(CLIP_RATE), "-f", "s16le", "-",
        ])
        pcm = np.frombuffer(out, dtype=np.int16)
        first = int(before * CLIP_RATE)
        pcm = pcm[first: first + int(length * CLIP_RATE)]

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out_wav:
        out_wav.setnchannels(1)
        out_wav.setsampwidth(2)
        out_wav.setframerate(CLIP_RATE)
        out_wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def set_denoise(voice: Voice, take_id: str, denoise: Any) -> Dict[str, Any]:
    take_dir = _take_dir(voice, take_id)
    if take_id in _live:
        raise RuntimeError("Still processing; wait for it to finish")
    take = _read(take_dir)
    take["denoise"] = _level(denoise)
    _write(take_dir, take)
    return _public(take_id, take)


def save(voice: Voice, take_id: str, keep: List[Dict[str, Any]]) -> int:
    """Save the kept clips (with edited text) as recordings, then drop the take."""
    take_dir = _take_dir(voice, take_id)
    segments = _read(take_dir)["segments"]
    saved = 0
    for item in keep:
        index = int(item.get("index", -1))
        text = " ".join(str(item.get("text", "")).split())
        if not 0 <= index < len(segments) or not text:
            continue
        voice.save_recording(GROUP, f"{take_id}_{index:04d}", text, clip_wav(voice, take_id, index), ".wav")
        saved += 1
    _LOGGER.info("Freeform take %s/%s: saved %d of %d clips as recordings", voice.name, take_id, saved, len(segments))
    discard(voice, take_id)
    return saved


def discard(voice: Voice, take_id: str) -> None:
    if take_id in _live:
        raise RuntimeError("Still processing; wait for it to finish")
    shutil.rmtree(_take_dir(voice, take_id), ignore_errors=True)
