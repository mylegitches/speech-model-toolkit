"""Freeform recording: one long take -> Whisper -> sentence-sized clips to review.

voices/<name>/freeform/<take id>/
  source.<ext>   what was recorded or uploaded (audio or video: only the sound is used)
  audio.wav      22050 Hz mono copy of the chosen audio track the clips are cut from
  take.json      {"state", "detail", "error", "duration", "created", "denoise",
                  "tracks", "track", "dialogue",
                  "segments": [{"start", "end", "text", "speaker"?}], "speakers"?}

Files with several audio tracks (movie languages, commentary) wait in state
"choose_track" until a track is picked. With surround sound (5.1/7.1) only the
center channel is used by default: that's where film dialogue is mixed, away
from the music and effects.

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
import difflib
import re
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
from . import identify
from . import speakers as people
from .voices import AUDIO_EXTENSIONS, Voice

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

# The helper scripts report their result as one JSON line; for a long video
# (thousands of clips + voice fingerprints) that's far over asyncio's 64 KB
# default line limit ("Separator is found, but chunk is longer than limit").
SUBPROCESS_LINE_LIMIT = 256 * 2**20

_live: Dict[str, Dict[str, Any]] = {}  # take id -> take being processed
_work: Optional[asyncio.Semaphore] = None  # one take at a time (Whisper is CPU-heavy)


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
        take = {**take, "state": "error", "error": "Interrupted by a restart. Press Try again to process it again."}
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


ENERGY_STEP = 0.01   # seconds per loudness reading for finding cut points


def loudness(audio: np.ndarray, rate: int) -> np.ndarray:
    """Loudness (dB) every ENERGY_STEP seconds: where the cuts between clips can go."""
    n = max(1, int(rate * ENERGY_STEP))
    frames = audio[: len(audio) // n * n].reshape(-1, n).astype(np.float64)
    db = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-10)
    # a little smoothing so one quiet sample between syllables doesn't look like a pause
    return np.convolve(db, np.ones(3) / 3, mode="same") if len(db) >= 3 else db


class _Cuts:
    """Find real pauses in the sound around Whisper's (approximate) word times."""

    def __init__(self, db: np.ndarray):
        self.db = db
        speech, floor = np.percentile(db, 90), np.percentile(db, 10)
        self.quiet = max(floor + 6, speech - 30)   # this quiet: no voice
        self.silent_enough = speech - 20           # quieter than this: an acceptable cut

    def _i(self, t: float) -> int:
        return int(min(max(t / ENERGY_STEP, 0), len(self.db) - 1))

    def valley(self, lo: float, hi: float):
        """The quietest moment between lo and hi: (time, its loudness)."""
        a, b = self._i(lo), self._i(hi)
        if b <= a:
            return lo, float(self.db[a])
        k = a + int(np.argmin(self.db[a:b + 1]))
        return k * ENERGY_STEP, float(self.db[k])

    def fade_out(self, t: float, limit: float) -> float:
        """From a word's estimated end, on until the voice has really stopped (or limit)."""
        i, stop = self._i(t), self._i(limit)
        run = 0
        while i < stop:
            run = run + 1 if self.db[i] < self.quiet else 0
            if run >= 5:              # 50 ms of quiet
                return min(limit, (i - 4) * ENERGY_STEP + 0.05)
            i += 1
        return limit

    def fade_in(self, t: float, limit: float) -> float:
        """From a word's estimated start, back until the voice really starts (or limit)."""
        i, stop = self._i(t), self._i(limit)
        run = 0
        while i > stop:
            run = run + 1 if self.db[i] < self.quiet else 0
            if run >= 5:
                return max(limit, (i + 4) * ENERGY_STEP - 0.05)
            i -= 1
        return limit


def split_words(words: List[Any], duration: float, turns: bool = False,
                db: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Group Whisper words (with .start/.end/.word) into clips.

    A clip ends at a sentence end once it is 2 s long, at any pause >= PAUSE,
    at a short pause once it is SOFT_MAX_CLIP long, and never exceeds MAX_CLIP.
    With turns=True it also ends wherever Whisper started a new segment
    (usually a change of speaker), words marked with .segment_start.

    With the audio's loudness (db, see loudness()), cuts go where the sound really
    is quiet rather than where Whisper's word times say: Whisper often ends a word
    early, and the rest of it would land in the next clip. A sentence end or short
    pause with no real quiet in it doesn't split (it's one stretch of speech).
    """
    cuts = _Cuts(db) if db is not None and len(db) > 10 else None
    clips: List[List[Any]] = []
    current: List[Any] = []
    for word in words:
        if current:
            gap = word.start - current[-1].end
            length = current[-1].end - current[0].start
            sentence_end = current[-1].word.strip()[-1:] in ".!?…"
            forced = (gap >= PAUSE or word.end - current[0].start > MAX_CLIP
                      or (turns and getattr(word, "segment_start", False)))
            optional = (sentence_end and length >= 2.0 and gap >= 0.1) or (length >= SOFT_MAX_CLIP and gap >= SHORT_PAUSE)
            if optional and not forced and cuts is not None:
                _, level = cuts.valley(current[-1].end - 0.1, word.start + 0.05)
                optional = level < cuts.silent_enough   # no real pause: keep going
            if forced or optional:
                clips.append(current)
                current = []
        current.append(word)
    if current:
        clips.append(current)

    # One cut between each pair of clips: the quietest moment between their words, looking
    # a little past Whisper's times on both sides (kept between the two words' estimates)
    bounds = []
    if cuts is not None:
        for left, right in zip(clips, clips[1:]):
            t = cuts.valley(left[-1].end - 0.1, right[0].start + 0.05)[0]
            lo, hi = sorted((left[-1].end, right[0].start))
            bounds.append(min(max(t, lo), hi))

    segments = []
    for i, clip in enumerate(clips):
        prev_end = clips[i - 1][-1].end if i > 0 else 0.0
        next_start = clips[i + 1][0].start if i + 1 < len(clips) else duration
        if cuts is None:
            # Pad around the words (Whisper timings are approximate), but only up
            # to the middle of the pause so neighbouring clips never overlap
            start = max((prev_end + clip[0].start) / 2 if i > 0 else 0.0, clip[0].start - 0.2, 0.0)
            end = min((clip[-1].end + next_start) / 2 if i + 1 < len(clips) else duration, clip[-1].end + 0.3, duration)
        else:
            before = bounds[i - 1] if i > 0 else max(0.0, clip[0].start - 0.4)
            after = bounds[i] if i + 1 < len(clips) else min(duration, clip[-1].end + 0.8)
            # Don't carry a long silence along: stop shortly after the voice fades
            start = cuts.fade_in(clip[0].start, before)
            end = cuts.fade_out(clip[-1].end, after)
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


# ISO 639-2 codes in media files -> the language prefixes voices use
_LANGUAGE_CODES = {
    "eng": "en", "spa": "es", "fra": "fr", "fre": "fr", "deu": "de", "ger": "de", "ita": "it",
    "por": "pt", "nld": "nl", "dut": "nl", "pol": "pl", "rus": "ru", "ukr": "uk", "tur": "tr",
    "ara": "ar", "hin": "hi", "zho": "zh", "chi": "zh", "jpn": "ja", "kor": "ko", "swe": "sv",
    "dan": "da", "nor": "no", "nob": "nb", "fin": "fi", "ces": "cs", "cze": "cs", "ell": "el",
    "gre": "el", "vie": "vi", "hun": "hu", "ron": "ro", "rum": "ro", "slk": "sk", "slo": "sk",
    "cat": "ca", "heb": "he", "ind": "id", "msa": "ms", "may": "ms", "srp": "sr", "hrv": "hr",
}


def probe_tracks(source: Path) -> List[Dict[str, Any]]:
    """The audio tracks in a file (via ffprobe), in stream order."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_name,channels,channel_layout:stream_tags=language,title:stream_disposition=default",
         "-of", "json", str(source)],
        capture_output=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")
        _LOGGER.warning("ffprobe failed (%s): %s", proc.returncode, stderr.strip()[-500:])
        raise ValueError(friendly_ffmpeg_error(stderr))
    tracks = []
    for n, stream in enumerate(json.loads(proc.stdout or b"{}").get("streams", [])):
        tags = stream.get("tags") or {}
        language = (tags.get("language") or "").lower()
        channels = int(stream.get("channels") or 0)
        tracks.append({
            "index": n,
            "codec": stream.get("codec_name") or "?",
            "channels": channels,
            "layout": stream.get("channel_layout") or (f"{channels} ch" if channels else ""),
            "language": language if language not in ("und", "") else "",
            "title": tags.get("title") or "",
            "default": bool((stream.get("disposition") or {}).get("default")),
            "surround": channels >= 5,  # 5.0 / 5.1 / 7.1 all have a center channel
        })
    return tracks


_USEFUL_TAGS = ("title", "show", "series", "season_number", "episode_id", "episode_sort", "date", "year",
                "description", "synopsis", "comment", "imdb", "imdb_id", "tmdb", "network", "artist", "album")


def file_tags(source: Path) -> Dict[str, str]:
    """The container's own metadata (title, show, episode, IMDb ID...), for identifying speakers."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(source)],
            capture_output=True, timeout=60,
        )
        tags = json.loads(proc.stdout or b"{}").get("format", {}).get("tags") or {}
    except (OSError, ValueError, subprocess.TimeoutExpired) as err:
        _LOGGER.warning("Couldn't read the metadata of %s: %s", source.name, err)
        return {}
    useful = {}
    for key, value in tags.items():
        key, value = key.lower(), " ".join(str(value).split())
        if value and (key in _USEFUL_TAGS or re.search(r"\btt\d{7,9}\b", value)):
            useful[key] = value[:200]
    return dict(list(useful.items())[:12])


def recommended_track(tracks: List[Dict[str, Any]], voice_language: str) -> int:
    """The voice's language, not a commentary, the default track, most channels."""
    want = voice_language.split("-")[0].lower()

    def score(track: Dict[str, Any]):
        lang = _LANGUAGE_CODES.get(track["language"], track["language"][:2])
        return (
            lang == want,
            "comment" not in track["title"].lower(),
            track["default"],
            track["channels"],
        )

    return max(tracks, key=score)["index"]


def track_language(track: Dict[str, Any]) -> str:
    """A track's language as a voice language prefix ("eng" -> "en"); "" when untagged."""
    code = (track.get("language") or "").lower()
    return _LANGUAGE_CODES.get(code, code[:2] if code not in ("", "und") else "")


def obvious_track(tracks: List[Dict[str, Any]], voice_language: str) -> Optional[int]:
    """The only track in the voice's language, if there's exactly one: no need to ask.
    None if there are several (e.g. a commentary in English too) or none."""
    want = voice_language.split("-")[0].lower()
    matching = [t for t in tracks if track_language(t) == want]
    return matching[0]["index"] if len(matching) == 1 else None


def best_track_in(tracks: List[Dict[str, Any]], language: str) -> Optional[int]:
    """The best track in a language (not a commentary, the default one, most channels)."""
    matching = [t for t in tracks if track_language(t) == language]
    if not matching:
        return None
    return max(matching, key=lambda t: ("comment" not in t["title"].lower(), t["default"], t["channels"]))["index"]


class _Word:
    __slots__ = ("start", "end", "word", "segment_start")

    def __init__(self, word: Any, segment_start: bool) -> None:
        self.start, self.end, self.word = word.start, word.end, word.word
        self.segment_start = segment_start


def _transcribe(take_dir: Path, source: Path, model_id: str, language: str, turns: bool,
                progress: Callable[[str], None], track: int = 0, dialogue: bool = False) -> Dict[str, Any]:
    progress("Extracting the dialogue channel…" if dialogue else "Extracting audio…")
    wav = take_dir / "audio.wav"
    # -map picks the audio track; the center channel (FC) of a surround mix is the dialogue
    mix = ["-af", "pan=mono|c0=FC"] if dialogue else ["-ac", "1"]
    _ffmpeg(["-i", str(source), "-map", f"0:a:{track}", "-vn", *mix,
             "-ar", str(CLIP_RATE), "-c:a", "pcm_s16le", str(wav)])
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
    return {"duration": round(duration, 1),
            "segments": split_words(words, duration, turns, loudness(audio, 16000))}


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
        limit=SUBPROCESS_LINE_LIMIT,
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


def start(voice: Voice, source: Path, denoise: Any = "light", diarize: bool = False,
          name: str = "", clone: bool = False) -> Dict[str, Any]:
    """Check the upload, then transcribe (and optionally diarize) it in the background.

    Files with several audio tracks wait in state "choose_track" (see choose_track)."""
    take_dir = source.parent
    take_id = take_dir.name
    size = source.stat().st_size
    # The upload limit isn't for media library files (linked, not uploaded)
    if (size > MAX_UPLOAD_BYTES and not source.is_symlink()) or size < 1000:
        shutil.rmtree(take_dir, ignore_errors=True)
        raise ValueError("The file is empty" if size < 1000 else "The file is larger than 8 GB")
    if not stt.available():
        shutil.rmtree(take_dir, ignore_errors=True)
        raise RuntimeError("Speech recognition (faster-whisper) is not installed")

    try:
        tracks = probe_tracks(source)
    except ValueError:
        shutil.rmtree(take_dir, ignore_errors=True)
        raise
    if not tracks:
        shutil.rmtree(take_dir, ignore_errors=True)
        raise ValueError("This file has no audio track.")

    recommended = recommended_track(tracks, voice.language)
    take: Dict[str, Any] = {
        "state": "choose_track", "detail": "", "error": None, "created": time.time(),
        "duration": None, "segments": [], "model": _model_for(voice),
        "denoise": _level(denoise), "diarize": bool(diarize),
        "source": source.name, "size": size, "name": " ".join(str(name).split())[:200],
        "tracks": tracks, "track": recommended, "dialogue": tracks[recommended]["surround"],
        "tags": file_tags(source),
        "clone": bool(clone),  # Character clone: reviewed per character, not per file
    }
    _LOGGER.info("Freeform take %s/%s uploaded: %s (%.1f MB), %d audio track(s)",
                 voice.name, take_id, source.name, size / 2**20, len(tracks))
    if len(tracks) > 1:
        obvious = obvious_track(tracks, voice.language)
        if obvious is None:
            _write(take_dir, take)  # wait for the track choice
            return _public(take_id, take)
        # One track in the voice's language: that's the one, no need to ask
        take.update(track=obvious, dialogue=tracks[obvious]["surround"])
        _LOGGER.info("Take %s: audio track %d picked (the only one in %s)", take_id, obvious, voice.language)
    return _begin(voice, take_dir, take)


def retry(voice: Voice, take_id: str) -> Dict[str, Any]:
    """Process a failed or interrupted take again from its uploaded file (no re-upload)."""
    take_dir = _take_dir(voice, take_id)
    if take_id in _live:
        raise RuntimeError("This take is still being processed")
    take = _public(take_id, _read(take_dir))
    if take["state"] != "error":
        raise RuntimeError("Only failed or interrupted takes can be tried again")
    source = take_dir / take.get("source", "")
    if not take.get("source") or not source.is_file():
        raise ValueError("The uploaded file is no longer on the server; import it again")
    take = {k: v for k, v in take.items() if k not in ("id", "speakers")}
    take.update(error=None, segments=[], duration=None, model=_model_for(voice))
    _LOGGER.info("Freeform take %s/%s: trying again", voice.name, take_id)
    return _begin(voice, take_dir, take)


def choose_track(voice: Voice, take_id: str, track: int, dialogue: bool) -> Dict[str, Any]:
    """Pick the audio track (and whether to use only its dialogue channel), then start."""
    take_dir = _take_dir(voice, take_id)
    take = _read(take_dir)
    if take["state"] != "choose_track":
        raise RuntimeError("This take is already being processed")
    tracks = take["tracks"]
    if not 0 <= int(track) < len(tracks):
        raise ValueError(f"No audio track {track}")
    take["track"] = int(track)
    take["dialogue"] = bool(dialogue) and tracks[int(track)]["surround"]
    return _begin(voice, take_dir, take)


def choose_tracks_for_all(voice: Voice, language: str) -> Dict[str, Any]:
    """Every file waiting for a track choice: its best track in `language`. Files without
    a track in that language keep waiting."""
    started, left = 0, 0
    for take_dir in sorted(_takes_dir(voice).iterdir()) if _takes_dir(voice).is_dir() else []:
        if not (take_dir / "take.json").is_file() or take_dir.name in _live:
            continue
        take = _read(take_dir)
        if take.get("state") != "choose_track":
            continue
        track = best_track_in(take.get("tracks") or [], language)
        if track is None:
            left += 1
            continue
        choose_track(voice, take_dir.name, track, take["tracks"][track]["surround"])
        started += 1
    _LOGGER.info("Voice %s: audio track in %s picked for %d files (%d without one)", voice.name, language, started, left)
    return {"started": started, "left": left}


def pick_obvious_tracks(voice: Voice) -> int:
    """Files that were left waiting for a track choice that is obvious (one track in the
    voice's language): start them. For files added before this rule existed."""
    started = 0
    for take_dir in sorted(_takes_dir(voice).iterdir()) if _takes_dir(voice).is_dir() else []:
        if not (take_dir / "take.json").is_file() or take_dir.name in _live:
            continue
        take = _read(take_dir)
        if take.get("state") != "choose_track":
            continue
        track = obvious_track(take.get("tracks") or [], voice.language)
        if track is not None:
            choose_track(voice, take_dir.name, track, take["tracks"][track]["surround"])
            started += 1
    if started:
        _LOGGER.info("Voice %s: %d waiting files had an obvious audio track: started", voice.name, started)
    return started


def _begin(voice: Voice, take_dir: Path, take: Dict[str, Any]) -> Dict[str, Any]:
    take_id = take_dir.name
    source = take_dir / take["source"]
    track, dialogue = take["track"], take["dialogue"]
    take.update(state="running", detail="Starting…")
    _write(take_dir, take)
    _live[take_id] = take

    def progress(line: str) -> None:
        take["detail"] = line

    _LOGGER.info("Freeform take %s/%s started: track %d%s, whisper=%s, noise=%s, diarize=%s",
                 voice.name, take_id, track, " (dialogue channel)" if dialogue else "",
                 take["model"], take["denoise"], take["diarize"])
    started = time.monotonic()

    async def run() -> None:
        global _work
        if _work is None:
            _work = asyncio.Semaphore(1)
        if _work.locked():
            progress("Waiting for the other files to finish…")
        try:
            async with _work:
                progress("Starting…")
                result = await asyncio.to_thread(
                    _transcribe, take_dir, source, take["model"],
                    voice.language.split("-")[0].lower(), take["diarize"], progress, track, dialogue,
                )
                take.update(result)
                if take["diarize"] and take["segments"]:
                    progress("Finding speakers…")
                    await _diarize(voice, take_dir, take, progress)
                    try:
                        people.link(voice, take_id, take)
                    except Exception:  # matching across files is a bonus, never fatal
                        _LOGGER.exception("Matching the speakers of %s/%s to known people failed", voice.name, take_id)
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
        if take.get("state") == "done" and take.get("speakers") and take.get("clone"):
            identify.after_take(voice)  # Character clone: who says each line, with AI

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


def save_clips(voice: Voice, clips: List[Dict[str, Any]]) -> int:
    """Save clips picked across files (one character's) as recordings. The files stay,
    so other characters' clips can still be picked; saved clips are marked."""
    by_take: Dict[str, List[Dict[str, Any]]] = {}
    for clip in clips:
        by_take.setdefault(str(clip.get("take", "")), []).append(clip)
    saved = 0
    for take_id, items in by_take.items():
        take_dir = _take_dir(voice, take_id)
        if take_id in _live:
            raise RuntimeError("That file is still being processed; wait for it to finish")
        take = _read(take_dir)
        segments = take["segments"]
        for item in items:
            index = int(item.get("index", -1))
            text = " ".join(str(item.get("text", "")).split())
            if not 0 <= index < len(segments) or not text:
                continue
            voice.save_recording(GROUP, f"{take_id}_{index:04d}", text, clip_wav(voice, take_id, index), ".wav")
            segments[index]["saved"] = True
            segments[index]["savedText"] = text  # as corrected in the review
            saved += 1
        _write(take_dir, take)
    _LOGGER.info("Voice %s: saved %d clips from %d files as recordings", voice.name, saved, len(by_take))
    return saved


MIN_TRIMMED = 0.4  # seconds a trimmed clip keeps at least


def words_still_heard(text: str, heard: str, front: bool, back: bool) -> str:
    """The text without the words a trim cut off: on the trimmed side(s), words that
    Whisper no longer hears in the clip are dropped. The rest (and any corrections the
    person made) stays as written. Nothing heard at all: the text is left alone."""
    tokens = text.split()
    norm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())  # "Chucky's" = "chuckys"
    mine = [norm(t) for t in tokens]
    theirs = [norm(w) for w in heard.split() if norm(w)]
    if not tokens or not theirs:
        return text
    matcher = difflib.SequenceMatcher(None, mine, theirs, autojunk=False)
    matched = [i for block in matcher.get_matching_blocks() for i in range(block.a, block.a + block.size)]
    if not matched:
        return text
    first = min(matched) if front else 0
    last = max(matched) if back else len(tokens) - 1
    kept = tokens[first:last + 1]
    if front and first > 0 and kept and tokens[0][:1].isupper():
        kept[0] = kept[0][:1].upper() + kept[0][1:]  # still starts like a sentence
    return " ".join(kept) or text


def _hear(voice: Voice, take: Dict[str, Any], take_id: str, index: int) -> str:
    """What Whisper hears in one clip on its own."""
    import io as _io
    wav = clip_wav(voice, take_id, index)
    with wave.open(_io.BytesIO(wav), "rb") as src:
        rate = src.getframerate()
        audio = np.frombuffer(src.readframes(src.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if rate != 16000:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, 16000, rate).astype(np.float32)
    model_id = take.get("model") or _model_for(voice)
    model = stt._load(model_id)
    segments, _ = model.transcribe(audio, language=None if model_id.endswith(".en") else voice.language.split("-")[0],
                                   beam_size=5, condition_on_previous_text=False, vad_filter=False)
    return " ".join(seg.text for seg in segments)


MIN_TRIMMED = 0.4  # seconds a trimmed clip keeps at least


def trim_clip(voice: Voice, take_id: str, index: int, front: float = 0.0, back: float = 0.0,
              reset: bool = False, text: Optional[str] = None) -> Dict[str, Any]:
    """Cut a bit off the start and/or end of a clip (a laugh, another voice), or undo
    all trims. Words cut off go from the text (`text`: the words as shown in the review,
    with any corrections). A clip already in the dataset gets its saved audio and text
    redone too."""
    take_dir = _take_dir(voice, take_id)
    if take_id in _live:
        raise RuntimeError("That file is still being processed; wait for it to finish")
    take = _read(take_dir)
    segments = take["segments"]
    if not 0 <= index < len(segments):
        raise KeyError(f"No clip {index}")
    seg = segments[index]
    original = seg.get("original") or [seg["start"], seg["end"]]
    current = " ".join((text if text is not None else (seg.get("savedText") or seg["text"])).split())
    if reset:
        start, end = original
        words = seg.get("originalText") or current
    else:
        start = seg["start"] + max(0.0, float(front))
        end = seg["end"] - max(0.0, float(back))
        if end - start < MIN_TRIMMED:
            raise ValueError(f"That would leave less than {MIN_TRIMMED:g} s of the clip")
        words = current
    if "originalText" not in seg and not reset:
        seg["originalText"] = current
    seg["start"], seg["end"] = round(start, 3), round(end, 3)
    if [seg["start"], seg["end"]] == [round(original[0], 3), round(original[1], 3)]:
        seg.pop("original", None)
        seg.pop("originalText", None)
    else:
        seg["original"] = original
    if not reset and (front or back):
        try:
            words = words_still_heard(words, _hear(voice, take, take_id, index), front > 0, back > 0)
        except Exception as err:  # noqa: BLE001 - the trim still counts; the words stay
            _LOGGER.warning("Couldn't re-check the words of %s/%d after a trim: %s", take_id, index, err)
    if seg.get("saved"):
        seg["savedText"] = words
    else:
        seg["text"] = words
    _write(take_dir, take)
    if seg.get("saved"):
        voice.save_recording(GROUP, f"{take_id}_{index:04d}", words, clip_wav(voice, take_id, index), ".wav")
    _LOGGER.info("Voice %s: clip %s/%d %s (%.2f-%.2f s)", voice.name, take_id, index,
                 "trims undone" if reset else "trimmed", seg["start"], seg["end"])
    return {"start": seg["start"], "end": seg["end"], "trimmed": "original" in seg, "text": words}


def set_saved_text(voice: Voice, take_id: str, index: int, text: str) -> str:
    """Correct the transcript of a clip that's already in the dataset."""
    text = " ".join(str(text).split())
    if not text:
        raise ValueError("The text can't be empty")
    take_dir = _take_dir(voice, take_id)
    take = _read(take_dir)
    segments = take["segments"]
    if not 0 <= index < len(segments) or not segments[index].get("saved"):
        raise ValueError("That clip isn't in the dataset")
    transcript = voice.recordings_dir / GROUP / f"{take_id}_{index:04d}.txt"
    if not transcript.parent.is_dir():
        raise ValueError("That clip isn't in the dataset")
    transcript.write_text(text, encoding="utf-8")
    segments[index]["savedText"] = text
    _write(take_dir, take)
    _LOGGER.info("Voice %s: corrected the text of saved clip %s/%d", voice.name, take_id, index)
    return text


def unsave_clips(voice: Voice, clips: List[Dict[str, Any]]) -> int:
    """Take saved clips back out of the dataset (their recordings are deleted; the clips
    stay in the review, with the text as it was saved)."""
    by_take: Dict[str, List[int]] = {}
    for clip in clips:
        by_take.setdefault(str(clip.get("take", "")), []).append(int(clip.get("index", -1)))
    removed = 0
    for take_id, indexes in by_take.items():
        take_dir = _take_dir(voice, take_id)
        if take_id in _live:
            raise RuntimeError("That file is still being processed; wait for it to finish")
        take = _read(take_dir)
        segments = take["segments"]
        for index in indexes:
            if not 0 <= index < len(segments):
                continue
            stem = voice.recordings_dir / GROUP / f"{take_id}_{index:04d}"
            found = False
            for ext in (*AUDIO_EXTENSIONS, ".txt"):
                path = stem.with_suffix(ext)
                if path.exists():
                    path.unlink()
                    found = True
            if found or segments[index].get("saved"):
                removed += 1
            segments[index]["saved"] = False
        _write(take_dir, take)
    _LOGGER.info("Voice %s: took %d clips from %d files out of the dataset", voice.name, removed, len(by_take))
    return removed


def retag_person(voice: Voice, take_id: str, old: str, new: str) -> None:
    """After merging two people, point a take's speakers at the merged one."""
    take_dir = _takes_dir(voice) / take_id
    if take_id in _live or not (take_dir / "take.json").is_file():
        return
    take = _read(take_dir)
    changed = False
    for speaker in take.get("speakers") or []:
        if speaker.get("person") == old:
            speaker["person"] = new
            changed = True
    if changed:
        _write(take_dir, take)


def discard(voice: Voice, take_id: str) -> None:
    if take_id in _live:
        raise RuntimeError("Still processing; wait for it to finish")
    take_dir = _take_dir(voice, take_id)
    name = _read(take_dir).get("name", "") if (take_dir / "take.json").is_file() else ""
    shutil.rmtree(take_dir, ignore_errors=True)
    people.drop_take(voice, take_id, name)
