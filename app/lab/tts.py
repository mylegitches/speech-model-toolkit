"""Voices the Test Lab can reply with: your trained Piper voices, or a stock default."""

import asyncio
import io
import logging
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..settings import store as settings_store
from ..voice import main as voice_app
from . import pauses
from ..voice.training import download_resumable

_LOGGER = logging.getLogger(__name__)

DEFAULT_VOICE = "default"
DEFAULT_NAME = "en_US-lessac-medium"
_DEFAULT_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
)
_download_lock = asyncio.Lock()


def default_dir() -> Path:
    return voice_app.DATA_DIR / "default-voices"


def _default_model() -> Path:
    return default_dir() / f"{DEFAULT_NAME}.onnx"


def options() -> List[Dict[str, Any]]:
    """Default voice first, then each trained voice with its newest export."""
    choices: List[Dict[str, Any]] = [
        {
            "id": DEFAULT_VOICE,
            "label": "Default: Piper Lessac (US English)",
            "ready": _default_model().exists(),
        }
    ]
    for voice in voice_app.store.list():
        exports = voice_app.trainer.get(voice).exports()
        if exports:
            newest = exports[0]
            choices.append(
                {
                    "id": f"voice:{voice.name}",
                    "label": f"{voice.name} (epoch {newest['epoch']})",
                    "ready": True,
                }
            )
    return choices


def _download(url: str, dest: Path) -> None:
    # Times out stalled connections and resumes from the partial file
    tmp = dest.with_name(dest.name + ".part")
    download_resumable(url, tmp, _LOGGER.info)
    tmp.replace(dest)


async def _ensure_default() -> Path:
    model = _default_model()
    async with _download_lock:
        if not (model.exists() and Path(f"{model}.json").exists()):
            default_dir().mkdir(parents=True, exist_ok=True)
            _LOGGER.info("Downloading default Piper voice %s", DEFAULT_NAME)
            await asyncio.to_thread(_download, f"{_DEFAULT_URL}.json", Path(f"{model}.json"))
            await asyncio.to_thread(_download, _DEFAULT_URL, model)
    return model


async def model_path(voice_id: str) -> Path:
    if voice_id in ("", DEFAULT_VOICE):
        return await _ensure_default()

    if not voice_id.startswith("voice:"):
        raise KeyError(f"Unknown voice {voice_id}")

    voice = voice_app.store.get(voice_id.split(":", 1)[1])
    workspace = voice_app.trainer.get(voice)
    exports = workspace.exports()
    if not exports:
        raise KeyError(f"{voice.name} has no exported version yet")
    return workspace.exports_dir / exports[0]["dir"] / exports[0]["model"]


def clamp_speed(speed: Any) -> int:
    try:
        return min(150, max(50, int(speed)))
    except (TypeError, ValueError):
        return 100


def _wav(pcm: bytes, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return buffer.getvalue()


async def _voice_path(voice_id: str) -> Path:
    try:
        return await model_path(voice_id)
    except OSError as err:
        raise RuntimeError(f"Could not download the default voice ({err}). Check the server's internet connection.") from err


def _piper_error(path: Path, err: Exception) -> RuntimeError:
    text = str(err).strip()
    return RuntimeError(f"Piper could not speak with {path.name}: {text.splitlines()[-1] if text else 'unknown error'}")


async def speak_events(voice_id: str, events: List[Any], speed: int = 100) -> Tuple[bytes, int]:
    """PCM (16-bit mono) and sample rate of speech and pause events, in one voice."""
    path = await _voice_path(voice_id)
    speed = clamp_speed(speed)
    try:
        return await voice_app.trainer.synthesize_events(path, events, round(100 / speed, 3) if speed != 100 else None)
    except RuntimeError as err:
        raise _piper_error(path, err) from err


async def synthesize(voice_id: str, text: str, speed: int = 100, pause_settings: Optional[Dict[str, Any]] = None) -> bytes:
    """WAV bytes of text spoken by the chosen voice, at speed % of its normal pace, with
    the pauses from Settings (or `pause_settings`) and any <break time="1s"/> tags."""
    cfg = pause_settings if pause_settings is not None else settings_store.load()["pauses"]
    events = pauses.parse(text, cfg)
    if not any(kind == "pause" for kind, _ in events):
        # Nothing to insert: Piper speaks the text as it is (tags, if any, already removed)
        text = " ".join(value for _, value in events) or text
        path = await _voice_path(voice_id)
        try:
            speed = clamp_speed(speed)
            return await voice_app.trainer.synthesize(path, text, round(100 / speed, 3) if speed != 100 else None)
        except RuntimeError as err:
            raise _piper_error(path, err) from err
    pcm, rate = await speak_events(voice_id, events, speed)
    return _wav(pcm, rate)


MAX_SCRIPT_LINES = 200


async def synthesize_script(lines: List[Dict[str, Any]], base: Dict[str, Any]) -> Tuple[bytes, List[Dict[str, float]]]:
    """A conversation: each line in its own voice, speed and pauses (the line's own pause
    settings over `base`), joined with each line's pause after it. Returns one WAV and
    where each line starts and ends in it (seconds)."""
    from scipy.signal import resample_poly

    pieces = []
    for line in lines[:MAX_SCRIPT_LINES]:
        cfg = pauses.clean_settings({**base, **(line.get("pauses") or {})})
        events = pauses.parse(str(line.get("text") or ""), cfg)
        if not events:
            pieces.append((None, 0, line))
            continue
        pcm, rate = await speak_events(str(line.get("voice") or DEFAULT_VOICE), events, line.get("speed", 100))
        pieces.append((pcm, rate, line))
    rate = max([r for _, r, _ in pieces if r] or [22050])
    out = []
    marks = []
    at = 0
    for pcm, r, line in pieces:
        start = at
        if pcm:
            audio = np.frombuffer(pcm, dtype=np.int16)
            if r != rate:  # voices at different rates (16 / 22 kHz): one rate for the file
                audio = np.clip(resample_poly(audio.astype(np.float32), rate, r), -32768, 32767).astype(np.int16)
            out.append(audio.tobytes())
            at += len(audio)
        marks.append({"start": round(start / rate, 3), "end": round(at / rate, 3)})
        gap = int(max(0, min(pauses.MAX_PAUSE_MS, float(line.get("gapAfter", 0) or 0))) * rate / 1000)
        if gap:
            out.append(bytes(2 * gap))
            at += gap
    return _wav(b"".join(out), rate), marks
