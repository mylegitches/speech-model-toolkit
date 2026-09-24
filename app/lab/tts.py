"""Voices the Test Lab can reply with: your trained Piper voices, or a stock default."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from ..voice import main as voice_app
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


async def synthesize(voice_id: str, text: str) -> bytes:
    """WAV bytes of text spoken by the chosen voice."""
    path = await model_path(voice_id)
    return await voice_app.trainer.synthesize(path, text)
