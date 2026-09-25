"""Local speech-to-text with faster-whisper (CPU, int8).

Models download on first use to $STT_MODELS_DIR (Hugging Face cache layout).
Runs on the CPU: faster-whisper's CUDA build wants a newer cuDNN than the
openWakeWord torch stack ships, and short questions transcribe in a second or
two on a CPU anyway.
"""

import asyncio
import logging
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_REPO_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("STT_MODELS_DIR", _REPO_DIR / "data" / "stt-models"))

MODELS = [
    {"id": "tiny.en", "label": "Tiny (English) · 75 MB · fastest"},
    {"id": "base.en", "label": "Base (English) · 145 MB · recommended"},
    {"id": "small.en", "label": "Small (English) · 480 MB · more accurate"},
    {"id": "medium.en", "label": "Medium (English) · 1.5 GB · slow on CPU"},
    {"id": "tiny", "label": "Tiny (multilingual) · 75 MB"},
    {"id": "base", "label": "Base (multilingual) · 145 MB"},
    {"id": "small", "label": "Small (multilingual) · 480 MB"},
    {"id": "medium", "label": "Medium (multilingual) · 1.5 GB"},
]

_LOGGER = logging.getLogger(__name__)
_models: Dict[str, Any] = {}
_lock = threading.Lock()
_downloading: Optional[str] = None


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def is_cached(model_id: str) -> bool:
    return any(MODELS_DIR.glob(f"models--*--faster-whisper-{model_id}/snapshots/*/model.bin"))


def status(model_id: str) -> Dict[str, Any]:
    return {
        "available": available(),
        "model": model_id,
        "cached": is_cached(model_id),
        "downloading": _downloading == model_id,
        "models": MODELS,
    }


def _load(model_id: str):
    global _downloading
    with _lock:
        if model_id not in _models:
            from faster_whisper import WhisperModel

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            _downloading = model_id
            started = time.monotonic()
            _LOGGER.info("Loading Whisper %s%s", model_id, "" if is_cached(model_id) else " (downloading first)")
            try:
                _models[model_id] = WhisperModel(
                    model_id,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(MODELS_DIR),
                )
            finally:
                _downloading = None
            _LOGGER.info("Whisper %s ready in %.0fs", model_id, time.monotonic() - started)
        return _models[model_id]


async def ensure(model_id: str) -> None:
    """Download (if needed) and load a model."""
    if not available():
        raise RuntimeError("faster-whisper is not installed")
    await asyncio.to_thread(_load, model_id)


def _transcribe(model_id: str, audio: np.ndarray, language: str) -> str:
    model = _load(model_id)
    lang = None if (language in ("", "auto") or model_id.endswith(".en")) else language
    segments, _info = model.transcribe(
        audio, language=lang, beam_size=1, condition_on_previous_text=False
    )
    return " ".join(s.text.strip() for s in segments).strip()


async def transcribe(pcm16: bytes, model_id: str = "base.en", language: str = "auto") -> str:
    """Transcribe 16 kHz mono s16le PCM."""
    if not available():
        raise RuntimeError("faster-whisper is not installed")
    audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    return await asyncio.to_thread(_transcribe, model_id, audio, language)
