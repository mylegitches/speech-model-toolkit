from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path

from .config import Settings, get_settings


class WhisperService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: object | None = None
        self._model_lock = threading.Lock()

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_model(self) -> object:
        with self._model_lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.settings.whisper_model,
                    device=self.settings.whisper_device,
                    compute_type=self.settings.whisper_compute_type,
                )
            return self._model

    async def transcribe(self, audio: bytes, suffix: str = ".webm") -> str:
        return await asyncio.to_thread(self._transcribe_sync, audio, suffix)

    def _transcribe_sync(self, audio: bytes, suffix: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(audio)
        try:
            model = self._get_model()
            segments, _ = model.transcribe(
                str(temp_path),
                beam_size=5,
                vad_filter=True,
                language="en",
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        finally:
            temp_path.unlink(missing_ok=True)


stt_service = WhisperService()
