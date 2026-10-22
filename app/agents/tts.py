from __future__ import annotations

import asyncio
import io
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

from .voice_catalog import VoiceCatalog, voice_catalog


@dataclass(slots=True)
class LoadedVoice:
    voice: object
    lock: threading.Lock


class PiperService:
    def __init__(self, catalog: VoiceCatalog | None = None) -> None:
        self.catalog = catalog or voice_catalog
        self._voices: dict[str, LoadedVoice] = {}
        self._cache_lock = threading.Lock()

    def available(self) -> bool:
        try:
            import piper  # noqa: F401
            return True
        except ImportError:
            return False

    async def synthesize(
        self,
        text: str,
        model_path: str,
        config_path: str,
        speaker_id: int = 0,
        length_scale: float = 1.0,
    ) -> bytes:
        if self.catalog.is_remote(model_path):
            # Wyoming has no per-request length_scale; the server's voice config applies.
            voice = self.catalog.resolve_remote(model_path, config_path)
            assert self.catalog.remote is not None
            return await asyncio.to_thread(
                self.catalog.remote.synthesize, text, voice.name, speaker_id
            )
        model, config = self.catalog.resolve_pair(model_path, config_path)
        return await asyncio.to_thread(
            self._synthesize_sync, text, model, config, speaker_id, length_scale
        )

    def _load(self, model: Path, config: Path) -> LoadedVoice:
        key = str(model)
        with self._cache_lock:
            if key in self._voices:
                return self._voices[key]
            try:
                from piper import PiperVoice

                loaded = LoadedVoice(
                    voice=PiperVoice.load(model, config_path=config), lock=threading.Lock()
                )
                self._voices[key] = loaded
                self.catalog.mark_runtime(model, "ready")
                return loaded
            except Exception as exc:
                self.catalog.mark_runtime(model, "error", str(exc))
                raise RuntimeError(f"Could not load voice {model.name}: {exc}") from exc

    def _synthesize_sync(
        self,
        text: str,
        model: Path,
        config: Path,
        speaker_id: int,
        length_scale: float,
    ) -> bytes:
        from piper import SynthesisConfig

        loaded = self._load(model, config)
        output = io.BytesIO()
        with loaded.lock, wave.open(output, "wb") as wav_file:
            loaded.voice.synthesize_wav(
                text,
                wav_file,
                syn_config=SynthesisConfig(
                    speaker_id=speaker_id,
                    length_scale=length_scale,
                ),
            )
        return output.getvalue()

    def invalidate(self, model_path: str) -> None:
        if self.catalog.is_remote(model_path):
            return
        try:
            model, _ = self.catalog.resolve_pair(model_path)
        except ValueError:
            return
        with self._cache_lock:
            self._voices.pop(str(model), None)


tts_service = PiperService()
