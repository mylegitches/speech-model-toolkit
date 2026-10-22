from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .schemas import VoiceInfo
from .wyoming import REMOTE_PREFIX, RemoteVoice, WyomingPiperClient


class VoiceCatalog:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._runtime_status: dict[str, tuple[str, str | None]] = {}
        self.remote: WyomingPiperClient | None = None
        if self.settings.piper_host:
            self.remote = WyomingPiperClient(
                self.settings.piper_host,
                self.settings.piper_port,
                languages=self.settings.piper_languages,
            )

    @staticmethod
    def is_remote(path: str) -> bool:
        return path.startswith(REMOTE_PREFIX)

    def resolve_remote(self, model_path: str, config_path: str | None = None) -> RemoteVoice:
        if self.remote is None:
            raise ValueError("No remote Piper server is configured")
        if config_path is not None and config_path != model_path:
            raise ValueError("Remote voice config must match the voice path")
        voice = self.remote.find(model_path.removeprefix(REMOTE_PREFIX))
        if voice is None:
            raise ValueError(self.remote.last_error or f"{model_path} is not on the Piper server")
        return voice

    def _scan_remote(self) -> list[VoiceInfo]:
        if self.remote is None:
            return []
        return [
            VoiceInfo(
                id=f"{REMOTE_PREFIX}{voice.name}",
                name=f"{voice.name.replace('_', ' ').replace('-', ' ')} (server)",
                model_path=f"{REMOTE_PREFIX}{voice.name}",
                config_path=f"{REMOTE_PREFIX}{voice.name}",
                sample_rate=None,
                num_speakers=max(1, len(voice.speakers)),
                language=voice.language,
                length_scale=1.0,
                status="remote",
            )
            for voice in self.remote.voices()
        ]

    @staticmethod
    def _read_metadata(config_path: Path) -> dict[str, Any]:
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _toolkit_voices(self) -> list[tuple[Path, str]]:
        """The newest exported version of each voice trained in the toolkit."""
        try:
            from ..voice import main as voice_app
        except Exception:  # standalone: no toolkit around
            return []
        out = []
        for voice in voice_app.store.list():
            try:
                workspace = voice_app.trainer.get(voice)
                exports = workspace.exports()
                if exports:
                    out.append((workspace.exports_dir / exports[0]["dir"] / exports[0]["model"],
                                f"{voice.name} (trained, epoch {exports[0]['epoch']})"))
            except Exception:  # noqa: BLE001 - one broken voice doesn't hide the others
                continue
        return out

    def scan(self) -> list[VoiceInfo]:
        found: dict[Path, VoiceInfo] = {}
        names: dict[Path, str] = {}
        candidates: list[Path] = []
        for model_path, label in self._toolkit_voices():
            candidates.append(model_path)
            names[model_path.resolve()] = label
        for model_root in (self.settings.uploads_dir, self.settings.default_voices_dir):
            if model_root.exists():
                candidates += list(model_root.rglob("*.onnx"))
        for model_path in candidates:
            model_path = model_path.resolve()
            if model_path in found:
                continue
            config_path = Path(f"{model_path}.json")
            voice_id = model_path.stem
            if not config_path.exists():
                found[model_path] = VoiceInfo(
                    id=voice_id,
                    name=voice_id,
                    model_path=self._display_path(model_path),
                    config_path=self._display_path(config_path),
                    sample_rate=None,
                    num_speakers=1,
                    language="unknown",
                    length_scale=1.0,
                    status="error",
                    error="Matching .onnx.json config is missing",
                )
                continue
            try:
                metadata = self._read_metadata(config_path)
                runtime_status, runtime_error = self._runtime_status.get(
                    str(model_path), ("untested", None)
                )
                found[model_path] = VoiceInfo(
                    id=voice_id,
                    name=names.get(model_path) or voice_id.replace("_", " ").replace("-", " "),
                    model_path=self._display_path(model_path),
                    config_path=self._display_path(config_path),
                    sample_rate=metadata.get("audio", {}).get("sample_rate"),
                    num_speakers=max(1, int(metadata.get("num_speakers", 1) or 1)),
                    language=metadata.get("language", {}).get("code", "unknown"),
                    length_scale=float(metadata.get("inference", {}).get("length_scale", 1.0)),
                    status=runtime_status,
                    error=runtime_error,
                )
            except Exception as exc:
                found[model_path] = VoiceInfo(
                    id=voice_id,
                    name=voice_id,
                    model_path=self._display_path(model_path),
                    config_path=self._display_path(config_path),
                    sample_rate=None,
                    num_speakers=1,
                    language="unknown",
                    length_scale=1.0,
                    status="error",
                    error=f"Invalid voice config: {exc}",
                )
        local = sorted(found.values(), key=lambda voice: voice.name.casefold())
        remote = sorted(self._scan_remote(), key=lambda voice: voice.name.casefold())
        return local + remote

    def resolve_pair(self, model_path: str, config_path: str | None = None) -> tuple[Path, Path]:
        resolved_model = self._resolve_allowed(model_path)
        resolved_config = self._resolve_allowed(config_path or f"{model_path}.json")
        if resolved_model.suffix.lower() != ".onnx":
            raise ValueError("Voice model must be an .onnx file")
        if resolved_config.name != f"{resolved_model.name}.json":
            raise ValueError("Voice config must use the matching .onnx.json basename")
        if not resolved_model.is_file() or not resolved_config.is_file():
            raise ValueError("Voice model/config pair does not exist")
        return resolved_model, resolved_config

    def mark_runtime(self, model_path: Path, status: str, error: str | None = None) -> None:
        self._runtime_status[str(model_path.resolve())] = (status, error)

    def _resolve_allowed(self, value: str) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.settings.root_dir / candidate
        candidate = candidate.resolve()
        allowed = [root.resolve() for root in self.settings.model_roots]
        if not any(candidate == root or root in candidate.parents for root in allowed):
            raise ValueError("Voice path must stay inside an allowed model directory")
        return candidate

    def _display_path(self, value: Path) -> str:
        return value.resolve().as_posix()  # absolute: voices live in the data folder


voice_catalog = VoiceCatalog()
