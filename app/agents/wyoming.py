from __future__ import annotations

import io
import json
import socket
import threading
import time
import wave
from dataclasses import dataclass
from typing import Any, BinaryIO


REMOTE_PREFIX = "wyoming:"


@dataclass(slots=True)
class RemoteVoice:
    name: str
    language: str
    speakers: list[str]


class WyomingPiperClient:
    """Minimal client for a Wyoming Piper server (e.g. rhasspy/wyoming-piper)."""

    def __init__(
        self,
        host: str,
        port: int = 10200,
        timeout: float = 30.0,
        languages: list[str] | None = None,
        cache_seconds: float = 60.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.languages = [item.casefold() for item in (languages or [])]
        self.cache_seconds = cache_seconds
        self._voices: list[RemoteVoice] | None = None
        self._voices_at = 0.0
        self._last_error: str | None = None
        self._lock = threading.Lock()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def voices(self) -> list[RemoteVoice]:
        with self._lock:
            if self._voices is not None and time.monotonic() - self._voices_at < self.cache_seconds:
                return self._voices
            try:
                self._voices = self._describe()
                self._last_error = None
            except (OSError, ValueError) as exc:
                self._last_error = f"Piper server {self.host}:{self.port} unreachable: {exc}"
                self._voices = self._voices or []
            self._voices_at = time.monotonic()
            return self._voices

    def find(self, name: str) -> RemoteVoice | None:
        return next((voice for voice in self.voices() if voice.name == name), None)

    def _describe(self) -> list[RemoteVoice]:
        with socket.create_connection((self.host, self.port), timeout=5) as sock:
            stream = sock.makefile("rwb")
            _write_event(stream, "describe")
            while True:
                event_type, data, _ = _read_event(stream)
                if event_type == "info":
                    break
        voices: list[RemoteVoice] = []
        for program in data.get("tts") or []:
            for item in program.get("voices") or []:
                languages = item.get("languages") or ["unknown"]
                language = str(languages[0])
                if self.languages and not any(
                    language.casefold().startswith(prefix) for prefix in self.languages
                ):
                    continue
                speakers = [str(s.get("name")) for s in item.get("speakers") or []]
                voices.append(RemoteVoice(name=str(item["name"]), language=language, speakers=speakers))
        return voices

    def synthesize(self, text: str, voice: str, speaker_id: int = 0) -> bytes:
        remote = self.find(voice)
        voice_data: dict[str, Any] = {"name": voice}
        if remote and remote.speakers:
            voice_data["speaker"] = remote.speakers[min(speaker_id, len(remote.speakers) - 1)]
        output = io.BytesIO()
        wav_file: wave.Wave_write | None = None
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            stream = sock.makefile("rwb")
            _write_event(stream, "synthesize", {"text": text, "voice": voice_data})
            while True:
                event_type, data, payload = _read_event(stream)
                if event_type == "audio-start":
                    wav_file = wave.open(output, "wb")
                    wav_file.setframerate(int(data["rate"]))
                    wav_file.setsampwidth(int(data["width"]))
                    wav_file.setnchannels(int(data["channels"]))
                elif event_type == "audio-chunk" and wav_file is not None:
                    wav_file.writeframes(payload)
                elif event_type == "audio-stop":
                    break
                elif event_type == "error":
                    raise RuntimeError(f"Piper server error: {data.get('text', data)}")
        if wav_file is None:
            raise RuntimeError("Piper server returned no audio")
        wav_file.close()
        return output.getvalue()


def _write_event(stream: BinaryIO, event_type: str, data: dict[str, Any] | None = None) -> None:
    header: dict[str, Any] = {"type": event_type, "version": "1.8.0"}
    body = json.dumps(data).encode("utf-8") if data else b""
    if body:
        header["data_length"] = len(body)
    stream.write(json.dumps(header).encode("utf-8") + b"\n" + body)
    stream.flush()


def _read_event(stream: BinaryIO) -> tuple[str, dict[str, Any], bytes]:
    line = stream.readline()
    if not line:
        raise ConnectionError("Piper server closed the connection")
    header = json.loads(line)
    data: dict[str, Any] = dict(header.get("data") or {})
    if header.get("data_length"):
        data.update(json.loads(_read_exact(stream, int(header["data_length"]))))
    payload = _read_exact(stream, int(header["payload_length"])) if header.get("payload_length") else b""
    return str(header["type"]), data, payload


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    chunk = stream.read(length)
    if chunk is None or len(chunk) != length:
        raise ConnectionError("Piper server closed the connection mid-event")
    return chunk
