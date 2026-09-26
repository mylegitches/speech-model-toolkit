"""One Test Lab conversation over a WebSocket.

Client → server
  text   {"type": "start", "wakeword": name, "voice": id}   first message
         {"type": "playback_done"}   the reply finished playing: listen again
         {"type": "reset"}           forget the conversation so far
         {"type": "voice", "voice": id}   switch the reply voice
         {"type": "speed", "speed": 90}   reply speed in % (start may carry it too)
         {"type": "ask", "text": "..."}   a typed question (while listening)
  binary 16 kHz mono s16le PCM from the microphone

Server → client (JSON)
  {"type": "ready"}
  {"type": "score", "score": 0.12}
  {"type": "state", "state": listening|heard|recording|transcribing|thinking|speaking, "detail": "..."}
  {"type": "transcript", "text": "..."}          what you said
  {"type": "reply", "text": "...", "audioUrl": "api/audio/<id>.wav"}
  {"type": "error", "message": "..."}

After the wake word: with an active AI connection it records your question
until you stop talking (Silero VAD), transcribes it (Whisper) and asks the AI;
otherwise it speaks the fixed reply. Mic audio is ignored while busy and while
the reply plays, so the reply can't trigger the wake word.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from ..settings import providers
from ..settings import store as settings_store
from ..wakeword import pipeline as ww
from . import stt, tts

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WAKE_FRAME = 1280  # openWakeWord: 80 ms
VAD_FRAME = 512  # Silero VAD: 32 ms
VAD_THRESHOLD = 0.5
NO_SPEECH_TIMEOUT = 5.0
HISTORY_MESSAGES = 12  # last 6 exchanges are sent to the AI

DIDNT_CATCH = "Sorry, I didn't catch that."
AI_FAILED = "Sorry, I couldn't reach the AI."

_VAD_MODEL = Path(__file__).resolve().parents[2] / "export_dataset" / "models" / "silero_vad.onnx"

# Generated replies, served by GET api/audio/{id}.wav
_AUDIO: "OrderedDict[str, bytes]" = OrderedDict()
_AUDIO_LIMIT = 40


def store_audio(wav: bytes) -> str:
    audio_id = uuid.uuid4().hex
    _AUDIO[audio_id] = wav
    while len(_AUDIO) > _AUDIO_LIMIT:
        _AUDIO.popitem(last=False)
    return audio_id


def get_audio(audio_id: str) -> Optional[bytes]:
    return _AUDIO.get(audio_id)


async def ask_ai(history: List[Dict[str, str]], question: str) -> str:
    """Send the question (plus recent history) to the active AI connection."""
    settings = settings_store.load()
    conn = settings_store.active_connection(settings)
    if conn is None:
        return settings["assistant"]["fixedReply"]

    assistant = settings["assistant"]
    messages = history[-HISTORY_MESSAGES:] + [{"role": "user", "content": question}]
    return await providers.chat(
        conn,
        messages,
        system=assistant["systemPrompt"],
        temperature=float(assistant["temperature"]),
        max_tokens=int(assistant["maxTokens"]),
    )


class LabSession:
    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        self.state = "idle"
        self.voice = tts.DEFAULT_VOICE
        self.speed = 100
        self.wakeword = ""
        self.detector: Any = None
        self.history: List[Dict[str, str]] = []
        self.task: Optional[asyncio.Task] = None

        self.settings = settings_store.load()
        self.wake_buf = np.array([], dtype=np.int16)
        self.cooldown_until = 0.0

        # Question recording
        self.vad: Any = None
        self.vad_buf = np.array([], dtype=np.int16)
        self.recording: List[bytes] = []
        self.record_started = 0.0
        self.speech_started = False
        self.silence_ms = 0.0

    # ---- Messages ----------------------------------------------------------

    async def send(self, **message: Any) -> None:
        try:
            await self.ws.send_json(message)
        except (RuntimeError, WebSocketDisconnect):
            pass

    async def set_state(self, state: str, detail: str = "") -> None:
        self.state = state
        await self.send(type="state", state=state, detail=detail)

    # ---- Main loop ----------------------------------------------------------

    async def run(self) -> None:
        await self.ws.accept()
        try:
            start = json.loads(await self.ws.receive_text())
            if start.get("type") != "start":
                raise ValueError("Expected a start message")
            await self.start(start)

            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await self.on_audio(message["bytes"])
                elif message.get("text") is not None:
                    await self.on_control(json.loads(message["text"]))
        except WebSocketDisconnect:
            pass
        except ValueError as err:  # e.g. unknown wake word
            _LOGGER.warning("Test Lab session rejected: %s", err)
            await self.send(type="error", message=str(err))
        except Exception as err:  # report, then close
            _LOGGER.exception("Test Lab session failed")
            await self.send(type="error", message=f"The session stopped unexpectedly: {err}. Details are in the server log.")
        finally:
            if self.task and not self.task.done():
                self.task.cancel()

    async def start(self, start: Dict[str, Any]) -> None:
        self.wakeword = str(start.get("wakeword") or "")
        self.voice = str(start.get("voice") or tts.DEFAULT_VOICE)
        self.speed = tts.clamp_speed(start.get("speed", 100))
        onnx_path = ww.find_model(self.wakeword)
        if onnx_path is None:
            raise ValueError(f"Wake word model '{self.wakeword}' not found. Was it deleted? Reload the page.")
        conn = settings_store.active_connection(self.settings)
        _LOGGER.info("Test Lab session: wake word %s, voice %s, replies: %s", self.wakeword, self.voice,
                     f"AI {conn['provider']}/{conn['model']}" if conn else "fixed")

        await self.set_state("loading", "Loading wake word model…")
        self.detector = await asyncio.to_thread(ww.load_detector, onnx_path)
        await self.send(type="ready")
        await self.listen()

    async def listen(self) -> None:
        self.settings = settings_store.load()
        if self.detector is not None:
            self.detector.reset()
        self.wake_buf = np.array([], dtype=np.int16)
        await self.set_state("listening", f"Say “{self.wakeword.replace('_', ' ')}”")

    async def on_control(self, message: Dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "playback_done" and self.state == "speaking":
            self.cooldown_until = time.monotonic() + float(self.settings["audio"]["cooldownSec"])
            await self.listen()
        elif kind == "reset":
            self.history.clear()
        elif kind == "voice":
            self.voice = str(message.get("voice") or tts.DEFAULT_VOICE)
            if "speed" in message:
                self.speed = tts.clamp_speed(message["speed"])
        elif kind == "speed":
            self.speed = tts.clamp_speed(message.get("speed"))
        elif kind == "ask" and self.state == "listening":
            # Typed question during a session: shares this conversation's history
            question = str(message.get("text") or "").strip()[:2000]
            if question:
                self.settings = settings_store.load()
                self.state = "thinking"
                await self.send(type="transcript", text=question)
                self.spawn(self.answer_text(question))

    # ---- Audio ----------------------------------------------------------------

    async def on_audio(self, data: bytes) -> None:
        if self.state == "listening":
            await self.detect_wake(np.frombuffer(data, dtype=np.int16))
        elif self.state == "recording":
            await self.record(data)
        # Busy or speaking: ignore the microphone

    async def detect_wake(self, chunk: np.ndarray) -> None:
        self.wake_buf = np.concatenate([self.wake_buf, chunk])
        threshold = float(self.settings["audio"]["threshold"])
        while len(self.wake_buf) >= WAKE_FRAME and self.state == "listening":
            frame = self.wake_buf[:WAKE_FRAME]
            self.wake_buf = self.wake_buf[WAKE_FRAME:]
            prediction = self.detector.predict(frame)
            score = float(list(prediction.values())[0])
            await self.send(type="score", score=round(score, 4))

            if score >= threshold and time.monotonic() >= self.cooldown_until:
                await self.on_wake()

    async def on_wake(self) -> None:
        _LOGGER.info("Test Lab: wake word %s detected", self.wakeword)
        await self.set_state("heard", "Wake word detected")
        if settings_store.active_connection(self.settings) is None:
            self.spawn(self.speak(self.settings["assistant"]["fixedReply"]))
            return

        from export_dataset.vad import SileroVoiceActivityDetector

        self.vad = SileroVoiceActivityDetector(_VAD_MODEL)
        self.vad_buf = np.array([], dtype=np.int16)
        self.recording = []
        self.record_started = time.monotonic()
        self.speech_started = False
        self.silence_ms = 0.0
        await self.set_state("recording", "Listening to your question…")

    async def record(self, data: bytes) -> None:
        self.recording.append(data)
        self.vad_buf = np.concatenate([self.vad_buf, np.frombuffer(data, dtype=np.int16)])
        speech = self.settings["speech"]
        frame_ms = VAD_FRAME * 1000 / SAMPLE_RATE

        while len(self.vad_buf) >= VAD_FRAME:
            frame = self.vad_buf[:VAD_FRAME].astype(np.float32) / 32768.0
            self.vad_buf = self.vad_buf[VAD_FRAME:]
            if float(self.vad(frame)) >= VAD_THRESHOLD:
                self.speech_started = True
                self.silence_ms = 0.0
            elif self.speech_started:
                self.silence_ms += frame_ms

        elapsed = time.monotonic() - self.record_started
        if self.speech_started and self.silence_ms >= float(speech["silenceMs"]):
            self.finish_recording()
        elif elapsed >= float(speech["maxSeconds"]):
            self.finish_recording()
        elif not self.speech_started and elapsed >= NO_SPEECH_TIMEOUT:
            self.state = "thinking"
            self.spawn(self.speak(DIDNT_CATCH))

    def finish_recording(self) -> None:
        self.state = "transcribing"
        pcm = b"".join(self.recording)
        self.recording = []
        self.spawn(self.answer_speech(pcm))

    # ---- Replies -----------------------------------------------------------------

    def spawn(self, coro) -> None:
        self.task = asyncio.create_task(coro)

    async def answer_speech(self, pcm: bytes) -> None:
        speech = self.settings["speech"]
        detail = "Transcribing…" if stt.is_cached(speech["sttModel"]) else "Downloading the speech model (first use)…"
        await self.set_state("transcribing", detail)
        try:
            question = await stt.transcribe(pcm, speech["sttModel"], speech["language"])
        except Exception as err:
            _LOGGER.exception("Speech recognition failed")
            await self.send(type="error", message=f"Speech recognition failed: {err}")
            await self.speak(DIDNT_CATCH)
            return

        if not question:
            await self.speak(DIDNT_CATCH)
            return

        await self.send(type="transcript", text=question)
        await self.answer_text(question)

    async def answer_text(self, question: str) -> None:
        await self.set_state("thinking", "Asking the AI…")
        try:
            reply = await ask_ai(self.history, question)
        except providers.ProviderError as err:
            _LOGGER.warning("Test Lab: AI request failed: %s", err)
            await self.send(type="error", message=f"The AI didn't answer: {err}")
            await self.speak(AI_FAILED)
            return

        self.history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": reply},
        ]
        await self.speak(reply)

    async def speak(self, text: str) -> None:
        await self.set_state("speaking", "Speaking…")
        try:
            wav = await tts.synthesize(self.voice, text, self.speed)
        except Exception as err:
            _LOGGER.exception("Speech synthesis failed")
            await self.send(type="reply", text=text, audioUrl=None)
            await self.send(type="error", message=f"Could not speak the reply: {err}")
            await self.listen()
            return

        audio_id = store_audio(wav)
        # The client plays it and answers with playback_done
        await self.send(type="reply", text=text, audioUrl=f"api/audio/{audio_id}.wav")
