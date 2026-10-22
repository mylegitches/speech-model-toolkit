from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from .bidding import BidCandidate, choose_speaker, was_directly_addressed
from .config import Settings, get_settings
from .db import SessionFactory
from .llm import OllamaLLM, llm_service
from .memory import needs_drift_nudge, render_recent, safe_word_matches
from .models import Character, Line
from .repositories import (
    append_line,
    get_room,
    get_session,
    list_lines,
    set_session_status,
    update_summary,
)
from .stt import WhisperService, stt_service
from .tts import PiperService, tts_service


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ActiveRoom:
    room_id: int
    session_id: int
    websocket: WebSocket
    state: str = "paused"
    revision: int = 0
    generation: int = 0
    pause_requested: bool = False
    loop_task: asyncio.Task[None] | None = None
    mic_task: asyncio.Task[None] | None = None
    mic_data: bytearray = field(default_factory=bytearray)
    mic_mime: str = "audio/webm"
    resume_after_mic: bool = False
    current_turn: str | None = None
    pending_audio: int = 0
    playback_ready: asyncio.Event = field(default_factory=asyncio.Event)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    command_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.playback_ready.set()


class SessionManager:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm: OllamaLLM | None = None,
        tts: PiperService | None = None,
        stt: WhisperService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or llm_service
        self.tts = tts or tts_service
        self.stt = stt or stt_service
        self.active: dict[int, ActiveRoom] = {}
        self._lock = asyncio.Lock()

    async def connect(self, room_id: int, session_id: int, websocket: WebSocket) -> ActiveRoom:
        async with SessionFactory() as db:
            room = await get_room(db, room_id)
            session = await get_session(db, session_id)
            if room is None or session is None or session.room_id != room_id:
                raise ValueError("Room/session pair was not found")
            if session.status == "stopped":
                raise ValueError("This session has already stopped")
            lines = await list_lines(db, session_id)

        async with self._lock:
            existing = self.active.get(room_id)
            if existing is not None and existing.websocket is not websocket:
                raise RuntimeError("This room already has a controlling connection")
            active = ActiveRoom(room_id=room_id, session_id=session_id, websocket=websocket)
            self.active[room_id] = active

        await self._send(
            active,
            {
                "type": "snapshot",
                "session_id": session_id,
                "state": active.state,
                "revision": active.revision,
                "lines": [
                    {
                        "id": line.id,
                        "sequence": line.sequence,
                        "speaker": line.speaker,
                        "character_id": line.character_id,
                        "text": line.text,
                        "ts": line.ts.isoformat(),
                    }
                    for line in lines
                ],
            },
        )
        return active

    async def disconnect(self, active: ActiveRoom) -> None:
        await self._cancel_task(active.loop_task)
        await self._cancel_task(active.mic_task)
        if active.state not in {"stopped", "paused"}:
            active.state = "paused"
            async with SessionFactory() as db:
                await set_session_status(db, active.session_id, "paused")
        async with self._lock:
            if self.active.get(active.room_id) is active:
                del self.active[active.room_id]

    async def handle_json(self, active: ActiveRoom, message: dict[str, Any]) -> None:
        command_id = str(message.get("command_id") or "")
        if command_id:
            if command_id in active.command_ids:
                return
            active.command_ids.add(command_id)
            if len(active.command_ids) > 500:
                active.command_ids = set(list(active.command_ids)[-250:])

        message_type = message.get("type")
        if message_type == "start":
            await self._start(active, str(message.get("topic") or "").strip())
        elif message_type == "pause":
            await self._pause(active)
        elif message_type == "stop":
            await self._stop(active, "Stopped by user")
        elif message_type == "interject_text":
            await self._interject(active, str(message.get("text") or "").strip())
        elif message_type == "mic_start":
            await self._mic_start(active, str(message.get("mime_type") or "audio/webm"))
        elif message_type == "mic_end":
            await self._mic_end(active)
        elif message_type in {"playback_finished", "playback_cancelled"}:
            active.pending_audio = max(0, active.pending_audio - 1)
            if active.pending_audio < 2:
                active.playback_ready.set()
        elif message_type == "ping":
            await self._send(active, {"type": "pong"})
        else:
            await self._error(active, "unknown_command", f"Unknown command: {message_type}")

    async def handle_binary(self, active: ActiveRoom, data: bytes) -> None:
        if len(active.mic_data) + len(data) > 25 * 1024 * 1024:
            active.mic_data.clear()
            await self._error(active, "mic_too_large", "Microphone recording exceeded 25 MB")
            return
        active.mic_data.extend(data)

    async def _start(self, active: ActiveRoom, topic: str) -> None:
        if active.state == "stopped":
            await self._error(active, "session_stopped", "Stopped sessions cannot be resumed")
            return
        if topic:
            await self._append_user(active, topic)
        if active.loop_task and not active.loop_task.done():
            return
        active.pause_requested = False
        await self._set_state(active, "playing")
        active.loop_task = asyncio.create_task(self._conversation_loop(active))

    async def _pause(self, active: ActiveRoom) -> None:
        if active.state != "playing":
            return
        active.pause_requested = True
        await self._set_state(active, "pause_requested", persist=False)

    async def _stop(self, active: ActiveRoom, reason: str) -> None:
        active.generation += 1
        active.pause_requested = False
        current = asyncio.current_task()
        if active.loop_task and active.loop_task is not current:
            await self._cancel_task(active.loop_task)
        active.loop_task = None
        await self._send(active, {"type": "cancel_audio", "reason": reason})
        active.pending_audio = 0
        active.playback_ready.set()
        await self._set_state(active, "stopped", reason=reason)

    async def _interject(self, active: ActiveRoom, text: str) -> None:
        if not text:
            await self._error(active, "empty_interjection", "Interjection text is empty")
            return
        was_playing = active.state in {"playing", "pause_requested"}
        await self._cancel_for_interjection(active)
        await self._append_user(active, text)
        async with SessionFactory() as db:
            room = await get_room(db, active.room_id)
        if room and safe_word_matches(text, room.safe_word):
            await self._stop(active, "Safe word detected")
            return
        if was_playing:
            await self._start(active, "")

    async def _cancel_for_interjection(self, active: ActiveRoom) -> None:
        active.generation += 1
        await self._cancel_task(active.loop_task)
        active.loop_task = None
        active.current_turn = None
        active.pending_audio = 0
        active.playback_ready.set()
        await self._send(active, {"type": "cancel_audio", "reason": "interjection"})
        await self._set_state(active, "paused")

    async def _mic_start(self, active: ActiveRoom, mime_type: str) -> None:
        active.resume_after_mic = active.state in {"playing", "pause_requested"}
        await self._cancel_for_interjection(active)
        active.mic_data.clear()
        active.mic_mime = mime_type
        await self._send(active, {"type": "mic_state", "state": "recording"})

    async def _mic_end(self, active: ActiveRoom) -> None:
        if not active.mic_data:
            await self._error(active, "empty_audio", "No microphone audio was received")
            return
        audio = bytes(active.mic_data)
        active.mic_data.clear()
        await self._send(active, {"type": "mic_state", "state": "transcribing"})
        active.mic_task = asyncio.create_task(self._finish_mic(active, audio))

    async def _finish_mic(self, active: ActiveRoom, audio: bytes) -> None:
        suffix = ".webm"
        if "ogg" in active.mic_mime:
            suffix = ".ogg"
        elif "wav" in active.mic_mime:
            suffix = ".wav"
        try:
            text = await self.stt.transcribe(audio, suffix=suffix)
            await self._send(active, {"type": "transcription", "text": text})
            if not text:
                await self._error(active, "no_speech", "No speech was detected")
                return
            await self._append_user(active, text)
            async with SessionFactory() as db:
                room = await get_room(db, active.room_id)
            if room and safe_word_matches(text, room.safe_word):
                await self._stop(active, "Safe word detected")
            elif active.resume_after_mic:
                await self._start(active, "")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Microphone transcription failed")
            await self._error(active, "transcription_failed", str(exc))
        finally:
            await self._send(active, {"type": "mic_state", "state": "idle"})

    async def _append_user(self, active: ActiveRoom, text: str) -> Line:
        async with SessionFactory() as db:
            line = await append_line(
                db, session_id=active.session_id, speaker="You", text=text, character_id=None
            )
        await self._send(
            active,
            {
                "type": "line_final",
                "line": self._line_payload(line),
                "turn_id": f"user-{line.id}",
            },
        )
        return line

    async def _conversation_loop(self, active: ActiveRoom) -> None:
        try:
            while active.state in {"playing", "pause_requested"}:
                generation = active.generation
                async with SessionFactory() as db:
                    room = await get_room(db, active.room_id)
                    session = await get_session(db, active.session_id)
                    lines = await list_lines(db, active.session_id)
                if room is None or session is None:
                    await self._error(active, "session_missing", "Room or session was deleted")
                    await self._stop(active, "Session missing")
                    return
                characters = [member.character for member in sorted(room.members, key=lambda x: x.position)]
                if len(characters) < 2:
                    await self._error(active, "not_enough_characters", "Room needs two characters")
                    await self._set_state(active, "paused")
                    return
                if not self.llm.configured:
                    await self._error(
                        active,
                        "llm_not_configured",
                        "No AI connection: add one in Settings → AI connections (and make it active) before starting a conversation",
                    )
                    await self._set_state(active, "paused")
                    return

                settings = {**self.settings.room_defaults, **(room.settings_json or {})}
                max_context = int(settings.get("max_context_lines", 30))
                transcript = render_recent(lines, max_context)
                last_character_id = next(
                    (line.character_id for line in reversed(lines) if line.character_id is not None), None
                )
                eligible = [c for c in characters if c.id != last_character_id]
                latest_text = lines[-1].text if lines else ""
                raw_bids = await asyncio.gather(
                    *[
                        self._safe_bid(c, session.summary, transcript)
                        for c in eligible
                    ]
                )
                candidates = [
                    BidCandidate(
                        character_id=character.id,
                        name=character.name,
                        raw_bid=raw_bid,
                        eagerness=character.eagerness,
                        turns_since_spoke=self._turns_since(lines, character.id),
                        directly_addressed=was_directly_addressed(character.name, latest_text),
                    )
                    for character, raw_bid in zip(eligible, raw_bids, strict=True)
                ]
                winner = choose_speaker(
                    candidates,
                    eagerness_weight=float(settings.get("eagerness_weight", 2.0)),
                    direct_address_bonus=float(settings.get("direct_address_bonus", 3.0)),
                )
                character = next(c for c in eligible if c.id == winner.character_id)
                bid_payload = {
                    "candidates": [
                        {
                            "character_id": candidate.character_id,
                            "name": candidate.name,
                            "raw_bid": round(candidate.raw_bid, 2),
                            "score": round(candidate.score, 2),
                        }
                        for candidate in candidates
                    ],
                    "winner": winner.character_id,
                }
                turn_id = uuid.uuid4().hex
                active.current_turn = turn_id
                await self._send(active, {"type": "bid", "turn_id": turn_id, **bid_payload})
                await self._send(
                    active,
                    {
                        "type": "line_start",
                        "turn_id": turn_id,
                        "speaker": character.name,
                        "character_id": character.id,
                        "color": character.color,
                    },
                )
                chunks: list[str] = []
                try:
                    async for chunk in self.llm.stream_line(
                        name=character.name,
                        personality=character.system_prompt,
                        max_words=character.max_words,
                        summary=session.summary,
                        transcript=transcript,
                        drift_nudge=needs_drift_nudge(lines),
                    ):
                        if generation != active.generation:
                            return
                        chunks.append(chunk)
                        await self._send(
                            active, {"type": "line_delta", "turn_id": turn_id, "text": chunk}
                        )
                except Exception as exc:
                    logger.exception("Character generation failed")
                    await self._error(active, "generation_failed", str(exc))
                    await self._set_state(active, "paused")
                    return

                if generation != active.generation:
                    return
                text = "".join(chunks).strip()
                if not text:
                    await self._error(active, "empty_generation", "The model returned an empty line")
                    await self._set_state(active, "paused")
                    return
                words = text.split()
                if len(words) > character.max_words:
                    text = " ".join(words[: character.max_words]).rstrip(" ,;:") + "…"
                async with SessionFactory() as db:
                    saved = await append_line(
                        db,
                        session_id=active.session_id,
                        speaker=character.name,
                        text=text,
                        character_id=character.id,
                        bid_json=bid_payload,
                    )
                await self._send(
                    active,
                    {"type": "line_final", "turn_id": turn_id, "line": self._line_payload(saved)},
                )
                active.current_turn = None

                try:
                    audio = await self.tts.synthesize(
                        text,
                        character.voice_path,
                        character.voice_config_path,
                        character.speaker_id,
                        character.length_scale,
                    )
                    if generation != active.generation:
                        return
                    if self.settings.save_session_audio:
                        await self._save_audio(active.session_id, saved.id, audio)
                    await self._send(
                        active,
                        {
                            "type": "audio_start",
                            "line_id": saved.id,
                            "turn_id": turn_id,
                            "mime_type": "audio/wav",
                            "byte_length": len(audio),
                        },
                    )
                    await self._send_bytes(active, audio)
                    active.pending_audio += 1
                    if active.pending_audio >= 2:
                        active.playback_ready.clear()
                except Exception as exc:
                    logger.exception("Speech synthesis failed")
                    await self._error(active, "tts_failed", str(exc), recoverable=True)

                await self._maybe_summarize(active, settings)
                if active.pause_requested:
                    active.pause_requested = False
                    await self._set_state(active, "paused")
                    return
                if active.pending_audio >= 2:
                    try:
                        await asyncio.wait_for(active.playback_ready.wait(), timeout=120)
                    except TimeoutError:
                        active.pending_audio = 0
                        active.playback_ready.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Room loop failed")
            with contextlib.suppress(Exception):
                await self._error(active, "room_loop_failed", str(exc))
                await self._set_state(active, "paused")

    async def _safe_bid(self, character: Character, summary: str, transcript: str) -> float:
        try:
            return await asyncio.wait_for(
                self.llm.bid(
                    name=character.name,
                    personality=character.system_prompt,
                    summary=summary,
                    transcript=transcript,
                ),
                timeout=self.settings.llm_timeout_seconds,
            )
        except Exception:
            logger.warning("Bid failed for %s; using neutral fallback", character.name)
            return 5.0

    async def _maybe_summarize(self, active: ActiveRoom, settings: dict[str, Any]) -> None:
        interval = max(1, int(settings.get("summary_interval", 20)))
        max_context = max(5, int(settings.get("max_context_lines", 30)))
        async with SessionFactory() as db:
            lines = await list_lines(db, active.session_id)
            session = await get_session(db, active.session_id)
        if session is None or len(lines) <= max_context or len(lines) % interval != 0:
            return
        old_lines = lines[:-max_context]
        try:
            summary = await self.llm.summarize(session.summary, render_recent(old_lines, interval))
            async with SessionFactory() as db:
                await update_summary(db, active.session_id, summary)
        except Exception:
            logger.warning("Summary refresh failed", exc_info=True)

    async def _save_audio(self, session_id: int, line_id: int, audio: bytes) -> None:
        folder = self.settings.data_dir / "sessions" / str(session_id)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{line_id}.wav"
        await asyncio.to_thread(target.write_bytes, audio)

    @staticmethod
    def _turns_since(lines: list[Line], character_id: int) -> int:
        count = 10
        for distance, line in enumerate(reversed(lines), start=1):
            if line.character_id == character_id:
                return distance
        return count

    async def _set_state(
        self, active: ActiveRoom, state: str, *, reason: str | None = None, persist: bool = True
    ) -> None:
        active.state = state
        active.revision += 1
        if persist:
            persisted = "playing" if state == "pause_requested" else state
            async with SessionFactory() as db:
                await set_session_status(db, active.session_id, persisted)
        await self._send(
            active,
            {"type": "state", "state": state, "revision": active.revision, "reason": reason},
        )

    async def _error(
        self,
        active: ActiveRoom,
        code: str,
        message: str,
        *,
        recoverable: bool = False,
    ) -> None:
        await self._send(
            active,
            {"type": "error", "code": code, "message": message, "recoverable": recoverable},
        )

    async def _send(self, active: ActiveRoom, payload: dict[str, Any]) -> None:
        async with active.send_lock:
            await active.websocket.send_json(payload)

    async def _send_bytes(self, active: ActiveRoom, payload: bytes) -> None:
        async with active.send_lock:
            await active.websocket.send_bytes(payload)

    @staticmethod
    async def _cancel_task(task: asyncio.Task[Any] | None) -> None:
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @staticmethod
    def _line_payload(line: Line) -> dict[str, Any]:
        return {
            "id": line.id,
            "sequence": line.sequence,
            "speaker": line.speaker,
            "character_id": line.character_id,
            "text": line.text,
            "ts": line.ts.isoformat(),
        }


session_manager = SessionManager()
