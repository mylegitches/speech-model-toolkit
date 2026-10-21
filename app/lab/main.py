"""Test Lab: say your wake word, hear your voice answer. Mounted at /lab."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import errors
from ..settings import providers
from ..settings import store as settings_store
from ..wakeword import pipeline as ww
from . import pauses, session, stt, tts

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Test Lab", docs_url=None, redoc_url=None)


errors.install(app, "lab")


def assistant_info(settings: Dict[str, Any]) -> Dict[str, Any]:
    conn = settings_store.active_connection(settings)
    if conn is None:
        return {"mode": "fixed", "fixedReply": settings["assistant"]["fixedReply"]}

    provider = providers.get_provider(conn["provider"])
    return {
        "mode": "ai",
        "connection": conn["name"],
        "provider": provider.label,
        "model": conn["model"],
    }


@app.get("/api/options")
async def api_options() -> Dict[str, Any]:
    settings = settings_store.load()
    return {
        "wakewords": [m["name"] for m in ww.list_models()],
        "voices": tts.options(),
        "assistant": assistant_info(settings),
        "stt": stt.status(settings["speech"]["sttModel"]),
        "allowDownloads": settings["audio"]["allowDownloads"],
        "audio": settings["audio"],
        "pauses": settings["pauses"],
        "pauseMarks": [{"key": k, "label": label, "default": ms, "max": top} for k, label, ms, top in pauses.MARKS],
    }


@app.websocket("/api/session")
async def api_session(websocket: WebSocket) -> None:
    await session.LabSession(websocket).run()


class AskRequest(BaseModel):
    text: str
    voice: str = tts.DEFAULT_VOICE
    speed: int = 100
    history: list = []
    repeat: bool = False  # speak the text as it is, no AI


@app.post("/api/ask")
async def api_ask(request: AskRequest) -> Dict[str, Any]:
    """A typed question: same AI + voice path as a spoken one, no microphone needed."""
    text = request.text.strip()[:2000]
    if not text:
        raise HTTPException(400, "Type a question")

    history = [
        {"role": m["role"], "content": str(m["content"])}
        for m in request.history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ]
    error = None
    if request.repeat:
        reply = text  # Repeat mode: the words as typed (break tags and pauses still apply)
    else:
        try:
            reply = await session.ask_ai(history, text)
        except providers.ProviderError as err:
            _LOGGER.warning("Test Lab typed question: AI request failed: %s", err)
            error = f"The AI didn't answer: {err}"
            reply = session.AI_FAILED

    try:
        wav = await tts.synthesize(request.voice, reply, request.speed)
        audio_url = f"api/audio/{session.store_audio(wav)}.wav"
    except Exception as err:
        _LOGGER.exception("Test Lab: speaking the reply with %s failed", request.voice)
        audio_url = None
        error = error or f"Could not speak the reply: {err}"

    return {"reply": reply, "audioUrl": audio_url, "error": error}


@app.put("/api/pauses")
async def api_pauses(values: Dict[str, Any]) -> Dict[str, Any]:
    """Save the punctuation pauses (Test Lab → Pauses)."""
    return settings_store.update({"pauses": values})["pauses"]


class ScriptRequest(BaseModel):
    lines: List[Dict[str, Any]]


@app.post("/api/script")
async def api_script(request: ScriptRequest) -> Dict[str, Any]:
    """A conversation, line by line in each line's voice, as one recording."""
    lines = [line for line in request.lines if str(line.get("text") or "").strip()]
    if not lines:
        raise HTTPException(400, "Write at least one line")
    if len(lines) > tts.MAX_SCRIPT_LINES:
        raise HTTPException(400, f"At most {tts.MAX_SCRIPT_LINES} lines")
    if sum(len(str(line.get("text") or "")) for line in lines) > 20000:
        raise HTTPException(400, "The script is too long (20,000 characters at most)")
    wav, marks = await tts.synthesize_script(lines, settings_store.load()["pauses"])
    return {"audioUrl": f"api/audio/{session.store_audio(wav)}.wav", "lines": marks}


@app.get("/api/audio/{audio_id}.wav")
async def api_audio(audio_id: str, download: bool = False) -> Response:
    wav = session.get_audio(audio_id)
    if wav is None:
        raise HTTPException(404, "This reply has expired")

    headers = {}
    if download:
        if not settings_store.load()["audio"]["allowDownloads"]:
            raise HTTPException(403, "Downloading generated audio is turned off in Settings")
        headers["Content-Disposition"] = f'attachment; filename="reply-{audio_id[:8]}.wav"'
    return Response(wav, media_type="audio/wav", headers=headers)


# -----------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=_DIR / "static"), name="static")
