"""Test Lab: say your wake word, hear your voice answer. Mounted at /lab."""

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..settings import providers
from ..settings import store as settings_store
from ..wakeword import pipeline as ww
from . import session, stt, tts

_DIR = Path(__file__).parent

app = FastAPI(title="Test Lab", docs_url=None, redoc_url=None)


@app.exception_handler(KeyError)
async def not_found(_request: Request, err: KeyError) -> Response:
    return Response(str(err.args[0]), status_code=404)


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
    }


@app.websocket("/api/session")
async def api_session(websocket: WebSocket) -> None:
    await session.LabSession(websocket).run()


class AskRequest(BaseModel):
    text: str
    voice: str = tts.DEFAULT_VOICE
    history: list = []


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
    try:
        reply = await session.ask_ai(history, text)
    except providers.ProviderError as err:
        error = str(err)
        reply = session.AI_FAILED

    try:
        wav = await tts.synthesize(request.voice, reply)
        audio_url = f"api/audio/{session.store_audio(wav)}.wav"
    except Exception as err:
        audio_url = None
        error = error or f"Could not speak the reply: {err}"

    return {"reply": reply, "audioUrl": audio_url, "error": error}


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
