"""Voice Room tab: characters (Piper voices + a personality) in rooms, talking to each other
live through the AI, interruptible by text or microphone. Mounted at /agents.

Built from github.com/mylegitches/voice-agent-freestyle: its Characters, Rooms and Live
Room screens (React, built into web/dist by the Docker image) and its server, adapted to
the toolkit's voices, AI connection, speech recognition and data folder (see config.py).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.characters import router as characters_router
from .api.characters import voices_router
from .api.rooms import router as rooms_router
from .api.sessions import router as sessions_router
from .config import get_settings
from .db import close_db, init_db
from .llm import llm_service
from .stt import stt_service
from .tts import tts_service
from .voice_catalog import voice_catalog
from .ws import router as ws_router

_LOGGER = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title="Voice Room", version="0.1.0", docs_url=None, redoc_url=None)
app.include_router(characters_router)
app.include_router(voices_router)
app.include_router(rooms_router)
app.include_router(sessions_router)
app.include_router(ws_router)


async def startup() -> None:
    """Called by the toolkit's lifespan (mounted sub-apps don't get their own)."""
    try:
        await init_db()
    except Exception:  # never stop the rest of the toolkit starting
        _LOGGER.exception("Voice Room: the database couldn't be prepared")


async def shutdown() -> None:
    await close_db()


@app.get("/api/health")
async def health(verify_llm: bool = Query(default=False)):
    voices = await asyncio.to_thread(voice_catalog.scan)
    conn = llm_service.connection()
    llm_ok = conn is not None
    llm_message = conn["label"] if conn else "No AI connection: add one in Settings → AI connections"
    if verify_llm and llm_ok:
        llm_ok, llm_message = await llm_service.verify_model()
    return {
        "status": "ready" if voices else "degraded",
        "voices": {
            "count": len(voices),
            "errors": [voice.id for voice in voices if voice.status == "error"],
            "tts_dependency": tts_service.available(),
            "piper_server": (
                {"host": f"{settings.piper_host}:{settings.piper_port}", "error": voice_catalog.remote.last_error}
                if voice_catalog.remote else None
            ),
        },
        "stt": {"dependency": stt_service.available(), "model": settings.whisper_model},
        "llm": {
            "configured": conn is not None,
            "verified": llm_ok if verify_llm else None,
            "model": conn["model"] if conn else None,
            "message": llm_message,
        },
    }


if settings.web_dist.exists():
    assets = settings.web_dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        requested = (settings.web_dist / path).resolve()
        if path and requested.is_file() and settings.web_dist.resolve() in requested.parents:
            return FileResponse(requested)
        return FileResponse(settings.web_dist / "index.html")
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse({
            "name": "Voice Room",
            "message": "The Voice Room screens aren't built. They're built by the Docker image "
                       "(or run npm install && npm run build in app/agents/web).",
        })
