"""Speech Model Toolkit: wake word and Piper voice training in one web app.

  /            landing page with tabs (static/index.html)
  /api/status  combined status of both tools, polled by the landing page
  /wakeword/   wake word trainer (app/wakeword, from easy-wakeword-trainer)
  /voice/      Piper voice trainer (app/voice, from piper-voice-helper)
  /lab/        Test Lab: wake word + voice (+ optional AI) together (app/lab)
  /settings/   AI connections, speech recognition and audio (app/settings)
"""

import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .lab import main as lab
from .settings import main as settings
from .settings import providers
from .settings import store as settings_store
from .voice import main as voice
from .wakeword import main as wakeword
from .wakeword import pipeline as ww

_DIR = Path(__file__).parent

# Not in every system mime table; browsers want this type for the web app manifest
mimetypes.add_type("application/manifest+json", ".webmanifest")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Mounted sub-apps don't get lifespan events, so run their startup here
    await wakeword.startup()
    await voice.startup()
    yield


app = FastAPI(
    title="Speech Model Toolkit", lifespan=lifespan, docs_url=None, redoc_url=None
)


class Revalidate:
    """Browsers must re-check pages/scripts/styles (a cheap 304 when unchanged),
    so an updated container never runs new HTML with stale cached JS or CSS.

    Plain ASGI (only touches response headers): unlike @app.middleware it leaves
    streaming responses and long background training tasks alone."""

    def __init__(self, app) -> None:  # Starlette passes the wrapped app as app=
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_header(message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                if not any(k.lower() == b"cache-control" for k, _ in headers):
                    headers.append((b"cache-control", b"no-cache"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_header)


app.add_middleware(Revalidate)


@app.get("/api/status")
async def api_status() -> Dict[str, Any]:
    ww_job = ww.active_job()
    ww_models = [m["name"] for m in ww.list_models()]
    voices = voice.store.list()
    conn = settings_store.active_connection()
    return {
        "wakeword": {
            "dataReady": ww.all_data_present(),
            "preparing": ww.prepare_running(),
            "training": ww_job.phrase if ww_job else None,
            "stage": ww_job.stage.value if ww_job else None,
            "models": ww_models,
        },
        "voice": {
            "device": voice.trainer.device,
            "training": voice.trainer.busy_voice(),
            "voices": [v.name for v in voices],
        },
        "assistant": {
            "connection": conn["name"] if conn else None,
            "provider": providers.get_provider(conn["provider"]).label if conn else None,
            "model": conn["model"] if conn else None,
        },
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_DIR / "static" / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(_DIR / "static" / "icon.svg", media_type="image/svg+xml")


app.mount("/wakeword", wakeword.app, name="wakeword")
app.mount("/voice", voice.app, name="voice")
app.mount("/lab", lab.app, name="lab")
app.mount("/settings", settings.app, name="settings")
app.mount("/static", StaticFiles(directory=_DIR / "static"), name="static")
