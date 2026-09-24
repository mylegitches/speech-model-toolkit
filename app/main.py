"""Speech Model Toolkit: wake word and Piper voice training in one web app.

  /            landing page with tabs (static/index.html)
  /api/status  combined status of both tools, polled by the landing page
  /wakeword/   wake word trainer (app/wakeword, from easy-wakeword-trainer)
  /voice/      Piper voice trainer (app/voice, from piper-voice-helper)
  /lab/        Test Lab: wake word + voice (+ optional AI) together (app/lab)
  /settings/   AI connections, speech recognition and audio (app/settings)
"""

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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Mounted sub-apps don't get lifespan events, so run their startup here
    await wakeword.startup()
    await voice.startup()
    yield


app = FastAPI(
    title="Speech Model Toolkit", lifespan=lifespan, docs_url=None, redoc_url=None
)


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
