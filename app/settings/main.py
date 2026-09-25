"""Settings: AI connections, assistant, speech recognition and audio. Mounted at /settings."""

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

import logging

from .. import errors
from ..lab import stt
from . import providers, store

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Settings", docs_url=None, redoc_url=None)


errors.install(app, "settings")


@app.exception_handler(providers.ProviderError)
async def provider_error(request: Request, err: providers.ProviderError) -> Response:
    # The provider's own message, e.g. "API key rejected (HTTP 401): ..."
    _LOGGER.warning("%s %s: %s", request.method, request.url.path, err)
    return Response(str(err), status_code=400)


@app.get("/api/settings")
async def api_settings() -> Dict[str, Any]:
    return store.public()


@app.put("/api/settings")
async def api_update(values: Dict[str, Any]) -> Dict[str, Any]:
    result = store.update(values)
    # What changed, never the values of keys
    changed = [k if k != "connections" else f"connections ({len(values[k])})" for k in values]
    _LOGGER.info("Settings updated: %s", ", ".join(changed))
    return result


@app.get("/api/providers")
async def api_providers() -> Dict[str, Any]:
    return {"providers": providers.catalog()}


@app.post("/api/connections/models")
async def api_models(conn: Dict[str, Any]) -> Dict[str, Any]:
    """Models for a draft connection (a blank key uses the saved connection's key)."""
    return {"models": await providers.list_models(store.fill_key(conn))}


@app.post("/api/connections/test")
async def api_test(conn: Dict[str, Any]) -> Dict[str, Any]:
    return await providers.test(store.fill_key(conn))


# ---- Speech recognition --------------------------------------------------------


@app.get("/api/stt")
async def api_stt(model: str = "") -> Dict[str, Any]:
    return stt.status(model or store.load()["speech"]["sttModel"])


@app.post("/api/stt/download")
async def api_stt_download(body: Dict[str, Any]) -> Dict[str, Any]:
    model = body.get("model") or store.load()["speech"]["sttModel"]
    if model not in {m["id"] for m in stt.MODELS}:
        raise ValueError(f"Unknown model: {model}")
    try:
        await stt.ensure(model)
    except Exception as err:  # download or load failure
        _LOGGER.exception("Downloading Whisper %s failed", model)
        raise HTTPException(500, f"Could not download or load Whisper {model}: {err}. "
                                 "Check the server's internet connection and free disk space.") from err
    return stt.status(model)


# -----------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=_DIR / "static"), name="static")
