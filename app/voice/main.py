"""Piper Voice Helper: record, train and test a Piper voice in the browser."""

import asyncio
import io
import json
import logging
import os
import re
import shutil
import zipfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import errors
from . import freeform, matching
from .training import (
    ACCELERATORS,
    AUTO_CHECKPOINT,
    DEFAULT_PRESET,
    LATEST_CHECKPOINT,
    PRESETS,
    TrainingManager,
    TrainingSettings,
    Workspace,
    catalog_groups,
    find_train_python,
    load_checkpoint_catalog,
    suggest_checkpoint,
)
from .voices import Voice, VoiceStore, load_prompts

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)
_DIR = Path(__file__).parent
_REPO_DIR = _DIR.parents[1]

DATA_DIR = Path(os.environ.get("VOICE_DATA_DIR", _REPO_DIR / "data" / "voice"))
PROMPTS_DIR = Path(os.environ.get("PROMPTS_DIR", _REPO_DIR / "prompts"))
MAX_UPLOAD_BYTES = 2 * 1024**3

LANGUAGES, PROMPTS = load_prompts(PROMPTS_DIR)
CATALOG = load_checkpoint_catalog()
store = VoiceStore(DATA_DIR / "voices")
trainer = TrainingManager(python=find_train_python())


async def startup() -> None:
    await trainer.detect_device()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Only runs standalone; mounted under app.main, which calls startup() itself
    await startup()
    yield


app = FastAPI(title="Piper Voice Helper", lifespan=lifespan)


errors.install(app, "voice")


# -----------------------------------------------------------------------------


def workspace_for(name: str) -> Workspace:
    return trainer.get(store.get(name))


def default_settings(voice: Voice, workspace: Workspace) -> Dict[str, Any]:
    settings = TrainingSettings()
    if workspace.latest_checkpoint() is not None:
        settings.checkpoint = LATEST_CHECKPOINT
    elif matching.candidates(voice):
        # Closest pretrained voice to the recordings, picked when training starts
        settings.checkpoint = AUTO_CHECKPOINT
    else:
        suggested = suggest_checkpoint(
            CATALOG, voice.language, voice.espeak_voice, voice.gender
        )
        settings.checkpoint = suggested.url if suggested else ""

    return asdict(settings)


def voice_json(voice: Voice) -> Dict[str, Any]:
    workspace = trainer.get(voice)
    return {
        **voice.to_json(),
        "languageName": LANGUAGES.get(voice.language, voice.language),
        "recorded": voice.num_recorded(),
        "prompts": len(PROMPTS.get(voice.language, [])),
        "modelName": voice.model_stem,
        "checkpointGroups": catalog_groups(voice.language, voice.espeak_voice),
        "startingVoices": matching.candidates(voice),
        "suggested": getattr(
            suggest_checkpoint(CATALOG, voice.language, voice.espeak_voice, voice.gender),
            "url",
            "",
        ),
        "defaults": default_settings(voice, workspace),
        "training": workspace.status(),
    }


# -----------------------------------------------------------------------------


@app.get("/api/info")
async def api_info() -> Dict[str, Any]:
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in sorted(LANGUAGES.items(), key=lambda kv: kv[1])
        ],
        "presets": PRESETS,
        "defaultPreset": DEFAULT_PRESET,
        "accelerators": ACCELERATORS,
        "checkpoints": {
            group: [asdict(e) for e in entries] for group, entries in CATALOG.items()
        },
        "device": trainer.device,
        "busy": trainer.busy_voice(),
    }


@app.get("/api/voices")
async def api_voices() -> Dict[str, Any]:
    return {"voices": [voice_json(v) for v in store.list()]}


class NewVoice(BaseModel):
    name: str
    language: str
    gender: str = "female"


@app.post("/api/voices")
async def api_create_voice(new_voice: NewVoice) -> Dict[str, Any]:
    if new_voice.language not in LANGUAGES:
        raise ValueError(f"Unknown language: {new_voice.language}")

    name = re.sub(r"[^a-z0-9_]+", "_", new_voice.name.strip().lower()).strip("_")
    voice = store.create(name, new_voice.language, new_voice.gender)
    return voice_json(voice)


@app.get("/api/voices/{name}")
async def api_voice(name: str) -> Dict[str, Any]:
    return voice_json(store.get(name))


@app.delete("/api/voices/{name}")
async def api_delete_voice(name: str) -> Dict[str, Any]:
    voice = store.get(name)
    trainer.forget(voice)
    store.delete(voice)
    return {"ok": True}


# ---- Recording ---------------------------------------------------------------


@app.get("/api/voices/{name}/prompt")
async def api_prompt(name: str, skip: int = 0) -> Dict[str, Any]:
    voice = store.get(name)
    prompt = voice.next_prompt(PROMPTS.get(voice.language, []), skip=max(0, skip))
    return {
        "prompt": asdict(prompt) if prompt else None,
        "recorded": voice.num_recorded(),
    }


@app.post("/api/voices/{name}/recordings")
async def api_record(
    name: str,
    group: str = Form(...),
    id: str = Form(...),
    text: str = Form(...),
    audio: UploadFile = File(...),
    mic: str = Form(""),
) -> Dict[str, Any]:
    voice = store.get(name)
    content_type = (audio.content_type or "").lower()
    extension = ".wav" if "wav" in content_type else ".webm"
    if "ogg" in content_type:
        extension = ".ogg"
    elif "mp4" in content_type:
        extension = ".m4a"

    data = await audio.read()
    if len(data) < 1000:
        raise ValueError("Recording is empty")

    voice.save_recording(group, id, text, data, extension)
    voice.remember_microphone(mic)
    return {"recorded": voice.num_recorded(), "microphone": voice.microphone}


@app.post("/api/voices/{name}/upload")
async def api_upload(name: str, dataset: UploadFile = File(...)) -> Dict[str, Any]:
    voice = store.get(name)
    data = await dataset.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload is larger than 2 GB")

    try:
        imported = await asyncio.to_thread(voice.import_zip, data)
    except zipfile.BadZipFile as err:
        raise ValueError("Upload must be a .zip file") from err

    return {"imported": imported, "recorded": voice.num_recorded()}


# ---- Freeform recording -------------------------------------------------------


@app.get("/api/voices/{name}/freeform")
async def api_freeform_list(name: str) -> Dict[str, Any]:
    return {"takes": freeform.list_takes(store.get(name))}


@app.post("/api/voices/{name}/freeform")
async def api_freeform_start(
    name: str,
    audio: UploadFile = File(...),
    denoise: str = Form("light"),
    diarize: bool = Form(False),
    mic: str = Form(""),
) -> Dict[str, Any]:
    """A long take (recorded, or an uploaded audio/video file): transcribe it,
    split it into clips and, with diarize, find who speaks in each clip."""
    voice = store.get(name)
    extension = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a"}.get(
        (audio.content_type or "").split(";")[0], ""
    )
    source = freeform.new_take(voice, audio.filename or f"take{extension}")
    with open(source, "wb") as out:
        await asyncio.to_thread(shutil.copyfileobj, audio.file, out, 4 * 2**20)
    take = freeform.start(voice, source, denoise, diarize)
    voice.remember_microphone(mic)
    return take


@app.get("/api/voices/{name}/freeform/{take_id}")
async def api_freeform_take(name: str, take_id: str) -> Dict[str, Any]:
    return freeform.get_take(store.get(name), take_id)


class TrackRequest(BaseModel):
    track: int
    dialogue: bool = True


@app.post("/api/voices/{name}/freeform/{take_id}/track")
async def api_freeform_track(name: str, take_id: str, request: TrackRequest) -> Dict[str, Any]:
    """Pick the audio track of a file with several (and whether to use only dialogue)."""
    return freeform.choose_track(store.get(name), take_id, request.track, request.dialogue)


class DenoiseRequest(BaseModel):
    denoise: str


@app.put("/api/voices/{name}/freeform/{take_id}")
async def api_freeform_update(name: str, take_id: str, request: DenoiseRequest) -> Dict[str, Any]:
    return freeform.set_denoise(store.get(name), take_id, request.denoise)


@app.get("/api/voices/{name}/freeform/{take_id}/clips/{index}.wav")
async def api_freeform_clip(name: str, take_id: str, index: int, denoise: str = "") -> Response:
    wav = await asyncio.to_thread(freeform.clip_wav, store.get(name), take_id, index, denoise or None)
    return Response(wav, media_type="audio/wav", headers={"Cache-Control": "no-store"})


class SaveTakeRequest(BaseModel):
    clips: list


@app.post("/api/voices/{name}/freeform/{take_id}/save")
async def api_freeform_save(name: str, take_id: str, request: SaveTakeRequest) -> Dict[str, Any]:
    voice = store.get(name)
    saved = await asyncio.to_thread(freeform.save, voice, take_id, request.clips)
    return {"saved": saved, "recorded": voice.num_recorded()}


@app.delete("/api/voices/{name}/freeform/{take_id}")
async def api_freeform_discard(name: str, take_id: str) -> Dict[str, Any]:
    freeform.discard(store.get(name), take_id)
    return {"ok": True}


# ---- Starting voice auto-detect ----------------------------------------------

_match_jobs: Dict[str, Dict[str, Any]] = {}


def _match_status(voice: Voice) -> Dict[str, Any]:
    job = _match_jobs.get(voice.name, {})
    return {
        "state": job.get("state", "idle"),
        "error": job.get("error"),
        "detail": job.get("detail", ""),
        "match": matching.cached(voice),
        "minRecordings": matching.MIN_RECORDINGS,
    }


@app.get("/api/voices/{name}/match")
async def api_match(name: str) -> Dict[str, Any]:
    return _match_status(store.get(name))


@app.post("/api/voices/{name}/match")
async def api_find_match(name: str) -> Dict[str, Any]:
    """Compare the recordings with the pretrained voices (runs in the background)."""
    voice = store.get(name)
    if _match_jobs.get(voice.name, {}).get("state") == "running":
        return _match_status(voice)
    if voice.num_recorded() < matching.MIN_RECORDINGS:
        raise ValueError(f"Record at least {matching.MIN_RECORDINGS} sentences first")

    job: Dict[str, Any] = {"state": "running", "detail": "Starting…"}
    _match_jobs[voice.name] = job

    async def run() -> None:
        def log(line: str) -> None:
            job["detail"] = line

        try:
            await matching.find(voice, DATA_DIR / "speaker-match", log)
            job["state"] = "done"
        except Exception as err:  # reported to the page (matching.find logs the details)
            if not isinstance(err, RuntimeError):
                _LOGGER.exception("Auto-detect for %s failed", voice.name)
            job.update(state="error", error=str(err))

    job["task"] = asyncio.create_task(run())
    return _match_status(voice)


# ---- Training ----------------------------------------------------------------


@app.post("/api/voices/{name}/train")
async def api_train(name: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    workspace = workspace_for(name)
    trainer.start(workspace, TrainingSettings.from_dict(settings))
    return workspace.status()


@app.post("/api/voices/{name}/stop")
async def api_stop(name: str) -> Dict[str, Any]:
    workspace = workspace_for(name)
    await trainer.stop(workspace)
    return workspace.status()


@app.post("/api/voices/{name}/export")
async def api_export(name: str) -> Dict[str, Any]:
    workspace = workspace_for(name)
    trainer.start_export(workspace)
    return workspace.status()


@app.get("/api/voices/{name}/events")
async def api_events(name: str, request: Request, since: int = 0) -> StreamingResponse:
    """Server-sent events: training status plus new log lines, once a second."""
    workspace = workspace_for(name)

    async def events():
        count = max(0, min(since, workspace.log_count))
        if since == 0:
            # New page: send recent history only
            count = max(0, workspace.log_count - 300)

        while not await request.is_disconnected():
            lines = workspace.lines_since(count)
            count = workspace.log_count
            data = {**workspace.status(), "lines": lines, "logCount": count}
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---- Testing and download ------------------------------------------------------


def export_dir_for(workspace: Workspace, export: str) -> Path:
    if not re.fullmatch(r"epoch_\d+", export):
        raise KeyError(f"No export {export}")

    export_dir = workspace.exports_dir / export
    if not export_dir.is_dir():
        raise KeyError(f"No export {export}")

    return export_dir


class SpeakRequest(BaseModel):
    text: str
    export: str


@app.post("/api/voices/{name}/speak")
async def api_speak(name: str, request: SpeakRequest) -> Response:
    workspace = workspace_for(name)
    export_dir_for(workspace, request.export)
    text = request.text.strip()[:500]
    if not text:
        raise ValueError("Type something to say")

    try:
        wav = await trainer.speak(workspace, request.export, text)
    except RuntimeError as err:
        raise HTTPException(500, f"Speaking failed: {err}") from err

    return Response(wav, media_type="audio/wav")


@app.get("/api/voices/{name}/exports/{export}/home-assistant.zip")
async def api_download_zip(name: str, export: str) -> Response:
    workspace = workspace_for(name)
    export_dir = export_dir_for(workspace, export)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(export_dir.glob("*.onnx*")):
            archive.write(path, path.name)

    filename = f"{workspace.voice.model_stem}-{export}.zip"
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/voices/{name}/exports/{export}/{filename}")
async def api_download_file(name: str, export: str, filename: str) -> FileResponse:
    workspace = workspace_for(name)
    path = export_dir_for(workspace, export) / filename
    if (not re.fullmatch(r"[\w.-]+\.onnx(\.json)?", filename)) or (not path.is_file()):
        raise KeyError(f"No file {filename}")

    return FileResponse(path, filename=filename)


# -----------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=_DIR / "static"), name="static")
