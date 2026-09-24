"""
main.py — wake word training (from easy-wakeword-trainer), mounted at /wakeword.

Endpoints:
  GET  /                        — serve the single-page UI
  GET  /api/health              — data asset check
  POST /api/train               — start a training job
  GET  /api/train/{job_id}/events  — SSE log stream
  GET  /api/train/{job_id}/status  — JSON status
  GET  /api/train/{job_id}/download        — zip of .onnx + .tflite
  GET  /api/train/{job_id}/download/onnx   — .onnx only
  GET  /api/train/{job_id}/download/tflite — .tflite only
  POST /api/preview             — generate a single Piper TTS WAV for a phrase
"""

import asyncio
import os
import shutil
import tempfile
import uuid
import struct
import numpy as np
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..settings import store as settings_store
from . import pipeline as pl

app = FastAPI(title="easy-wakeword-trainer", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Startup hooks — run once at container start, skip if files already exist.
# ---------------------------------------------------------------------------

async def startup() -> None:
    """
    Two idempotent tasks on every startup (called by app.main's lifespan,
    since mounted sub-apps don't receive startup events):

    1. Download OWW base ONNX models (melspectrogram + embedding) if missing.
       They are kept in $WAKEWORD_DATA_DIR/oww-models so they survive restarts;
       openWakeWord's resources/models directory is symlinked there.

    2. Symlink each *.pt Piper TTS weight from the data volume into the
       piper-sample-generator scripts dir so generate_samples.py's hardcoded
       default path (Path(__file__).parent / "models") resolves correctly.
    """
    import asyncio
    import logging
    import urllib.request

    loop = asyncio.get_event_loop()

    # ── 1. OWW ONNX base models ────────────────────────────────────────────
    OWW_MODELS_BASE = (
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
    )
    oww_models_dir = pl.DATA_DIR / "oww-models"
    oww_models_dir.mkdir(parents=True, exist_ok=True)
    resources_dir = pl.OWW_DIR / "openwakeword" / "resources" / "models"
    if not resources_dir.is_symlink():
        try:
            if resources_dir.is_dir():
                for existing in resources_dir.iterdir():
                    if not (oww_models_dir / existing.name).exists():
                        shutil.move(str(existing), str(oww_models_dir / existing.name))
                shutil.rmtree(resources_dir)
            resources_dir.parent.mkdir(parents=True, exist_ok=True)
            resources_dir.symlink_to(oww_models_dir, target_is_directory=True)
        except OSError as exc:
            logging.warning("[startup] Could not link %s: %s", resources_dir, exc)

    for fname in ("melspectrogram.onnx", "embedding_model.onnx"):
        dest = oww_models_dir / fname
        if dest.exists():
            continue
        url = OWW_MODELS_BASE + fname
        logging.info("[startup] Downloading %s …", fname)
        try:
            await loop.run_in_executor(None, urllib.request.urlretrieve, url, dest)
            logging.info("[startup] Saved %s (%d KB)", fname, dest.stat().st_size // 1024)
        except Exception as exc:
            logging.error("[startup] Failed to download %s: %s", fname, exc)

    # ── 2. Piper .pt weight symlinks ────────────────────────────────────────
    pl.link_piper_models()


# ---------------------------------------------------------------------------
# Root — serve UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Health — data asset check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    present = pl.check_data()
    missing = [k for k, v in present.items() if not v]
    descriptions = pl.REQUIRED_DATA
    return {
        "ready": len(missing) == 0,
        "assets": {
            name: {
                "present": present[name],
                "description": descriptions[name],
            }
            for name in present
        },
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Prepare — first-run data download
# ---------------------------------------------------------------------------

@app.post("/api/prepare", status_code=202)
async def start_prepare(background_tasks: BackgroundTasks):
    """Start the data-download job. Idempotent if data is already present."""
    if pl.prepare_running():
        job = pl.get_prepare_job()
        return {"job_id": job.job_id, "status": "already_running"}

    if pl.all_data_present():
        return {"job_id": None, "status": "already_ready"}

    job = pl.PrepareJob(job_id=str(uuid.uuid4()))
    background_tasks.add_task(pl.run_prepare, job)
    return {"job_id": job.job_id, "status": "started"}


@app.get("/api/prepare/status")
async def prepare_status():
    job = pl.get_prepare_job()
    if not job:
        return {"stage": "not_started", "log_lines": []}
    return {
        "job_id": job.job_id,
        "stage": job.stage.value,
        "error": job.error,
        "log_lines": job.log_lines[-50:],  # last 50 lines for polling fallback
    }


@app.get("/api/prepare/events")
async def prepare_events():
    job = pl.get_prepare_job()
    if not job:
        raise HTTPException(status_code=404, detail="No prepare job started yet.")
    return StreamingResponse(
        pl.stream_prepare_events(job),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Start training job
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    phrase: str


@app.post("/api/train", status_code=202)
async def start_train(req: TrainRequest, background_tasks: BackgroundTasks):
    # Validate phrase
    try:
        phrase = pl.validate_phrase(req.phrase)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Only one job at a time
    if pl.active_job():
        raise HTTPException(
            status_code=409,
            detail="A training job is already running. Wait for it to finish.",
        )

    # Check data is present
    if not pl.all_data_present():
        missing = [k for k, v in pl.check_data().items() if not v]
        raise HTTPException(
            status_code=428,
            detail=f"Required data assets are missing: {missing}. "
                   "Run scripts/prepare_data.py first.",
        )

    model_name = pl.phrase_to_model_name(phrase)
    job_id = str(uuid.uuid4())
    job = pl.TrainJob(job_id=job_id, phrase=phrase, model_name=model_name)
    pl._jobs[job_id] = job

    background_tasks.add_task(pl.run_training, job)

    return {"job_id": job_id, "model_name": model_name, "phrase": phrase}


# ---------------------------------------------------------------------------
# SSE event stream
# ---------------------------------------------------------------------------

@app.get("/api/train/{job_id}/events")
async def events(job_id: str):
    job = pl.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return StreamingResponse(
        pl.stream_events(job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.get("/api/train/{job_id}/status")
async def status(job_id: str):
    job = pl.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job.job_id,
        "phrase": job.phrase,
        "model_name": job.model_name,
        "stage": job.stage.value,
        "error": job.error,
        "artifacts_ready": job.artifacts_ready(),
    }


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def _require_job(job_id: str) -> pl.TrainJob:
    job = pl.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.stage == pl.Stage.ERROR:
        raise HTTPException(status_code=500, detail=f"Job failed: {job.error}")
    if job.stage != pl.Stage.DONE:
        raise HTTPException(status_code=202, detail="Training not finished yet.")
    return job


@app.get("/api/train/{job_id}/download")
async def download_zip(job_id: str):
    job = _require_job(job_id)
    if not job.zip_path.exists():
        raise HTTPException(status_code=404, detail="Zip not found.")
    return FileResponse(
        str(job.zip_path),
        media_type="application/zip",
        filename=f"{job.model_name}.zip",
    )


@app.get("/api/train/{job_id}/download/onnx")
async def download_onnx(job_id: str):
    job = _require_job(job_id)
    if not job.onnx_path.exists():
        raise HTTPException(status_code=404, detail="ONNX not found.")
    return FileResponse(
        str(job.onnx_path),
        media_type="application/octet-stream",
        filename=f"{job.model_name}.onnx",
    )


@app.get("/api/train/{job_id}/download/tflite")
async def download_tflite(job_id: str):
    job = _require_job(job_id)
    if not job.tflite_path.exists():
        raise HTTPException(status_code=404, detail="TFLite not found.")
    return FileResponse(
        str(job.tflite_path),
        media_type="application/octet-stream",
        filename=f"{job.model_name}.tflite",
    )


# ---------------------------------------------------------------------------
# Model browser — list available trained .onnx models
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def list_models():
    """Return all trained .onnx models in the output directory."""
    return {"models": pl.list_models()}


# ---------------------------------------------------------------------------
# Phrase preview — generate a single Piper TTS clip and stream it back
# ---------------------------------------------------------------------------

_PIPER_MODEL = "en_US-libritts_r-medium"

class PreviewRequest(BaseModel):
    phrase: str

@app.post("/api/preview")
async def preview_phrase(req: PreviewRequest):
    """
    Generate one Piper TTS sample for the given phrase and return it as audio/wav.
    Uses the same model checkpoint already downloaded for training — no extra downloads.
    """
    phrase = req.phrase.strip()
    if not phrase:
        raise HTTPException(status_code=400, detail="phrase is empty")

    # Check the model is available
    model_path = pl.PIPER_GEN_DIR / "models" / f"{_PIPER_MODEL}.pt"
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Piper model not found. Run the data download step first.",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="eww_preview_"))
    try:
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(pl.PIPER_GEN_DIR) + (":" + existing_pp if existing_pp else "")

        cmd = [
            "python3",
            str(pl.PIPER_GEN_DIR / "generate_samples.py"),
            phrase,
            "--max-samples", "1",
            "--model", str(model_path),
            "--output-dir", str(tmp_dir),
            "--batch-size", "1",
        ]

        loop = asyncio.get_event_loop()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(pl.OWW_DIR),
            env=env,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)

        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Piper generation failed:\n{stdout.decode(errors='replace')}",
            )

        wavs = list(tmp_dir.glob("*.wav"))
        if not wavs:
            raise HTTPException(status_code=500, detail="No WAV file generated.")

        wav_path = wavs[0]
        wav_bytes = wav_path.read_bytes()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        iter([wav_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="preview.wav"'},
    )


# ---------------------------------------------------------------------------
# Wake word tester — WebSocket live inference
# ---------------------------------------------------------------------------
# Protocol:
#   Client → binary: raw 16-bit signed PCM at 16 kHz, any chunk size
#   Server → JSON:   {"score": float, "detected": bool}
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000


@app.websocket("/api/test/{model_name}")
async def test_wakeword(websocket: WebSocket, model_name: str):
    """
    Stream microphone audio (16 kHz s16le PCM) and receive detection scores.
    model_name must match the stem of a .onnx file in /outputs.
    """
    # Locate the model
    onnx_path = pl.find_model(model_name)
    if onnx_path is None:
        await websocket.close(code=4004, reason=f"Model '{model_name}' not found")
        return

    await websocket.accept()
    try:
        # Load OWW model (this blocks for a moment — run in executor)
        loop = asyncio.get_event_loop()
        oww_model = await loop.run_in_executor(None, pl.load_detector, onnx_path)

        threshold = float(settings_store.load()["audio"]["threshold"])
        audio_buf = np.array([], dtype=np.int16)

        while True:
            data = await websocket.receive_bytes()
            # Decode s16le PCM
            chunk = np.frombuffer(data, dtype=np.int16)
            audio_buf = np.concatenate([audio_buf, chunk])

            # OWW needs at least 1280 samples per call (80 ms at 16 kHz)
            while len(audio_buf) >= 1280:
                frame = audio_buf[:1280]
                audio_buf = audio_buf[1280:]

                prediction = oww_model.predict(frame)
                score = float(list(prediction.values())[0])
                detected = score >= threshold

                await websocket.send_json({"score": round(score, 4), "detected": detected})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        import logging
        logging.error("Test WebSocket error: %s", exc)
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass
