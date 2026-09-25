"""
pipeline.py — fully programmatic openWakeWord training pipeline.

No LLM or cloud AI involved. Phrase → YAML → generate → augment → train → onnx2tf export.

Steps mirror train_wakewords.sh from the source repo, translated to Python subprocesses
so FastAPI can stream stdout/stderr line-by-line via an asyncio queue.
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional

import yaml

from ..errors import explain_failure

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment (set by Docker / docker-compose)
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("WAKEWORD_DATA_DIR", "./data/wakeword")).resolve()
OUTPUT_DIR = Path(os.environ.get("WAKEWORD_OUTPUT_DIR", "./data/wakeword-models")).resolve()
OWW_DIR = Path(os.environ.get("OPENWAKEWORD_DIR", "./openwakeword")).resolve()
PIPER_GEN_DIR = Path(os.environ.get("PIPER_GENERATOR_DIR", "./piper-sample-generator")).resolve()

TRAIN_SCRIPT = OWW_DIR / "openwakeword" / "train.py"
TEMPLATE_PATH = Path(__file__).parent / "config_template.yaml"


# ---------------------------------------------------------------------------
# Phrase validation and slugification
# ---------------------------------------------------------------------------

_ALLOWED = re.compile(r"^[a-z0-9 ]+$")
MAX_WORDS = 6


def validate_phrase(phrase: str) -> str:
    """
    Clean and validate a wake phrase.
    Returns the cleaned phrase (lowercase, stripped).
    Raises ValueError with a user-friendly message on invalid input.
    """
    cleaned = phrase.strip().lower()
    if not cleaned:
        raise ValueError("Phrase must not be empty.")
    if not _ALLOWED.match(cleaned):
        raise ValueError(
            "Phrase may only contain letters, numbers, and spaces."
        )
    words = cleaned.split()
    if len(words) > MAX_WORDS:
        raise ValueError(f"Phrase must be at most {MAX_WORDS} words.")
    return cleaned


def phrase_to_model_name(phrase: str) -> str:
    """'yo hal' → 'yo_hal'"""
    return phrase.strip().lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------


class Stage(str, Enum):
    PENDING = "pending"
    GENERATE = "generate"
    AUGMENT = "augment"
    TRAIN = "train"
    EXPORT = "export"
    DONE = "done"
    ERROR = "error"


@dataclass
class TrainJob:
    job_id: str
    phrase: str
    model_name: str
    stage: Stage = Stage.PENDING
    error: Optional[str] = None
    log_lines: list[str] = field(default_factory=list)
    # asyncio.Queue of (stage, line) tuples; None sentinel = done
    _queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    @property
    def onnx_path(self) -> Path:
        return OUTPUT_DIR / self.model_name / f"{self.model_name}.onnx"

    @property
    def tflite_path(self) -> Path:
        return OUTPUT_DIR / self.model_name / f"{self.model_name}.tflite"

    @property
    def zip_path(self) -> Path:
        return OUTPUT_DIR / self.model_name / f"{self.model_name}.zip"

    def is_done(self) -> bool:
        return self.stage in (Stage.DONE, Stage.ERROR)

    def artifacts_ready(self) -> bool:
        return self.stage == Stage.DONE and self.onnx_path.exists()


# ---------------------------------------------------------------------------
# Job registry (in-memory, single process)
# ---------------------------------------------------------------------------

_jobs: dict[str, TrainJob] = {}
_active_job_id: Optional[str] = None


def get_job(job_id: str) -> Optional[TrainJob]:
    return _jobs.get(job_id)


def active_job() -> Optional[TrainJob]:
    if _active_job_id and _active_job_id in _jobs:
        j = _jobs[_active_job_id]
        if not j.is_done():
            return j
    return None


# ---------------------------------------------------------------------------
# Data health check
# ---------------------------------------------------------------------------

REQUIRED_DATA = {
    "features_neg.npy": "Negative training features (~4–17 GB from HuggingFace)",
    "validation_set_features.npy": "Full validation features (will be truncated on first run)",
    "mit_rirs": "MIT room impulse responses (directory)",
    "fma": "FMA background music (directory)",
    "piper-sample-generator/models/en_US-libritts_r-medium.pt": "Piper TTS checkpoint",
}


def _asset_complete(p: Path) -> bool:
    """
    True if the asset at path p is fully present.
    - Directories (mit_rirs, fma): must exist and be non-empty.
    - Files downloaded by stream_download: must have a sibling .done marker.
    - Symlinks (features_neg.npy → ACAV file): both the link and the target's
      .done marker must exist.
    """
    if not p.exists():
        return False
    if p.is_dir():
        return any(p.iterdir())
    if p.is_symlink():
        target = p.resolve()
        return target.exists() and target.with_name(target.name + ".done").exists()
    # Regular file — check for .done marker
    marker = p.with_name(p.name + ".done")
    return marker.exists()


def check_data() -> dict[str, bool]:
    """Return {asset_name: present_and_complete} for each required asset."""
    result = {}
    for name in REQUIRED_DATA:
        p = DATA_DIR / name
        result[name] = _asset_complete(p)
    return result


def all_data_present() -> bool:
    return all(check_data().values())


def link_piper_models() -> None:
    """
    Symlink each *.pt Piper TTS weight from the data volume into the
    piper-sample-generator scripts dir so generate_samples.py's hardcoded
    default path (Path(__file__).parent / "models") resolves correctly.
    Runs at startup and again after the prepare job downloads the weights.
    """
    src_dir = DATA_DIR / "piper-sample-generator" / "models"
    dst_dir = PIPER_GEN_DIR / "models"
    if not src_dir.exists():
        _LOGGER.info("Wake word training data not downloaded yet (Wake Word tab → Download Data)")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pt_file in src_dir.glob("*.pt"):
        dst = dst_dir / pt_file.name
        if dst.is_symlink() or dst.exists():
            continue
        try:
            dst.symlink_to(pt_file)
            _LOGGER.info("[startup] Linked piper model: %s", pt_file.name)
        except Exception as exc:
            _LOGGER.warning("[startup] Could not symlink %s: %s", pt_file.name, exc)


def find_model(model_name: str) -> Optional[Path]:
    """The trained .onnx for a model name (file stem) in OUTPUT_DIR, if any."""
    if not OUTPUT_DIR.is_dir():
        return None
    candidates = list(OUTPUT_DIR.rglob(f"{model_name}.onnx"))
    return candidates[0] if candidates else None


def list_models() -> list[dict]:
    """Trained wake word models in OUTPUT_DIR."""
    if not OUTPUT_DIR.is_dir():
        return []
    return [
        {
            "name": onnx_file.stem,
            "path": str(onnx_file.relative_to(OUTPUT_DIR)),
            "size_kb": round(onnx_file.stat().st_size / 1024),
        }
        for onnx_file in sorted(OUTPUT_DIR.rglob("*.onnx"))
    ]


def load_detector(onnx_path: Path):
    """openWakeWord detector for one trained model (blocking: run in a thread)."""
    from openwakeword.model import Model as OWWModel

    return OWWModel(wakeword_models=[str(onnx_path)], inference_framework="onnx")


# ---------------------------------------------------------------------------
# YAML config generation
# ---------------------------------------------------------------------------

def _write_config(job: TrainJob) -> Path:
    """Write a training YAML for this job and return its path."""
    with open(TEMPLATE_PATH) as f:
        text = f.read()

    model_output = OUTPUT_DIR / job.model_name
    model_output.mkdir(parents=True, exist_ok=True)

    replacements = {
        "{data_dir}": str(DATA_DIR),
        "{model_name}": job.model_name,
        "{output_dir}": str(model_output),
        "{piper_generator_dir}": str(DATA_DIR / "piper-sample-generator"),
        "{target_phrase}": job.phrase,
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)

    config_path = model_output / f"{job.model_name}.yaml"
    with open(config_path, "w") as f:
        f.write(text)
    return config_path


# ---------------------------------------------------------------------------
# OOM-safe validation set truncation
# ---------------------------------------------------------------------------

def _ensure_small_validation() -> None:
    """
    Create validation_set_features_small.npy (50k rows) if it does not exist.
    This prevents the silent OOM at training step 7500 with the full 481k-row file.
    """
    src = DATA_DIR / "validation_set_features.npy"
    dst = DATA_DIR / "validation_set_features_small.npy"
    if dst.exists():
        return
    import numpy as np
    data = np.load(src, mmap_mode="r")
    small = data[:50000]
    np.save(str(dst), small)


# ---------------------------------------------------------------------------
# Subprocess runner with live log streaming
# ---------------------------------------------------------------------------

async def _run_step(
    job: TrainJob,
    stage: Stage,
    cmd: list[str],
    allow_nonzero: bool = False,
) -> None:
    """
    Run a subprocess, stream stdout+stderr lines to job._queue and job.log_lines.
    Raises RuntimeError on failure (unless allow_nonzero=True).
    """
    job.stage = stage
    await job._queue.put((stage.value, f"--- [{stage.value.upper()}] starting ---"))

    # The generate step needs generate_samples.py (from piper-sample-generator)
    # on the Python path. We expose PIPER_GEN_DIR for all steps — harmless for others.
    import os as _os
    env = _os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PIPER_GEN_DIR) + (":" + existing_pp if existing_pp else "")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(OWW_DIR),
        env=env,
        limit=64 * 1024 * 1024,  # 64 MB — default 64 KB causes LimitOverrunError on long tqdm lines
    )

    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        job.log_lines.append(line)
        await job._queue.put((stage.value, line))

    rc = await proc.wait()
    if rc != 0 and not allow_nonzero:
        msg = explain_failure(f"The {stage.value} step", rc, job.log_lines[-20:])
        _LOGGER.error("Wake word %s: %s", job.model_name, msg)
        job.log_lines.append(msg)
        await job._queue.put((stage.value, msg))
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------

async def _export_tflite(job: TrainJob) -> None:
    """Run onnx2tf on the trained ONNX to produce a .tflite file."""
    onnx_path = job.onnx_path
    if not onnx_path.exists():
        raise RuntimeError(f"ONNX not found at {onnx_path}")

    out_dir = onnx_path.parent
    await _run_step(
        job,
        Stage.EXPORT,
        [
            sys.executable, "-m", "onnx2tf",
            "-i", str(onnx_path),
            "-o", str(out_dir),
            "-kat", "onnx____Flatten_0",
        ],
        allow_nonzero=False,
    )

    # onnx2tf may write <name>_<hash>_float32.tflite — normalise to <name>.tflite
    tflite_dst = job.tflite_path
    if not tflite_dst.exists():
        candidates = sorted(
            out_dir.glob(f"{job.model_name}*float32*.tflite")
        ) or sorted(
            p for p in out_dir.glob(f"{job.model_name}*.tflite")
            if "int8" not in p.name and "float16" not in p.name
        )
        if candidates:
            shutil.move(str(candidates[0]), str(tflite_dst))


# ---------------------------------------------------------------------------
# Zip artifacts
# ---------------------------------------------------------------------------

def _make_zip(job: TrainJob) -> None:
    with zipfile.ZipFile(job.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if job.onnx_path.exists():
            zf.write(job.onnx_path, job.onnx_path.name)
        if job.tflite_path.exists():
            zf.write(job.tflite_path, job.tflite_path.name)


# ---------------------------------------------------------------------------
# Main training coroutine
# ---------------------------------------------------------------------------

async def run_training(job: TrainJob) -> None:
    """
    Full pipeline: validate data → write config → generate → augment → train → export → zip.
    All steps are programmatic; no LLM or cloud API is called.
    """
    global _active_job_id
    _active_job_id = job.job_id
    _jobs[job.job_id] = job
    started = time.monotonic()
    _LOGGER.info("Wake word training started: %r -> %s", job.phrase, job.model_name)

    try:
        # Ensure OOM-safe validation set exists
        _ensure_small_validation()

        config_path = _write_config(job)
        await job._queue.put(("info", f"Config written: {config_path}"))

        py = sys.executable

        # Step 1: Generate positive TTS clips via Piper (local offline model)
        await _run_step(
            job, Stage.GENERATE,
            [py, str(TRAIN_SCRIPT), "--training_config", str(config_path), "--generate_clips"],
        )

        # Step 2: Augment clips + compute mel features
        await _run_step(
            job, Stage.AUGMENT,
            [py, str(TRAIN_SCRIPT), "--training_config", str(config_path), "--augment_clips"],
        )

        # Step 3: Train DNN model (10k+1k+1k steps).
        # allow_nonzero=True because train.py exits non-zero after ONNX is written,
        # when it tries to call the missing onnx_tf module. The ONNX itself is saved.
        await _run_step(
            job, Stage.TRAIN,
            [py, str(TRAIN_SCRIPT), "--training_config", str(config_path), "--train_model"],
            allow_nonzero=True,
        )

        # Step 4: Export TFLite via onnx2tf (not onnx_tf which is unavailable)
        await _export_tflite(job)

        # Zip both artifacts
        _make_zip(job)

        job.stage = Stage.DONE
        _LOGGER.info("Wake word %s trained in %.1f min", job.model_name, (time.monotonic() - started) / 60)
        await job._queue.put(("done", f"Training complete: {job.model_name}"))

    except Exception as exc:
        job.stage = Stage.ERROR
        job.error = str(exc)
        _LOGGER.error("Wake word %s failed after %.1f min: %s", job.model_name,
                      (time.monotonic() - started) / 60, exc,
                      exc_info=not isinstance(exc, RuntimeError))
        await job._queue.put(("error", f"ERROR: {exc}"))

    finally:
        # None sentinel signals the SSE generator to close
        await job._queue.put(None)


# ---------------------------------------------------------------------------
# SSE event generator
# ---------------------------------------------------------------------------

async def stream_events(job: TrainJob) -> AsyncIterator[str]:
    """
    Yield Server-Sent Events for the given job.
    Replays buffered log lines first, then follows live.
    """
    # Replay historical lines for late-joining clients
    for line in job.log_lines:
        yield _sse("log", line)

    if job.is_done():
        yield _sse(job.stage.value, job.error or job.model_name)
        return

    while True:
        try:
            item = await asyncio.wait_for(job._queue.get(), timeout=25)
        except asyncio.TimeoutError:
            # No output for 25 s — send an SSE comment to keep the connection alive
            # and let the client know training is still running.
            if job.is_done():
                yield _sse(job.stage.value, job.error or job.model_name)
                return
            yield f": keepalive stage={job.stage.value}\n\n"
            continue
        if item is None:
            # Re-enqueue sentinel so multiple consumers all see it
            await job._queue.put(None)
            yield _sse(job.stage.value, job.error or job.model_name)
            return
        stage_name, line = item
        yield _sse(stage_name, line)


def _sse(event: str, data: str) -> str:
    safe = data.replace("\n", " ")
    return f"event: {event}\ndata: {safe}\n\n"


# ---------------------------------------------------------------------------
# Prepare job — first-run data download with live SSE progress
# ---------------------------------------------------------------------------

class PrepareStage(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class PrepareJob:
    job_id: str
    stage: PrepareStage = PrepareStage.PENDING
    error: Optional[str] = None
    log_lines: list[str] = field(default_factory=list)
    _queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    def is_done(self) -> bool:
        return self.stage in (PrepareStage.DONE, PrepareStage.ERROR)


_prepare_job: Optional[PrepareJob] = None


def get_prepare_job() -> Optional[PrepareJob]:
    return _prepare_job


def prepare_running() -> bool:
    return _prepare_job is not None and not _prepare_job.is_done()


PREPARE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_wakeword_data.py"


async def run_prepare(job: PrepareJob) -> None:
    """
    Stream scripts/prepare_data.py as a subprocess, forwarding every output line
    to the SSE queue. Fully programmatic — no LLM or cloud-AI calls.
    """
    global _prepare_job
    _prepare_job = job
    job.stage = PrepareStage.RUNNING

    await job._queue.put(("running", "Starting data download…"))
    _LOGGER.info("Wake word training data download started -> %s", DATA_DIR)

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(PREPARE_SCRIPT),
            "--data-dir", str(DATA_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            job.log_lines.append(line)
            await job._queue.put(("log", line))

        rc = await proc.wait()
        if rc != 0:
            # Last few log lines contain the traceback — surface them in the error message
            raise RuntimeError(explain_failure("The data download", rc, job.log_lines[-20:])
                               + " Click Download Data again to resume.")

        link_piper_models()
        _LOGGER.info("Wake word training data ready")
        job.stage = PrepareStage.DONE
        await job._queue.put(("done", "All assets downloaded and ready."))

    except Exception as exc:
        job.stage = PrepareStage.ERROR
        job.error = str(exc)
        _LOGGER.error("Wake word data download failed: %s", exc, exc_info=not isinstance(exc, RuntimeError))
        await job._queue.put(("error", f"ERROR: {exc}"))

    finally:
        await job._queue.put(None)


async def stream_prepare_events(job: PrepareJob) -> AsyncIterator[str]:
    """SSE generator for the prepare job — replays history then follows live."""
    for line in job.log_lines:
        yield _sse("log", line)

    if job.is_done():
        yield _sse(job.stage.value, job.error or "done")
        return

    while True:
        item = await asyncio.wait_for(job._queue.get(), timeout=60)
        if item is None:
            await job._queue.put(None)
            yield _sse(job.stage.value, job.error or "done")
            return
        event, data = item
        yield _sse(event, data)
