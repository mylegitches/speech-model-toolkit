"""Consistent errors and logging for every sub-app.

Every API error is a short, user-facing message in plain text (the pages show
it as is). Expected problems use exception types; anything unexpected is
logged with its traceback and reported as "Unexpected error: ...".

  KeyError      404  something doesn't exist ("No voice dad")
  ValueError    400  bad input ("Record at least 5 sentences first")
  RuntimeError  409  can't do that right now ("A training job is already running")
  OSError(ENOSPC) 507  the server's disk is full
  anything else 500  bug or environment problem (logged with traceback)
"""

import errno
import logging
import os
import re
import sys

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging() -> None:
    """One log format for the whole app; LOG_LEVEL=DEBUG|INFO|WARNING (default INFO)."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S",
                        stream=sys.stdout, force=True)
    # One access line per request (the pages poll every few seconds) only at DEBUG;
    # failed requests are logged by the handlers below
    debug = getattr(logging, level, logging.INFO) <= logging.DEBUG
    logging.getLogger("uvicorn.access").setLevel(logging.INFO if debug else logging.WARNING)
    for noisy in ("httpx", "httpcore", "faster_whisper", "speechbrain", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _message(err: BaseException) -> str:
    if isinstance(err, KeyError) and err.args:
        return str(err.args[0])
    return str(err) or err.__class__.__name__


def install(app: FastAPI, name: str) -> None:
    """Register the standard exception handlers on a (sub-)app."""
    logger = logging.getLogger(f"app.{name}")

    @app.exception_handler(KeyError)
    async def not_found(request: Request, err: KeyError) -> PlainTextResponse:
        logger.info("%s %s: not found: %s", request.method, request.url.path, _message(err))
        return PlainTextResponse(_message(err), status_code=404)

    @app.exception_handler(ValueError)
    async def bad_request(request: Request, err: ValueError) -> PlainTextResponse:
        logger.info("%s %s: rejected: %s", request.method, request.url.path, _message(err))
        return PlainTextResponse(_message(err), status_code=400)

    @app.exception_handler(RuntimeError)
    async def conflict(request: Request, err: RuntimeError) -> PlainTextResponse:
        logger.warning("%s %s: %s", request.method, request.url.path, _message(err))
        return PlainTextResponse(_message(err), status_code=409)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, err: Exception) -> PlainTextResponse:
        if isinstance(err, OSError) and err.errno == errno.ENOSPC:
            logger.error("%s %s: disk full", request.method, request.url.path)
            return PlainTextResponse("The server's disk is full. Free some space and try again.", status_code=507)
        logger.exception("%s %s failed", request.method, request.url.path)
        return PlainTextResponse(
            f"Unexpected error: {_message(err)}. Details are in the server log (docker compose logs).",
            status_code=500,
        )


def friendly_ffmpeg_error(stderr: str) -> str:
    """Turn ffmpeg's stderr into something a person can act on."""
    text = stderr.lower()
    if "does not contain any stream" in text or "matches no streams" in text or "no audio" in text:
        return "This file has no audio track."
    if any(k in text for k in ("invalid data found", "could not find codec", "moov atom not found",
                               "invalid argument", "error opening input", "end of file")):
        return "This isn't a readable audio or video file (it may be damaged or only partly uploaded)."
    if "no space left" in text:
        return "The server's disk is full. Free some space and try again."
    if "no such file" in text:
        return "The audio file went missing on the server; upload it again."
    tail = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
    tail = re.sub(r"(/[^\s:]+)+:?\s*", "", tail).strip() or "unknown error"  # no server paths
    return f"Could not read the audio ({tail[:200]})."


_FAILURE_HINTS = [
    (("cuda out of memory", "outofmemoryerror", "cublas_status_alloc_failed"),
     "the GPU ran out of memory. Lower the batch size in Advanced settings, or close other programs using the GPU"),
    (("no space left on device",), "the server's disk is full. Free some space and try again"),
    (("killed", "memoryerror", "cannot allocate memory"),
     "the server ran out of memory (RAM). Close other programs or use a smaller model/batch"),
    (("temporary failure in name resolution", "urlopen error", "connection refused", "connectionerror",
      "max retries exceeded", "read timed out", "network is unreachable"),
     "a download failed. Check the server's internet connection and try again"),
    (("modulenotfounderror", "importerror", "no module named"),
     "a required component is missing. Rebuild the image: docker compose up -d --build"),
    (("permission denied",), "permission denied writing files. Check the owner/permissions of the ./data folder"),
]


def explain_failure(what: str, code, tail) -> str:
    """A clear message for a failed step, from its exit code and last output lines."""
    lines = [line.strip() for line in tail if line and line.strip()]
    text = "\n".join(lines).lower()
    if code in (-9, 137):
        text += "\nkilled"
    for needles, hint in _FAILURE_HINTS:
        if any(n in text for n in needles):
            return f"{what} failed: {hint}."
    # Prefer an actual error line over progress output
    errors = [l for l in lines if any(k in l.lower() for k in ("error", "exception", "failed", "traceback"))]
    last = (errors or lines or ["no output"])[-1][:300]
    code_text = f" (exit code {code})" if code not in (None, 0) else ""
    return f"{what} failed{code_text}: {last}"
