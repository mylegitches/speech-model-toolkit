"""Resumable uploads: a big file is sent in pieces, so a phone that sleeps or drops its
connection mid-upload carries on from where it stopped instead of starting over.

uploads/<id>.part   the bytes received so far
uploads/<id>.json   {"name", "size", "created"}

The browser keys an upload by file name + size + last-modified date, so picking the
same file again (even after a reload) resumes it. Unfinished uploads go after two days.
"""

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from .voices import Voice

STALE_SECONDS = 2 * 24 * 3600
MAX_CHUNK = 64 * 2**20
_ID = re.compile(r"[0-9a-f]{32}")


def _dir(voice: Voice) -> Path:
    return voice.root / "uploads"


def _paths(voice: Voice, upload_id: str):
    if not _ID.fullmatch(upload_id or ""):
        raise KeyError("No such upload")
    base = _dir(voice) / upload_id
    meta, part = base.with_suffix(".json"), base.with_suffix(".part")
    if not meta.is_file() or not part.is_file():
        raise KeyError("No such upload (it may have expired): start it again")
    return meta, part


def _clean(voice: Voice) -> None:
    folder = _dir(voice)
    if not folder.is_dir():
        return
    cutoff = time.time() - STALE_SECONDS
    for path in folder.iterdir():
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def create(voice: Voice, name: str, size: int) -> Dict[str, Any]:
    _clean(voice)
    folder = _dir(voice)
    folder.mkdir(parents=True, exist_ok=True)
    upload_id = uuid.uuid4().hex
    (folder / f"{upload_id}.part").touch()
    (folder / f"{upload_id}.json").write_text(
        json.dumps({"name": name[:300], "size": int(size), "created": time.time()}), encoding="utf-8")
    return {"id": upload_id, "received": 0}


def status(voice: Voice, upload_id: str) -> Dict[str, Any]:
    meta, part = _paths(voice, upload_id)
    info = json.loads(meta.read_text(encoding="utf-8"))
    return {"id": upload_id, "received": part.stat().st_size, "size": info["size"]}


def append(voice: Voice, upload_id: str, offset: int, data: bytes) -> int:
    """Add a piece at `offset`; returns how much has arrived. A piece for the wrong place
    (sent twice, or one went missing) raises ValueError with the right offset."""
    meta, part = _paths(voice, upload_id)
    size = json.loads(meta.read_text(encoding="utf-8"))["size"]
    received = part.stat().st_size
    if offset != received:
        raise ValueError(received)
    if received + len(data) > size:
        raise OverflowError("More data than the file's size")
    with open(part, "ab") as out:
        out.write(data)
    return received + len(data)


def finish(voice: Voice, upload_id: str) -> Path:
    """The complete file (the caller moves it); its metadata goes."""
    meta, part = _paths(voice, upload_id)
    size = json.loads(meta.read_text(encoding="utf-8"))["size"]
    received = part.stat().st_size
    if received != size:
        raise ValueError(received)
    meta.unlink()
    return part
