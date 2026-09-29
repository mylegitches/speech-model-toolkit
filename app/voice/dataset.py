"""The clips already in a voice's dataset: list, play, correct, delete (Build Dataset tab).

A clip is recordings/<group>/<stem>.<audio ext> + <stem>.txt; its id is "<group>/<stem>".
Groups: prompt groups from Read sentences, "freeform" (Speak freely, Import file and
Character clone: <take id>_<clip index>) and "upload" (a prepared dataset zip).
Changing or deleting a freeform clip is mirrored in its take, when that file is still
there, so its review shows the clip as it now is.
"""

import logging
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .voices import AUDIO_EXTENSIONS, Voice

_LOGGER = logging.getLogger(__name__)
_PART = re.compile(r"[\w.-]+")
_FREEFORM_STEM = re.compile(r"^(.+)_(\d{4})$")


def _split(clip_id: str) -> Tuple[str, str]:
    group, _, stem = clip_id.partition("/")
    for part in (group, stem):
        if not _PART.fullmatch(part) or part.startswith("."):
            raise KeyError(f"No clip {clip_id}")
    return group, stem


def audio_path(voice: Voice, clip_id: str) -> Path:
    group, stem = _split(clip_id)
    base = voice.recordings_dir / group / stem
    for ext in AUDIO_EXTENSIONS:
        if base.with_suffix(ext).is_file():
            return base.with_suffix(ext)
    raise KeyError(f"No clip {clip_id}")


_SECONDS: Dict[str, Tuple[float, int, Optional[float]]] = {}  # path -> (mtime, size, seconds)


def _probe(path: Path) -> Optional[float]:
    if path.suffix == ".wav":
        try:
            with wave.open(str(path)) as w:
                return w.getnframes() / w.getframerate()
        except (OSError, EOFError, wave.Error):
            pass
    try:
        import soundfile
        return soundfile.info(str(path)).duration
    except Exception:  # webm / m4a: libsndfile can't read them
        pass
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return float(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _seconds(path: Path) -> Optional[float]:
    """The clip's length (cached: a dataset of thousands of clips is listed often)."""
    try:
        stat = path.stat()
    except OSError:
        return None
    key = str(path)
    cached = _SECONDS.get(key)
    if cached and cached[:2] == (stat.st_mtime, stat.st_size):
        return cached[2]
    seconds = _probe(path)
    seconds = round(seconds, 2) if seconds and seconds > 0 else None
    _SECONDS[key] = (stat.st_mtime, stat.st_size, seconds)
    return seconds


def list_clips(voice: Voice) -> List[Dict[str, Any]]:
    """Every clip with its transcript, by group and name."""
    clips = []
    root = voice.recordings_dir
    for text_path in sorted(root.rglob("*.txt")) if root.is_dir() else []:
        audio = next((p for ext in AUDIO_EXTENSIONS if (p := text_path.with_suffix(ext)).exists()), None)
        if audio is None:
            continue
        group = text_path.parent.relative_to(root).as_posix()
        clips.append({
            "id": f"{group}/{text_path.stem}", "group": group,
            "text": text_path.read_text(encoding="utf-8").strip(),
            "seconds": _seconds(audio),
        })
    return clips


def _take_of(voice: Voice, group: str, stem: str) -> Optional[Tuple[Any, Path, Dict[str, Any], int]]:
    """For a freeform clip whose file is still there: (freeform module, take dir, take, index)."""
    if group != "freeform":
        return None
    m = _FREEFORM_STEM.match(stem)
    if not m:
        return None
    from . import freeform  # freeform imports this package's modules

    take_dir = freeform._takes_dir(voice) / m.group(1)
    if m.group(1) in freeform._live or not (take_dir / "take.json").is_file():
        return None
    take = freeform._read(take_dir)
    index = int(m.group(2))
    if not 0 <= index < len(take.get("segments") or []):
        return None
    return freeform, take_dir, take, index


def set_text(voice: Voice, clip_id: str, text: str) -> str:
    text = " ".join(str(text).split())
    if not text:
        raise ValueError("The text can't be empty")
    group, stem = _split(clip_id)
    audio_path(voice, clip_id)  # exists?
    (voice.recordings_dir / group / f"{stem}.txt").write_text(text, encoding="utf-8")
    found = _take_of(voice, group, stem)
    if found:
        freeform, take_dir, take, index = found
        take["segments"][index]["savedText"] = text
        freeform._write(take_dir, take)
    return text


def delete_clips(voice: Voice, clip_ids: List[str]) -> int:
    deleted = 0
    for clip_id in clip_ids:
        group, stem = _split(clip_id)
        base = voice.recordings_dir / group / stem
        found = False
        for ext in (*AUDIO_EXTENSIONS, ".txt"):
            if base.with_suffix(ext).exists():
                base.with_suffix(ext).unlink()
                found = True
        if not found:
            continue
        deleted += 1
        take = _take_of(voice, group, stem)
        if take:
            freeform, take_dir, data, index = take
            data["segments"][index]["saved"] = False
            freeform._write(take_dir, data)
    _LOGGER.info("Voice %s: deleted %d clips from the dataset", voice.name, deleted)
    return deleted


def delete_dataset(voice: Voice) -> None:
    """All clips, the imported files still waiting and the people found in them.
    Trained models (training/, exports/) are kept."""
    from . import freeform

    if any(take_id in freeform._live for take_id in _take_ids(voice)):
        raise RuntimeError("A file of this dataset is still being processed; wait for it to finish")
    for path in (voice.recordings_dir, voice.root / "freeform"):
        shutil.rmtree(path, ignore_errors=True)
    (voice.root / "speakers.json").unlink(missing_ok=True)
    _LOGGER.info("Voice %s: dataset deleted", voice.name)


def _take_ids(voice: Voice) -> List[str]:
    root = voice.root / "freeform"
    return [p.name for p in root.iterdir()] if root.is_dir() else []
