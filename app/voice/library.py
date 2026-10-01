"""The user's media library (TV and Movies), mounted read only, as a source of files.

Nothing here ever writes to the library: a file picked from it becomes a take whose
source is a symlink to the library file (no copy, no upload). Deleting the take only
removes the link. The mounts are read only too (docker-compose: `:ro`), so even a bug
couldn't change a media file.

Paths in the API are relative to a library root ("The Sopranos/Season 1/x.mkv");
anything that would resolve outside the root (.., symlinks out) is refused.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from .voices import Voice

ROOTS = {
    "tv": ("TV", Path(os.environ.get("MEDIA_TV_DIR", "/media/tv"))),
    "movies": ("Movies", Path(os.environ.get("MEDIA_MOVIES_DIR", "/media/movies"))),
}
MEDIA = re.compile(
    r"\.(mkv|mp4|m4v|avi|mov|wmv|flv|webm|ts|m2ts|mts|mpg|mpeg|vob|3gp|ogv|mp3|wav|flac|ogg|opus|m4a|m4b|aac|ac3|eac3|dts|wma|aiff?)$",
    re.IGNORECASE,
)
MAX_FILES = 1000  # one pick (a whole series is a few hundred)


def _natural(name: str) -> List[Any]:
    """Episode 2 before episode 10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _base(root: str) -> Path:
    if root not in ROOTS:
        raise KeyError(f"No media library {root}")
    return ROOTS[root][1]


def status() -> List[Dict[str, Any]]:
    """Each library: mounted (a folder with something in it) or not."""
    result = []
    for root, (label, path) in ROOTS.items():
        try:
            mounted = path.is_dir() and any(True for _ in path.iterdir())
        except OSError:
            mounted = False
        result.append({"id": root, "label": label, "path": str(path), "mounted": mounted})
    return result


def resolve(root: str, rel: str) -> Path:
    base = _base(root).resolve()
    path = (base / (rel or "").lstrip("/")).resolve()
    if path != base and base not in path.parents:
        raise KeyError("That path is outside the media library")
    return path


def _rel(root: str, path: Path) -> str:
    return path.relative_to(_base(root).resolve()).as_posix()


def linked_files(voice: Voice) -> Set[str]:
    """The library files this dataset already uses (their real paths)."""
    found = set()
    takes = voice.root / "freeform"
    for source in takes.glob("*/source.*") if takes.is_dir() else []:
        if source.is_symlink():
            found.add(os.path.realpath(source))
    return found


def browse(voice: Voice, root: str, rel: str) -> Dict[str, Any]:
    folder = resolve(root, rel)
    if not folder.is_dir():
        raise KeyError("No such folder in the media library")
    used = linked_files(voice)
    folders, files = [], []
    for entry in sorted(folder.iterdir(), key=lambda p: _natural(p.name)):
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir():
                folders.append({"name": entry.name, "path": _rel(root, entry.resolve())})
            elif MEDIA.search(entry.name) and entry.is_file():
                real = entry.resolve()
                files.append({"name": entry.name, "path": _rel(root, real), "size": entry.stat().st_size,
                              "added": str(real) in used})
        except (OSError, KeyError, ValueError):
            continue  # unreadable, or a link out of the library
    return {"root": root, "label": ROOTS[root][0], "path": "" if folder == _base(root).resolve() else _rel(root, folder),
            "folders": folders, "files": files}


def expand(voice: Voice, root: str, paths: Iterable[str]) -> List[Dict[str, Any]]:
    """Picked files and folders (with their subfolders) as a flat list of media files."""
    used = linked_files(voice)
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def add(path: Path) -> None:
        real = path.resolve()
        key = str(real)
        if key in seen or not MEDIA.search(real.name):
            return
        seen.add(key)
        rel = _rel(root, real)  # raises for links out of the library
        out.append({"root": root, "path": rel, "size": real.stat().st_size, "added": key in used})

    for rel in paths:
        path = resolve(root, rel)
        if path.is_file():
            add(path)
            continue
        for folder, subfolders, names in os.walk(path):
            subfolders[:] = sorted((d for d in subfolders if not d.startswith(".")), key=_natural)
            for name in sorted((n for n in names if not n.startswith(".")), key=_natural):
                if len(out) >= MAX_FILES:
                    raise ValueError(f"That's more than {MAX_FILES} files; pick fewer folders at a time")
                try:
                    add(Path(folder) / name)
                except (OSError, KeyError, ValueError):
                    continue
    return out


def source_for(root: str, rel: str) -> Path:
    """The library file to link a new take to (checked: inside the library, a media file)."""
    path = resolve(root, rel)
    if not path.is_file() or not MEDIA.search(path.name):
        raise KeyError("No such audio or video file in the media library")
    return path
