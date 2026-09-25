"""App-wide settings in one JSON file ($SETTINGS_FILE, default data/settings.json).

API keys are stored here in plain text (the file is chmod 600 and lives on the
user's own data volume), but they never go back to the browser: public()
replaces them with a hint, and update() keeps a stored key when the browser
sends an empty one.
"""

import copy
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_DIR = Path(__file__).resolve().parents[2]
SETTINGS_FILE = Path(
    os.environ.get("SETTINGS_FILE", _REPO_DIR / "data" / "settings.json")
)

FIXED_REPLY = "Your voice wakeword model was trained and triggered successfully."

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your answers are spoken aloud by a "
    "text-to-speech voice, so reply in one to three short sentences of plain "
    "text: no markdown, lists, code, emoji or URLs."
)

DEFAULTS: Dict[str, Any] = {
    "connections": [],
    "activeConnection": None,
    "assistant": {
        "systemPrompt": DEFAULT_SYSTEM_PROMPT,
        "temperature": 0.7,
        "maxTokens": 300,
        "fixedReply": FIXED_REPLY,
    },
    "speech": {
        "sttModel": "base.en",
        "language": "auto",
        "silenceMs": 900,
        "maxSeconds": 15,
    },
    "identify": {
        # Name the people found in imported files with the active AI connection
        "enabled": False,
        "webSearch": True,     # let the AI search the web, where the provider can
        "castLookup": True,    # look up the show's cast on TVmaze (IMDb ID or folder name)
        "autoName": False,     # rename untouched "Person N" cards when the AI is sure
        "autoMerge": False,    # merge cards the AI is sure are one character (voices alike)
    },
    "audio": {
        "threshold": 0.5,
        "cooldownSec": 2.0,
        "allowDownloads": True,
        "echoCancellation": True,
        "noiseSuppression": True,
        "autoGainControl": True,
    },
}

CONNECTION_FIELDS = ("id", "name", "provider", "baseUrl", "apiKey", "model")

_lock = threading.Lock()


def _merge(defaults: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    """Defaults overlaid with values, one level deep for the section dicts."""
    merged = copy.deepcopy(defaults)
    for key, value in values.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key].update(
                {k: v for k, v in value.items() if k in merged[key]}
            )
        elif key in merged:
            merged[key] = value
    return merged


def load() -> Dict[str, Any]:
    """Current settings, including API keys (server side only)."""
    try:
        values = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        values = {}
    except (OSError, ValueError):
        values = {}
    return _merge(DEFAULTS, values)


def _save(settings: Dict[str, Any]) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_name(SETTINGS_FILE.name + ".tmp")
    tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, SETTINGS_FILE)


def key_hint(key: str) -> str:
    if not key:
        return ""
    return f"{key[:3]}…{key[-4:]}" if len(key) > 10 else "••••"


def public_connection(conn: Dict[str, Any]) -> Dict[str, Any]:
    safe = {k: v for k, v in conn.items() if k != "apiKey"}
    safe["hasKey"] = bool(conn.get("apiKey"))
    safe["keyHint"] = key_hint(conn.get("apiKey", ""))
    return safe


def public(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Settings safe to send to the browser (no API keys)."""
    settings = settings if settings is not None else load()
    safe = copy.deepcopy(settings)
    safe["connections"] = [public_connection(c) for c in settings["connections"]]
    return safe


def find_connection(
    settings: Dict[str, Any], conn_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    for conn in settings["connections"]:
        if conn["id"] == conn_id:
            return conn
    return None


def active_connection(settings: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The connection the Test Lab uses, if one is chosen and has a model."""
    settings = settings if settings is not None else load()
    conn = find_connection(settings, settings.get("activeConnection"))
    return conn if conn and conn.get("model") else None


def _clean_connection(
    new: Dict[str, Any], existing: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    conn = {k: str(new.get(k) or "").strip() for k in CONNECTION_FIELDS}
    conn["id"] = conn["id"] or uuid.uuid4().hex[:12]
    if not conn["provider"]:
        raise ValueError("Choose a provider")
    conn["name"] = conn["name"] or conn["provider"]
    if not conn["apiKey"] and existing and not new.get("clearKey"):
        conn["apiKey"] = existing.get("apiKey", "")
    return conn


def fill_key(conn: Dict[str, Any]) -> Dict[str, Any]:
    """A draft connection from the browser, with the stored key if it left it blank."""
    conn = {k: str(conn.get(k) or "").strip() for k in CONNECTION_FIELDS}
    if not conn["apiKey"] and conn["id"]:
        stored = find_connection(load(), conn["id"])
        if stored:
            conn["apiKey"] = stored.get("apiKey", "")
    return conn


def update(values: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a (partial) settings update from the browser; returns public()."""
    with _lock:
        current = load()
        if "connections" in values:
            connections: List[Dict[str, Any]] = []
            for new in values["connections"]:
                existing = find_connection(current, new.get("id"))
                connections.append(_clean_connection(new, existing))
            current["connections"] = connections

        for section in ("assistant", "speech", "identify", "audio"):
            if isinstance(values.get(section), dict):
                current[section].update(
                    {
                        k: v
                        for k, v in values[section].items()
                        if k in DEFAULTS[section]
                    }
                )

        if "activeConnection" in values:
            current["activeConnection"] = values["activeConnection"] or None
        if not find_connection(current, current["activeConnection"]):
            current["activeConnection"] = None

        _save(current)
        return public(current)
