"""Voice Room settings (from github.com/mylegitches/voice-agent-freestyle, built into the toolkit).

Inside the toolkit:
  - its data (database, uploaded voices, session audio) lives in AGENTS_DATA_DIR (data/agents);
  - voices are the toolkit's trained voices (newest version of each), the stock default voice,
    and voices uploaded in the Voice Room (plus a remote Wyoming Piper server, if set);
  - the AI is the toolkit's active AI connection (Settings → AI connections); VOICE_ROOM_*
    environment variables still override, as in the standalone app;
  - speech recognition uses the toolkit's Whisper model (Settings → Speech recognition).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
_TOOLKIT_DATA = Path(os.environ.get("VOICE_DATA_DIR", PACKAGE_DIR.parents[1] / "data" / "voice")).parent

ROOM_DEFAULTS: dict[str, Any] = {
    "max_context_lines": 30,
    "summary_interval": 20,
    "bid_threshold": 3,
    "eagerness_weight": 2.0,
    "direct_address_bonus": 3.0,
    "interjection_mode": "cut",
}


@dataclass(slots=True)
class Settings:
    root_dir: Path = PACKAGE_DIR
    data_dir: Path = _TOOLKIT_DATA / "agents"
    database_url: str = ""
    llm_timeout_seconds: float = 45.0
    model_roots: list[Path] = field(default_factory=list)
    piper_host: str = ""
    piper_port: int = 10200
    piper_languages: list[str] = field(default_factory=list)
    save_session_audio: bool = False
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    room_defaults: dict[str, Any] = field(default_factory=lambda: dict(ROOM_DEFAULTS))

    @property
    def web_dist(self) -> Path:
        return self.root_dir / "web" / "dist"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def toolkit_voices_dir(self) -> Path:
        return _TOOLKIT_DATA / "voice" / "voices"

    @property
    def default_voices_dir(self) -> Path:
        return _TOOLKIT_DATA / "voice" / "default-voices"

    # The AI and speech model come from the toolkit's Settings, read when used, so a
    # change there applies straight away
    @property
    def whisper_model(self) -> str:
        override = os.getenv("VOICE_ROOM_WHISPER_MODEL")
        if override:
            return override
        from ..settings import store
        return store.load()["speech"]["sttModel"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.environ.get("AGENTS_DATA_DIR", _TOOLKIT_DATA / "agents"))
    database_url = os.getenv(
        "VOICE_ROOM_DATABASE_URL",
        f"sqlite+aiosqlite:///{(data_dir / 'voice-room.sqlite3').as_posix()}",
    )
    settings = Settings(
        data_dir=data_dir,
        database_url=database_url,
        piper_host=os.getenv("VOICE_ROOM_PIPER_HOST", ""),
        piper_port=int(os.getenv("VOICE_ROOM_PIPER_PORT", "10200")),
        piper_languages=[lang for lang in os.getenv("VOICE_ROOM_PIPER_LANGUAGES", "en").split(",") if lang],
        save_session_audio=os.getenv("VOICE_ROOM_SAVE_AUDIO", "") == "1",
    )
    settings.model_roots = [settings.uploads_dir, settings.toolkit_voices_dir, settings.default_voices_dir]
    return settings
