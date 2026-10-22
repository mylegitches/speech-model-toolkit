"""Voice Room voices: uploads and default voices are found; paths can't leave the voice folders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.config import Settings
from app.agents.voice_catalog import VoiceCatalog


def _voice(folder: Path, stem: str, rate: int = 22050) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.onnx").write_bytes(b"onnx")
    (folder / f"{stem}.onnx.json").write_text(json.dumps({"audio": {"sample_rate": rate}, "num_speakers": 1}))
    return folder / f"{stem}.onnx"


@pytest.fixture
def catalog(tmp_path):
    settings = Settings(data_dir=tmp_path / "agents")
    settings.model_roots = [settings.uploads_dir, tmp_path / "voice" / "voices", tmp_path / "default-voices"]
    cat = VoiceCatalog(settings)
    cat._toolkit_voices = lambda: []  # no toolkit voices in this test
    return cat, settings, tmp_path


def test_uploads_and_default_voices_are_found(catalog, monkeypatch):
    cat, settings, tmp_path = catalog
    _voice(settings.uploads_dir / "1", "en_US-glados-high")
    monkeypatch.setattr(type(settings), "default_voices_dir", property(lambda self: tmp_path / "default-voices"))
    _voice(tmp_path / "default-voices", "en_US-lessac-medium")
    ids = {v.id for v in cat.scan()}
    assert ids == {"en_US-glados-high", "en_US-lessac-medium"}


def test_catalog_rejects_path_escape(catalog):
    cat, _settings, _ = catalog
    with pytest.raises(ValueError, match="allowed model directory"):
        cat.resolve_pair("/etc/outside.onnx")
