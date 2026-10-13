"""Clicks and decisions from the page are appended to LOG_DIR/clicks.jsonl."""

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def test_events_are_appended(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    from app.voice import main
    client = TestClient(main.app)
    events = [{"t": "2026-10-12T23:00:00Z", "action": "rejected", "take": "t1", "index": 4, "words": "Hi"},
              {"t": "2026-10-12T23:00:02Z", "action": "kept", "take": "t1", "index": 5}]
    assert client.post("/api/voices/alfv1/events", json={"events": events}).json() == {"logged": 2}
    client.post("/api/voices/alfv1/events", json={"events": [{"action": "click", "label": "✓ Keep"}]})
    rows = [json.loads(line) for line in (tmp_path / "logs" / "clicks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["action"] for r in rows] == ["rejected", "kept", "click"]
    assert rows[0]["voice"] == "alfv1" and rows[2]["label"] == "✓ Keep"
