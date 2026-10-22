"""Voice Room: characters → room → session, and the live room without an AI connection."""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")
from fastapi.testclient import TestClient  # noqa: E402

from app.agents import main as agents  # noqa: E402
from app.agents.config import get_settings  # noqa: E402


def character(name: str) -> dict:
    voice = get_settings().uploads_dir / "test" / "en_US-test-medium.onnx"
    voice.parent.mkdir(parents=True, exist_ok=True)
    voice.write_bytes(b"onnx")
    (voice.parent / "en_US-test-medium.onnx.json").write_text('{"audio": {"sample_rate": 22050}}')
    return {
        "name": name, "voice_path": voice.as_posix(), "voice_config_path": f"{voice.as_posix()}.json",
        "speaker_id": 0, "length_scale": 1, "system_prompt": f"You are {name}.", "eagerness": 0.5,
        "max_words": 50, "color": "#7c5cff", "avatar": None,
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(agents.llm_service, "connection", lambda: None)  # no AI connection
    import asyncio
    asyncio.run(agents.startup())
    with TestClient(agents.app) as c:
        yield c


def test_characters_rooms_and_sessions(client):
    assert client.get("/api/health").status_code == 200
    first = client.post("/api/characters", json=character("Beavis"))
    second = client.post("/api/characters", json=character("Butt-Head"))
    assert first.status_code == 201 and second.status_code == 201
    room = client.post("/api/rooms", json={"name": "Couch", "safe_word": "shut up", "settings": {},
                                          "character_ids": [first.json()["id"], second.json()["id"]]})
    assert room.status_code == 201
    assert [c["name"] for c in room.json()["characters"]] == ["Beavis", "Butt-Head"]
    session = client.post(f"/api/rooms/{room.json()['id']}/session", json={})
    assert session.status_code == 200
    assert client.get(f"/api/sessions/{session.json()['id']}/transcript").json() == []

    # The live room says plainly that there's no AI connection
    with client.websocket_connect(f"/ws/rooms/{room.json()['id']}?session_id={session.json()['id']}") as socket:
        assert socket.receive_json()["type"] == "snapshot"
        socket.send_json({"type": "start", "command_id": "1", "topic": "Nachos"})
        messages = [socket.receive_json() for _ in range(4)]
        assert any(m.get("code") == "llm_not_configured" for m in messages)
