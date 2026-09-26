"""Slower or faster downloads: the speed goes into the .onnx.json and the file name."""

import json
import pytest

from app.voice import main, speed
from app.voice.voices import Voice


def test_speed_files(tmp_path, monkeypatch):
    voice = Voice(name="tony", language="en-US", espeak_voice="en-us", root=tmp_path)
    export = tmp_path / "exports" / "epoch_10"
    export.mkdir(parents=True)
    (export / "en_US-tony-medium.onnx").write_bytes(b"onnx")
    (export / "en_US-tony-medium.onnx.json").write_text(
        json.dumps({"inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8}}))
    baked = []
    monkeypatch.setattr(speed, "bake", lambda src, dest, factor: (baked.append(factor), dest.write_bytes(b"baked")))

    files = speed.speed_files(voice, export, 90)
    assert set(files) == {"en_US-tony_speed90-medium.onnx", "en_US-tony_speed90-medium.onnx.json"}
    # The speed is built into the model; the json keeps length_scale 1 so it isn't applied twice
    assert baked == [1.111] and files["en_US-tony_speed90-medium.onnx"].read_bytes() == b"baked"
    config = json.loads(files["en_US-tony_speed90-medium.onnx.json"])
    assert config["inference"] == {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8}
    speed.speed_files(voice, export, 90)
    assert len(baked) == 1  # cached
    assert set(speed.speed_files(voice, export, 100)) == {"en_US-tony-medium.onnx", "en_US-tony-medium.onnx.json"}


def test_speed_limits():
    assert speed.length_scale(80) == 1.25
    with pytest.raises(ValueError):
        speed.length_scale(20)
