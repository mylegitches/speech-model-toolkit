"""Slower or faster downloads: the speed goes into the .onnx.json and the file name."""

import json
from types import SimpleNamespace

import pytest

from app.voice import main
from app.voice.voices import Voice


def test_speed_files(tmp_path):
    voice = Voice(name="tony", language="en-US", espeak_voice="en-us", root=tmp_path)
    export = tmp_path / "exports" / "epoch_10"
    export.mkdir(parents=True)
    (export / "en_US-tony-medium.onnx").write_bytes(b"onnx")
    (export / "en_US-tony-medium.onnx.json").write_text(
        json.dumps({"inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8}}))
    ws = SimpleNamespace(voice=voice)

    files = main.speed_files(ws, export, 90)
    assert set(files) == {"en_US-tony_speed90-medium.onnx", "en_US-tony_speed90-medium.onnx.json"}
    config = json.loads(files["en_US-tony_speed90-medium.onnx.json"])
    assert config["inference"] == {"noise_scale": 0.667, "length_scale": 1.111, "noise_w": 0.8}
    assert set(main.speed_files(ws, export, 100)) == {"en_US-tony-medium.onnx", "en_US-tony-medium.onnx.json"}


def test_speed_limits():
    assert main.length_scale(80) == 1.25
    with pytest.raises(ValueError):
        main.length_scale(20)
