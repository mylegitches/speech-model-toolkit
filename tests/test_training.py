"""Starting-voice downloads: damaged files are replaced, not trained on."""

import asyncio
import zipfile

from app.voice import training
from app.voice.voices import Voice

URL = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/hfc_male/medium/epoch%3D2785-step%3D2128064.ckpt"


def fake_download(url, tmp_path, log):
    with zipfile.ZipFile(tmp_path, "w") as archive:
        archive.writestr("archive/data.pkl", b"weights")


def test_damaged_cached_checkpoint_is_downloaded_again(tmp_path, monkeypatch):
    root = tmp_path / "voices" / "tony"
    root.mkdir(parents=True)
    ws = training.Workspace(voice=Voice(name="tony", language="en-US", espeak_voice="en-us", root=root))
    cached = tmp_path / "checkpoints" / "hfc_male_medium_epoch=2785-step=2128064.ckpt"
    cached.parent.mkdir()
    cached.write_bytes(b"PK half a download")  # what two overlapping downloads left behind

    downloads = []
    monkeypatch.setattr(training, "download_resumable", lambda *a: (downloads.append(a), fake_download(*a)))
    manager = object.__new__(training.TrainingManager)
    path = asyncio.run(manager._get_checkpoint(ws, URL))

    assert path == cached and len(downloads) == 1 and training._checkpoint_ok(path)
    assert any("damaged" in line for line in ws.log)
    # Next time the good copy is used as is
    assert asyncio.run(manager._get_checkpoint(ws, URL)) == cached and len(downloads) == 1
