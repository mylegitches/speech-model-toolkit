"""Voices that speak slower or faster than they were trained, without retraining.

Piper models take the speaking pace as an input on every call ("scales":
noise_scale, length_scale, noise_w). Players are meant to fill it from the
.onnx.json, but some send their own (usually 1.0), so a speed that only lives
in the .onnx.json can be ignored. Downloads at another speed therefore get it
built into the model: a Mul node multiplies whatever length_scale the player
sends by the chosen factor. The .onnx.json keeps its length_scale, so players
that do read it don't apply the speed twice.

Baked models are cached next to the export: exports/<epoch>/speeds/<name>.onnx
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict

from .voices import Voice

_LOGGER = logging.getLogger(__name__)
_lock = threading.Lock()


def length_scale(speed: int) -> float:
    """Piper's length_scale for a speed in % of the trained pace (> 1 is slower)."""
    if not 50 <= speed <= 150:
        raise ValueError("Speed must be between 50% and 150%")
    return round(100 / speed, 3)


def speed_stem(voice: Voice, speed: int) -> str:
    """Piper file name with the speed in it: en_US-tony_speed90-medium (Home Assistant's pattern)."""
    if speed == 100:
        return voice.model_stem
    return f"{voice.language.replace('-', '_')}-{voice.name}_speed{speed}-medium"


def bake(source: Path, dest: Path, factor: float) -> None:
    """Copy of a Piper model whose length_scale input is multiplied by factor."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(source))
    graph = model.graph
    if not any(i.name == "scales" for i in graph.input):
        raise RuntimeError("This model has no 'scales' input, so its speed can't be built in")
    for node in graph.node:
        node.input[:] = ["smt_scales" if name == "scales" else name for name in node.input]
    graph.initializer.append(
        numpy_helper.from_array(np.array([1.0, factor, 1.0], dtype=np.float32), "smt_speed"))
    graph.node.insert(0, helper.make_node("Mul", ["scales", "smt_speed"], ["smt_scales"], name="smt_speed_mul"))
    onnx.checker.check_model(model)
    tmp = dest.with_suffix(".tmp")
    onnx.save(model, str(tmp))
    tmp.replace(dest)


def speed_files(voice: Voice, export_dir: Path, speed: int) -> Dict[str, Any]:
    """{download name: bytes or Path} for an export at a speed (blocking: may build the model)."""
    stem = speed_stem(voice, speed)
    files: Dict[str, Any] = {}
    for path in sorted(export_dir.glob("*.onnx*")):
        if path.name.endswith(".onnx.json"):
            config = json.loads(path.read_text(encoding="utf-8"))
            if speed != 100:
                config["dataset"] = f"{voice.name}_speed{speed}"
            files[f"{stem}.onnx.json"] = json.dumps(config, indent=2, ensure_ascii=False).encode("utf-8")
        elif path.suffix == ".onnx":
            if speed == 100:
                files[f"{stem}.onnx"] = path
                continue
            baked = export_dir / "speeds" / f"{stem}.onnx"
            with _lock:
                if not baked.is_file() or baked.stat().st_mtime < path.stat().st_mtime:
                    baked.parent.mkdir(exist_ok=True)
                    bake(path, baked, length_scale(speed))
                    _LOGGER.info("Built %s (length_scale x%g)", baked.name, length_scale(speed))
            files[f"{stem}.onnx"] = baked
    return files
