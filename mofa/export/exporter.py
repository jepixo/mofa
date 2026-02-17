from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image


def _write_obj(vertices: np.ndarray, faces: np.ndarray, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            idx = [int(i) + 1 for i in face]
            f.write("f " + " ".join(str(i) for i in idx) + "\n")


def export_glb(vertices, faces, uv_coords, albedo, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_obj(vertices, faces, path.with_suffix(".obj"))
    Image.fromarray((np.clip(albedo, 0, 1) * 255).astype(np.uint8)).save(path.with_name("albedo.png"))
    path.write_text("placeholder glb container", encoding="utf-8")
    return path


def export_fbx(vertices, faces, uv_coords, albedo, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_obj(vertices, faces, path.with_suffix(".obj"))
    path.write_text("placeholder fbx container", encoding="utf-8")
    return path


def export_identity_json(beta, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"beta": np.asarray(beta).tolist()}), encoding="utf-8")
    return path
