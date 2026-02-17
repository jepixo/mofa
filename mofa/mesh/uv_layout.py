from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def generate_uv_layout(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    u = (np.arctan2(z, x) / (2.0 * np.pi) + 0.5) % 1.0
    v = (y - y.min()) / (y.max() - y.min() + 1e-8)

    # Face-priority remap for center front region
    front = np.clip((x - x.min()) / (x.max() - x.min() + 1e-8), 0, 1)
    u = 0.15 + 0.7 * (0.65 * u + 0.35 * front)
    return np.stack([np.clip(u, 0, 1), np.clip(v, 0, 1)], axis=1).astype(np.float32)


def render_uv_preview(uv_coords: np.ndarray, faces: np.ndarray, size: int = 1024) -> Image.Image:
    img = Image.new("RGB", (size, size), "black")
    draw = ImageDraw.Draw(img)
    for face in faces[::8]:
        pts = [(float(uv_coords[idx, 0] * (size - 1)), float((1 - uv_coords[idx, 1]) * (size - 1))) for idx in face]
        draw.polygon(pts, outline=(0, 255, 0))
    return img
