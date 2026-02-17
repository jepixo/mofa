from __future__ import annotations

import numpy as np


def _radius_profile(y: np.ndarray) -> np.ndarray:
    core = 0.55 * np.sqrt(np.clip(1.0 - (y / 1.05) ** 2, 0.0, None))
    chin_taper = 1.0 - 0.25 * np.clip((-y - 0.2) / 0.8, 0.0, 1.0)
    forehead = 1.0 + 0.08 * np.exp(-((y - 0.55) ** 2) / 0.05)
    return core * chin_taper * forehead


def generate_base_head(lat_steps: int = 80, lon_steps: int = 150):
    """Generate a closed quad-dominant procedural head mesh."""
    y = np.linspace(-1.1, 1.1, lat_steps)
    theta = np.linspace(0.0, 2.0 * np.pi, lon_steps, endpoint=False)
    yy, tt = np.meshgrid(y, theta, indexing="ij")

    r = _radius_profile(yy)
    x = r * np.cos(tt)
    z = 0.85 * r * np.sin(tt)

    # Feature sculpting (coarse, deterministic)
    nose = np.exp(-((yy - 0.05) ** 2) / 0.03) * np.exp(-((np.mod(tt + np.pi, 2 * np.pi) - np.pi) ** 2) / 0.15)
    x += 0.09 * nose
    cheek = np.exp(-((yy + 0.05) ** 2) / 0.15) * np.exp(-((np.abs(np.sin(tt)) - 1.0) ** 2) / 0.12)
    z *= 1.0 + 0.10 * cheek

    vertices = np.stack([x, yy, z], axis=-1).reshape(-1, 3).astype(np.float32)

    faces = []
    for i in range(lat_steps - 1):
        for j in range(lon_steps):
            jn = (j + 1) % lon_steps
            v0 = i * lon_steps + j
            v1 = i * lon_steps + jn
            v2 = (i + 1) * lon_steps + jn
            v3 = (i + 1) * lon_steps + j
            faces.append([v0, v1, v2, v3])
    faces = np.asarray(faces, dtype=np.int32)

    normals = vertices.copy()
    n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.clip(n_norm, 1e-8, None)
    return vertices, faces, normals
