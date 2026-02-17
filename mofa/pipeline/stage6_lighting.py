from __future__ import annotations

import numpy as np


def neutralize_lighting(partial_uv: np.ndarray, visibility_mask: np.ndarray) -> np.ndarray:
    albedo = partial_uv.copy()
    mean = np.maximum(albedo.mean(axis=(0, 1), keepdims=True), 1e-4)
    albedo = albedo / mean * 0.5
    return np.clip(albedo * visibility_mask + partial_uv * (1 - visibility_mask), 0.0, 1.0)
