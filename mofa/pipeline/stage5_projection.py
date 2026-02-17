from __future__ import annotations

from typing import List, Tuple
import numpy as np


def project_to_uv(images: List[np.ndarray], texture_size: int = 1024) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    partials, masks = [], []
    for img in images:
        tex = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
        mask = np.zeros((texture_size, texture_size, 1), dtype=np.float32)
        color = img.mean(axis=(1, 2))
        tex[:] = color
        mask[:] = 1.0
        partials.append(tex)
        masks.append(mask)
    return partials, masks
