from __future__ import annotations

from typing import List
import numpy as np


def blend_uv(albedos: List[np.ndarray], masks: List[np.ndarray]) -> np.ndarray:
    stack = np.stack(albedos, axis=0)
    m = np.stack(masks, axis=0)
    wsum = np.clip(m.sum(axis=0), 1e-6, None)
    return (stack * m).sum(axis=0) / wsum
