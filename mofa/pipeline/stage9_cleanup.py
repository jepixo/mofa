from __future__ import annotations

import numpy as np


def cleanup_texture(completed_uv: np.ndarray) -> np.ndarray:
    out = completed_uv.copy()
    out = np.clip(out, 0.0, 1.0)
    return out
