from __future__ import annotations

import numpy as np


def complete_texture(blended_uv: np.ndarray, visibility_mask: np.ndarray, beta: np.ndarray) -> np.ndarray:
    fill = np.clip(beta[:3].reshape(1, 1, 3), 0.0, 1.0)
    return np.where(visibility_mask > 0.5, blended_uv, fill).astype(np.float32)
