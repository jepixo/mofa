from __future__ import annotations

from typing import List
import numpy as np


class IdentityEncoder:
    def __init__(self, beta_dim: int = 128):
        self.beta_dim = beta_dim

    def encode(self, tensors: List[np.ndarray]) -> np.ndarray:
        if not tensors:
            return np.zeros(self.beta_dim, dtype=np.float32)
        feats = []
        for t in tensors:
            pooled = t.mean(axis=(1, 2))
            reps = np.tile(pooled, int(np.ceil(self.beta_dim / pooled.size)))[: self.beta_dim]
            feats.append(reps)
        return np.mean(np.stack(feats, axis=0), axis=0).astype(np.float32)
