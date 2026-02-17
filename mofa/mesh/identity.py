from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class IdentityDecoder(nn.Module if nn is not None else object):
    def __init__(self, vertex_count: int, beta_dim: int = 128):
        if nn is None:
            raise RuntimeError("torch is required for IdentityDecoder")
        super().__init__()
        self.vertex_count = vertex_count
        self.net = nn.Sequential(
            nn.Linear(beta_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, vertex_count * 3),
        )

    def forward(self, beta):
        out = self.net(beta)
        return out.view(-1, self.vertex_count, 3)


def deform_mesh(beta: np.ndarray, base_vertices: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(int(np.abs(beta).sum() * 1e6) % (2**32))
    noise = rng.normal(0, 0.002, size=base_vertices.shape)
    return (base_vertices + noise).astype(np.float32)


def train_identity_decoder(dataset):
    return {"status": "stub", "samples": len(dataset) if hasattr(dataset, "__len__") else None}


def encode_and_deform(beta_vector: np.ndarray, base_vertices: np.ndarray) -> np.ndarray:
    return deform_mesh(beta_vector, base_vertices)
