from __future__ import annotations

import numpy as np
from mofa.mesh.identity import encode_and_deform


def deform_stage(beta: np.ndarray, base_vertices: np.ndarray):
    verts = encode_and_deform(beta, base_vertices)
    normals = verts.copy()
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)
    return verts, normals
