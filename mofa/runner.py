from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import numpy as np
from PIL import Image

from mofa.config import MofaConfig
from mofa.mesh.base_head import generate_base_head
from mofa.mesh.uv_layout import generate_uv_layout
from mofa.pipeline.stage1_preprocess import preprocess_images
from mofa.pipeline.stage2_identity import IdentityEncoder
from mofa.pipeline.stage3_deform import deform_stage
from mofa.pipeline.stage4_camera import solve_cameras
from mofa.pipeline.stage5_projection import project_to_uv
from mofa.pipeline.stage6_lighting import neutralize_lighting
from mofa.pipeline.stage7_blend import blend_uv
from mofa.pipeline.stage8_completion import complete_texture
from mofa.pipeline.stage9_cleanup import cleanup_texture
from mofa.export.exporter import export_fbx, export_glb, export_identity_json


@dataclass
class MofaResult:
    vertices: np.ndarray
    faces: np.ndarray
    uv_coords: np.ndarray
    albedo: np.ndarray
    identity: np.ndarray
    timings: Dict[str, float]


class MofaPipeline:
    def __init__(self, config: MofaConfig | None = None):
        self.config = config or MofaConfig()
        self.encoder = IdentityEncoder(beta_dim=self.config.beta_dim)
        self.base_vertices, self.faces, self.normals = generate_base_head()
        self.uv_coords = generate_uv_layout(self.base_vertices, self.faces)

    def run(self, images: List[Image.Image], output_dir: str | None = None) -> MofaResult:
        timings = {}
        t0 = perf_counter()
        stage1 = preprocess_images(images, size=self.config.image_size)
        timings["stage1_preprocess"] = perf_counter() - t0

        t0 = perf_counter()
        beta = self.encoder.encode([s.image_tensor for s in stage1])
        timings["stage2_identity"] = perf_counter() - t0

        t0 = perf_counter()
        vertices, normals = deform_stage(beta, self.base_vertices)
        timings["stage3_deform"] = perf_counter() - t0

        t0 = perf_counter()
        cameras = solve_cameras(len(stage1))
        timings["stage4_camera"] = perf_counter() - t0

        t0 = perf_counter()
        partials, masks = project_to_uv([s.image_tensor for s in stage1], texture_size=self.config.texture_size)
        timings["stage5_projection"] = perf_counter() - t0

        t0 = perf_counter()
        albedo_partials = [neutralize_lighting(p, m) for p, m in zip(partials, masks)]
        timings["stage6_lighting"] = perf_counter() - t0

        t0 = perf_counter()
        blended = blend_uv(albedo_partials, masks)
        timings["stage7_blend"] = perf_counter() - t0

        t0 = perf_counter()
        completion_mask = np.maximum.reduce(masks)
        completed = complete_texture(blended, completion_mask, beta)
        timings["stage8_completion"] = perf_counter() - t0

        t0 = perf_counter()
        final_albedo = cleanup_texture(completed)
        timings["stage9_cleanup"] = perf_counter() - t0

        result = MofaResult(vertices, self.faces, self.uv_coords, final_albedo, beta, timings)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            if self.config.output_format.lower() == "fbx":
                export_fbx(vertices, self.faces, self.uv_coords, final_albedo, out / "face.fbx")
            else:
                export_glb(vertices, self.faces, self.uv_coords, final_albedo, out / "face.glb")
            if self.config.export_identity:
                export_identity_json(beta, out / "identity.json")
        _ = cameras, normals
        return result
