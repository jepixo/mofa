import numpy as np
from mofa.mesh.base_head import generate_base_head
from mofa.mesh.uv_layout import generate_uv_layout
from mofa.export.exporter import export_glb, export_fbx


def test_export_files_exist(tmp_path):
    v, f, _ = generate_base_head()
    uv = generate_uv_layout(v, f)
    albedo = np.ones((128, 128, 3), dtype=np.float32) * 0.5
    glb = export_glb(v, f, uv, albedo, tmp_path / "face.glb")
    fbx = export_fbx(v, f, uv, albedo, tmp_path / "face.fbx")
    assert glb.exists()
    assert fbx.exists()
