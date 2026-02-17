import numpy as np
from PIL import Image

from mofa.runner import MofaPipeline


def test_end_to_end_pipeline(tmp_path):
    img = Image.fromarray(np.full((256, 256, 3), 160, dtype=np.uint8))
    pipe = MofaPipeline()
    result = pipe.run([img], output_dir=tmp_path)
    assert result.vertices.shape[1] == 3
    assert result.albedo.shape[:2] == (pipe.config.texture_size, pipe.config.texture_size)
    assert (tmp_path / "face.glb").exists()
    assert (tmp_path / "identity.json").exists()
