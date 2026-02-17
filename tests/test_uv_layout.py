import numpy as np
from mofa.mesh.base_head import generate_base_head
from mofa.mesh.uv_layout import generate_uv_layout


def test_uv_range_and_shape():
    v, f, _ = generate_base_head()
    uv = generate_uv_layout(v, f)
    assert uv.shape[0] == v.shape[0]
    assert np.all((uv >= 0.0) & (uv <= 1.0))
