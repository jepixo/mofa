import numpy as np
from mofa.mesh.base_head import generate_base_head


def test_base_head_shape_and_quads():
    v, f, n = generate_base_head()
    assert 10000 <= len(v) <= 20000
    assert f.shape[1] == 4
    assert n.shape == v.shape


def test_base_head_symmetric_stats():
    v, _, _ = generate_base_head()
    assert np.isclose(np.abs(v[:, 2]).mean(), np.abs(v[:, 2]).mean(), atol=1e-6)
