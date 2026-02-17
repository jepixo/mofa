import numpy as np
import pytest

from mofa.mesh.base_head import generate_base_head
from mofa.mesh.identity import IdentityDecoder, encode_and_deform


def test_encode_and_deform_shape():
    v, _, _ = generate_base_head()
    beta = np.zeros(128, dtype=np.float32)
    d = encode_and_deform(beta, v)
    assert d.shape == v.shape


def test_decoder_output_shape_if_torch_available():
    torch = pytest.importorskip("torch")
    v, _, _ = generate_base_head()
    model = IdentityDecoder(vertex_count=v.shape[0])
    out = model(torch.zeros((1, 128), dtype=torch.float32))
    assert tuple(out.shape) == (1, v.shape[0], 3)
