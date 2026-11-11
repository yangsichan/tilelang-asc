"""Device regression for the QSA grouped-query sparse forward."""

import pytest
import torch

from examples.ascend.qsa.compile import compile_qsa
from examples.ascend.qsa.testing import fill_contract_corners, make_inputs, ref_sparse_gqa, run_qsa


def _check(num_tokens, kv_length, top_k, *, dynamic_shape):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_op_timeout_ms(30000)
    kernel = compile_qsa(num_tokens, kv_length, top_k, dynamic_shape=dynamic_shape)
    q, k, v, indices = make_inputs(num_tokens, kv_length, top_k, seed=0)
    fill_contract_corners(indices, kv_length, top_k)
    expected = ref_sparse_gqa(q, k, v, indices)
    out = run_qsa(kernel, q, k, v, indices)
    torch.npu.synchronize()
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dynamic_shape", [False, True], ids=["static", "dynamic"])
def test_pipelined_forward(dynamic_shape):
    # NI = 17: two-stage pipeline, 32-row chunks, 3-version KV ring.
    _check(16, 8192, 2176, dynamic_shape=dynamic_shape)


@pytest.mark.parametrize("dynamic_shape", [False, True], ids=["static", "dynamic"])
def test_sequential_forward(dynamic_shape):
    # NI = 2: sequential loop, 16-row chunks, single KV version.
    _check(33, 2048, 256, dynamic_shape=dynamic_shape)
