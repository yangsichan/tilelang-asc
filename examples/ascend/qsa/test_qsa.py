"""Device regression tests for the fixed QSA grouped-query forward."""

import pytest
import torch

from examples.ascend.qsa.compile import compile_qsa
from examples.ascend.qsa.testing import fill_contract_corners, make_inputs, ref_sparse_gqa, run_qsa


def _check(num_tokens, kv_length, *, dynamic_shape):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_op_timeout_ms(30000)
    kernel = compile_qsa(num_tokens, kv_length, dynamic_shape=dynamic_shape, aic_count=32)
    q, k, v, indices = make_inputs(num_tokens, kv_length, seed=0)
    fill_contract_corners(indices, kv_length)
    expected = ref_sparse_gqa(q, k, v, indices)
    output = run_qsa(kernel, q, k, v, indices)
    torch.npu.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dynamic_shape", [False, True], ids=["static", "dynamic"])
def test_qwen_prefill_contract(dynamic_shape):
    _check(4096, 32768, dynamic_shape=dynamic_shape)


@pytest.mark.parametrize("dynamic_shape", [False, True], ids=["static", "dynamic"])
def test_qwen_decode_contract(dynamic_shape):
    _check(128, 65536, dynamic_shape=dynamic_shape)


def test_dynamic_shape_reuse():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_op_timeout_ms(30000)
    kernel = compile_qsa(16, 2048, dynamic_shape=True, aic_count=32)
    for seed, (num_tokens, kv_length) in enumerate(((16, 2048), (64, 4096))):
        q, k, v, indices = make_inputs(num_tokens, kv_length, seed=seed)
        fill_contract_corners(indices, kv_length)
        expected = ref_sparse_gqa(q, k, v, indices)
        output = run_qsa(kernel, q, k, v, indices)
        torch.npu.synchronize()
        torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1e-2)


def test_two_tile_handoff():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_op_timeout_ms(30000)
    tokens, kv_length = 16, 2048
    kernel = compile_qsa(tokens, kv_length, aic_count=32)
    query = torch.zeros(tokens, 24, 256, dtype=torch.bfloat16, device="npu")
    key = torch.zeros(kv_length, 2, 256, dtype=torch.bfloat16, device="npu")
    indices = torch.full((tokens, 2176), -1, dtype=torch.int32, device="npu")
    indices[:, :256] = torch.arange(256, dtype=torch.int32, device="npu")
    masked = indices.clone()
    masked[:, 128:] = -1
    for first, second in ((0, 2), (4, -2), (-2, 0)):
        value = torch.zeros_like(key)
        value[:128].fill_(first)
        value[128:256].fill_(second)
        for selection, expected in ((indices, (first + second) / 2), (masked, first)):
            output = run_qsa(kernel, query, key, value, selection)
            torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
