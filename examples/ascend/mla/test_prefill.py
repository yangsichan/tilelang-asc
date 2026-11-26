"""Correctness tests for Ascend sparse MLA prefill."""

import pytest
import torch

from .example_prefill import SM_SCALE, _check_prefill, make_prefill_inputs, make_sink, reference_prefill
from .prefill import flash_mla_sparse_fwd


@pytest.fixture
def device():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_op_timeout_ms(30000)
    return torch.device("npu", torch.npu.current_device())


def _prefill(q, kv, indices, lengths, *, sink=None):
    out, max_logits, lse = flash_mla_sparse_fwd(
        q, kv, indices, SM_SCALE, attn_sink=sink, topk_length=lengths)
    torch.npu.synchronize()
    return out, max_logits, lse


@pytest.mark.parametrize("seq_len,kv_len,index_width,sink", [
    (16, 1024, 128, False),
    (8, 4096, 512, True),
    (3, 512, 192, False),
])
def test_prefill_matches_reference(device, seq_len, kv_len, index_width, sink):
    """Sq/Skv dynamic, T a multiple of 64, optional sink."""
    q, kv, indices, lengths = make_prefill_inputs(seq_len, kv_len, index_width)
    attn_sink = make_sink(enabled=sink)
    expected_out, expected_max, expected_lse = reference_prefill(
        q, kv, indices, topk_length=lengths, attn_sink=attn_sink)
    q, kv, indices, lengths = (tensor.to(device) for tensor in (q, kv, indices, lengths))
    if attn_sink is not None:
        attn_sink = attn_sink.to(device)
    out, max_logits, lse = _prefill(q, kv, indices, lengths, sink=attn_sink)
    _check_prefill(out, max_logits, lse, expected_out, expected_max, expected_lse, seq_len)


def test_prefill_variable_lengths_and_empty_rows(device):
    """A zero length row emits zeros, -inf max logits and an infinite LSE."""
    seq_len, kv_len, index_width = 4, 1024, 192
    q, kv, indices, _ = make_prefill_inputs(seq_len, kv_len, index_width)
    lengths = torch.tensor([0, 1, 65, 131], dtype=torch.int32)
    expected_out, expected_max, expected_lse = reference_prefill(
        q, kv, indices, topk_length=lengths)
    q, kv, indices, lengths = (tensor.to(device) for tensor in (q, kv, indices, lengths))
    out, max_logits, lse = _prefill(q, kv, indices, lengths)
    _check_prefill(out, max_logits, lse, expected_out, expected_max, expected_lse, seq_len)
    assert torch.count_nonzero(out[0]) == 0
    assert bool(torch.isinf(max_logits[0]).all() and (max_logits[0] < 0).all())
    assert bool(torch.isinf(lse[0]).all())


def test_prefill_all_invalid_indices(device):
    """Negative and out-of-range indices are masked instead of rejected."""
    seq_len, kv_len, index_width = 2, 512, 64
    q, kv, indices, lengths = make_prefill_inputs(seq_len, kv_len, index_width)
    indices = torch.full_like(indices, -1)
    indices[:, 0, : index_width // 2] = kv_len  # out-of-range tail
    expected_out, expected_max, expected_lse = reference_prefill(
        q, kv, indices, topk_length=lengths)
    q, kv, indices, lengths = (tensor.to(device) for tensor in (q, kv, indices, lengths))
    out, max_logits, lse = _prefill(q, kv, indices, lengths)
    _check_prefill(out, max_logits, lse, expected_out, expected_max, expected_lse, seq_len)
    assert torch.count_nonzero(out) == 0
    assert bool(torch.isinf(lse).all())


def test_prefill_implicit_full_lengths(device):
    """The implicit full prefix can be wider than the model indexer cap."""
    seq_len, kv_len, index_width = 5, 777, 576
    q, kv, indices, _ = make_prefill_inputs(seq_len, kv_len, index_width)
    expected_out, expected_max, expected_lse = reference_prefill(q, kv, indices)
    q, kv, indices = (tensor.to(device) for tensor in (q, kv, indices))
    out, max_logits, lse = _prefill(q, kv, indices, None)
    _check_prefill(out, max_logits, lse, expected_out, expected_max, expected_lse, seq_len)
