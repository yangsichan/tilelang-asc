"""Correctness tests for Ascend sparse MLA decode."""

import pytest
import torch

from .decode import flash_mla_with_kvcache, get_mla_metadata
from .example_decode import (
    FP4_RECORD_BYTES,
    FP8_RECORD_BYTES,
    HEAD_DIM,
    NUM_HEADS,
    _check_decode,
    make_decode_inputs,
    make_sink,
    reference_decode,
    run_decode,
)


@pytest.fixture
def device():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_op_timeout_ms(30000)
    return torch.device("npu", torch.npu.current_device())


def _decode(q, cache, indices, lengths, *, sink=None):
    metadata, _ = get_mla_metadata()
    out, lse = flash_mla_with_kvcache(q, cache, None, None, HEAD_DIM, metadata, indices=indices, attn_sink=sink, topk_length=lengths)
    torch.npu.synchronize()
    return out, lse


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.float8_e4m3fn])
def test_decode_reference_shapes_and_byte_containers(dtype):
    q = torch.zeros((1, 1, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16)
    indices = torch.full((1, 1, 64), -1, dtype=torch.int32)
    indices[0, 0, 0] = 0
    records = torch.full((2, FP8_RECORD_BYTES), 0x38, dtype=torch.uint8)
    records[:, HEAD_DIM:] = 128  # scale 2, E4M3 payload 1

    expected_out, expected_lse = reference_decode(q, indices, records.view(dtype))
    _check_decode(torch.full_like(q, 2), torch.zeros((1, NUM_HEADS, 1), dtype=torch.float32), expected_out, expected_lse, batch=1)


def test_decode_reference_fp4_nibbles_scales_and_prefix():
    q = torch.zeros((2, 1, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16)
    indices = torch.full((2, 1, 64), -1, dtype=torch.int32)
    extra_indices = torch.zeros((2, 1, 17), dtype=torch.int32)
    main = torch.zeros((1, 2, 1, FP8_RECORD_BYTES), dtype=torch.uint8)
    extra = torch.full((1, 1, 1, FP4_RECORD_BYTES), 0xB2, dtype=torch.uint8)
    extra[..., HEAD_DIM // 2 :] = 0x40  # E4M3 scale 2
    lengths = torch.tensor([1, 0], dtype=torch.int32)

    out, lse = reference_decode(q, indices, main, extra_records=extra, extra_indices=extra_indices, extra_topk_length=lengths)
    expected_out = torch.zeros_like(q)
    expected_out[0, ..., 0::2] = 2
    expected_out[0, ..., 1::2] = -3
    expected_lse = torch.zeros((2, NUM_HEADS, 1), dtype=torch.float32)
    expected_lse[1] = torch.inf
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)


@pytest.mark.parametrize(
    "batch,index_width,tokens,page_size,sink,cache_dtype",
    [
        (1, 64, 256, 2, False, torch.uint8),
        (2, 128, 1024, 3, True, torch.int8),
        (1, 192, 512, 8, False, torch.float8_e4m3fn),
    ],
)
def test_decode_matches_reference(device, batch, index_width, tokens, page_size, sink, cache_dtype):
    """B dynamic, P > 1, T a multiple of 64, optional sink."""
    q, cache, indices, _ = make_decode_inputs(batch, index_width, tokens=tokens, page_size=page_size)
    attn_sink = make_sink(enabled=sink)
    expected_out, expected_lse = reference_decode(q, indices, cache, attn_sink=attn_sink)
    q, cache, indices = (tensor.to(device) for tensor in (q, cache, indices))
    if attn_sink is not None:
        attn_sink = attn_sink.to(device)
    out, lse = _decode(q, cache.view(cache_dtype), indices, None, sink=attn_sink)
    _check_decode(out, lse, expected_out, expected_lse, batch)


def test_decode_variable_lengths_and_empty_rows(device):
    """topk_length limits the visible prefix, and a zero-length row emits zeros."""
    batch, index_width = 3, 128
    q, cache, indices, _ = make_decode_inputs(batch, index_width, tokens=512, page_size=2)
    lengths = torch.tensor([index_width, 0, index_width // 2], dtype=torch.int32)
    expected_out, expected_lse = reference_decode(q, indices, cache.view(-1, FP8_RECORD_BYTES), topk_length=lengths)
    q, cache, indices, lengths = (tensor.to(device) for tensor in (q, cache, indices, lengths))
    out, lse = _decode(q, cache, indices, lengths)
    _check_decode(out, lse, expected_out, expected_lse, batch)
    assert torch.count_nonzero(out[1]) == 0
    assert bool(torch.isinf(lse[1]).all())


def test_decode_all_invalid_rows(device):
    """Every selected position is -1: output zeros and an infinite LSE."""
    batch, index_width = 2, 64
    q, cache, indices, lengths = make_decode_inputs(batch, index_width, tokens=256, page_size=2)
    indices = torch.full_like(indices, -1)
    expected_out, expected_lse = reference_decode(q, indices, cache.view(-1, FP8_RECORD_BYTES), topk_length=lengths)
    q, cache, indices, lengths = (tensor.to(device) for tensor in (q, cache, indices, lengths))
    out, lse = _decode(q, cache, indices, lengths)
    _check_decode(out, lse, expected_out, expected_lse, batch)
    assert torch.count_nonzero(out) == 0
    assert bool(torch.isinf(lse).all())


@pytest.mark.usefixtures("device")
@pytest.mark.parametrize("extra_format,extra_width", [("fp8", 64), ("fp4", 67)])
def test_decode_extra_cache(extra_format, extra_width):
    """Exercise both cache formats and independently limit each prefix."""
    run_decode(
        2,
        128,
        tokens=36,
        page_size=3,
        sink=True,
        topk_length=[61, 128],
        extra_index_width=extra_width,
        extra_tokens=6,
        extra_page_size=2,
        extra_format=extra_format,
        extra_topk_length=[extra_width, 0],
        bench=False,
    )
