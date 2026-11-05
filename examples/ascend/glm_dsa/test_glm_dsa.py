"""Correctness tests for the GLM-5.3 DSA forward kernel.

Every case runs on the NPU: the kernel output is compared against the float32
per-query reference in ``example_glm_dsa``.  The reference loops over query rows
in Python, so the shapes stay small.
"""

import pytest
import torch

from example_glm_dsa import (
    NUM_HEADS,
    compile_kernel,
    make_inputs,
    reference_attention,
)


def _check(batch_size, seq_len, kv_len, num_blocks, mask_mode="mixed"):
    kernel = compile_kernel(batch_size, seq_len, kv_len, num_blocks=num_blocks)
    q, k, v, indices, attn_mask = make_inputs(batch_size, seq_len, kv_len)
    if mask_mode == "all_keep":
        attn_mask = torch.ones_like(attn_mask)
    elif mask_mode == "all_drop":
        attn_mask = torch.zeros_like(attn_mask)

    expected = reference_attention(q, k, v, indices, attn_mask)
    actual = kernel(q, k, v, indices, attn_mask.view(torch.uint8))
    torch.npu.synchronize()

    assert actual.dtype == torch.bfloat16
    assert actual.shape == (batch_size, seq_len, NUM_HEADS, 256)
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)
    return actual


@pytest.mark.parametrize("mask_mode", ["mixed", "all_keep"])
def test_single_topk_block(mask_mode):
    """Ksel <= 128: one index tile, no online-softmax update."""
    _check(1, 32, 96, 8, mask_mode=mask_mode)


@pytest.mark.parametrize("kv_len,seq_len", [(1024, 32), (2048, 16)])
def test_multi_topk_block(kv_len, seq_len):
    """Several 128-wide tiles with running max/sum and output rescaling."""
    _check(1, seq_len, kv_len, 8)


def test_batch_and_query_blocks():
    """Two batches, more query rows than cores."""
    _check(2, 32, 512, 16)


def test_fully_masked_query_emits_zeros():
    """Every selected position is dropped: the query must emit zeros, not NaN."""
    actual = _check(1, 32, 512, 8, mask_mode="all_drop")
    assert torch.count_nonzero(actual.float()) == 0


def test_partial_tile_tail():
    """Ksel is not a multiple of the 128-wide tile, so the last tile is partial."""
    _check(1, 16, 160, 8)


@pytest.mark.parametrize("kv_len,seq_len", [(257, 8), (2049, 4)])
def test_unaligned_kv_len(kv_len, seq_len):
    """T need not be 32-aligned or <= 32768.

    257 is neither, so both the mask and the indices fall back to scalar GM
    reads; 2049 keeps a staged index tile (Ksel = 2048) but not a staged mask
    row, and spans several index tiles.
    """
    _check(1, seq_len, kv_len, seq_len)


def test_both_halves_of_a_tile_contribute():
    """Uniform weights over one tile must average both AIV halves.

    Q and K are zero, so all 128 selected positions carry the same weight, and
    V is 0 over the first half of the tile and 1 over the second.  The output is
    the mean, 0.5.  A keep-flag vector that only covers the half gathered by one
    AIV yields 0 (or 1) instead, because the other 64 columns stay masked.
    """
    batch_size, seq_len, kv_len, num_blocks = 1, 8, 128, 8
    kernel = compile_kernel(batch_size, seq_len, kv_len, num_blocks=num_blocks)
    q = torch.zeros(batch_size, NUM_HEADS, seq_len, 256, dtype=torch.bfloat16, device="npu")
    k = torch.zeros(batch_size, NUM_HEADS, kv_len, 256, dtype=torch.bfloat16, device="npu")
    v = torch.zeros(batch_size, NUM_HEADS, kv_len, 256, dtype=torch.bfloat16, device="npu")
    v[:, :, 64:, :] = 1.0
    indices = (
        torch.arange(kv_len, dtype=torch.int32, device="npu")
        .view(1, 1, kv_len)
        .expand(batch_size, seq_len, kv_len)
        .contiguous()
    )
    attn_mask = torch.ones(batch_size, 1, seq_len, kv_len, dtype=torch.uint8, device="npu")

    actual = kernel(q, k, v, indices, attn_mask)
    torch.npu.synchronize()

    expected = torch.full_like(actual.float(), 0.5)
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)
