"""DSA forward example with per-head Q/K/V on Ascend 950.

The DSL and tiling configuration live in core.py. This example provides the
FP32 reference and a standalone accuracy check and benchmark; regression cases
live in test_glm_dsa.py. Each input shape is specialized at
compile time; mask bytes are nonzero for participating positions.
"""

import argparse

import torch
import tilelang
from tilelang.profiler import do_bench

from core import FwdTiling, HEAD_DIM, INDEX_TOPK, NUM_HEADS, SOFTMAX_SCALE, glm_dsa_fwd, pick_num_blocks

# Re-export the builder/configuration for existing callers of this example.
__all__ = ["FwdTiling", "HEAD_DIM", "INDEX_TOPK", "NUM_HEADS", "SOFTMAX_SCALE", "glm_dsa_fwd", "pick_num_blocks", "compile_kernel"]


def make_inputs(batch_size: int, seq_len: int, kv_len: int):
    torch.manual_seed(88888888)
    top_k = min(INDEX_TOPK, kv_len)
    q = torch.randn((batch_size, NUM_HEADS, seq_len, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    k = torch.randn((batch_size, NUM_HEADS, kv_len, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    v = torch.randn((batch_size, NUM_HEADS, kv_len, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    # Dense top-k, exactly what the caller produces:
    #   topk = min(index_topk, index_scores.shape[-1])
    #   ind = index_scores.topk(topk, dim=-1).indices.to(torch.int32)
    index_scores = torch.rand(batch_size, seq_len, kv_len, device="npu")
    indices = index_scores.topk(top_k, dim=-1).indices.to(torch.int32)
    mask = torch.rand(batch_size, 1, seq_len, kv_len, device="npu") > 0.1
    return q, k, v, indices, mask


def reference_attention(q, k, v, indices, mask):
    """Per-query reference in float32; the gathered tiles are too large to hold at once."""
    batch_size, heads, seq_len, dim = q.shape
    mask_u8 = mask.view(torch.uint8) if mask.dtype == torch.bool else mask
    out = torch.empty((batch_size, seq_len, heads, dim), dtype=torch.float32, device=q.device)
    for batch in range(batch_size):
        k_batch = k[batch]
        v_batch = v[batch]
        for query in range(seq_len):
            # torch.gather/advanced indexing require int64 indices.
            idx = indices[batch, query].long()
            gathered_k = k_batch[:, idx, :].float()  # [H, Ksel, D]
            gathered_v = v_batch[:, idx, :].float()
            scores = torch.einsum("hd,hkd->hk", q[batch, :, query, :].float(), gathered_k) * SOFTMAX_SCALE
            keep = mask_u8[batch, 0, query][idx].ne(0)  # [Ksel]
            scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
            # Softmax over an all-masked row is NaN; the kernel emits zeros there.
            probabilities = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
            out[batch, query] = torch.einsum("hk,hkd->hd", probabilities, gathered_v)
    return out


def compile_kernel(batch_size, seq_len, kv_len, *, num_blocks=None, tiling=None):
    return tilelang.compile(
        glm_dsa_fwd(batch_size, seq_len, kv_len, min(INDEX_TOPK, kv_len), num_blocks=num_blocks, tiling=tiling),
        target="ascend",
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


def run(batch_size=1, seq_len=448, *, kv_len=4096, num_blocks=None, tiling=None):
    """Check one input against Golden, then profile the resident kernel call."""
    import torch_npu  # noqa: F401 - register the NPU tensor backend

    kernel = compile_kernel(batch_size, seq_len, kv_len, num_blocks=num_blocks, tiling=tiling)
    q, k, v, indices, mask = make_inputs(batch_size, seq_len, kv_len)
    mask_u8 = mask.view(torch.uint8)
    expected = reference_attention(q, k, v, indices, mask)
    output = kernel(q, k, v, indices, mask_u8)
    torch.npu.synchronize()
    assert output.dtype == torch.bfloat16
    assert output.shape == expected.shape and output.is_contiguous()
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1e-2)
    print("Accuracy PASS", flush=True)

    # Compile, input preparation, and Golden are outside the timed callable.
    prof = do_bench(lambda: kernel(q, k, v, indices, mask_u8), backend="msprof_detail", _n_warmup=30, _n_repeat=100)
    flops = 4.0 * batch_size * seq_len * NUM_HEADS * HEAD_DIM * min(INDEX_TOPK, kv_len)
    print(f"{prof.dur_us:.2f} us/iter | {prof.tflops(flops):.1f} TFLOPS", flush=True)
    return prof.dur_ns / 1e6


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=4096)
    args = parser.parse_args()
    run(batch_size=args.batch_size, seq_len=args.seq_len, kv_len=args.kv_len)
