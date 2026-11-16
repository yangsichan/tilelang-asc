"""DSA forward example with per-head Q/K/V on Ascend 950.

The DSL and tiling configuration live in core.py. Correctness references and
regression cases live in test_glm_dsa.py. Each input shape is specialized at
compile time; mask bytes are nonzero for participating positions.
"""

import argparse

import torch
import tilelang

from core import FwdTiling, HEAD_DIM, INDEX_TOPK, NUM_HEADS, SOFTMAX_SCALE, glm_dsa_fwd, pick_num_blocks

# Re-export the builder/configuration for existing callers of this example.
__all__ = ["FwdTiling", "HEAD_DIM", "INDEX_TOPK", "NUM_HEADS", "SOFTMAX_SCALE", "glm_dsa_fwd", "pick_num_blocks", "compile_kernel"]


def compile_kernel(batch_size, seq_len, kv_len, *, num_blocks=None, tiling=None):
    return tilelang.compile(
        glm_dsa_fwd(batch_size, seq_len, kv_len, min(INDEX_TOPK, kv_len), num_blocks=num_blocks, tiling=tiling),
        target="ascend",
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


def run(batch_size=1, seq_len=448, *, kv_len=4096, num_blocks=None, tiling=None):
    """Prepare one input and demonstrate the compiled kernel call."""
    import torch_npu  # noqa: F401 - register the NPU tensor backend

    kernel = compile_kernel(batch_size, seq_len, kv_len, num_blocks=num_blocks, tiling=tiling)
    torch.manual_seed(88888888)
    q = torch.randn((batch_size, NUM_HEADS, seq_len, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    k = torch.randn((batch_size, NUM_HEADS, kv_len, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    v = torch.randn_like(k)
    top_k = min(INDEX_TOPK, kv_len)
    indices = torch.arange(top_k, dtype=torch.int32, device="npu").view(1, 1, top_k).expand(batch_size, seq_len, top_k).contiguous()
    mask = torch.ones((batch_size, 1, seq_len, kv_len), dtype=torch.bool, device="npu")
    output = kernel(q, k, v, indices, mask.view(torch.uint8))
    torch.npu.synchronize()
    print(f"Output: shape={tuple(output.shape)}, dtype={output.dtype}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=4096)
    args = parser.parse_args()
    run(batch_size=args.batch_size, seq_len=args.seq_len, kv_len=args.kv_len)
