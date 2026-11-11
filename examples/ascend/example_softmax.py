"""Standalone row-wise softmax on Ascend NPU (SIMT, register-resident fusions).

Follows the ``example_rmsnorm.py`` skeleton: a 64-core ``T.Kernel`` grid with
pipelined GM->UB copies and a ``T.SimtVF`` thread-parallel region. The softmax
math reuses the numerically stable idea from the FlashAttention ``softmax``
macro in ``flash_attention/core.py``: subtract the per-row maximum before
``exp`` so large inputs cannot overflow, then normalize by the row sum of the
shifted exponentials.

The row axis is the flattened batch dimension; each core processes ``rows``
contiguous rows of ``d`` columns per pipeline iteration.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def softmax_fwd(batch, d):
    if batch <= 0 or batch % 56 or d <= 0:
        raise ValueError("softmax expects a positive batch divisible by 56 and d > 0")
    N_CORES = 56
    TILE = tilelang.next_power_of_2(d)
    # Two compact row tiles (input + output), each rows * d float32 elements.
    # Fragments live in registers, so only the UB tiles count against the budget.
    ub_budget = 208 * 1024
    rows = 4
    while rows > 1 and (batch % (N_CORES * rows) or 2 * rows * d * 4 > ub_budget):
        rows //= 2
    if rows == 1:
        threads = 256 if TILE > 4096 else 128
    else:
        threads = 256 if rows == 4 else 512
    threads = min(threads, TILE)
    iterations = batch // (N_CORES * rows)
    N = batch * d

    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(N_CORES) as core_id:
            x_ub = T.alloc_shared(rows * d, "float32")
            y_ub = T.alloc_shared(rows * d, "float32")
            T.annotate_buffer_versions({x_ub: 2, y_ub: 1})
            for t in T.Pipelined(iterations, num_stages=2):
                row_id = (t * N_CORES + core_id) * rows
                base = row_id * d
                T.copy(X[base : base + rows * d], x_ub)
                with T.SimtVF(threads=threads):
                    # Only fragments need power-of-two padding; the DMA tiles stay
                    # compact, so padded lanes are masked out below via `i < d`.
                    x_frag = T.alloc_fragment((rows, TILE), "float32")
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        x_frag[r, i] = T.if_then_else(i < d, x_ub[r * d + i], T.float32(0))

                    # Pass 1: per-row maximum, for numerical stability only.
                    max_acc = T.alloc_reducer((rows,), "float32", op="max")
                    T.reducer_init(max_acc)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            T.reducer_update(max_acc[r], x_frag[r, i])
                    max_val = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(max_acc, max_val)

                    # Pass 2: per-row sum of exp(x - max).
                    sum_acc = T.alloc_reducer((rows,), "float32", op="sum")
                    T.reducer_init(sum_acc)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            T.reducer_update(sum_acc[r], T.exp(x_frag[r, i] - max_val[r]))
                    sum_val = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(sum_acc, sum_val)

                    # Normalize each element by the row sum.
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            y_ub[r * d + i] = T.exp(x_frag[r, i] - max_val[r]) / sum_val[r]
                T.copy(y_ub, Y[base : base + rows * d])

    return main


def ref_program(x):
    """Reference: PyTorch softmax along the trailing dimension."""
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def run_regression_perf(batch=3584, d=4096):
    kernel = tilelang.compile(softmax_fwd(batch, d), out_idx=-1)
    x = torch.randn(batch * d, device="npu")
    latency = do_bench(lambda: kernel(x), backend="msprof", _n_warmup=30, _n_repeat=50)
    # Logical X read and Y write.
    num_bytes = 2 * x.numel() * x.element_size()
    bandwidth = num_bytes / (latency * 1e6)
    print(f"softmax (batch={batch}, d={d}): {latency * 1000:.2f} us | {bandwidth:.2f} GB/s")
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True, help="benchmark after correctness checks")
    args = parser.parse_args()
    batch, d = 224, 4096
    x = torch.randn(batch, d, device="npu")
    kernel = tilelang.compile(softmax_fwd(batch, d), out_idx=-1)
    y = kernel(x.flatten())
    torch.testing.assert_close(y.reshape(batch, d), ref_program(x), rtol=1e-4, atol=1e-4)
    print("softmax: correctness passed")
    if args.bench:
        run_regression_perf()