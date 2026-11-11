"""Standalone row-wise softmax on Ascend NPU (SIMD + fused online softmax).

SIMD rewrite of ``example_softmax.py``: instead of the SIMT reducer path, this
kernel drives the AIV vector unit explicitly with 64-lane fp32 intrinsics
(``T.simd.*``), mirroring the ``softmax`` macro in ``flash_attention/core.py``.

Online-softmax fusion
---------------------
A naive SIMD softmax touches each column value three times (max scan, sum scan,
normalize). This version fuses the max scan and sum scan into a single pass over
64-lane column chunks, maintaining a running max ``m`` and running sum ``l``::

    m_i   = max(m_{i-1}, max_local(chunk_i))
    l_i   = l_{i-1} * exp(m_{i-1} - m_i) + sum(exp(chunk_i - m_i))

The rescale factor ``exp(m_{i-1} - m_i)`` is the online-softmax trick from
FlashAttention; it drops out to ``0`` on the first chunk (``m_0 = -inf``). A
second lightweight pass then writes ``exp(x - m_final) / l_final``.

Constraint: ``d`` must be a multiple of the fp32 SIMD lane count (64); no tail
masking is performed.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def softmax_simd_fwd(batch, d):
    if batch <= 0 or batch % 56 or d <= 0:
        raise ValueError("softmax expects a positive batch divisible by 56 and d > 0")
    if d % 64:
        raise ValueError("SIMD softmax requires d to be a multiple of the fp32 lane count (64)")
    N_CORES = 56
    VL = 64  # fp32 SIMD lane count
    num_chunks = d // VL
    # Two compact row tiles (input + output), each rows * d float32 elements.
    ub_budget = 208 * 1024
    rows = 4
    while rows > 1 and (batch % (N_CORES * rows) or 2 * rows * d * 4 > ub_budget):
        rows //= 2
    iterations = batch // (N_CORES * rows)
    N = batch * d

    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(N_CORES) as core_id:
            x_ub = T.alloc_shared(rows * d, "float32")
            y_ub = T.alloc_shared(rows * d, "float32")
            T.annotate_buffer_versions({x_ub: 2, y_ub: 1})
            for t in T.Pipelined(iterations, num_stages=2):
                base = (t * N_CORES + core_id) * rows * d
                T.copy(X[base : base + rows * d], x_ub)
                with T.SimdVF():
                    full = T.simd.pset(32, "PAT_ALL")
                    for r in range(rows):
                        # Fused online-softmax scan: single pass over the column
                        # chunks, updating running max m and running sum l.
                        m = T.simd.vdup(-T.infinity("float32"), "float32", full)
                        l = T.simd.vdup(T.float32(0.0), "float32", full)
                        for c in range(num_chunks):
                            s = T.simd.vld(x_ub[r * d + c * VL])
                            local_max = T.simd.vdupv(T.simd.vcmax(s, full), full, pos="POS_LOWEST")
                            m_new = T.simd.vmax(m, local_max, full)
                            e = T.simd.vexpdif(s, m_new, full)
                            row_sum = T.simd.vdupv(T.simd.vcadd(e, full), full, pos="POS_LOWEST")
                            alpha = T.simd.vexpdif(m, m_new, full)  # exp(m_old - m_new)
                            l = T.simd.vadd(T.simd.vmul(l, alpha, full), row_sum, full)
                            m = m_new
                        # Normalize: exp(x - m_final) / l_final.
                        for c in range(num_chunks):
                            s = T.simd.vld(x_ub[r * d + c * VL])
                            y = T.simd.vdiv(T.simd.vexpdif(s, m, full), l, full)
                            T.simd.vsts(y_ub[r * d + c * VL], y, full)
                T.copy(y_ub, Y[base : base + rows * d])

    return main


def ref_program(x):
    """Reference: PyTorch softmax along the trailing dimension."""
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def run_regression_perf(batch=3584, d=4096):
    kernel = tilelang.compile(softmax_simd_fwd(batch, d), out_idx=-1)
    x = torch.randn(batch * d, device="npu")
    latency = do_bench(lambda: kernel(x), backend="msprof", _n_warmup=30, _n_repeat=50)
    num_bytes = 2 * x.numel() * x.element_size()
    bandwidth = num_bytes / (latency * 1e6)
    print(f"softmax_simd (batch={batch}, d={d}): {latency * 1000:.2f} us | {bandwidth:.2f} GB/s")
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True, help="benchmark after correctness checks")
    args = parser.parse_args()
    batch, d = 224, 4096
    x = torch.randn(batch, d, device="npu")
    kernel = tilelang.compile(softmax_simd_fwd(batch, d), out_idx=-1)
    y = kernel(x.flatten())
    torch.testing.assert_close(y.reshape(batch, d), ref_program(x), rtol=1e-4, atol=1e-4)
    print("softmax_simd: correctness passed")
    if args.bench:
        run_regression_perf()