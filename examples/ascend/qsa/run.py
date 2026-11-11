"""Correctness checks and benchmarks for the QSA kernels."""

import argparse

import torch
from tilelang.profiler import do_bench

from examples.ascend.qsa.compile import compile_qsa
from examples.ascend.qsa.kernel import HEAD_DIM, NUM_HEADS
from examples.ascend.qsa.testing import (
    fill_contract_corners,
    make_inputs,
    pad_q_heads,
    ref_sparse_gqa,
    run_qsa,
    split_indices,
)


def run(num_tokens, kv_length, top_k=2176, *, dynamic_shape=False, label=None):
    kernel = compile_qsa(num_tokens, kv_length, top_k, dynamic_shape=dynamic_shape)
    q, k, v, indices = make_inputs(num_tokens, kv_length, top_k, seed=0)
    fill_contract_corners(indices, kv_length, top_k)
    expected = ref_sparse_gqa(q, k, v, indices)
    actual = run_qsa(kernel, q, k, v, indices)
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)
    print(f"Verification passed; max error: {(actual.float() - expected).abs().max().item():.6f}")

    q, k, v, indices = make_inputs(num_tokens, kv_length, top_k, seed=1, fixed_budget=min(2051, top_k))
    safe, valid = split_indices(indices)
    qp = pad_q_heads(q)
    latency_ms = do_bench(lambda: kernel(qp, k, v, safe, valid), backend="msprof", _n_warmup=30, _n_repeat=20)
    # Useful flops: 24 real query heads x average valid slots per query
    # (QSA selects fewer than top_k slots; pad rows do no useful work).
    n_valid = float((indices >= 0).sum()) / num_tokens
    flops = 4 * num_tokens * NUM_HEADS * HEAD_DIM * n_valid
    label = label or f"q{num_tokens}-kv{kv_length}"
    print(f"[{label}] {latency_ms * 1000:.2f} us/iter | {flops / (latency_ms * 1e-3) / 1e12:.1f} TFLOPS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, help="run a single custom shape instead of the default scenarios")
    parser.add_argument("--kv-length", type=int, help="KV length for the custom shape (required with --num-tokens)")
    parser.add_argument("--top-k", type=int, default=2176)
    parser.add_argument("--dynamic", action="store_true", help="symbolic num_tokens/kv_length (one kernel serves all shapes)")
    args = parser.parse_args()

    if args.num_tokens is not None:
        assert args.kv_length is not None, "--kv-length is required with --num-tokens"
        run(args.num_tokens, args.kv_length, args.top_k, dynamic_shape=args.dynamic)
        return
    # The small run covers the sequential path (NI <= 2); decode/prefill
    # cover the pipelined path (NI = 17) and the two real-world scenarios.
    run(16, 2048, 256, dynamic_shape=args.dynamic, label="correctness-small")
    run(128, 65536, args.top_k, dynamic_shape=args.dynamic, label="decode-qwen3.8")
    run(4096, 32768, args.top_k, dynamic_shape=args.dynamic, label="prefill-qwen3.8")


if __name__ == "__main__":
    main()
