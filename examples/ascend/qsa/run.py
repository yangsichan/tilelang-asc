"""Run correctness checks and benchmarks for the final QSA kernel."""

import argparse

import torch
from tilelang.profiler import do_bench

from examples.ascend.qsa.compile import compile_qsa
from examples.ascend.qsa.kernel import (
    HEAD_DIM,
    NUM_HEADS,
    SUPPORTED_AIC_COUNTS,
)
from examples.ascend.qsa.testing import (
    fill_contract_corners,
    make_inputs,
    pad_q_heads,
    ref_sparse_gqa,
    split_indices,
)


def prepare_inputs(q, k, v, indices):
    safe, valid = split_indices(indices)
    q_padded = pad_q_heads(q)
    k_padded = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, 3)).contiguous()
    v_padded = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 3)).contiguous()
    return q_padded, k_padded, v_padded, safe, valid


def run(num_tokens, kv_length, *, dynamic_shape, aic_count, check_only, label):
    kernel = compile_qsa(
        num_tokens,
        kv_length,
        dynamic_shape=dynamic_shape,
        aic_count=aic_count,
    )
    q, k, v, indices = make_inputs(num_tokens, kv_length, seed=0)
    fill_contract_corners(indices, kv_length)
    expected = ref_sparse_gqa(q, k, v, indices)
    prepared = prepare_inputs(q, k, v, indices)
    actual = kernel(*prepared)
    actual = actual[:, :, :12, :].reshape(num_tokens, NUM_HEADS, HEAD_DIM)
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)
    max_error = (actual.float() - expected).abs().max().item()
    print(f"[{label}, dynamic={dynamic_shape}, blocks={aic_count}] correctness passed; max error {max_error:.6f}")
    if check_only:
        return

    torch.npu.synchronize()
    latency_ms = do_bench(lambda: kernel(*prepared), backend="msprof", _n_warmup=30, _n_repeat=20)
    valid_slots = float((indices >= 0).sum()) / num_tokens
    useful_flops = 4 * num_tokens * NUM_HEADS * HEAD_DIM * valid_slots
    tflops = useful_flops / (latency_ms * 1e-3) / 1e12
    print(f"[{label}] {latency_ms * 1000:.2f} us/iter | {tflops:.1f} useful TFLOPS")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, help="packed query rows for one custom shape")
    parser.add_argument("--kv-length", type=int, help="packed KV rows, required with --num-tokens")
    parser.add_argument("--dynamic", action="store_true", help="compile symbolic num_tokens and kv_length")
    parser.add_argument("--aic-count", type=int, choices=SUPPORTED_AIC_COUNTS, default=32)
    parser.add_argument("--check-only", action="store_true", help="run correctness checks without benchmarking")
    args = parser.parse_args(argv)

    if args.num_tokens is not None:
        assert args.kv_length is not None, "--kv-length is required with --num-tokens"
        run(
            args.num_tokens,
            args.kv_length,
            dynamic_shape=args.dynamic,
            aic_count=args.aic_count,
            check_only=args.check_only,
            label=f"q{args.num_tokens}-kv{args.kv_length}",
        )
        return

    run(16, 2048, dynamic_shape=args.dynamic, aic_count=args.aic_count, check_only=args.check_only, label="small")
    run(128, 65536, dynamic_shape=args.dynamic, aic_count=args.aic_count, check_only=args.check_only, label="decode")
    run(4096, 32768, dynamic_shape=args.dynamic, aic_count=args.aic_count, check_only=args.check_only, label="prefill")


if __name__ == "__main__":
    main()
