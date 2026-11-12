# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Q4Kernels contributors.
"""Standalone mHC Post device benchmark: three hidden sizes and 20 lengths."""

import argparse
import csv
from contextlib import nullcontext
import math
from pathlib import Path

from examples.ascend.mhc._bench import capture_output, open_log


HIDDEN_SIZES = (4096, 5120, 7168)
LENGTHS = (1, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2304, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192)


def _positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _length(value):
    number = _positive(value)
    if number > 8192:
        raise argparse.ArgumentTypeError("length must be in 1..8192")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", choices=HIDDEN_SIZES, default=HIDDEN_SIZES)
    parser.add_argument("--lengths", type=_length, nargs="+", default=LENGTHS)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--vec-cores", type=_positive, default=56)
    parser.add_argument("--warmup", type=_positive, default=30)
    parser.add_argument("--repeat", type=_positive, default=500)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--output", type=Path, help="optional new CSV file; completed rows are flushed after every case")
    parser.add_argument("--log-file", type=Path, help="optional new log file; defaults to a unique file in the temporary directory")
    parser.add_argument("--list-cases", action="store_true", help="list shapes without importing NPU dependencies")
    return parser


def _latency_us(profile):
    duration = getattr(profile, "dur_ns", float("nan"))
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("msprof_detail did not return a positive finite device duration")
    return duration / 1000


def benchmark(n, hidden, args):
    """Return one msprof_detail batch mean in microseconds, with fresh outputs."""
    import torch
    import torch_npu  # noqa: F401
    from tilelang.profiler import do_bench

    from examples.ascend.mhc.mhc_post import _compiled_kernel

    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(args.seed + hidden * 8193 + n)
    with torch.inference_mode():
        x = torch.randn((n, hidden), dtype=torch.bfloat16, device=device)
        residual = torch.randn((n, 4, hidden), dtype=torch.bfloat16, device=device)
        post = 2 * torch.randn((n, 4), dtype=torch.float32, device=device).sigmoid()
        comb = torch.randn((n, 4, 4), dtype=torch.float32, device=device).softmax(-1)
        kernel = _compiled_kernel(hidden, args.vec_cores)

        def run():
            output = torch.empty_like(residual)
            kernel(comb, residual, post, x, output)
            return output

        output = run()
        torch.npu.synchronize(device)
        del output
        profile = do_bench(
            run, backend="msprof_detail", device=device, _n_warmup=args.warmup, _n_repeat=args.repeat, early_stop_baseline=None
        )
    return _latency_us(profile)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.device.startswith("npu:"):
        parser.error("--device must select an NPU, for example npu:0")
    cases = [(hidden, n) for hidden in dict.fromkeys(args.hidden_sizes) for n in dict.fromkeys(args.lengths)]
    if args.list_cases:
        print("hidden,length")
        for hidden, n in cases:
            print(f"{hidden},{n}")
        return 0
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    destination = args.output.open("x", newline="") if args.output else nullcontext(None)
    with destination as output, open_log(args.log_file, "post") as log:
        print(f"Detailed log: {log.name}", flush=True)
        writer = csv.writer(output) if output else None
        if writer:
            writer.writerow(("hidden", "length", "device_mean_us"))
            output.flush()
        for index, (hidden, n) in enumerate(cases, 1):
            label = f"[{index:2d}/{len(cases)}] Post H={hidden} N={n:4d}"
            print(label, end=" ... ", flush=True)
            print(label, file=log, flush=True)
            try:
                with capture_output(log):
                    latency = benchmark(n, hidden, args)
            except Exception as exc:
                print(f"FAILED; see {log.name}", flush=True)
                raise RuntimeError(f"mHC Post benchmark failed for H={hidden}, N={n}; details: {log.name}") from exc
            print(f"device_mean_us={latency:10.3f}", flush=True)
            print(f"device_mean_us={latency:.3f}", file=log, flush=True)
            if writer:
                writer.writerow((hidden, n, latency))
                output.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
