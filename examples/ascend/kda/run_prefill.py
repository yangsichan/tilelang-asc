"""Correctness and performance runner for the Ascend KDA chunk prefill.

``smoke`` checks one small aligned shape. ``correctness`` checks the 15 packed
shapes from ``bench_kda_ascendc_vs_tilelang.py``. ``bench`` times the fixed
profile from ``test_chunk_prefill_ascend_perf.py`` (``S=16384``, ``H=96``).
Every mode launches ``KDAChunkPrefill``. Smoke and correctness compare the
output and the final state with the float32 reference below, on CPU. Their
inputs are a seeded ``randn`` in the bf16-safe range. Bench keeps the constant
AscendC profile.

The reference is the TileOPs ``kda_ref``. The Ascend kernel multiplies
``1/sqrt(K)`` only onto ``qg @ h``, so the comparison passes
``scale_state_only=True``.

From the repository root, on an A5 NPU:

    export ASCEND_RT_VISIBLE_DEVICES=0
    export PYTHONPATH=.
    python examples/kda/run_prefill.py --mode smoke
    python examples/kda/run_prefill.py --mode correctness
    python examples/kda/run_prefill.py --mode bench
"""

import argparse
import functools
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_REPO_ROOT), str(Path(__file__).resolve().parent)]

import tilelang  # noqa: E402
from kda_prefill import (  # noqa: E402
    ATTN_SCALE,
    BLOCK_T,
    DIM_K,
    DIM_V,
    DT_LOWER_BOUND,
    HEADS,
    SEQ_LEN,
    KDAChunkPrefill,
    build,
)

try:
    import torch_npu  # noqa: E402
except ImportError:
    torch_npu = None

# (total_tokens, num_seqs, heads). Sequence length is total_tokens / num_seqs.
# Same packed shapes as bench_kda_ascendc_vs_tilelang.py. Every sequence is a
# multiple of 64, so the launch takes the aligned kernel.
_CORRECTNESS: tuple[tuple[int, int, int], ...] = (
    (1024, 16, 12),
    (2048, 16, 48),
    (2048, 32, 24),
    (4096, 16, 12),
    (4096, 32, 24),
    (4096, 64, 96),
    (8192, 32, 24),
    (8192, 64, 12),
    (8192, 128, 96),
    (16384, 32, 24),
    (16384, 64, 12),
    (16384, 128, 96),
    (32768, 64, 48),
    (32768, 128, 12),
    (65536, 128, 96),
)
# Two chunks and one head pair: both AIVs run, and the state term is nonzero.
_SMOKE: tuple[tuple[int, int, int], ...] = ((128, 1, 2),)
# The AscendC profile timed by test_chunk_prefill_ascend_perf.py.
_BENCH: tuple[tuple[int, int, int], ...] = ((SEQ_LEN, 1, HEADS),)

# Mixed tolerance from the Ascend float-compute standard.
_BF16 = {"atol": 2**-6, "rtol": 2**-6, "ratio": 0.99, "limit": 1.0}
_FP32 = {"atol": 2**-16, "rtol": 2**-10, "ratio": 0.99, "limit": 1e-2}


def kda_ref(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    a_log=None,
    dt_bias=None,
    *,
    scale=None,
    raw_gate=False,
    lower_bound=None,
    beta_sigmoid=False,
    allow_neg_eigval=False,
    l2norm=False,
    state_v_first=False,
    scale_state_only=False,
):
    """The KDA recurrence evaluated independently of the in-tree kernels and of FLA.

    Per token: the state decays by ``exp(g)`` along its key axis, the delta rule writes
    ``beta * (v - k^T S)`` along ``k``, and the output reads ``q^T S``. The recurrence runs
    chunk by chunk in float32: within a chunk the written values solve the unit
    lower-triangular system exactly, with every per-channel decay gap materialized, so no
    gap is exponentiated with a positive argument. The sequence boundaries come from the
    call's offsets, and a sequence with no token ends on the state it started from.

    ``scale_state_only`` multiplies ``scale`` onto the query-state term and leaves the
    intra-chunk term unscaled. The default applies ``scale`` to both, which is the
    TileOPs recurrence. The Ascend kernel uses the former: it stores ``qg = q * exp(g)``
    and multiplies ``1/sqrt(K)`` only onto ``qg @ h``.
    """
    dtype = torch.float32
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    group = v.shape[2] // q.shape[2]
    if cu_seqlens is None:
        rows = list(range(q.shape[0]))
        starts, lengths = [0] * q.shape[0], [q.shape[1]] * q.shape[0]
    else:
        bounds = (cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu).tolist()
        rows, starts = [0] * (len(bounds) - 1), bounds[:-1]
        lengths = [end - start for start, end in zip(bounds[:-1], bounds[1:], strict=True)]
    g = g.to(dtype)
    if raw_gate:
        heads, dim = v.shape[2], q.shape[-1]
        x = g + (0.0 if dt_bias is None else dt_bias.to(dtype).view(heads, dim))
        rate = a_log.to(dtype).exp()[:, None]
        g = lower_bound * torch.sigmoid(rate * x) if lower_bound is not None else -rate * F.softplus(x)
    beta = beta.to(dtype)
    if beta_sigmoid:
        beta = torch.sigmoid(beta) * (2.0 if allow_neg_eigval else 1.0)
    longest = max(lengths, default=0)
    # The exact per-channel decay gap is [sequences, value heads, chunk, chunk, K] in float32,
    # which bounds the chunk; a decode call runs one token per step and needs no gap tensor.
    chunk = 1 if longest <= 1 else 16
    # Longest first, so the sequences still running at a step are a prefix.
    order = sorted(range(len(lengths)), key=lambda sequence: -lengths[sequence])
    row = torch.tensor([rows[s] for s in order], dtype=torch.long, device=q.device)
    start = torch.tensor([starts[s] for s in order], dtype=torch.long, device=q.device)
    length = torch.tensor([lengths[s] for s in order], dtype=torch.long, device=q.device)
    if initial_state is None:
        state = torch.zeros(len(order), v.shape[2], q.shape[-1], v.shape[-1], dtype=dtype, device=q.device)
    else:
        state = initial_state[order].to(dtype)
        state = state.transpose(-1, -2).contiguous() if state_v_first else state
    output = torch.zeros_like(v)
    offset = torch.arange(chunk, device=q.device)
    inclusive = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    strict = inclusive.tril(-1)
    running = len(order)
    for first in range(0, longest, chunk):
        while lengths[order[running - 1]] <= first:
            running -= 1
        r = row[:running, None]
        valid = first + offset < length[:running, None]
        t = (start[:running, None] + first + offset).clamp(max=q.shape[1] - 1)
        mask = valid.to(dtype)[..., None]
        # [running, chunk, value heads, K]; value head h reads the key head its group shares.
        q_c = q[r, t].to(dtype).repeat_interleave(group, dim=2)
        k_c = k[r, t].to(dtype).repeat_interleave(group, dim=2)
        if l2norm:
            q_c = q_c * torch.rsqrt(q_c.square().sum(-1, keepdim=True) + 1e-6)
            k_c = k_c * torch.rsqrt(k_c.square().sum(-1, keepdim=True) + 1e-6)
        # [running, value heads, chunk, ...]. Padding carries no key, value, step or decay.
        q_c = q_c.transpose(1, 2)
        q_state = q_c * scale
        q_attn = q_c if scale_state_only else q_state
        k_c = (k_c * mask[..., None]).transpose(1, 2)
        v_c = (v[r, t].to(dtype) * mask[..., None]).transpose(1, 2)
        step = (beta[r, t] * mask).transpose(1, 2)
        log_decay = (g[r, t] * mask[..., None]).transpose(1, 2).cumsum(2)
        live = state[:running]
        # gap[i, j, c] = log_decay[i, c] - log_decay[j, c], used only where j <= i.
        gap = log_decay[..., :, None, :] - log_decay[..., None, :, :]
        weight = gap.masked_fill(~inclusive[..., None], -torch.inf).exp()
        system = step[..., None] * torch.einsum("nhic,nhijc,nhjc->nhij", k_c, weight, k_c)
        system = system.masked_fill(~strict, 0.0)
        decayed = log_decay.exp()
        rhs = step[..., None] * (v_c - (k_c * decayed) @ live)
        written = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
        attention = torch.einsum("nhic,nhijc,nhjc->nhij", q_attn, weight, k_c)
        out = ((q_state * decayed) @ live + attention @ written).transpose(1, 2)
        carry = (log_decay[..., -1:, :] - log_decay).exp()
        live = decayed[..., -1, :, None] * live + (k_c * carry).transpose(-1, -2) @ written
        state[:running] = live
        output[r.expand_as(t)[valid], t[valid]] = out[valid].to(v.dtype)
    final = torch.empty_like(state)
    final[order] = state
    final = final.transpose(-1, -2) if state_v_first else final
    return output, final.float()


def _cases(mode: str) -> list[tuple[int, int, int, int]]:
    table = {"smoke": _SMOKE, "correctness": _CORRECTNESS, "bench": _BENCH}[mode]
    selected = []
    for total, num_seqs, heads in table:
        if total % num_seqs:
            raise ValueError(f"T={total} is not divisible by N={num_seqs}")
        seq_len = total // num_seqs
        if seq_len % BLOCK_T:
            raise ValueError(f"sequence length {seq_len} is not a multiple of {BLOCK_T}")
        if heads % 2:
            raise ValueError(f"H={heads} must be even")
        selected.append((total, num_seqs, heads, seq_len))
    return selected


_SEED = 20260808


def _profile_inputs(device: torch.device, total: int, heads: int):
    """The AscendC profile values: constant q/k, v, raw gate, and beta."""

    shape = (1, total, heads, DIM_K)
    level = 1.0 / math.sqrt(DIM_K)
    q = torch.full(shape, level, dtype=torch.bfloat16, device=device)
    k = torch.full_like(q, level)
    v = torch.full_like(q, 0.01)
    raw_g = torch.full(shape, -1.0, dtype=torch.float32, device=device)
    beta = torch.full(shape[:-1], 0.5, dtype=torch.bfloat16, device=device)
    a_log = torch.zeros((heads,), dtype=torch.float32, device=device)
    dt_bias = torch.zeros((heads * DIM_K,), dtype=torch.float32, device=device)
    return q, k, v, raw_g, beta, a_log, dt_bias


def _randn_inputs(device: torch.device, total: int, heads: int, seed: int):
    """bf16-safe random inputs. Drawn on CPU so the seed does not depend on the NPU.

    ``q/k/v`` are ``level * randn``. ``beta`` is ``sigmoid(randn)``, so it stays in
    ``(0, 1)``. ``raw_g`` is ``0.02 * randn - 0.5``. ``A_log`` and ``dt_bias`` are
    ``0.1 * randn``.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shape = (1, total, heads, DIM_K)
    q = (torch.randn(shape, generator=generator) * 0.05).to(torch.bfloat16)
    k = (torch.randn(shape, generator=generator) * 0.05).to(torch.bfloat16)
    v = (torch.randn(shape, generator=generator) * 0.05).to(torch.bfloat16)
    raw_g = torch.randn(shape, generator=generator) * 0.02 - 0.5
    beta = torch.sigmoid(torch.randn(shape[:-1], generator=generator)).to(torch.bfloat16)
    a_log = torch.randn((heads,), generator=generator) * 0.1
    dt_bias = torch.randn((heads * DIM_K,), generator=generator) * 0.1
    return tuple(tensor.to(device) for tensor in (q, k, v, raw_g, beta, a_log, dt_bias))


def _cu_seqlens(num_seqs: int, seq_len: int, device: torch.device) -> torch.Tensor:
    host = [index * seq_len for index in range(num_seqs + 1)]
    return torch.tensor(host, dtype=torch.int64, device=device)


def _reference(inputs, cu_seqlens: torch.Tensor | None):
    """Float32 reference on CPU. The NPU workspace for this recurrence stalls."""

    q, k, v, raw_g, beta, a_log, dt_bias = (tensor.detach().to("cpu") for tensor in inputs)
    cu = None if cu_seqlens is None else cu_seqlens.detach().to("cpu")
    print("computing float32 kda_ref on CPU", flush=True)
    return kda_ref(
        q,
        k,
        v,
        raw_g,
        beta,
        None,
        cu,
        cu,
        a_log,
        dt_bias,
        scale=float(ATTN_SCALE),
        raw_gate=True,
        lower_bound=float(DT_LOWER_BOUND),
        scale_state_only=True,
    )


def _mixed_tolerance(actual: torch.Tensor, expected: torch.Tensor, kind: str) -> dict:
    bound = _BF16 if kind == "bf16" else _FP32
    got = actual.detach().float().cpu()
    ref = expected.detach().float().cpu()
    if got.shape != ref.shape:
        return {
            "pass": False,
            "error": f"shape {tuple(got.shape)} != {tuple(ref.shape)}",
        }
    diff = (got - ref).abs()
    matched = diff <= bound["atol"] + bound["rtol"] * ref.abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    ratio = float(matched.float().mean()) if matched.numel() else 1.0
    return {
        "pass": ratio >= bound["ratio"] and max_abs <= bound["limit"],
        "matched_ratio": ratio,
        "max_abs": max_abs,
        "atol": bound["atol"],
        "rtol": bound["rtol"],
        "max_abs_limit": bound["limit"],
    }


def _check(actual: torch.Tensor, expected: torch.Tensor, kind: str, name: str) -> dict:
    report = _mixed_tolerance(actual, expected, kind)
    if "error" in report:
        print(f"{name} {report['error']}", flush=True)
        return report
    flag = "PASS" if report["pass"] else "FAIL"
    print(
        f"{name} {flag} max_abs={report['max_abs']:.3e} matched={report['matched_ratio']:.6f}",
        flush=True,
    )
    return report


def _bench(fn, warmup: int, repeat: int) -> float:
    from tilelang.profiler import do_bench

    return float(
        do_bench(
            fn,
            _n_warmup=warmup,
            _n_repeat=repeat,
            backend="msprof",
            device=torch.device("npu"),
            early_stop_baseline=None,
        )
    )


def _launch(op: KDAChunkPrefill, inputs, cu: torch.Tensor | None):
    if cu is None:
        return op(*inputs)
    return op(*inputs, cu_seqlens=cu)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "correctness", "bench"], default="smoke")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--case", type=int, help="Run only this zero-based case number")
    args = parser.parse_args()
    if torch_npu is None or not torch.npu.is_available():
        raise SystemExit("This runner requires torch_npu and an NPU.")

    torch.npu.set_device(args.device)
    device = torch.device("npu", torch.npu.current_device())
    torch.npu.set_op_timeout_ms(30000)
    cases = _cases(args.mode)
    result = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "tilelang": tilelang.__version__,
        "device": torch.npu.get_device_name(device),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "mode": args.mode,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    operators: dict[int, KDAChunkPrefill] = {}
    failed = False
    for case_id, (total, num_seqs, heads, seq_len) in enumerate(cases):
        if args.case is not None and case_id != args.case:
            continue
        print(
            f"CASE T={total} N={num_seqs} H={heads} seq={seq_len}",
            flush=True,
        )
        seed = _SEED + case_id
        if args.mode == "bench":
            inputs = _profile_inputs(device, total, heads)
        else:
            inputs = _randn_inputs(device, total, heads, seed)
        cu = _cu_seqlens(num_seqs, seq_len, device) if num_seqs > 1 else None
        op = operators.setdefault(heads, KDAChunkPrefill(heads))
        print("Launching tilelang", flush=True)
        attn, final = _launch(op, inputs, cu)
        torch.npu.synchronize()
        print("Finished tilelang", flush=True)
        case = {
            "T": total,
            "N": num_seqs,
            "H": heads,
            "seq": seq_len,
            "seed": None if args.mode == "bench" else seed,
            "correctness": {},
        }
        if args.mode != "bench":
            expected_o, expected_final = _reference(inputs, cu)
            case["correctness"]["o"] = _check(attn, expected_o, "bf16", "o")
            case["correctness"]["final"] = _check(final, expected_final, "fp32", "final")
            if not all(item["pass"] for item in case["correctness"].values()):
                failed = True
        else:
            # Compile and the first launch stay outside the timed interval.
            samples = []
            for round_idx in range(args.rounds):
                elapsed = _bench(functools.partial(_launch, op, inputs, cu), args.warmup, args.repeat)
                samples.append(elapsed)
                print(f"end_to_end round {round_idx}: {elapsed:.3f} ms", flush=True)
            prepare, state, output = build(heads, num_seqs, True)
            q, k, v, raw_g, beta, a_log, dt_bias = inputs
            if cu is None:
                cu_kernel = torch.tensor([0, total], dtype=torch.int64, device=device)
            else:
                cu_kernel = cu
            h0 = torch.zeros((num_seqs, heads, DIM_K, DIM_V), dtype=torch.float32, device=device)
            prepare_ms = _bench(
                functools.partial(prepare, q, k, v, raw_g, beta, cu_kernel, a_log, dt_bias),
                args.warmup,
                args.repeat,
            )
            w, u, qg, kg, aqk, dec = prepare(q, k, v, raw_g, beta, cu_kernel, a_log, dt_bias)
            torch.npu.synchronize()
            state_ms = _bench(functools.partial(state, w, u, kg, dec, h0, cu_kernel), args.warmup, args.repeat)
            h, vn, _final = state(w, u, kg, dec, h0, cu_kernel)
            torch.npu.synchronize()
            output_ms = _bench(functools.partial(output, qg, aqk, h, vn, cu_kernel), args.warmup, args.repeat)
            case["performance"] = {
                "round_ms": samples,
                "end_to_end_ms": statistics.median(samples),
                "prepare_ms": prepare_ms,
                "state_ms": state_ms,
                "output_ms": output_ms,
                "warmup": args.warmup,
                "repeat": args.repeat,
            }
            print(
                f"stages_ms prepare:{prepare_ms:.3f} state:{state_ms:.3f} output:{output_ms:.3f}",
                flush=True,
            )
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2))
        torch.npu.empty_cache()
        if failed:
            raise AssertionError("Correctness failure")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
