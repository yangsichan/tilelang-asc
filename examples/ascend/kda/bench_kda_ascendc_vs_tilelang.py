"""Compare TileLang and AscendC KDA prefill on 15 packed shapes.

``tilelang_perf_pct`` is ``100 * ascend_ms / tilelang_ms``, the reciprocal of
the time ratio. Above 100 means TileLang is faster. Both times are the sum of
NPU kernel durations from ``do_bench(..., backend="msprof")``, the same
total used by ``test_chunk_prefill_ascend_perf.py --stages``.

By default every sequence length is ``T / N`` and a multiple of 64, so
TileLang takes its aligned kernel. ``--varlen`` keeps the same 15 shapes but
gives each sequence a deterministic length in ``[75%, 125%]`` of ``T / N``.
Those lengths are not all multiples of 64, so TileLang takes the varlen
kernel. Both sides always receive the packed ``[1, T, H, K]`` tensors and the
same ``cu_seqlens``. ``T=128/256/512`` cannot satisfy both ``N >= 16`` and a
64-aligned sequence, so the sweep starts at 1024.

TileLang is ``kda_prefill.KDAChunkPrefill``. A ``--ops`` wheel is
``cann_ops_transformer_custom``:

    from cann_ops_transformer_custom.ops import chunk_kda_fwd

Run on an A5 NPU, from this directory:

    conda run -n tilelang python -u bench_kda_ascendc_vs_tilelang.py
    conda run -n tilelang python -u bench_kda_ascendc_vs_tilelang.py --varlen
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path[:0] = [str(_REPO_ROOT), str(_HERE)]

import torch  # noqa: E402

try:
    import torch_npu  # noqa: F401,E402
except ImportError:
    torch_npu = None

from kda_prefill import ATTN_SCALE, BLOCK_T, DIM_K, DT_LOWER_BOUND, KDAChunkPrefill  # noqa: E402
from run_prefill import _cu_seqlens_from_lengths, _varlen_lengths  # noqa: E402

# (total_tokens, num_seqs, heads). Sequence length is total_tokens / num_seqs.
# N grows with T. Heads were drawn once from {12, 24, 48, 96} with seed 20260808.
_CASES: tuple[tuple[int, int, int], ...] = (
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
_CSV_PATH = Path(__file__).with_name("bench_kda_ascendc_vs_tilelang.csv")
_VARLEN_CSV_PATH = Path(__file__).with_name("bench_kda_ascendc_vs_tilelang_varlen.csv")
_HEADER_NOTES = (
    "tilelang_perf_pct = 100 * ascend_ms / tilelang_ms; "
    "above 100 means TileLang is faster.",
    "Times are summed NPU kernel durations from do_bench(backend='msprof').",
    "Each sequence length is T/N and a multiple of 64, so TileLang uses the aligned kernel.",
    "Both sides receive packed [1, T, H, K] tensors and the same cu_seqlens.",
)
_VARLEN_NOTES = (
    "tilelang_perf_pct = 100 * ascend_ms / tilelang_ms; "
    "above 100 means TileLang is faster.",
    "Times are summed NPU kernel durations from do_bench(backend='msprof').",
    "Each sequence length is in [75%, 125%] of T/N and not all are multiples of 64, "
    "so TileLang uses the varlen kernel.",
    "Both sides receive packed [1, T, H, K] tensors and the same cu_seqlens.",
)


def cases() -> list[tuple[int, int, int, int]]:
    """Return ``(total_tokens, num_seqs, heads, seq_len)`` for the 15 shapes."""

    selected = []
    for total, num_seqs, heads in _CASES:
        if total % num_seqs:
            raise ValueError(f"T={total} is not divisible by N={num_seqs}")
        seq_len = total // num_seqs
        if seq_len % BLOCK_T:
            raise ValueError(f"sequence length {seq_len} is not a multiple of {BLOCK_T}")
        selected.append((total, num_seqs, heads, seq_len))
    if len(selected) != 15:
        raise ValueError(f"expected 15 cases, got {len(selected)}")
    return selected


def _load_chunk_kda_fwd():
    """A ``--ops`` wheel is ``cann_ops_transformer_custom``. The full package is the fallback."""

    for name in ("cann_ops_transformer_custom", "cann_ops_transformer"):
        try:
            module = __import__(f"{name}.ops", fromlist=["chunk_kda_fwd"])
        except ModuleNotFoundError:
            continue
        fn = getattr(module, "chunk_kda_fwd", None)
        if fn is not None:
            return fn
    raise ImportError("chunk_kda_fwd was not found in cann_ops_transformer or cann_ops_transformer_custom")


def _make_inputs(device: torch.device, total: int, heads: int):
    shape = (1, total, heads, DIM_K)
    level = float(ATTN_SCALE)
    q = torch.full(shape, level, dtype=torch.bfloat16, device=device)
    k = torch.full_like(q, level)
    v = torch.full_like(q, 0.01)
    raw_g = torch.full(shape, -1.0, dtype=torch.float32, device=device)
    beta = torch.full(shape[:-1], 0.5, dtype=torch.bfloat16, device=device)
    a_log = torch.zeros((heads,), dtype=torch.float32, device=device)
    dt_bias = torch.zeros((heads * DIM_K,), dtype=torch.float32, device=device)
    return q, k, v, raw_g, beta, a_log, dt_bias


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


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "oom" in text


def _empty_cache() -> None:
    if torch_npu is not None and hasattr(torch, "npu"):
        torch.npu.empty_cache()


def _time_one(fn, warmup: int, repeat: int) -> tuple[float | None, str]:
    try:
        return _bench(fn, warmup, repeat), ""
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        _empty_cache()
        return None, "OOM"


def _perf_pct(tilelang_ms: float | None, ascend_ms: float | None) -> float | None:
    """TileLang performance as a percentage of AscendC. Higher is faster."""

    if tilelang_ms is None or ascend_ms is None or tilelang_ms == 0:
        return None
    return 100.0 * ascend_ms / tilelang_ms


def _format_ms(value: float | None) -> str:
    return "OOM" if value is None else f"{value:.3f}"


def _format_pct(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}"


def _print_table(rows: list[dict[str, object]], notes: tuple[str, ...], varlen: bool) -> None:
    for note in notes:
        print(note, flush=True)
    seq_width = 11 if varlen else 6
    print(
        f"{'T':>7} {'N':>5} {'H':>4} {'seq':>{seq_width}} "
        f"{'tilelang_ms':>12} {'ascend_ms':>12} {'tilelang/ascend %':>18}",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['T']:>7} {row['N']:>5} {row['H']:>4} {row['seq']:>{seq_width}} "
            f"{_format_ms(row['tilelang_ms']):>12} {_format_ms(row['ascend_ms']):>12} "
            f"{_format_pct(row['tilelang_perf_pct']):>18}",
            flush=True,
        )


def _write_csv(rows: list[dict[str, object]], path: Path, notes: tuple[str, ...], varlen: bool) -> None:
    fieldnames = [
        "T",
        "N",
        "H",
        "seq",
        "tilelang_ms",
        "ascend_ms",
        "tilelang_perf_pct",
        "note",
    ]
    if varlen:
        fieldnames.insert(4, "seqlens")
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("# " + " ".join(notes) + "\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {
                "T": row["T"],
                "N": row["N"],
                "H": row["H"],
                "seq": row["seq"],
                "tilelang_ms": "" if row["tilelang_ms"] is None else f"{row['tilelang_ms']:.6f}",
                "ascend_ms": "" if row["ascend_ms"] is None else f"{row['ascend_ms']:.6f}",
                "tilelang_perf_pct": _format_pct(row["tilelang_perf_pct"]),
                "note": row["note"],
            }
            if varlen:
                record["seqlens"] = row["seqlens"]
            writer.writerow(record)


def run(device_index: int, warmup: int, repeat: int, varlen: bool) -> None:
    chunk_kda_fwd = _load_chunk_kda_fwd()

    torch.npu.set_device(device_index)
    device = torch.device("npu", device_index)
    operators: dict[int, KDAChunkPrefill] = {}
    notes = _VARLEN_NOTES if varlen else _HEADER_NOTES
    csv_path = _VARLEN_CSV_PATH if varlen else _CSV_PATH
    rows: list[dict[str, object]] = []
    for note in notes:
        print(note, flush=True)
    for total, num_seqs, heads, seq_len in cases():
        lengths = _varlen_lengths(total, num_seqs) if varlen else [seq_len] * num_seqs
        seq_label = f"{min(lengths)}-{max(lengths)}" if varlen else str(seq_len)
        print(
            f"case T={total} N={num_seqs} H={heads} seq={seq_label}",
            flush=True,
        )
        note = ""
        tilelang_ms = None
        ascend_ms = None
        try:
            q, k, v, raw_g, beta, a_log, dt_bias = _make_inputs(device, total, heads)
            cu, cu_host = _cu_seqlens_from_lengths(lengths, device)
            op = operators.setdefault(heads, KDAChunkPrefill(heads))

            def tilelang_fn(
                q=q, k=k, v=v, raw_g=raw_g, beta=beta, a_log=a_log, dt_bias=dt_bias, cu=cu, op=op,
            ):
                return op(q, k, v, raw_g, beta, a_log, dt_bias, cu_seqlens=cu)

            def ascend_fn(
                q=q, k=k, v=v, raw_g=raw_g, beta=beta, a_log=a_log, dt_bias=dt_bias,
                cu=cu, cu_host=cu_host,
            ):
                return chunk_kda_fwd(
                    q,
                    k,
                    v,
                    raw_g,
                    beta,
                    float(ATTN_SCALE),
                    A_log=a_log,
                    dt_bias=dt_bias,
                    cu_seqlens=cu,
                    cu_seqlens_cpu=cu_host,
                    layout="BSND",
                    chunk_size=BLOCK_T,
                    safe_gate=True,
                    lower_bound=DT_LOWER_BOUND,
                    use_gate_in_kernel=True,
                    output_final_state=True,
                )

            tilelang_ms, tilelang_note = _time_one(tilelang_fn, warmup, repeat)
            ascend_ms, ascend_note = _time_one(ascend_fn, warmup, repeat)
            note = " ".join(
                part for part in (
                    f"tilelang {tilelang_note}" if tilelang_note else "",
                    f"ascend {ascend_note}" if ascend_note else "",
                )
                if part
            )
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            note = "OOM"
            _empty_cache()
        row = {
            "T": total,
            "N": num_seqs,
            "H": heads,
            "seq": seq_label,
            "seqlens": " ".join(str(length) for length in lengths),
            "tilelang_ms": tilelang_ms,
            "ascend_ms": ascend_ms,
            "tilelang_perf_pct": _perf_pct(tilelang_ms, ascend_ms),
            "note": note,
        }
        rows.append(row)
        _write_csv(rows, csv_path, notes, varlen)
        print(
            f"  tilelang_ms={_format_ms(tilelang_ms)} "
            f"ascend_ms={_format_ms(ascend_ms)} "
            f"tilelang/ascend %={_format_pct(row['tilelang_perf_pct']) or note}",
            flush=True,
        )
        _empty_cache()
    _print_table(rows, notes, varlen)
    print(f"wrote {csv_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--varlen",
        action="store_true",
        help=(
            "Use the same 15 shapes, but give each sequence a length between "
            "75 and 125 percent of T/N so TileLang runs the varlen kernel."
        ),
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be >= 0 and --repeat must be >= 1")
    if torch_npu is None or not torch.npu.is_available():
        raise SystemExit("This comparison requires torch_npu and an NPU.")
    run(args.device, args.warmup, args.repeat, args.varlen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
