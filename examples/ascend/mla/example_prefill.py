"""Run and check sparse MLA prefill with shared BF16 KV on Ascend 950."""

import argparse
import math

import torch

if __package__:
    from ._example_inputs import add_sink_options, load_sink, make_sink, make_topk_length
    from .prefill import flash_mla_sparse_fwd
else:
    from _example_inputs import add_sink_options, load_sink, make_sink, make_topk_length
    from prefill import flash_mla_sparse_fwd

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 64, 1, 512
INDEX_TOPK = 512
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)
OUT_RTOL, OUT_ATOL = 2e-2, 2e-2
STAT_ATOL = 3e-2


def make_prefill_inputs(seq_len=32, kv_len=4096, index_width=INDEX_TOPK, *, seed=88888888):
    """Generate ``(q, kv, indices, topk_length)`` on the CPU."""
    if min(seq_len, kv_len) <= 0:
        raise ValueError("Sq and Skv must be positive")
    if index_width <= 0 or index_width % 64:
        raise ValueError("index width T must be a positive multiple of 64")
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, generator=generator, dtype=torch.bfloat16)
    kv = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, generator=generator, dtype=torch.bfloat16)
    indices = torch.randint(0, kv_len, (seq_len, 1, index_width), generator=generator, dtype=torch.int32)
    indices[:, 0, ::32] = -1  # exercise the invalid-index mask
    lengths = torch.full((seq_len,), index_width, dtype=torch.int32)
    return q, kv, indices, lengths


def reference_prefill(q, kv, indices, *, topk_length=None, attn_sink=None, scale=SM_SCALE):
    """Per-query float32 reference for the contiguous-BF16 prefill contract.

    Negative and out-of-range indices are masked, and so are positions beyond
    ``topk_length``. ``max_logits`` and ``lse`` exclude the sink; an empty row
    returns zeros, ``-inf`` max logits and an infinite LSE.
    """
    seq_len, heads, dim = q.shape
    tokens = kv.shape[0]
    keys_all = kv.reshape(tokens, dim).float()
    out = torch.zeros((seq_len, heads, dim), dtype=torch.float32)
    max_logits = torch.full((seq_len, heads), -torch.inf, dtype=torch.float32)
    lse = torch.full((seq_len, heads), torch.inf, dtype=torch.float32)
    sink = None if attn_sink is None else attn_sink.float()
    for row in range(seq_len):
        length = indices.shape[-1] if topk_length is None else int(topk_length[row])
        index = indices[row, 0, :length]
        visible = (index >= 0) & (index < tokens)
        if not bool(visible.any()):
            continue
        keys = keys_all[index[visible].long()]
        scores = q[row].float() @ keys.T * scale
        row_max = scores.amax(dim=-1)
        probs = torch.exp(scores - row_max[:, None])
        total = probs.sum(dim=-1)
        denominator = total if sink is None else total + torch.exp(sink - row_max)
        out[row] = (probs.to(torch.bfloat16).float() @ keys) / denominator[:, None]
        max_logits[row] = row_max
        lse[row] = row_max + torch.log(total)
    return out.to(torch.bfloat16), max_logits, lse


def _check_prefill(out, max_logits, lse, expected_out, expected_max, expected_lse, seq_len):
    assert out.dtype == torch.bfloat16 and out.shape == (seq_len, NUM_HEADS, HEAD_DIM), out.shape
    assert max_logits.dtype == torch.float32 and max_logits.shape == (seq_len, NUM_HEADS)
    assert lse.dtype == torch.float32 and lse.shape == (seq_len, NUM_HEADS)
    torch.testing.assert_close(out.float().cpu(), expected_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL)
    torch.testing.assert_close(max_logits.cpu(), expected_max, rtol=0, atol=STAT_ATOL)
    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=STAT_ATOL)


def _npu_device():
    import torch_npu  # noqa: F401 - register the NPU tensor backend

    return torch.device("npu", torch.npu.current_device())


def run_prefill(seq_len=32, kv_len=4096, index_width=INDEX_TOPK, *, sink=False, attn_sink=None,
                topk_length=None, seed=88888888, bench=True):
    """Check the prefill kernel against the reference, then profile it."""
    q, kv, indices, _ = make_prefill_inputs(seq_len, kv_len, index_width, seed=seed)
    topk_length = make_topk_length(topk_length, seq_len, index_width)
    attn_sink = make_sink(seed, enabled=sink, weights=attn_sink)
    expected_out, expected_max, expected_lse = reference_prefill(
        q, kv, indices, topk_length=topk_length, attn_sink=attn_sink)
    device = _npu_device()
    q, kv, indices = (tensor.to(device) for tensor in (q, kv, indices))
    if topk_length is not None:
        topk_length = topk_length.to(device)
    if attn_sink is not None:
        attn_sink = attn_sink.to(device)
    print(f"Prefill: Sq={seq_len}, Skv={kv_len}, T={index_width}, sink={attn_sink is not None}", flush=True)
    out, max_logits, lse = flash_mla_sparse_fwd(q, kv, indices, SM_SCALE, attn_sink=attn_sink,
                                                topk_length=topk_length)
    torch.npu.synchronize()
    _check_prefill(out, max_logits, lse, expected_out, expected_max, expected_lse, seq_len)
    print("Accuracy PASS", flush=True)
    if bench:
        from tilelang.profiler import do_bench

        call = lambda: flash_mla_sparse_fwd(  # noqa: E731
            q, kv, indices, SM_SCALE, attn_sink=attn_sink, topk_length=topk_length,
            validate_values=False)
        profile = do_bench(call, backend="msprof_detail", _n_warmup=30, _n_repeat=100)
        flops = 4.0 * seq_len * NUM_HEADS * index_width * HEAD_DIM
        print(f"{profile.dur_us:.2f} us/iter | {profile.tflops(flops):.1f} TFLOPS", flush=True)
    return out, max_logits, lse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=32, help="Prefill query length Sq")
    parser.add_argument("--kv-len", type=int, default=4096, help="Prefill KV length Skv")
    parser.add_argument("--index-width", type=int, default=INDEX_TOPK, help="Index width T, a positive multiple of 64")
    parser.add_argument("--topk-length", nargs="+", type=int, metavar="N", help="One prefix length or one per query")
    add_sink_options(parser)
    parser.add_argument("--seed", type=int, default=88888888)
    parser.add_argument("--no-bench", action="store_true", help="Skip profiling")
    args = parser.parse_args()
    run_prefill(args.seq_len, args.kv_len, args.index_width, sink=args.sink, attn_sink=load_sink(args.sink_file),
                topk_length=args.topk_length, seed=args.seed, bench=not args.no_bench)


if __name__ == "__main__":
    main()
