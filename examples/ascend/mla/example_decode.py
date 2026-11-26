"""Check Ascend sparse MLA decode with FP8 main and optional FP8 or FP4 extra KV."""

import argparse
import math

import torch

if __package__:
    from ._example_inputs import add_sink_options, load_sink, make_sink, make_topk_length
    from .decode import flash_mla_with_kvcache, get_mla_metadata
else:
    from _example_inputs import add_sink_options, load_sink, make_sink, make_topk_length
    from decode import flash_mla_with_kvcache, get_mla_metadata

NUM_HEADS, HEAD_DIM = 64, 512
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)
FP8_MAX = 448.0
SCALE_BIAS = 127
MAX_SCALE_CODE = 224
FP8_RECORD_BYTES = HEAD_DIM + HEAD_DIM // 32
FP4_RECORD_BYTES = HEAD_DIM // 2 + HEAD_DIM // 16
CACHE_DTYPES = (torch.uint8, torch.int8, torch.float8_e4m3fn)
OUT_RTOL, OUT_ATOL = 2e-2, 2e-2
STAT_ATOL = 3e-2


def pack_fp8_records(kv):
    """Quantize BF16 rows ``[N, 512]`` into 528-byte E4M3/E8M0 records."""
    rows = kv.shape[0]
    values = kv.float().view(rows, HEAD_DIM // 32, 32)
    amax = values.abs().amax(dim=-1)
    # The kernel decodes each scale byte as 2 ** (byte - 127) and clamps the
    # byte at 224, so the producer stores the same biased, clamped exponent.
    exponent = torch.ceil(torch.log2(amax.clamp_min(torch.finfo(torch.float32).tiny) / FP8_MAX))
    code = (exponent + SCALE_BIAS).to(torch.int32).clamp(1, MAX_SCALE_CODE)
    scale = torch.pow(2.0, (code - SCALE_BIAS).to(torch.float32))
    data = (values / scale[..., None]).to(torch.float8_e4m3fn)
    records = torch.zeros((rows, FP8_RECORD_BYTES), dtype=torch.uint8)
    records[:, :HEAD_DIM] = data.view(torch.uint8).view(rows, HEAD_DIM)
    records[:, HEAD_DIM:] = code.to(torch.uint8)
    return records


def pack_fp8_pages(kv, page_size):
    """Pad records to whole pages and reshape them to ``[Nc, P, 1, 528]``."""
    rows = kv.shape[0]
    pages = (rows + page_size - 1) // page_size
    records = pack_fp8_records(kv)
    padded = torch.zeros((pages * page_size, FP8_RECORD_BYTES), dtype=torch.uint8)
    padded[:rows] = records
    padded[rows:, HEAD_DIM:] = SCALE_BIAS  # empty token: zero codes, unit scale
    return padded.view(pages, page_size, 1, FP8_RECORD_BYTES)


def dequantize_records(records):
    """Read packed FP8 or FP4 records and return BF16 KV rows."""
    width = records.shape[-1]
    if records.dtype not in CACHE_DTYPES or width not in (FP8_RECORD_BYTES, FP4_RECORD_BYTES):
        raise ValueError("Expected a byte container with 528 or 288 bytes per record")
    flat = records.contiguous().view(torch.uint8).reshape(-1, width)
    if width == FP8_RECORD_BYTES:
        values = flat[:, :HEAD_DIM].contiguous().view(torch.float8_e4m3fn).float()
        scale_codes = flat[:, HEAD_DIM:].to(torch.int16).clamp_max(MAX_SCALE_CODE)
        scales = torch.where(scale_codes == 0, 0.0, torch.exp2((scale_codes - SCALE_BIAS).float()))
        group_size = 32
    else:
        packed = flat[:, : HEAD_DIM // 2]
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(-1, HEAD_DIM)
        magnitude = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=flat.device)
        values = magnitude[(codes & 7).long()]
        values = torch.where((codes & 8) != 0, -values, values)
        scales = flat[:, HEAD_DIM // 2 :].contiguous().view(torch.float8_e4m3fn).float()
        group_size = 16
    values = values.reshape(-1, HEAD_DIM // group_size, group_size)
    return (values * scales[..., None]).reshape(-1, HEAD_DIM).bfloat16()


def make_decode_inputs(batch=1, index_width=128, *, tokens=512, page_size=2, seed=88888888):
    """Generate ``(q, packed_cache, indices, topk_length)`` on the CPU."""
    if min(batch, tokens) <= 0:
        raise ValueError("B and the main cache token count must be positive")
    if index_width <= 0 or index_width % 64:
        raise ValueError("index width T must be a positive multiple of 64")
    if page_size <= 1:
        raise ValueError("decode page size P must be greater than 1")
    generator = torch.Generator().manual_seed(seed)
    pages = (tokens + page_size - 1) // page_size
    tokens = pages * page_size
    kv = torch.randn(tokens, HEAD_DIM, generator=generator, dtype=torch.bfloat16)
    cache = pack_fp8_pages(kv, page_size)
    q = torch.randn(batch, 1, NUM_HEADS, HEAD_DIM, generator=generator, dtype=torch.bfloat16)
    indices = torch.randint(0, tokens, (batch, 1, index_width), generator=generator, dtype=torch.int32)
    indices[:, :, ::16] = -1  # exercise the invalid-index mask
    lengths = torch.full((batch,), index_width, dtype=torch.int32)
    return q, cache, indices, lengths


def make_extra_inputs(batch, index_width, *, tokens=128, page_size=2, cache_format="fp8", seed=88888888):
    """Generate an extra packed KV cache and its physical indices on the CPU."""
    if min(batch, index_width, tokens, page_size) <= 0:
        raise ValueError("B, Te, extra token count and Pe must be positive")
    generator = torch.Generator().manual_seed(seed)
    if cache_format == "fp8":
        kv = torch.randn(tokens, HEAD_DIM, generator=generator, dtype=torch.bfloat16)
        cache = pack_fp8_pages(kv, page_size)
    elif cache_format == "fp4":
        pages = (tokens + page_size - 1) // page_size
        records = torch.zeros((pages * page_size, FP4_RECORD_BYTES), dtype=torch.uint8)
        records[:tokens, : HEAD_DIM // 2] = torch.randint(0, 256, (tokens, HEAD_DIM // 2), generator=generator, dtype=torch.uint8)
        # Positive finite E4M3 scale codes in [0.25, 0.5].
        records[:tokens, HEAD_DIM // 2 :] = torch.randint(0x28, 0x31, (tokens, HEAD_DIM // 16), generator=generator, dtype=torch.uint8)
        cache = records.view(pages, page_size, 1, FP4_RECORD_BYTES)
    else:
        raise ValueError("Extra cache format must be fp8 or fp4")
    indices = torch.randint(0, tokens, (batch, 1, index_width), generator=generator, dtype=torch.int32)
    indices[..., 15::16] = -1
    return cache, indices


def _selected_rows(q, sources):
    """Yield the selected rows for each query, per the index and length rules.

    A position is invalid when its index is negative, out of range for its
    cache or beyond that source's length prefix. Each source has its own index
    width, so main and extra caches can use different ``T``.
    """
    for row in range(q.shape[0]):
        selected = []
        for kv, idx, lengths in sources:
            index = idx[row, 0]
            valid = (index >= 0) & (index < kv.shape[0])
            if lengths is not None:
                positions = torch.arange(idx.shape[-1], device=q.device)
                valid = valid & (positions < lengths[row])
            if bool(valid.any()):
                selected.append(kv[index[valid].long()])
        yield selected


def reference_decode(
    q, indices, records, *, topk_length=None, attn_sink=None, extra_indices=None, extra_records=None, extra_topk_length=None, scale=SM_SCALE
):
    """Per-query float32 reference for FP8 main and optional FP8 or FP4 extra records.

    Return O[B, 1, 64, 512] and LSE[B, 64, 1]. The sink contributes to the
    output denominator. A query with no valid position returns zeros and +inf LSE.
    """
    sources = [(dequantize_records(records).float(), indices, topk_length)]
    if extra_records is not None:
        sources.append((dequantize_records(extra_records).float(), extra_indices, extra_topk_length))
    batch = q.shape[0]
    out = torch.zeros((batch, 1, NUM_HEADS, HEAD_DIM), dtype=torch.float32)
    lse = torch.full((batch, NUM_HEADS, 1), torch.inf, dtype=torch.float32)
    sink = None if attn_sink is None else attn_sink.float()
    for row, selected in enumerate(_selected_rows(q, sources)):
        if not selected:
            continue
        keys = torch.cat(selected, dim=0)
        scores = q[row, 0].float() @ keys.T * scale
        row_max = scores.amax(dim=-1)
        probs = torch.exp(scores - row_max[:, None])
        total = probs.sum(dim=-1)
        # The kernel hands BF16 probabilities to PV and divides by an FP32 sum.
        denominator = total if sink is None else total + torch.exp(sink - row_max)
        out[row, 0] = (probs.to(torch.bfloat16).float() @ keys) / denominator[:, None]
        lse[row, :, 0] = row_max + torch.log(total)
    return out.to(torch.bfloat16), lse


def _check_decode(out, lse, expected_out, expected_lse, batch):
    assert out.dtype == torch.bfloat16
    assert out.shape == (batch, 1, NUM_HEADS, HEAD_DIM), out.shape
    assert lse.dtype == torch.float32 and lse.shape == (batch, NUM_HEADS, 1), lse.shape
    torch.testing.assert_close(out.float().cpu(), expected_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL)
    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=STAT_ATOL)


def _npu_device():
    import torch_npu  # noqa: F401 - register the NPU tensor backend

    return torch.device("npu", torch.npu.current_device())


def run_decode(
    batch=1,
    index_width=128,
    *,
    tokens=512,
    page_size=2,
    sink=False,
    attn_sink=None,
    topk_length=None,
    cache_dtype=torch.uint8,
    extra_index_width=0,
    extra_tokens=128,
    extra_page_size=2,
    extra_format="fp8",
    extra_topk_length=None,
    seed=88888888,
    bench=True,
):
    """Check the decode kernel against the reference, then profile it."""
    if cache_dtype not in CACHE_DTYPES:
        raise ValueError("k_cache must use a uint8, int8, or float8_e4m3fn byte container")
    if extra_index_width < 0:
        raise ValueError("Extra index width (Te) must be nonnegative")
    if extra_index_width == 0 and extra_topk_length is not None:
        raise ValueError("extra_topk_length requires a positive extra index width (Te)")
    q, cache, indices, _ = make_decode_inputs(batch, index_width, tokens=tokens, page_size=page_size, seed=seed)
    topk_length = make_topk_length(topk_length, batch, index_width)
    attn_sink = make_sink(seed, enabled=sink, weights=attn_sink)
    extra_cache, extra_indices = None, None
    if extra_index_width > 0:
        extra_cache, extra_indices = make_extra_inputs(
            batch, extra_index_width, tokens=extra_tokens, page_size=extra_page_size, cache_format=extra_format, seed=seed + 2
        )
    extra_topk_length = make_topk_length(extra_topk_length, batch, extra_index_width, "extra_topk_length")
    expected_out, expected_lse = reference_decode(
        q,
        indices,
        cache,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_records=extra_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
    )
    device = _npu_device()
    q, cache, indices = (tensor.to(device) for tensor in (q, cache, indices))
    cache = cache.view(cache_dtype)  # Transfer the packed bytes before reinterpreting them.
    if topk_length is not None:
        topk_length = topk_length.to(device)
    if attn_sink is not None:
        attn_sink = attn_sink.to(device)
    if extra_cache is not None:
        extra_cache = extra_cache.to(device).view(cache_dtype)
        extra_indices = extra_indices.to(device)
    if extra_topk_length is not None:
        extra_topk_length = extra_topk_length.to(device)
    metadata, _ = get_mla_metadata()
    print(f"Decode: B={batch}, T={index_width}, Nc={cache.shape[0]}, P={page_size}, sink={attn_sink is not None}", flush=True)
    if extra_cache is not None:
        print(f"Extra cache: {extra_format}, Ne={extra_cache.shape[0]}, Pe={extra_page_size}, Te={extra_index_width}", flush=True)

    def call(*, validate_values=True):
        return flash_mla_with_kvcache(
            q,
            cache,
            None,
            None,
            HEAD_DIM,
            metadata,
            num_splits=None,
            softmax_scale=SM_SCALE,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
            attn_sink=attn_sink,
            topk_length=topk_length,
            extra_k_cache=extra_cache,
            extra_indices_in_kvcache=extra_indices,
            extra_topk_length=extra_topk_length,
            validate_values=validate_values,
        )

    out, lse = call()
    torch.npu.synchronize()
    _check_decode(out, lse, expected_out, expected_lse, batch)
    print("Accuracy PASS", flush=True)
    if bench:
        from tilelang.profiler import do_bench

        # validate_values=False keeps the host-side index copy out of the loop;
        # the accuracy call above already validated this input.
        profile = do_bench(lambda: call(validate_values=False), backend="msprof_detail", _n_warmup=30, _n_repeat=100)
        flops = 4.0 * batch * NUM_HEADS * (index_width + extra_index_width) * HEAD_DIM
        print(f"{profile.dur_us:.2f} us/iter | {profile.tflops(flops):.1f} TFLOPS", flush=True)
    return out, lse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1, help="Decode batch size B")
    parser.add_argument("--index-width", type=int, default=512, help="Index width T, a positive multiple of 64")
    parser.add_argument("--tokens", type=int, default=512, help="Decode cache token count before page padding")
    parser.add_argument("--page-size", type=int, default=2, help="Tokens per page P > 1")
    parser.add_argument(
        "--cache-dtype",
        choices=[str(dtype).removeprefix("torch.") for dtype in CACHE_DTYPES],
        default="uint8",
        help="Byte container for the main and extra cache",
    )
    parser.add_argument("--topk-length", nargs="+", type=int, metavar="N", help="One main prefix length or one per batch row")
    parser.add_argument("--extra-index-width", type=int, default=0, metavar="Te", help="Extra index width, 0 disables the extra cache")
    parser.add_argument("--extra-tokens", type=int, default=128, help="Extra cache token count before page padding")
    parser.add_argument("--extra-page-size", type=int, default=2, help="Tokens per extra page Pe")
    parser.add_argument("--extra-format", choices=("fp8", "fp4"), default="fp8", help="FP8 (528 bytes) or FP4 (288 bytes)")
    parser.add_argument("--extra-topk-length", nargs="+", type=int, metavar="N", help="One extra prefix length or one per batch row")
    add_sink_options(parser)
    parser.add_argument("--seed", type=int, default=88888888)
    parser.add_argument("--no-bench", action="store_true", help="Skip profiling")
    args = parser.parse_args()
    run_decode(
        args.batch_size,
        args.index_width,
        tokens=args.tokens,
        page_size=args.page_size,
        sink=args.sink,
        attn_sink=load_sink(args.sink_file),
        topk_length=args.topk_length,
        cache_dtype=getattr(torch, args.cache_dtype),
        extra_index_width=args.extra_index_width,
        extra_tokens=args.extra_tokens,
        extra_page_size=args.extra_page_size,
        extra_format=args.extra_format,
        extra_topk_length=args.extra_topk_length,
        seed=args.seed,
        bench=not args.no_bench,
    )


if __name__ == "__main__":
    main()
