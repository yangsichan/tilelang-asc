# Sparse MLA on Ascend 950

Sparse MLA for DeepSeek-V4.1-Flash with one shared KV head. The specification
uses 64 query heads, `head_dim = d_v = 512`, `sm_scale = 1/sqrt(512)`, and the
model indexer cap `index_topk = 512`. The index width `T` is a separate dynamic
value; it is a positive multiple of 64 and need not equal `index_topk`.

| Stage | Implementation | Example | Tests |
| --- | --- | --- | --- |
| Prefill | `prefill.py` | `example_prefill.py` | `test_prefill.py` |
| Decode | `decode.py` | `example_decode.py` | `test_decode.py` |

## Prefill contract

`flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None)`

| Tensor | Shape | dtype |
| --- | --- | --- |
| Q | `[Sq, 64, 512]` | BF16 |
| KV | `[Skv, 1, 512]` | BF16 |
| indices | `[Sq, 1, T]` | int32 |
| attn_sink | `[64]`, optional | FP32 |
| topk_length | `[Sq]`, optional | int32 |
| O | `[Sq, 64, 512]` | BF16 |
| max_logits | `[Sq, 64]` | FP32 |
| lse | `[Sq, 64]` | FP32 |

`Sq` and `Skv` are dynamic. `T` is a positive multiple of 64. Negative and
out-of-range indices are masked, and so are positions at or beyond
`topk_length`. A query with no valid position returns `O = 0`, `max_logits =
-inf` and `lse = +inf`. `max_logits` and `lse` exclude the sink; the sink only
adds to the output denominator. When the model has a sink, pass its weights
instead of relying on the default `-inf`.

## Decode contract

`flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v,
tile_scheduler_metadata, indices=..., attn_sink=..., topk_length=...)`

| Tensor | Shape | dtype |
| --- | --- | --- |
| Q | `[B, 1, 64, 512]` | BF16 |
| k_cache | `[Nc, P, 1, 528]` | float8_e4m3fn / int8 / uint8 |
| indices | `[B, 1, T]` | int32 |
| attn_sink | `[64]`, optional | FP32 |
| topk_length | `[B]`, optional | int32 |
| O | `[B, 1, 64, 512]` | BF16 |
| lse | `[B, 64, 1]` | FP32 |

`B`, `Nc` and `T` are dynamic; the current Ascend kernel keeps the query length
fixed at 1. `P > 1` is a cache layout choice. `causal` stays `False`,
`is_fp8_kvcache` stays `True`, and `num_splits` stays `None`. `indices` are
physical positions in `[0, Nc*P)`; `-1` marks an invalid slot. `block_table` and
`cache_seqlens` are accepted for signature compatibility and ignored.
`tile_scheduler_metadata` must come from `get_mla_metadata()`.

Two optional inputs extend the main cache: `extra_k_cache` with shape
`[Ne, Pe, 1, Ce]` and `extra_indices_in_kvcache` with shape `[B, 1, Te]`, plus
an optional `extra_topk_length [B]`. Both caches contribute selected positions
to the same softmax.

## Packed KV page layout

Each main-cache token is 528 bytes:

1. 512 E4M3 code bytes, in column order.
2. 16 E8M0 scale bytes, one per 32-column group.

The Ascend kernel decodes scale byte `0` as zero and other scale bytes `b`
as `2 ** (min(b, 224) - 127)`. An extra cache may use the same 528-byte layout
or the 288-byte FP4 layout with 256 packed E2M1 bytes followed by 32 E4M3
scale bytes, one per 16-column group. The even-indexed E2M1 value occupies
the low nibble. `example_decode.py` can generate either extra cache format.

## Run

Run the examples and tests from the workspace root with a compatible
TileLang/CANN environment and an idle Ascend 950 device.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0  # Replace 0 with an idle physical NPU.

python -m mla_test.mla.example_prefill --seq-len 32 --kv-len 4096 --index-width 512
python -m mla_test.mla.example_decode --batch-size 1 --tokens 512 --page-size 2 --index-width 128

python -m mla_test.mla.example_decode --batch-size 2 --tokens 36 --page-size 3 --index-width 128 \
    --extra-format fp4 --extra-tokens 6 --extra-page-size 2 --extra-index-width 67 \
    --topk-length 61 128 --extra-topk-length 67 0 --no-bench

python -m pytest -v -s mla_test/mla/test_prefill.py
python -m pytest -v -s mla_test/mla/test_decode.py
```

Pass `--extra-format fp8` to generate a 528-byte extra cache. `--cache-dtype`
selects the byte container used for both caches. The example enables the extra
cache when `--extra-index-width` is positive, including widths unaligned to 64.
Each length option accepts one value for all rows or one value per row.
Omitting a length option makes the full index width its valid prefix.

Use `--sink-file PATH` to pass the model's actual FP32 `[64]` sink tensor
saved with `torch.save`. `--sink` generates synthetic weights. Both examples
accept `--no-bench` for accuracy checks alone. The profiling call uses
`validate_values=False` after checking accuracy. `msprof_detail` reports
device kernel timings. The nominal QK and PV FLOP estimate uses the main
index width plus the extra index width when an extra cache is present.

## Notes

- Both entry points compile internally and cache by launch configuration. Each
  shape is specialised, so unaligned `Sq`, `Skv`, `B` or `T` do not reuse one
  precompiled binary.
- Decode selects the pipelined gather schedule from the launch geometry, the
  static/dynamic length mode and the Cube core count. Host validation checks
  every index in the consumed prefix; `validate_values=False` turns that into a
  caller precondition while keeping in-kernel bounds guards.
- Prefill derives its scheduling metadata from `topk_length`; without it the
  whole index width is a valid prefix. Explicit lengths must be in `[0, T]`.
