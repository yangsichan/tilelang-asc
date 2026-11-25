# QSA Sparse Attention on Ascend 950

This example implements the fixed QSA grouped-query attention contract used by
Qwen3.8-Flash-Next. The TileLang kernel schedules packed query and KV rows across persistent cores. The Cube and Vector paths
use explicit pipe events and mode-4 flags between one AIC and its two AIVs.
The compiler emits the DMA, MAD and SIMD instructions with automatic scheduling
disabled.

## Model contract

| Parameter | Supported value |
| --- | --- |
| Query heads | 24 |
| KV heads | 2 |
| GQA group size | 12 |
| Head dimension | 256 |
| Input and output dtype | BF16 |
| Accumulation and softmax | FP32, scale `1 / sqrt(256)` |
| Logical selection width | 2051 |
| Physical index width | 2176 |
| Sparse unit | Four consecutive KV tokens, with a partial final unit allowed |
| Q rows per KV head | 16 physical rows, 12 logical rows plus four zero rows |
| Software pipeline | Three stages |

The logical width is the Qwen selection contract. The physical width is fixed
at 2176 slots so the explicit schedule can use 17 tiles of 128 slots. The host
stores invalid physical slots as `-1` and supplies the corresponding `Valid`
mask. K and V each receive three zero-padded physical rows at the end.

The runtime-varying dimensions are packed `num_tokens` and `kv_length`.
Static and dynamic builds use the same kernel contract. Each request in a
batch must be packed into the tensors together with indices that already carry
its KV offset. Per-request `cu_seqlens` metadata is outside this interface.

The implementation accepts positive packed query lengths and KV lengths of at
least four. Its explicit GM addressing uses 32-bit offsets, giving the current
implementation bounds of `num_tokens <= 262144` and `kv_length <= 4194301`.
The native model context limit is 262144 tokens per request. A packed batch can contain multiple requests.

`aic_count` selects the persistent block count and accepts 28, 32, or 36. The
command-line default is 32 for the standard community 950 configuration.
The model-facing dimensions, logical selection width, row padding, and stage
count are fixed in the kernel and are not runtime options.

## Build and run

Build TileLang with Ascend support and configure CANN before running. From the
repository root, select an idle NPU and set `PYTHONPATH`:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD"

python examples/ascend/qsa/run.py --check-only
python examples/ascend/qsa/run.py
python examples/ascend/qsa/run.py --dynamic --check-only
python examples/ascend/qsa/run.py \
  --num-tokens 4096 --kv-length 32768 --aic-count 32
```

The default run checks and benchmarks 16:2048, 128:65536, and 4096:32768.
Use `--check-only` to avoid the `msprof` benchmark path. `--dynamic` makes the
packed row dimensions symbolic for that invocation.

## Regression tests

These tests require an Ascend NPU and `torch_npu`:

```bash
PYTHONPATH="$PWD" python -m pytest examples/ascend/qsa/test_qsa.py -q
```

The regression suite covers static and dynamic Qwen prefill and decode shapes,
same-kernel dynamic shape reuse, empty and partial index tails, and the
two-tile handoff behavior.
