# QSA Sparse Attention on Ascend

Grouped-query sparse attention for Qwen 3.8 on Ascend 950, written in
TileLang. Each query attends to `top_k` whole 4-token KV blocks chosen by
an index tensor; 24 query heads share 2 KV heads with head_dim = 256.

| Files | Purpose |
| --- | --- |
| `kernel.py` | Persistent QSA forward with compiler-managed buffer versions |
| `compile.py` | Compilation entry point for static and dynamic-shape kernels |
| `testing.py` | Input generation, contract corners, and a grouped-query reference |
| `run.py` | Correctness checks and benchmarks |
| `test_qsa.py` | Kernel regression tests |

## Kernel contract

- `top_k` must be a positive multiple of 128.
- Indices are per query and shared across KV heads. Valid slots form whole
  4-token blocks: slots `4i..4i+3` hold four consecutive tokens; only the
  last run of a query may be short. Ascending order is allowed (it improves
  L2 reuse); 4-alignment is not.
- Invalid slots hold -1 in `Indices` and 0 in `Valid`. The host passes
  clamped indices (`split_indices`), so invalid quads gather token 0 and the
  `(valid - 1) * 1e30` pen mask drops their softmax contribution.
- Q is pre-grouped per KV head and zero-padded to 32 rows (`pad_q_heads`);
  the output keeps that layout and the host slices the first group rows
  per head.
- With `dynamic_shape=True`, `num_tokens` and `kv_length` are symbolic and
  one compiled kernel serves all shapes.

On an Ascend 950PR, the kernel reaches about **7.6 ms in BF16 prefill**
(4,096 query tokens, 32,768 KV, topk = 2,176) and **308 us in BF16 decode**
(128 query tokens, 65,536 KV), or roughly 27 and 21 useful TFLOPS. The flop
counts use the 24 real query heads and the average number of valid slots
per query; timings are kernel-only with L2 flushed before each launch.

## Run

Build TileLang with Ascend support; see the [backend guide](../../../tilelang/ascend/README.md).
From the TileLang repository root, select an idle NPU with
`ASCEND_RT_VISIBLE_DEVICES`, then:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0  # Replace 0 with an idle physical NPU.
export PYTHONPATH=.
python examples/ascend/qsa/run.py            # correctness + decode/prefill benchmarks
python examples/ascend/qsa/run.py --dynamic  # one symbolic kernel serves all shapes
python -m pytest examples/ascend/qsa/test_qsa.py -q
```

`--num-tokens N --kv-length M --top-k K` benchmarks a single custom shape.
