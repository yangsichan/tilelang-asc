# GLM DSA forward on Ascend 950

This directory contains the DSL kernel (`core.py`), standalone example (`example_glm_dsa.py`), correctness tests (`test_glm_dsa.py`), and this README. The implementation follows the per-head Q/K/V contract below.

## Contract

| Tensor | Shape | dtype |
| --- | --- | --- |
| Q | `[B, 64, sq, 256]` | BF16 |
| K, V | `[B, 64, T, 256]` | BF16 |
| indices | `[B, sq, Ksel]`, where `Ksel = min(2048, T)` | int32 |
| attention_mask | `[B, 1, sq, T]` | bool, passed as a contiguous uint8 view |
| O | `[B, sq, 64, 256]`, contiguous | BF16 |

The fixed attributes are 64 heads, head dimension 256, scale 0.0625, maximum TopK 2048, and dropout 0. K/V heads may contain independent values. Indices must be unique valid positions in `[0, T)`; `-1` padding is not supported. The kernel assumes this index contract rather than filtering invalid inputs.

A nonzero mask byte means that the position participates. Causal and padding restrictions must already be encoded by the caller. A query whose selected positions are all masked produces zero. The internal penalty convention is different: zero means keep and one means drop.

Comparison tools must preserve this True-means-keep contract. In the separately archived `NativeGBSA` adapter, `pack_indices` folds the mask into valid sparse indices, then sorts, compacts, and counts them. That adapter does not invert the bool mask or pass it directly to the native GBSA call. The adapter is maintained outside this directory; verify the deployed adapter version when reproducing results. This does not define the `atten_mask` convention of other operators.

B, sq, and T are supplied to the builder and specialized at compilation. Positive unaligned lengths use supported fallback paths; this is not one precompiled binary for arbitrary shapes. This kernel does not generate sparse indices or integrate an entire model.

## Implementation and entry points

`glm_dsa_fwd` constructs the device function. A task handles one query and a group of heads, with four heads per task by default. `pick_num_blocks` selects up to the available Cube core count. Each core loops over exactly its valid tasks, so task counts need not be divisible by the grid size.

Each head gathers 128 selected KV positions per tile and runs QK, masking, online softmax, and PV. K and V retain separate gathers in that order. Scores, running statistics, and the output numerator use FP32; probabilities used by PV and the final output use BF16. The store writes query-major output directly. Each AIV writes a disjoint half of 128 output columns.

Cube M contains 16 copies of the same query. The two-stage pipeline and five Cube `PIPE_ALL` barriers are retained. Changing those barriers requires separate correctness and performance validation.

For aligned mask rows within the UB budget, the vector mask path:

1. Widens the raw uint8 mask row to uint32 with vector operations once per query/head-group task.
2. Copies each full 128-index tile into a separate buffer consumed only by SIMD loads; K/V gathers still read their indices from GM.
3. Uses vector min/max to match the ascending pairs used by the DMA, gathers the corresponding mask words, and writes 128 penalty flags directly.

Eligibility requires T to be divisible by 256, T no larger than 8192, and `Ksel % 128 == 0`. The widened row and double-buffered index tile add at most approximately 33 KiB per AIV. There is no full sparse-selection flag cache. Other lengths and partial tiles retain the scalar mask path. The original raw mask-row staging rule remains T divisible by 32 and T no larger than 32768; larger or unaligned rows are read from GM. These are implementation choices, not new restrictions on valid input lengths.

`compile_kernel` in the example returns the callable for the five input tensors. `make_inputs` and `reference_attention` in the test module provide test inputs and the FP32 Golden. `run` demonstrates one kernel call and prints the output shape and dtype; correctness and performance validation are separate. GBSA deployment, adaptation, and performance scripts are maintained separately from this directory.

## Run

Use the existing compatible TileLang/CANN environment and an idle device:

```bash
ASCEND_RT_VISIBLE_DEVICES=2 python -m pytest -v -s -x test_glm_dsa.py
ASCEND_RT_VISIBLE_DEVICES=2 python example_glm_dsa.py --seq-len 448 --kv-len 4096
```

## Validation and source traceability

The server-validated snapshot is commit `7b552cb7`, tagged `glm-dsa-vector-mask-stable-20261106`. Its example MD5 is `8f354544cfd366749c986e50d6cf3c43`. All 17 server tests passed: three exact vector-mask probes and 14 attention regressions. Probes change masks and indices between calls and cover mixed nonzero bytes, all-keep, and all-drop masks.

The style-review snapshot dated 2026-11-06 has example MD5 `3471b18e53490861a730f4ad2f49a089`. This hash identifies that snapshot only and must not be assumed to cover later edits. It removes unused bindings, applies the repository formatter, adds the explicit full-tile eligibility condition, and clears an unnecessary executable bit. This revision changes the repository example and test files. The frozen Git tag and the separately archived `glm-dsa-vector-mask-stable.zip` release archive remain unchanged; the archive is not the repository `test_glm_dsa.py` file. Under the fixed TopK and current eligibility rules, the added condition does not change the selected path. Representative vector/fallback kernels and the three flag probes were checked for identical generated device source. No new NPU run is claimed for this style revision; the performance results below belong to the frozen snapshot.

## Recorded performance scope

The recorded setup reported 28 available Cube cores. All rows use B=1 and the fixed contract above, with the default four heads per task.

The ratio is GBSA complete-call resident host latency divided by TileLang latency. Inputs are already on the NPU. GBSA includes its adaptation, Metadata, and bridge work; this is not an isolated AICore ratio or model speedup. The measured points exceeded 1.0x, which does not establish all-shape or model-level performance acceptance.

| sq / T | TileLang latency, ms | GBSA / TileLang |
| --- | --- | --- |
| 1 / 4097 | approximately 0.286 | 2.20-2.32x |
| 16 / 2048 | 1.618-1.621 | 1.87-1.93x |
| 17 / 2049 | approximately 2.921 | approximately 1.11x |
| 448 / 4096 | 53.676-54.411 | 2.40-2.44x |
| 8192 / 8192 | approximately 875.532 | approximately 2.95x |

The first four points had two rounds with five warmups and 20 repeats. The sq=8192 point had one round with one warmup and three repeats, so it is a short sample. Large shapes compare 16 query rows against Golden; small shapes compare all query rows. Optional test configurations reached approximately 2.01x at sq=16 with one head per task and 3.15x at sq=448 with 64 heads per task. Neither changes the default configuration.

## File organization

- `core.py`: DSL kernel, sparse gathers, online softmax, head-group scheduling, and tiling configuration.
- `example_glm_dsa.py`: compilation interface and a single runnable usage example.
- `test_glm_dsa.py`: FP32 reference, test input generation, mask probes, and correctness regressions.

The file split preserves the device computation and does not introduce extra kernel launches.
Run correctness checks with `python -m pytest -v -s test_glm_dsa.py`. The example entry point demonstrates a call; it does not perform accuracy or performance acceptance.

The 2026-11-16 organization revision moves functions without changing the device algorithm. The hashes above describe historical single-file snapshots, not the split files. Local checks produced identical device source before and after the split for eight shapes. On Ascend 950 with 28 available Cube cores, the split revision passed all 17 regression tests and six supplementary boundary tests. Six performance shapes were measured in three rounds with four heads per task, using five warmups and 20 repeats (three warmups and five repeats for sq=8192). The framework base was `32684998`; the custom GBSA source was ops-transformer commit `aaad76f2b1a2f0d4aaf2a5fd531cff7aa9c1cc68`, in the same September 30 CANN environment used for the preceding run.


| B / sq / T | Split TileLang median, ms | Minimum resident GBSA / TileLang across three rounds |
| --- | --- | --- |
| 1 / 1 / 4097 | 0.260 | 2.518x |
| 1 / 16 / 2048 | 1.608 | 1.903x |
| 1 / 17 / 2049 | 2.707 | 1.186x |
| 1 / 448 / 4096 | 34.539 | 3.766x |
| 2 / 32 / 512 | 0.958 | 2.388x |
| 1 / 8192 / 8192 | 838.655 | 3.057x |

TileLang's median of the three round medians differed by less than 0.5% from the preceding single-file run for every measured shape. These remain complete-call resident comparisons, not isolated AICore measurements. E2E did not meet 1.0x in every round for sq=17/T=2049. The largest shape retained cross-round variation (approximately 696-845 ms); large-shape accuracy checks sampled 16 query rows. No all-shape or model-level acceptance is claimed.
