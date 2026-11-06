# GLM DSA forward on Ascend 950

This directory contains one implementation (`example_glm_dsa.py`), its
correctness tests (`test_glm_dsa.py`), and this README. This stable checkpoint is based on
b63342c1 with a bounded vector mask path (MD5 8f354544cfd366749c986e50d6cf3c43).

## Contract

- Q: BF16 [B, 64, sq, 256].
- K/V: BF16 [B, 64, T, 256], with independent heads supported.
- indices: int32 [B, sq, min(2048, T)], valid unique positions in [0, T).
  No -1 padding is supported.
- attention_mask: [B, 1, sq, T], boolean True means keep. Pass its
  contiguous uint8 view to the compiled kernel.
- O: BF16 [B, sq, 64, 256]. Scale 0.0625; dropout 0.

B/sq/T are supplied to the builder and specialized at compilation. They are
not restricted to aligned lengths. Causality must be encoded in the mask;
all-masked query rows produce zero.

## Implementation

`glm_dsa_fwd` builds the kernel. Each task processes one query and a group
of heads (four by default, configured through `FwdTiling.heads_per_task`).
Each head gathers sparse K/V in 128-position tiles and performs QK,
masking, online softmax and PV. The Cube M dimension repeats one query to
16 rows; head grouping does not make those rows distinct useful queries.
The output store writes the required query-major layout directly.

K/V gathers still read indices directly from GM. The vector mask path has a
separate128-index tile consumed only by SIMD loads, not scalar UB index reads. Mask rows use UB staging
only for T divisible by 32 and no larger than 32768; otherwise mask reads
come from GM. The two-stage pipeline retains Cube barriers protecting the
shared L0 resources used by QK and PV.

`make_inputs` and `reference_attention` supply sample inputs and an FP32
Golden. `compile_kernel` compiles the kernel; `pick_num_blocks` selects a
grid of min(available cores, B*sq*(64/heads_per_task)).
Each core loops over exactly its valid tasks, including its part of the final
round; no divisibility constraint or invalid padded task is needed. When num_blocks is omitted, the builder selects the grid using the actual
FwdTiling.heads_per_task; standalone run uses this same path. `run` is the standalone correctness and
TileLang timing entry; it does not compare against GBSA.

## Run

Activate the existing TileLang/CANN 9.3 weekly environment first, then run:

```bash
ASCEND_RT_VISIBLE_DEVICES=2 python -m pytest -v -s -x test_glm_dsa.py
ASCEND_RT_VISIBLE_DEVICES=2 python example_glm_dsa.py --seq-len 448 --kv-len 4096
```

GBSA deployment, adapters and comparison scripts remain outside this example
directory. Their timings depend on the bridge and adapter boundary and must
not be described as isolated AICore kernel timings without profiler evidence.
The current version passed all 14 server regression cases on the existing
Ascend 950 / CANN environment. This does not establish full-scene performance
acceptance.


## Grid correction (2026-11-06)

The default group size remains four. Automatic selection uses the actual tiling
and available Cube core count. No index staging or barrier removal is enabled.
The grid change includes a nondivisible 48-task/28-core case; all 14 regression
instances passed on the server. Local checks additionally covered 140 task
mappings and six lowering/codegen configurations.
The old PDFs retain old snapshot line numbers. Benchmark and server scripts
remain in the separate test package, not this directory.


## Stable vector mask checkpoint (2026-11-06)

Aligned T divisible by256 and no larger than8192 uses a bounded vector path.
The original raw uint8 mask row is widened to uint32 once per query/head-group
using vector bit extraction/interleave/stores. Each full128-index tile is DMA
staged for SIMD only; vld2/min/max/vgather2 generate paired0/1 flags in registers
and write them directly to penalty_ub. No full sparse-selection flag cache is
used. The uint32 row and index tile add at most approximately33 KiB per AIV.
Other positive lengths, partial tiles and larger T use the original scalar path.
These are internal eligibility rules, not input alignment restrictions.

K/QK/V/mask/softmax/PV ordering, independent per-head K/V, default grouping4,
balanced grid, math and five Cube barriers remain unchanged. No external
operator argument is added. This checkpoint passed all17 server tests. The test
suite has3 exact vector-flag probes first and14 original attention regressions.
Probe calls change masks and indices and cover all-keep/drop and nonzero byte
values. The server run passed all flag probes and attention regressions. Local
lowering/codegen and CPU models alone would not prove device correctness.


## Recorded performance and freeze boundary

Stable label: glm-dsa-vector-mask-stable-20261106. Default heads_per_task=4.
All cases below use B=1, BF16, H=64, D=256 and the recorded28-Cube-core setup.
The ratio is GBSA complete-call resident host latency / TileLang latency, not
isolated AICore execution. Inputs/mask adaptation and Metadata remain inside
GBSA. These are measured cases, not all-shape or model-level acceptance.

| sq / T | TileLang ms | GBSA / TileLang |
| --- | --- | --- |
| 1 / 4097 | about0.286 | 2.20-2.32x |
| 16 / 2048 | 1.618-1.621 | 1.87-1.93x |
| 17 / 2049 | about2.921 | about1.11x |
| 448 / 4096 | 53.676-54.411 | 2.40-2.44x |
| 8192 / 8192 | about875.532 | about2.95x |

The first four points had two rounds, warmup5/repeat20. The8192 point is one
round, warmup1/repeat3 only. Large cases compare16 Golden query rows; small
cases compare all query rows. Optional group1 at sq16 reached about2.01x;
optional group64 at sq448 reached about3.15x. Neither changes the default.
The example/test bytes are frozen to the server-verified snapshot. Failed
pair-cache and fused-gather experiments are not part of this checkpoint.
