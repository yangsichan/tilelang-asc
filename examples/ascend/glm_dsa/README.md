# GLM DSA forward on Ascend 950

This directory contains one implementation (`example_glm_dsa.py`), its
correctness tests (`test_glm_dsa.py`), and this README. The implementation is
synchronized with the tested head-groups package (MD5 cb2db7cd8dcd395cb70690229f4cf8a7).

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

Indices are read from GM (`INDEX_STAGED=False`). Mask rows use UB staging
only for T divisible by 32 and no larger than 32768; otherwise mask reads
come from GM. The two-stage pipeline retains Cube barriers protecting the
shared L0 resources used by QK and PV.

`make_inputs` and `reference_attention` supply sample inputs and an FP32
Golden. `compile_kernel` compiles the kernel; `pick_num_blocks` selects a
grid dividing B*sq*(64/heads_per_task). When overriding tiling, pass the same
heads_per_task to grid selection. `run` is the standalone correctness and
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
The existing server regression passed 10 cases; local syntax checks do not
constitute new NPU validation or full-scene performance acceptance.
