---
name: ascend950-simulation
description: Simulate Ascend950 kernels with msprof op simulator, validate outputs, analyze instruction cycles and pipeline overlap, and supplement GET/RLS from the same dump for MindStudio Insight traces. Use npusim for fallback collection and cross-validation.
---

# Ascend950 Simulation

Default workflow: TileLang-generated `.asc` -> Bisheng compilation -> CANN runtime launcher ->
`msprof op simulator` -> output and collection validation -> enhanced merged trace -> Insight validation.
AutoSchedule latency/II cost estimates do not automatically run simulation.

## Compilation and Runtime Environment

- Use compatible versions of Bisheng, the CANN runtime, simulator, and msprof. Source `set_env.sh`
  and check `bisheng --version`, `msprof op simulator --help`, and the actual target architecture/SoC.
  `Ascend950PR_9589` was validated with CANN 9.2.0; do not reuse npusim's `Ascend950` spelling directly.
- Reuse existing compilation entry points; save the `.asc`, compilation command, and binary hash.
  Native ACL launchers do not require torch_npu. For Python/JIT, separately verify interpreter,
  module, and shared-library compatibility.
- Pass arguments according to the generated source's actual ABI. Preserve `__mix__(1, 2)` for mixed
  kernels and confirm that AIC, AIV0, and AIV1 all perform the intended work. When isolating operations,
  preserve layouts, memory spaces, aliases, synchronization, VF masks/threads, and loops.
- Prepare inputs and CPU references outside collection. Synchronize the stream after the kernel launch
  and validate outputs. Cycle experiments need inputs that exercise results and buffer versions;
  avoid collecting extra NPU initialization kernels.
- For initialization failures, distinguish the launcher environment from the profiler subprocess
  environment and prevent development stubs from shadowing the real runtime. Source mapping requires
  debug information; do not claim source call stacks when debug_line is absent.

## Collection and Export

Use a fresh directory for each kernel/variant. The SoC, frequency, and `--dump` syntax below are
examples from a validated installation; confirm them against the current tools and simulator model:

```bash
msprof op simulator --soc-version=Ascend950PR_9589 \
  --launch-count=1 --aic-metrics=PipeUtilization,ResourceConflictRatio \
  --dump=true --output=/path/to/new-profile /absolute/path/to/acl_launcher

python /path/to/ascend950-simulation/scripts/enhance_msprof_trace.py \
  /path/to/OPPROF_SINGLE_LAUNCH /path/to/new-enhanced/trace.json \
  --frequency-ghz 1.65
```

- Synchronization analysis requires `ResourceConflictRatio`; enabling only `PipeUtilization` disables
  synchronization details. Record target identity, launch/block counts, tool versions, output errors,
  and logs. A successful exit does not prove complete collection.
- Cover all participating cores by default. If `--core-id` is specified or a timeout truncates
  collection, explicitly report the scope and completeness limits.
- The input root must contain `dump/` and the top-level `simulator/trace.json` from the same launch.
  A per-core trace cannot replace the merged AIC/AIV trace. Do not splice ticks across collections,
  launches, or variants.
- Verify the simulator clock domain and frequency. `1.65` is neither the host CPU frequency nor a
  universal model default. Native JSON timestamps use microseconds; do not infer `ts/dur` units from
  `displayTimeUnit`.

The enhancement script preserves all native events and writes separate `trace.json` and
`trace.validation.json` files:

- Compare a **multiset** of core/subcore, PC, name, and completion tick against the dump; count B/E
  as one instruction. IDs can collide across instruction streams; do not deduplicate repeated PCs.
  Equal totals can still hide different instruction-name compositions.
- Stop on omissions other than GET/RLS, extra records, clock mismatches, or unknown dump formats;
  do not guess missing instructions. If native GET/RLS already exist, audit the version before adding
  duplicates. Use a fresh directory when output files already exist.
- Coverage validation applies only to cores in the trace; inspect `excluded_dump_cores`. It establishes
  coverage relative to the completion dump. Also inspect logs and dispatch dumps; this check cannot
  prove that the dump itself is complete.
- Supplement GET/RLS with their original completion tick, PC, ID, file, and line number; do not invent
  unknown dispatch times. Pair by core, target PIPE, buf_id, buf_mode, and buf_cnt epoch. Report missing
  fields, duplicates, or reversed ordering as `unlinked`.
- Preserve the GET/RLS names, zero semantic duration, and **1 tick** display width. Add explicitly
  marked 1-tick SET/WAIT anchors for Insight flows; exclude anchors from real instruction counts and
  performance statistics.
- For intra-block synchronization, preserve native B/E events, waiting intervals, IDs, target PIPEs,
  and completion-point flows. Add no compatibility events, re-pairing, or display-track remapping.
  A WAIT may start blocking before its SET begins.

Before importing into Insight, read [Viewer Compatibility and Validation](references/insight.md)
and verify actual parsing and endpoints. Successful JSON pairing does not prove GUI clickability.
For npusim microbenchmarks, SQLite queries, VF IPC, or independent cross-validation, see the
[Fallback Workflow](references/npusim-fallback.md).

## Analysis and Deliverables

- Dispatch-to-completion intervals can include queueing and dependency waits; do not treat them as
  intrinsic execution time. Use long native synchronization intervals to locate blocking. Exclude
  GET/RLS display widths from elapsed-time measurements.
- Calculate the kernel window as `max(start + duration) - min(start)`; do not sum overlapping intervals.
  Retain cycles and the frequency basis. Host wall-clock time is not device latency.
- Preserve buffer versions and producer/consumer dependencies across iterations; analyze fill,
  steady state, drain, and waits. `num_stages=2` does not prove overlap. Compare variants with the same
  workload/inputs and validate each output. Estimate steady-state II from same-kind instruction
  spacing over enough iterations, then verify it by varying iteration counts. Utilization alone
  does not establish a speedup.
- Deliver shape/dtype/layout, ABI/launch/core scope, commands/versions/hashes, correctness results,
  native output directory, enhanced trace, coverage and unlinked reports, cycles, and measurement
  boundaries. Native `visualize_data.bin` does not contain script-added GET/RLS and is not an
  equivalent substitute for the enhanced trace.
- **Simulation-test replies must disclose GET/RLS zero duration, 1-tick display width, and the extra
  1-tick SET/WAIT anchors. Report the actual anchor count and state that anchors are excluded from
  real instruction counts and performance statistics.** Also state whether collection was reused
  or simulation was rerun. Separately report correctness, collection coverage, Insight parsing/API/GUI
  checks, hardware validation status, and quantization limits.

For TileLang operation isolation and II measurements, consult Operation cycle measurements in
`.agents/skills/tilelang-ascend/references/profiling.md` as needed.
