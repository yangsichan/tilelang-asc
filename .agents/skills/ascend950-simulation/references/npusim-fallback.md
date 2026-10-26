# npusim Fallback Workflow

Use this path for microbenchmarks, raw SQLite queries, VF IPC, or independent cross-validation with
msprof. For the default Insight instruction-detail deliverable, use msprof as described in SKILL.md.

```bash
npusim record -s Ascend950 -o ./sim-new -g -n all /absolute/path/to/acl_launcher
# For Python/JIT, use a validated interpreter and confirm the quoted invocation with current npusim help.
npusim record -s Ascend950 -o ./sim-python-new -g -n all \
  '/absolute/path/to/python /absolute/path/to/kernel_launcher.py'
```

`-n` controls logged cores; the launcher determines the launch core count. Explicitly report partial
scope when only core 0 is present. The validated CLI also exports VF IPC JSON with `-g vf`. Reuse the
original archive when regenerating reports:

```bash
npusim report -e /path/to/npusim_ARCHIVE -o ./new-pipeline-report -n all \
  --smart-pipes --merge-trace
```

Inspect actual per-core reports and `trace_allchips.json`; a top-level index warning cannot replace
file inspection. Native reports have omitted SQLite-recorded RVEC intervals in a SIMT case. Export
from SQLite when necessary:

```bash
mkdir -p ./new-sqlite-detail
python /path/to/ascend950-simulation/scripts/export_pipe_trace.py \
  /path/to/npusim_ARCHIVE/record/kernel.db ./new-sqlite-detail/trace.json --layout pipe
```

- In validated CANN 9.2.0 data, `ExecutedInstructions` contains start/end ticks; the clock comes from
  `KernelInfo.frequency_ghz`. The legacy exporter shares tracks by core/hardware PIPE and merges RVEC
  subtypes into VECTOR. `--layout slots` only creates virtual display slots; `--overview` optionally
  writes interval unions separately.
- Helper `status=ok` proves only successful writing. Inspect `start_overwrites`, `end_overwrites`,
  `start_only`, `end_only`, `mismatched`, `invalid`, and `write_failed`. SIMD/mixed cases have shown
  overwrites and unmatched records; SQLite export cannot recover missing events.
- Identify each `end_only` record individually. Distinguish it from compute/data-movement gaps only
  after proving it is zero-duration GET/RLS. Preserve raw pairing metrics; a count equal to the number
  of GET/RLS records is not identity verification.
- The validated helper 0.8.0 normalizes synchronization records to 1 tick, including WAIT_FLAG and
  intra-block events. These intervals cannot reveal actual waits. msprof ResourceConflictRatio
  intervals have different semantics.
- The legacy SQLite exporter still emits intra-block events as X slices. JSON pairing counts do not
  prove clickable associations in older Insight versions. Use the primary msprof merged-trace workflow
  for complete waiting intervals and validated flows.
- If the archive contains `instr.bin`/`chip*_instr.bin` instead of SQLite, use a matching-version decoder
  and verify fields and units first. Validated records use `is_popped=1` for dispatch and 0 for
  completion; revalidate after changing readers.
- Check target identity, inputs/outputs, launch/chip/core, ID/PC, and provenance. Do not align or splice
  absolute ticks across independent runs. Report correctness, collection completeness, simulated
  cycles, and hardware validation status separately.
