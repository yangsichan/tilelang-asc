"""Regression checks for native interval preservation and conservative dump supplementation."""

import json
from pathlib import Path
import tempfile
import unittest

from enhance_msprof_trace import BUFFER_COLOR, BUFFER_FLOW_CATEGORY, supplement


class EnhancementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="msprof-skill-test-")
        self.addCleanup(self.tmp.cleanup)
        self.profile = Path(self.tmp.name)
        (self.profile / "simulator").mkdir()
        (self.profile / "dump").mkdir()
        self.events = [dict(name="MMAD", ph="X", pid="core0.cubecore0", tid="CUBE", ts=0.001, dur=0.009, args=dict(pc_addr="0x100"))]
        for name, pid, tid, start, end, pc, sync_id in [
            ("SET_INTRA_BLOCK", "core0.cubecore0", "FIXP", 0.08, 0.10, "0x104", 0),
            ("WAIT_INTRA_BLOCK", "core0.veccore0", "MTE3", 0.01, 0.11, "0x108", 0),
        ]:
            event = dict(
                name=name, ph="B", id="intra_0", pid=pid, tid=tid, ts=start, args=dict(pc_addr=pc, detail=json.dumps(dict(sync_id=sync_id)))
            )
            self.events.extend([event, dict(event, ph="E", ts=end)])
        for phase, pid, tid, ts in [("s", "core0.cubecore0", "FIXP", 0.10), ("t", "core0.veccore0", "MTE3", 0.11)]:
            self.events.append(dict(name="flow", ph=phase, id="intra_0", cat="FIXPToMTE3", pid=pid, tid=tid, ts=ts))
        self.rows = [
            ("core0.cubecore0", 10, "0x100", "MMAD", {}),
            ("core0.cubecore0", 100, "0x104", "SET_INTRA_BLOCK", {}),
            ("core0.veccore0", 110, "0x108", "WAIT_INTRA_BLOCK", {}),
        ]
        for epoch, get, release in [(0, 20, 30), (1, 40, 50)]:
            for name, tick, pc in [("GET_BUF", get, "0x110"), ("RLS_BUF", release, "0x114")]:
                self.rows.append(
                    ("core0.cubecore0", tick, pc, name, dict(instr_type="MTE1", target_pipe="MTE1", buf_id=0, buf_mode=0, buf_cnt=epoch))
                )

    def write_case(self):
        self.source = self.profile / "simulator/trace.json"
        self.source.write_text(json.dumps(dict(profilingType="op", traceEvents=self.events)), encoding="utf-8")
        by_core = {}
        for core, tick, pc, name, params in self.rows:
            # Deliberately repeated IDs/PCs: IDs alone must not deduplicate events.
            by_core.setdefault(core, []).append(
                f"[info] [{tick:08d}] (PC: {pc}) FLOWCTRL : (Binary: 0x00000000) (ID: 000007) {name}  {json.dumps(params)}"
            )
        for core, lines in by_core.items():
            (self.profile / "dump" / f"{core}.instr_log.dump").write_text("\n".join(lines), encoding="utf-8")

    def test_native_intervals_and_flows_survive_early_wait(self):
        self.write_case()
        before = self.source.read_bytes()
        trace, audit = supplement(self.profile, 1.0)
        self.assertEqual(trace["traceEvents"][: len(self.events)], self.events)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual((audit["source_completion_records"], audit["enhanced_logical_instructions"]), (7, 7))
        self.assertEqual((audit["buffer_pairs"], audit["compatibility_anchors"]), (2, 4))

    def test_intra_blockI_preserves_native_events_without_extra_endpoints(self):
        rename = {"SET_INTRA_BLOCK": "SET_INTRA_BLOCKI", "WAIT_INTRA_BLOCK": "WAIT_INTRA_BLOCKI"}
        for event in self.events:
            event["name"] = rename.get(event["name"], event["name"])
        self.rows = [(core, tick, pc, rename.get(name, name), params) for core, tick, pc, name, params in self.rows]
        self.write_case()
        trace, audit = supplement(self.profile, 1.0)
        self.assertEqual(trace["traceEvents"][: len(self.events)], self.events)
        added = trace["traceEvents"][len(self.events) :]
        anchors = [e for e in added if e.get("ph") == "B"]
        self.assertEqual(len(anchors), 4)
        self.assertTrue(all(e["args"]["original_instruction_name"] in {"GET_BUF", "RLS_BUF"} for e in anchors))
        self.assertEqual(audit["native_events_preserved"], len(self.events))

    def test_nonbuffer_omission_is_not_silently_accepted(self):
        self.rows.append(("core0.cubecore0", 120, "0x120", "LOAD_2Dv2", {}))
        self.write_case()
        with self.assertRaisesRegex(ValueError, "unsupported omissions"):
            supplement(self.profile, 1.0)

    def test_wrong_run_or_clock_fails_completion_matching(self):
        self.write_case()
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            supplement(self.profile, 1.65)

    def test_invalid_frequency(self):
        for frequency in [0, -1, float("nan"), float("inf")]:
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                supplement(self.profile, frequency)

    def test_missing_epoch_metadata_is_visible_but_not_guessed(self):
        self.rows[3][4].pop("buf_cnt")
        self.write_case()
        trace, audit = supplement(self.profile, 1.0)
        self.assertEqual(audit["buffer_pairs"], 1)
        self.assertEqual(len(audit["unlinked"]), 2)
        self.assertEqual(audit["added_buffer_events"], 4)

    def test_duplicate_epoch_is_not_paired_by_occurrence(self):
        self.rows.append(self.rows[3])
        self.write_case()
        trace, audit = supplement(self.profile, 1.0)
        self.assertEqual(audit["buffer_pairs"], 1)
        self.assertEqual(len(audit["unlinked"]), 3)
        self.assertEqual(audit["added_buffer_events"], 5)

    def test_color_hash_matches_insight_palette(self):
        self.assertEqual(sum(31 * ord(c) for c in BUFFER_COLOR) % 12, 9)
        self.assertEqual(sum(31 * ord(c) for c in BUFFER_FLOW_CATEGORY) % 12, 9)


if __name__ == "__main__":
    unittest.main()
