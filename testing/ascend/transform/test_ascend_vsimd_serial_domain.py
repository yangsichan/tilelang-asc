"""P1 proves independence across lanes over full serial domains and guards."""

import itertools

import pytest

import tilelang.ascend.language as T
import tvm
from tilelang.ascend import transform as at
from testing.ascend.transform.test_ascend_vsimd_lift import (
    _lift,
    _regions,
    _strip_for_annotations,
)


def _check(func, status):
    report = str(at.AscendAnalyzeSimdVFMemoryReport(func))
    assert f"status={status}" in report, report
    mod = _lift(func)
    regions = _regions(mod)
    assert len(regions) == 1
    region = regions[0]
    assert region.status == ("LIFTED" if status == "LEGAL" else "OPAQUE"), region.reason
    if status != "LEGAL":
        assert region.reason_code == status
    tvm.ir.assert_structural_equal(_strip_for_annotations(mod["main"].body), func.body)
    return region


@pytest.mark.parametrize("scale,trip_count", [(0, 3), (1, 3), (2, 3), (3, 3), (5, 3), (0, 0), (0, 1)])
def test_serial_store_matches_finite_cross_lane_oracle(scale, trip_count):
    """Independent serial times can collide even if each fixed-time map is injective."""

    @T.prim_func
    def main(C: T.Tensor((64,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((64,), "int32")
            with T.SimdVF():
                for i in T.Parallel(4):
                    for k in T.serial(trip_count):
                        c[scale * i + k] = i + k
            T.copy(c, C)

    executions = list(itertools.product(range(4), range(trip_count)))
    conflict = any(i != j and scale * i + k == scale * j + t for (i, k), (j, t) in itertools.product(executions, repeat=2))
    _check(main, "ILLEGAL" if conflict else "LEGAL")


@pytest.mark.parametrize("factor", [1, 2])
def test_serial_reduction_and_ordered_recurrence(factor):
    """No reduction recognizer is needed; same-lane temporal dependence is legal."""

    @T.prim_func
    def main(A: T.Tensor((8, 4), "float32"), C: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((8, 4), "float32")
            c = T.alloc_shared((8,), "float32")
            T.copy(A, a)
            with T.SimdVF():
                for i in T.Parallel(8):
                    c[i] = 0.0
                    for k in T.serial(4):
                        c[i] = factor * c[i] + a[i, k]
                    c[i] = c[i] + 1.0
            T.copy(c, C)

    region = _check(main, "LEGAL")
    assert [op.kind for op in region.ops] == ["store", "serial", "store"]


@pytest.mark.parametrize("shift", [0, 1])
def test_sibling_shell_raw_is_checked_in_one_region(shift):
    @T.prim_func
    def main(C: T.Tensor((8,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((9, 2), "int32")
            c = T.alloc_shared((8,), "int32")
            with T.SimdVF():
                for i in T.Parallel(8):
                    for k in T.serial(2):
                        a[i, k] = i + k
                    for k in T.serial(2):
                        c[i] = a[i + shift, k]
            T.copy(c, C)

    region = _check(main, "ILLEGAL" if shift else "LEGAL")
    if not shift:
        assert [op.kind for op in region.ops] == ["serial", "serial"]


@pytest.mark.parametrize("trip_count", [1, 3])
def test_sibling_shell_cache_includes_domain(trip_count):
    @T.prim_func
    def main(C: T.Tensor((32,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((32,), "int32")
            with T.SimdVF():
                for i in T.Parallel(8):
                    for k in T.serial(1):
                        c[i + k] = 0
                    for k in T.serial(trip_count):
                        c[i + k] = 1
            T.copy(c, C)

    _check(main, "LEGAL" if trip_count == 1 else "ILLEGAL")


@pytest.mark.parametrize("bound", [1, 3])
def test_nested_serial_bound_uses_own_outer_counter(bound):
    @T.prim_func
    def main(C: T.Tensor((32,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((32,), "int32")
            with T.SimdVF():
                for i in T.Parallel(8):
                    for k in T.serial(bound):
                        for t in T.serial(k + 1):
                            c[i + t] = k
            T.copy(c, C)

    _check(main, "LEGAL" if bound == 1 else "ILLEGAL")


@pytest.mark.parametrize("trip_count", [1, 2])
def test_serial_guards_do_not_share_time_between_lanes(trip_count):
    @T.prim_func
    def main(C: T.Tensor((16,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((16,), "int32")
            with T.SimdVF():
                for i in T.Parallel(8):
                    for k in T.serial(trip_count):
                        if k == 0:
                            c[2 * i] = 1
                        else:
                            c[i] = 2
            T.copy(c, C)

    region = _check(main, "LEGAL" if trip_count == 1 else "ILLEGAL")
    if trip_count == 1:
        branch = region.ops[0].ops[0]
        assert branch.kind == "if"
        assert len(branch.ops) == len(branch.else_ops) == 1


def test_guard_restricts_overlapping_serial_map():
    @T.prim_func
    def main(C: T.Tensor((16,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((16,), "int32")
            with T.SimdVF():
                for i in T.Parallel(8):
                    for k in T.serial(4):
                        if k == 0:
                            c[i + k] = 1
            T.copy(c, C)

    _check(main, "LEGAL")


@pytest.mark.parametrize("shift", [0, 1])
def test_same_shell_cross_time_raw(shift, capfd):
    @T.prim_func
    def main(C: T.Tensor((32,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((32,), "int32")
            with T.SimdVF():
                for i in T.Parallel(8):
                    for k in T.serial(2):
                        c[2 * i + k] = c[2 * i + k + shift] + 1
            T.copy(c, C)

    region = _check(main, "ILLEGAL" if shift else "LEGAL")
    warning = capfd.readouterr().err
    if shift:
        assert "pair W c[" in region.reason
        assert " / R c[" in region.reason
        assert "counterexample" in region.reason
        assert region.reason in warning
        assert warning.count("[vsimd][P1][ILLEGAL]") == 1
    else:
        assert "[vsimd][P1]" not in warning


def test_outer_serial_counter_is_shared_context():
    @T.prim_func
    def main(C: T.Tensor((16,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((16,), "int32")
            with T.SimdVF():
                for k in T.serial(2):
                    for i in T.Parallel(8):
                        if k == 0:
                            c[2 * i] = 1
                        else:
                            c[i] = 2
            T.copy(c, C)

    _check(main, "LEGAL")


def test_serial_domain_outside_vf_restricts_guard():
    @T.prim_func
    def main(C: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((1,), "int32")
            for k in T.serial(1):
                with T.SimdVF():
                    for i in T.Parallel(8):
                        if i <= k:
                            c[0] = i
            T.copy(c, C)

    _check(main, "LEGAL")


def test_guard_and_serial_bound_only_scalars_are_captured():
    @T.prim_func
    def main(C: T.Tensor((8,), "int32"), bound: T.int32, flag: T.int32):
        with T.Kernel(1):
            c = T.alloc_shared((8,), "int32")
            with T.SimdVF():
                for i in T.Parallel(8):
                    for _k in T.serial(bound):
                        if flag > 0:
                            c[i] = 1
                        else:
                            c[i] = 2
            T.copy(c, C)

    region = _check(main, "LEGAL")
    assert {entry.var.name for entry in region.outer_vars} >= {"bound", "flag"}
    assert region.ops[0].ops[0].kind == "if"


def test_rmw_syntax_cannot_hide_shared_accumulator():
    @T.prim_func
    def main(C: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((1,), "int32")
            with T.SimdVF():
                for i in T.Parallel(8):
                    for k in T.serial(4):
                        c[0] = c[0] + i + k
                        c[0] = 1
            T.copy(c, C)

    _check(main, "ILLEGAL")


@pytest.mark.parametrize("active_lanes", [1, 2])
def test_guard_controls_ownership_of_serial_accumulator(active_lanes, capfd):
    @T.prim_func
    def main(C: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((1,), "int32")
            with T.SimdVF():
                for i in T.Parallel(8):
                    for k in T.serial(4):
                        if i < active_lanes:
                            c[0] = c[0] + k
            T.copy(c, C)

    region = _check(main, "LEGAL" if active_lanes == 1 else "ILLEGAL")
    warning = capfd.readouterr().err
    if active_lanes == 1:
        assert "[vsimd][P1]" not in warning
        assert region.reason == ""
    else:
        assert warning.count("[vsimd][P1][ILLEGAL]") == 1


def test_read_only_uniform_guard_is_stable_across_serial_time():
    @T.prim_func
    def main(F: T.Tensor((1,), "int32"), C: T.Tensor((16,), "int32")):
        with T.Kernel(1):
            flags = T.alloc_shared((1,), "int32")
            c = T.alloc_shared((16,), "int32")
            T.copy(F, flags)
            with T.SimdVF():
                for i in T.Parallel(8):
                    for k in T.serial(2):
                        if flags[0] == 0:
                            c[2 * i] = k
                        else:
                            c[i] = k
            T.copy(c, C)

    _check(main, "LEGAL")


def test_mutable_guard_does_not_supply_an_immutability_proof(capfd):
    @T.prim_func
    def main(C: T.Tensor((16,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((16,), "int32")
            flags = T.alloc_shared((8,), "int32")
            with T.SimdVF():
                for i in T.Parallel(8):
                    for k in T.serial(2):
                        flags[i] = k
                        if flags[i] == 0:
                            c[2 * i] = 1
                        else:
                            c[i] = 2
            T.copy(c, C)

    region = _check(main, "UNPROVEN")
    assert "guard reads storage written in the region" in region.reason
    warning = capfd.readouterr().err
    assert warning.count("[vsimd][P1][UNPROVEN]") == 1
    assert "independence could not be proven" in warning
    assert "[ILLEGAL]" not in warning
