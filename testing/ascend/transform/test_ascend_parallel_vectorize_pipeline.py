"""P1 delivery and P1→P2→P3 pass-pipeline contract tests.

Offline: checks the annotation hand-off between ``AscendLiftVSimd`` →
``AscendPlanVSimd`` → ``AscendEmitVSimd``, idempotence, rejected nests staying
byte-identical, and the ``AscendVectorizeVSimd`` pipeline switch.
"""

from __future__ import annotations


import pytest

import tilelang
import tilelang.ascend.language as T
import tvm

from tilelang.ascend import transform as at
from tilelang.ascend.analysis.parallel_vectorize import (
    build_plan,
    emit_plan,
    render_plan,
)
from testing.ascend.transform._parallel_vectorize_helpers import (
    _count_parallel_nests,
    _emitted_ops,
    _lower_source,
    _outer_parallel_nests,
    _p1_items,
    _p1_kernel_1d,
    _p1_kernel_extent,
    _region_annotations,
    _region_plans,
    _run_lift,
    _run_new_path,
    _run_plan,
    _single_segment,
)


# ===========================================================================
# P1 delivery, the three-staged passes, and pipeline integration
# ===========================================================================
#
# This group pins the interfaces *outside* the nest. The full P1
# item/group/capture fields are covered by
# ``test_ascend_vsimd_{lift,region,capture_scope,memory_legality}.py``, and the
# P2/P3 planning/emission semantics are covered by the scenario cases in the
# sections above (multi-store, ternary/if, reductions, broadcast,
# transpose/flatten, mixed width, NZ, ...). This group does not repeat those
# assertions and only covers three things:
#
#   * delivery: a LIFTED nest can be fed to ``build_plan`` directly; a rejected
#     nest is recorded as OPAQUE/with a reason (P1 keeps one representative case);
#   * staging: P1/P2/P3 hand off through the ``tl.vsimd.region`` /
#     ``tl.vsimd.plan`` annotations (both attached to the Parallel), are
#     idempotent, and leave a rejected nest byte-identical;
#   * pipeline: ``AscendVectorizeVSimd`` is the only entry point of the new path
#     and runs under the default PassContext.


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Delivery contract: a LIFTED stmt is directly plannable; a rejected nest carries only its status
# ---------------------------------------------------------------------------


def test_plan_tail_chunk_math():
    """100 f32 elements: one full chunk plus a 36-lane tail."""
    func = _p1_kernel_extent(100)
    segment = build_plan(_outer_parallel_nests(func)[0]).segments[0]
    assert (segment.chunks, segment.tail) == (2, 36)


def test_plan_symbolic_extent():
    """Symbolic extent: a runtime chunk loop plus a symbolic tail; the plan does not fold it into a constant."""
    e = T.dynamic("E")

    @T.prim_func
    def main(A: T.Tensor((e,), "float32"), C: T.Tensor((e,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((e,), "float32")
            c_ub = T.alloc_shared((e,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(e):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    segment = build_plan(_outer_parallel_nests(main)[0]).segments[0]
    assert segment.chunks is None and segment.tail is None


def test_p1_serial_reduction_is_lifted_and_planned():
    """P1 proves distinct i instances independent while j remains serial.

    P2 can then choose its existing short reduction lowering. RMW syntax alone
    does not authorize replacing the serial j loop with a Parallel axis.
    """
    m, n = 8, 64

    @T.prim_func
    def main(A: T.Tensor((m, n), "float32"), C: T.Tensor((m,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((m, n), "float32")
            c_ub = T.alloc_shared((m,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(m):
                    for j in T.serial(n):
                        c_ub[i] += a_ub[i, j]
            T.copy(c_ub, C)

    assert _p1_items(main)[0].status == "LIFTED"
    segment = _single_segment(main)
    assert segment.vector_axis.name == "j"
    stmt = emit_plan(segment)
    ops = _emitted_ops(stmt)
    assert ops.count("tl.simd.vcadd") == 1
    assert ops.count("tl.simd.vadds") == 1
    assert ops.count("tl.simd.vsts") == 1
    # The reduction result lives only in lane 0; the write-back must use a lane-0-only predicate.
    assert 'pset(32, "PAT_VL1")' in str(stmt)


# ---------------------------------------------------------------------------
# Three-staged passes: annotation handoff, idempotence, rejected nests unchanged
# ---------------------------------------------------------------------------


def test_staged_passes_exchange_annotations():
    func = _p1_kernel_1d()

    lifted = _run_lift(func)
    regions = _region_annotations(lifted.body)
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"
    assert _count_parallel_nests(lifted.body) == 1, "P1 must not rewrite the nest"
    assert _emitted_ops(lifted.body) == []

    planned = _run_plan(lifted)
    planned_regions = _region_annotations(planned.body)
    assert len(planned_regions) == 1
    assert planned_regions[0].status == "LIFTED", "P2 must keep the P1 annotation"
    plans = _region_plans(planned.body)
    assert len(plans) == 1
    plan = plans[0]
    assert int(plan.version) == 1
    assert (int(plan.vf_index), int(plan.region_index)) == (0, 0)
    assert len(plan.program.segments) == 1
    assert _count_parallel_nests(planned.body) == 1
    # P2's plan matches the plan from directly planning the handed-off stmt, field by field.
    assert render_plan(plan.program) == render_plan(build_plan(_outer_parallel_nests(func)[0]))

    emitted = _run_new_path(func)
    assert _region_plans(emitted.body) == []
    assert _region_annotations(emitted.body) == []
    assert _count_parallel_nests(emitted.body) == 0
    assert "tl.simd.vsts" in _emitted_ops(emitted.body)
    tvm.ir.assert_structural_equal(emitted.body, _run_new_path(func).body, map_free_vars=True)


def test_passes_are_idempotent():
    once = _run_lift(_p1_kernel_1d())
    twice = _run_lift(once)
    assert len(_region_annotations(twice.body)) == 1

    combined = _run_new_path(_p1_kernel_1d())
    again = _run_new_path(combined)
    tvm.ir.assert_structural_equal(again.body, combined.body, map_free_vars=True)


def test_rejected_nests_fall_through_every_stage():
    # P1 rejects it (illegal memory) => recorded as OPAQUE (not liftable) and the IR
    # stays byte-identical; the new chain leaves it alone too.
    n = 256

    @T.prim_func
    def main(A: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n - 32):
                    a_ub[i] = a_ub[i + 32]

    illegal = main
    lifted = _run_lift(illegal)
    # P1 still records the nest, but as an OPAQUE region (memory legality is
    # ILLEGAL): nothing is liftable, so the body is byte-identical.
    regions = _region_annotations(lifted.body)
    assert regions and all(region.status != "LIFTED" for region in regions)
    # P2 writes no plan (nothing is liftable) and P3 restores the original body.
    assert _region_plans(_run_plan(lifted).body) == []
    tvm.ir.assert_structural_equal(_run_new_path(illegal).body, illegal.body, map_free_vars=True)


# ---------------------------------------------------------------------------
# Pipeline integration: P1/P2/P3 share one switch
# ---------------------------------------------------------------------------


@pytest.fixture
def pass_calls(monkeypatch) -> list:
    """Record how often the pipeline constructs the pass."""
    calls: list = []
    real = at.AscendVectorizeVSimd

    def spy():
        calls.append(True)
        return real()

    monkeypatch.setattr(at, "AscendVectorizeVSimd", spy)
    return calls


def test_disabling_the_lift_skips_the_whole_chain(pass_calls):
    _lower_source(
        _p1_kernel_1d(),
        config={tilelang.PassConfigKey.TL_ASCEND_SIMD_LIFT: False},
    )

    assert pass_calls == [], "P2/P3 must not run without the P1 annotation"
