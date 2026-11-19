"""P1 delivery and P1→P2→P3 pass-pipeline contract tests.

Offline: checks the annotation hand-off between ``AscendLiftVSimd`` →
``AscendPlanVSimd`` → ``AscendEmitVSimd``, idempotence, and the
``TL_ASCEND_SIMD_LIFT`` switch (on: the three staged passes; off: the legacy
``AscendSimdVFLowerParallel``).
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
#   * delivery: a LIFTED nest can be fed to ``build_plan`` directly;
#   * staging: P1/P2/P3 hand off through the ``tl.vsimd.region`` /
#     ``tl.vsimd.plan`` annotations (both attached to the Parallel) and are
#     idempotent;
#   * pipeline: the three staged passes run under the default PassContext and
#     ``TL_ASCEND_SIMD_LIFT=False`` selects the legacy lowering.


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


# ---------------------------------------------------------------------------
# Pipeline integration: P1/P2/P3 share one switch
# ---------------------------------------------------------------------------


@pytest.fixture
def pass_calls(monkeypatch) -> dict:
    """Record how often the pipeline constructs each Parallel lowering pass."""
    calls = {"plan": 0, "emit": 0, "legacy": 0}

    def spy(key, real):
        def wrapped():
            calls[key] += 1
            return real()

        return wrapped

    monkeypatch.setattr(at, "AscendPlanVSimd", spy("plan", at.AscendPlanVSimd))
    monkeypatch.setattr(at, "AscendEmitVSimd", spy("emit", at.AscendEmitVSimd))
    monkeypatch.setattr(at, "AscendSimdVFLowerParallel", spy("legacy", at.AscendSimdVFLowerParallel))
    return calls


def test_disabling_the_lift_uses_the_legacy_lowering(pass_calls):
    source = _lower_source(
        _p1_kernel_1d(),
        config={tilelang.PassConfigKey.TL_ASCEND_SIMD_LIFT: False},
    )

    assert pass_calls["plan"] == 0 and pass_calls["emit"] == 0, "P2/P3 must not run without the P1 annotation"
    assert pass_calls["legacy"] == 1, "the switch selects the legacy Parallel lowering"
    assert "simd_inst::" in source, "the legacy lowering still reaches the vector ISA"


def test_disabling_the_lift_lowers_a_fragment_parallel():
    """P1 rejects fragments; the legacy path is the documented escape hatch."""
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                frag = T.alloc_fragment((n,), "float32")
                for i in T.Parallel(n):
                    frag[i] = a_ub[i] + 1.0
                for i in T.Parallel(n):
                    c_ub[i] = frag[i] * 2.0
            T.copy(c_ub, C)

    source = _lower_source(main, config={tilelang.PassConfigKey.TL_ASCEND_SIMD_LIFT: False})
    assert "simd_inst::" in source


def test_default_path_runs_the_three_staged_chain(pass_calls):
    source = _lower_source(_p1_kernel_1d())

    assert pass_calls["plan"] == 1 and pass_calls["emit"] == 1
    assert pass_calls["legacy"] == 0
    assert "simd_inst::" in source
