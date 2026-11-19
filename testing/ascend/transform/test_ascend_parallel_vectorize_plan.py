"""Offline structural tests for the P2 plan / P3 emit entry points.

Covers the transform-level contract (multi-axis nests, tail chunks, serial
shells, reductions, guards, access-shape variants) and the rejection surface
of ``build_plan``.  Hardware independent.
"""

from __future__ import annotations

import contextlib

import pytest

import tilelang.ascend.language as T
from tvm import tirx

from tilelang.ascend.analysis.parallel_vectorize import (
    AccessClass,
    Unsupported,
    VectorProgram,
    build_plan,
    emit_plan,
    render_plan,
    verify_no_scalar_data_ops,
)
from testing.ascend.transform._parallel_vectorize_helpers import (
    _alloc_buffers,
    _call_args,
    _chunk_iterations,
    _emitted_ops,
    _eval_lane,
    _first_parallel,
    _for_extents,
    _for_kinds,
    _loop_var_names,
    _negative_immediate_muls,
    _one_dimensional_copy,
    _over_reading_contiguous_load,
    _plan,
    _rejects,
    _single_segment,
)


# ---------------------------------------------------------------------------
# transform-level entry point
# ---------------------------------------------------------------------------


def test_plan_emit_replaces_the_parallel_nest():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i]

    stmt = emit_plan(_plan(k))

    kinds = _for_kinds(stmt)
    assert tirx.ForKind.PARALLEL not in kinds, "the T.Parallel nest must be replaced"
    assert kinds.count(tirx.ForKind.SERIAL) == 1  # the single axis -> chunk loop only
    assert "tl.simd.vld" in _emitted_ops(stmt)


def test_plan_emit_lowers_the_multidim_nest_as_a_whole():
    """Top-down lowering: a 2-axis ``T.Parallel(M, K)`` must be consumed *as a
    whole* (outer axis → serial, vector axis ``j`` → strip-mined chunks).  A
    bottom-up walk would replace the inner axis first, stranding the outer
    ``T.Parallel``."""
    M, K = 16, 128

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M, K), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            c = T.alloc_shared((M, K), "float32")
            for i, j in T.Parallel(M, K):
                c[i, j] = a[i, j]

    stmt = emit_plan(_plan(k))

    kinds = _for_kinds(stmt)
    assert tirx.ForKind.PARALLEL not in kinds, "the whole nest must be lowered"
    assert kinds.count(tirx.ForKind.SERIAL) == 2, "outer axis + chunk loop, no residue"
    ops = _emitted_ops(stmt)
    assert "tl.simd.vld" in ops and "tl.simd.vsts" in ops


def test_multidim_nest_keeps_every_axis():
    """The plan built from the nest head keeps all axes (outer ``i`` serial,
    inner ``j`` as the vector axis), which is what makes the whole-nest lowering
    possible."""
    M, K = 16, 128

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M, K), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            c = T.alloc_shared((M, K), "float32")
            for i, j in T.Parallel(M, K):
                c[i, j] = a[i, j]

    plan = _single_segment(k)
    assert [a.name for a in plan.axes] == ["i", "j"]
    assert plan.vector_axis.name == "j"


def test_plan_emit_lowers_multi_serial_program():
    """Top-down lowering also consumes a multi-serial ``T.Parallel`` body: the shared
    parallel axis becomes one serial loop, each segment contributes its own chunk loop,
    and source order (``c`` before ``d``) is preserved."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32"), D: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            d = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    d[i, l] = c[i, l] + 1.0

    stmt = emit_plan(_plan(k))

    kinds = _for_kinds(stmt)
    assert tirx.ForKind.PARALLEL not in kinds
    # Both segments are flattened into a full-lane composite axis: one chunk loop per segment.
    assert kinds.count(tirx.ForKind.SERIAL) == 2
    ops = _emitted_ops(stmt)
    assert ops.index("tl.simd.vmuls") < ops.index("tl.simd.vadds")


# ---------------------------------------------------------------------------
# Known-defect regression baseline (probes A-M)
#
# Every case below asserts the **correct contract** and marks it with
# ``xfail(strict=True)`` while it is still unmet:
#   * once fixed the test XPASSes, and ``strict=True`` turns that into an error --
#     forcing the xfail marker to be removed at the same time;
#   * if a defect is fixed "in another reasonable way" (e.g. an explicit rejection
#     instead of a silent wrong emission), the case already accepts "rejection" as
#     a valid outcome, so it needs no change.
# Reproduction entry points: the ``d*`` cases at the end of
# ``tilelang/ascend/analysis/test_drive_plan.py``; the matching diagnostics are in
# ``log_poly.log``.
# ---------------------------------------------------------------------------


def test_defect_d1_product_reduction_is_not_emitted_as_sum():
    """D1 fixed: a product reduction is **no longer silently emitted as a sum**.

    Full-lane vectorization along the parallel axis i emits ``s[i] *= a[i,j]`` as a
    per-lane ``vmul`` read-modify-write (the serial j loop stays a real loop), so
    the product semantics are fully preserved and no ``vcadd`` appears."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] * a[i, j]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vmul" in ops and "tl.simd.vcadd" not in ops


def test_defect_d2_plain_overwrite_is_not_treated_as_reduction():
    """D2 fixed: an overwrite ``s[i] = a[i,j]`` is not treated as a reduction (no ``vcadd``).

    ``j`` is a serial shell (last write wins in memory order), so after vectorizing
    along the parallel axis i the plan is a per-lane assignment inside a plan-level
    ``Loop(j)``: the semantics match the scalar program exactly and no cross-lane
    ``vcadd`` appears.  (A plain overwrite over a **parallel** j would be a racing
    write; P1 rejects that nest as ILLEGAL, so it never reaches P2.)"""
    MS, KS = 64, 16

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = a[i, j]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vsts" in ops and "tl.simd.vcadd" not in ops


def test_defect_d3_int_max_reduction_does_not_crash():
    """D3 fixed: the identity is derived from the dtype (``tirx.min_value``), so an int32 max reduction no longer raises ValueError."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "int32"), S: T.Tensor((MS,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "int32")
            s = T.alloc_shared((MS,), "int32")
            for i, j in T.Parallel(MS, KS):
                s[i] = T.max(s[i], a[i, j])

    # Correct contract: emission either succeeds or rejects cleanly with Unsupported;
    # it must never raise ValueError.
    with contextlib.suppress(Unsupported):
        emit_plan(_single_segment(k))


def test_defect_d3_f16_max_reduction_does_not_crash():
    """D3 fixed: the identity is derived from the dtype, so an f16 max reduction no longer raises ValueError by overflowing the value range."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float16"), S: T.Tensor((MS,), "float16")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float16")
            s = T.alloc_shared((MS,), "float16")
            for i, j in T.Parallel(MS, KS):
                s[i] = T.max(s[i], a[i, j])

    with contextlib.suppress(Unsupported):
        emit_plan(_single_segment(k))


def test_defect_d4_uniform_addend_inside_reduction_is_not_dropped():
    """D4/D5 fixed: a lane-uniform addend inside a reduction is no longer dropped.

    After vectorizing along the parallel axis i, ``s[i] = s[i] + a[i,j] + 1.0``
    keeps the ``+1.0`` (the scalar side of ``vadds`` is the imm 1.0), and the old
    value ``s[i]`` participates directly in the read-modify-write as a ``vload``."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] + a[i, j] + 1.0

    segment = _single_segment(k)
    assert segment.reduce is None
    value = segment.stores[0].value
    assert value.recipe == "vadds"
    assert "1.0" in str(value.args[-1].scalar)
    ops = _emitted_ops(emit_plan(segment))
    assert "tl.simd.vadds" in ops and "tl.simd.vcadd" not in ops


def test_defect_d6_two_reductions_in_one_nest_are_both_emitted():
    """D6 fixed: two reductions in one nest are both preserved (no longer only the first).

    After vectorizing along the parallel axis i, both stores are per-lane
    read-modify-writes: two ``vadd`` + two ``vsts``."""
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        B: T.Tensor((MS, KS), "float32"),
        S1: T.Tensor((MS,), "float32"),
        S2: T.Tensor((MS,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            s1 = T.alloc_shared((MS,), "float32")
            s2 = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s1[i] = s1[i] + a[i, j]
                    s2[i] = s2[i] + b[i, j]

    segment = _single_segment(k)
    assert [s.reduce for s in segment.stores] == [None, None]
    ops = _emitted_ops(emit_plan(segment))
    # Both stores must be written back (two vsts) with two per-lane additions; no
    # store may be silently dropped.
    assert ops.count("tl.simd.vsts") >= 2
    assert ops.count("tl.simd.vadd") == 2
    assert "tl.simd.vcadd" not in ops


def test_defect_d13_lane_varying_overwrite_is_not_a_reduction():
    """D13 fixed (unified criterion): ``s[i] = a[i,j] + 1.0`` is a "last write wins" overwrite and is not silently emitted as a sum.

    ``j`` is a serial shell, so after vectorizing along the parallel axis i it is a
    per-lane ``vadds`` + ``vsts`` inside ``Loop(j)``, with no ``vcadd``.  (Over a
    parallel j this would be a racing write, which P1 rejects as ILLEGAL.)"""
    MS, KS = 64, 16

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = a[i, j] + 1.0

    plan = _single_segment(k)
    assert plan.reduce is None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vadds" in ops and "tl.simd.vcadd" not in ops


def test_defect_d5_scaled_old_value_is_not_a_reduction():
    """D5/D13: a scaled old value (``s[i] = s[i] * 2.0 + a[i,j]``) no longer needs to be rejected.

    Vectorizing along the parallel axis i is a per-lane read-modify-write:
    ``vmuls(s, 2.0) -> vadd(a) -> vsts``, with the scale factor fully preserved
    and matching the scalar semantics."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] * 2.0 + a[i, j]

    plan = _single_segment(k)
    assert plan.reduce is None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vmuls" in ops and "tl.simd.vadd" in ops and "tl.simd.vcadd" not in ops


def test_defect_d6b_mixed_reductions_keep_their_own_recipe():
    """D6 fixed: a sum and a max in the same nest **each** keep their own operator.

    After vectorizing along the parallel axis they are ``vadd`` and ``vmax``
    respectively (no longer cross-lane ``vcadd``/``vcmax``)."""
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        B: T.Tensor((MS, KS), "float32"),
        S1: T.Tensor((MS,), "float32"),
        S2: T.Tensor((MS,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            s1 = T.alloc_shared((MS,), "float32")
            s2 = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s1[i] = s1[i] + a[i, j]
                    s2[i] = T.max(s2[i], b[i, j])

    segment = _single_segment(k)
    assert [store.reduce for store in segment.stores] == [None, None]
    ops = _emitted_ops(emit_plan(segment))
    assert "tl.simd.vadd" in ops and "tl.simd.vmax" in ops
    assert "tl.simd.vcadd" not in ops and "tl.simd.vcmax" not in ops


def test_broadcast_sibling_store_is_not_dropped():
    """A lane-invariant sibling store inside a serial shell is a broadcast write: it
    neither takes part in the accumulation of ``s1`` nor gets dropped.

    ``j`` is a serial shell (last write wins), so after vectorizing along the parallel
    axis i: ``s1[i] += a[i,j]`` is a per-lane read-modify-write and ``s2[i] = 3.0``
    is a broadcast write (``vdup`` + ``vsts``) inside ``Loop(j)``.  The same shape
    over a parallel j would race on ``s2[i]`` and P1 rejects it as ILLEGAL."""
    MS, KS = 64, 16

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S1: T.Tensor((MS,), "float32"), S2: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s1 = T.alloc_shared((MS,), "float32")
            s2 = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s1[i] = s1[i] + a[i, j]
                    s2[i] = 3.0

    segment = _single_segment(k)
    assert segment.reduce is None
    assert [store.reduce for store in segment.stores] == [None, None]
    ops = _emitted_ops(emit_plan(segment))
    assert "tl.simd.vadd" in ops and "tl.simd.vdup" in ops
    assert ops.count("tl.simd.vsts") >= 2  # one read-modify-write and one broadcast write


def test_defect_d13_stencil_shifted_read_is_not_an_accumulation():
    """D13/D5: in ``s[i] = s[i-1] + a[i,j]``, ``s[i-1]`` is **not at the same address**
    as the store, so this is not an accumulation.

    Stripping the old value requires reading back from the same address as the
    destination; a shifted (stencil) read stays in ``varying`` and triggers an
    explicit rejection, instead of being mistaken for ``dst op= x`` and silently
    dropping ``s[i-1]``.
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i, j in T.Parallel(MS, KS):
                s[i] = s[i - 1] + a[i, j]

    with pytest.raises(Unsupported, match="accumulate position"):
        _single_segment(k)


def test_partial_reduction_uses_parallel_axis_in_multi_segment():
    """A0: for ``s[2*i] += a[i,j]`` + ``u[i,j] = s[2*i]``, the segment is a "partial
    reduction" along the serial axis j (lane-uniform address + lane-varying value)
    -> that axis is rejected and the axis falls back to the parallel axis i:
    ``s[2*i]`` is a strided address along i with no per-lane conflicts, which is
    semantically correct."""
    MS, KS = 8, 128

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), U: T.Tensor((MS, KS), "float32"), S: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            u = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((16,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[2 * i] = s[2 * i] + a[i, j]
                    u[i, j] = s[2 * i]
                for j in T.serial(KS):
                    u[i, j] = a[i, j] * 2.0

    program = _plan(k)
    assert isinstance(program, VectorProgram)
    assert len(program.segments) == 2
    assert program.segments[0].vector_axis.name == "i"
    assert program.segments[0].reduce is None
    emit_plan(program)  # build succeeded => emit must succeed


def test_partial_reduction_falls_back_to_the_parallel_axis():
    """A0 (new): on the single-segment path, rejecting the serial axis falls back to the parallel axis and the result is still correct (not a silent wrong emission)."""
    MS, KS = 8, 128

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), U: T.Tensor((MS, KS), "float32"), S: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            u = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((16,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[2 * i] = s[2 * i] + a[i, j]
                    u[i, j] = s[2 * i]

    segment = _single_segment(k)
    # Do not vectorize along the serial axis j (that would lose the accumulation);
    # vectorize along the parallel axis i instead, keeping j as an ordinary serial loop.
    assert segment.vector_axis.name == "i"
    ops = _emitted_ops(emit_plan(segment))
    assert "tl.simd.vcadd" not in ops  # not a cross-lane reduction


def test_defect_d7_build_success_implies_emit_success_for_mixed_stores():
    """D7 fixed: `is_uniform` explicitly rejects argument-less leaves, so `vload_strided` is no longer misclassified as lane-uniform."""
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        D: T.Tensor((MS, KS), "float32"),
        S: T.Tensor((MS,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            d = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] + a[i, j]
                    d[i, j] = s[i]

    # Correct contract: if build_plan succeeds, emit_plan must succeed (no
    # "OK during analysis, Unsupported during emission").
    plan = build_plan(_first_parallel(k))
    emit_plan(plan)


def test_defect_d8_negative_stride_below_minus_one():
    """D8 fixed: a negative stride with ``Delta <= -2`` now goes through the per-lane
    index-vector channel instead of an unsigned ramp times a negative stride.

    The reproducer now uses a **legal** index (``126 - 2*i``, ``i < 64`` -> 126..0);
    the old case ``a[N-2-2*i]`` had a negative index for ``i >= 64``, i.e. the
    source program itself was out of bounds.
    """
    M = 64

    @T.prim_func
    def k(A: T.Tensor((128,), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((128,), "float32")
            c = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                c[i] = a[126 - 2 * i]

    try:
        plan = _single_segment(k)
        stmt = emit_plan(plan)
    except Unsupported:
        return
    ops = _emitted_ops(stmt)
    # Correct contract: either reject, or never express a negative-stride access as
    # "unsigned lane ramp x negative stride".
    assert not _negative_immediate_muls(stmt)
    assert "tl.simd.vgather2" in ops
    # Numeric golden: lane 0..3 indices must be 126, 124, 122, 120 (P0-1: no global negation).
    assert [_eval_lane(plan.stores[0].value, lane, plan) for lane in range(4)] == [126, 124, 122, 120]


def test_defect_d8_negative_stride_store_is_scattered():
    """D8 write side: a negative-stride store with ``Delta <= -2`` also uses index-vector ``vscatter`` (no negative-stride ramp)."""
    M = 64

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((128,), "float32")
            for i in T.Parallel(M):
                c[126 - 2 * i] = a[i]

    try:
        plan = _single_segment(k)
        stmt = emit_plan(plan)
    except Unsupported:
        return
    ops = _emitted_ops(stmt)
    assert not _negative_immediate_muls(stmt)
    assert "tl.simd.vscatter" in ops
    # Numeric golden (write side): scatter indices for lane 0..3 must be 126, 124, 122, 120.
    assert [_eval_lane(plan.stores[0].index, lane, plan) for lane in range(4)] == [126, 124, 122, 120]


def test_defect_d9_symbolic_extent_is_not_silently_truncated():
    """D9 fixed: a symbolic extent goes through a runtime chunk loop (``ceildiv(n, lanes)``) instead of a fixed single chunk."""
    n = tirx.Var("n", "int32")

    @T.prim_func
    def k(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float32")
            c = T.alloc_shared((n,), "float32")
            for i in T.Parallel(n):
                c[i] = a[i]

    try:
        stmt = emit_plan(_single_segment(k))
    except Unsupported:
        return
    # Correct contract: either reject, or the chunk loop must be driven by the
    # symbolic extent (never hard-coded to a single chunk).
    assert any(not isinstance(extent, tirx.IntImm) for extent in _for_extents(stmt))
    # The tail lane count must be clamped to lanes at runtime (A5: update_mask does
    # not accept an over-wide lane count).
    assert "tl.simd.update_mask" in _emitted_ops(stmt)
    assert any(isinstance(node, tirx.Min) for node in _call_args(stmt, "tl.simd.update_mask"))


def test_short_reduction_is_unrolled_with_multiple_accumulator_chains():
    """Short reduction (``chunks <= 4``): compile-time unrolling + multiple SSA accumulator chains + a tree merge.

    ``(4, 256)`` / lanes 64 -> 4 chunks, no tail: each of the 4 chains holds only
    one chunk (the chain head takes the ``vcadd`` result directly, so no ``vdup``
    identity is needed), then they are merged pairwise. Serial latency drops from
    4 levels to 2.
    """
    M, K = 4, 256

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), S: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            s = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                s[i] = s[i] + a[i, j]

    plan = _single_segment(k)
    assert plan.chunks == 4 and plan.tail == 0
    stmt = emit_plan(plan)
    ops = _emitted_ops(stmt)
    assert ops.count("tl.simd.vcadd") == 4  # unrolled: one per chunk
    # The 3 tree merges are vector `vadd` (PAT_VL1, lane 0 only); merging the old value uses scalar-vector `vadds`.
    assert ops.count("tl.simd.vadd") == 3
    assert ops.count("tl.simd.vadds") == 1
    assert "tl.simd.vdup" not in ops  # the chain head takes the vcadd result directly; no identity needed
    assert not _alloc_buffers(stmt)  # the unrolled path introduces no mutable registers
    # The old value is read once as a scalar and merged in; no BRC broadcast load remains.
    assert "BRC_B32" not in str(stmt)
    # Merges and write-backs after a cross-lane reduction use a lane-0-only predicate.
    assert 'pset(32, "PAT_VL1")' in str(stmt)


def test_long_reduction_keeps_a_real_chunk_loop_with_a_mutable_accumulator():
    """Long reduction (``chunks > 4``): no unrolling; a real chunk loop plus a ``local.var`` accumulator.

    ``(2, 512)`` / lanes 64 -> 8 chunks. Unrolling would grow linearly to 8x3
    instructions while the critical path stays 8 levels; the loop version has a
    single ``vcadd``, and the accumulator becomes a writable vector register
    (codegen emits ``vector_f32``).
    """
    M, K = 2, 512

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), S: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            s = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                s[i] = s[i] + a[i, j]

    plan = _single_segment(k)
    assert plan.chunks == 8
    stmt = emit_plan(plan)
    ops = _emitted_ops(stmt)
    assert ops.count("tl.simd.vcadd") == 1  # exactly one inside the loop body
    assert "tl.simd.vadd" in ops
    accumulators = _alloc_buffers(stmt)
    assert [buffer.scope() for buffer in accumulators] == ["local.var"]
    assert str(accumulators[0].dtype) == "float32x64"
    kinds = _for_kinds(stmt)
    assert kinds.count(tirx.ForKind.SERIAL) == 2  # outer parallel axis i + chunk loop
    # With tail == 0 the chunk loop must cover **all** chunks (the old implementation
    # ran only chunks-1, silently skipping the last one; after strip-mine convergence
    # the same skeleton guarantees it).
    chunk_loops = [extent for extent in _for_extents(stmt) if isinstance(extent, tirx.IntImm) and int(extent.value) == plan.chunks]
    assert chunk_loops, f"expected a {plan.chunks}-iteration full-chunk loop (tail == 0)"


def test_symbolic_extent_reduction_uses_a_runtime_chunk_loop():
    """Symbolic-extent reduction (D9 reduction branch): no longer rejected; uses a full-chunk main loop + peeled tail.

    The main loop runs ``E // lanes`` full-lane chunks; the tail runs once under
    ``if E % lanes > 0``, with ``update_mask`` clamping at runtime before
    ``vcadd`` (the Add identity is 0, so no ``vsel`` padding is needed).
    """
    n = tirx.Var("n", "int32")

    @T.prim_func
    def k(A: T.Tensor((n,), "float32"), S: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float32")
            s = T.alloc_shared((1,), "float32")
            for i in T.Parallel(n):
                s[0] = s[0] + a[i]

    segment = _single_segment(k)
    assert segment.chunks is None
    stmt = emit_plan(segment)
    ops = _emitted_ops(stmt)
    assert ops.count("tl.simd.vcadd") == 2  # one in the main loop body and one in the peeled tail (not unrolled by extent)
    assert "tl.simd.vsel" not in ops  # Add needs no identity padding
    assert any(not isinstance(extent, tirx.IntImm) for extent in _for_extents(stmt))
    assert any(isinstance(node, tirx.Min) for node in _call_args(stmt, "tl.simd.update_mask"))
    assert [buffer.scope() for buffer in _alloc_buffers(stmt)] == ["local.var"]


def test_defect_d10_non_reduction_tail_load_is_bounded():
    """D10 fixed: a tail load is predicated (the unpredicated ``vld`` form degrades to a predicated ``vgather2``)."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0

    try:
        stmt = emit_plan(_single_segment(k))
    except Unsupported:
        return
    # Correct contract: either reject, or never emit a whole-lane read that crosses
    # the innermost dimension of the source buffer.
    assert not _over_reading_contiguous_load(stmt)


def test_defect_d10_one_dimensional_tail_load_is_bounded():
    """The 1-D form of D10: on ``(100,)`` the tail ``vld(a[64], 64)`` would cross the allocation."""
    M = 100

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                c[i] = a[i]

    try:
        stmt = emit_plan(_single_segment(k))
    except Unsupported:
        return
    assert not _over_reading_contiguous_load(stmt)


def test_tail_chunk_loop_runs_exactly_one_iteration():
    """P0: the tail must cover exactly the last chunk ``[chunks-1, chunks)``.

    The third argument of ``tirx.For(var, min, extent)`` is an extent, not an end.
    The old implementation wrote ``For(tailchunk, full, chunks)``, so it started at
    ``full`` and ran ``chunks`` times, causing out-of-bounds reads/writes. The tail
    is now unrolled into **straight-line code**, so two forms are checked here: the
    union of full-chunk loops must be exactly ``[0, chunks-1)`` (when a tail
    exists), or every chunk is written exactly once when fully unrolled.
    """
    lanes = 64
    for M in (100, 200, 300):
        chunks = (M + lanes - 1) // lanes
        stmt = emit_plan(_single_segment(_one_dimensional_copy(M)))
        assert not _over_reading_contiguous_load(stmt), f"M={M}: contiguous load escapes the buffer"
        iterations = _chunk_iterations(stmt)
        if not iterations:
            # Single/double chunk fully unrolled: each chunk is written exactly once
            # (including the tail masked write).
            assert _emitted_ops(stmt).count("tl.simd.vsts") == chunks, f"M={M}"
            continue
        covered: list = []
        for name, lo, extent in iterations:
            assert lo is not None and extent is not None, f"{name}: non-constant chunk span"
            assert lo >= 0 and lo + extent <= chunks, f"M={M}: chunk loop {name} escapes [0, {chunks})"
            covered.extend(range(lo, lo + extent))
        # After peeling the tail, the remaining full-chunk loops must cover exactly
        # [0, chunks-1): no extra iteration (out of bounds) and no missing one.
        expected = list(range(chunks - 1)) if M % lanes else list(range(chunks))
        assert sorted(covered) == expected, f"M={M}: chunk coverage {sorted(covered)} != {expected}"


def test_tail_chunk_guard_is_proved_free():
    """The tail predicate is provable: ``if i >= 128: c[i] = a[i]`` on a 130-element axis.

    ``i >= 128`` cannot be proven for the whole axis (``i = 0`` is a counterexample),
    but it always holds on the tail ``[128, 130)``, so ``mask_free_from`` should
    point at the last chunk and the tail should emit no comparison.

    Note that the probe range in this fork is ``Range(begin, end)``, i.e.
    ``Range(128, 130)``; interpreting it as TVM's traditional
    ``Range(min, extent)`` and writing ``Range(128, 2)`` would yield an empty range
    and this case would fail immediately (``mask_free_from is None``).
    """
    N = 130  # 2 full chunks + a 2-lane tail chunk

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if i >= 128:
                    c[i] = a[i]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert (plan.chunks, plan.tail) == (3, 2)
    store = plan.stores[0]
    # until stays 0 when the whole axis is unprovable; from points at chunk 2 when
    # the tail is provable.
    assert store.mask_free_until == 0
    assert store.mask_free_from == 2
    # The prefix of two chunks has an always-false predicate and is removed entirely
    # (see the dead_guard case); the remaining tail uses the tail mask directly and
    # emits no comparison.
    ops = _emitted_ops(emit_plan(plan))
    assert ops.count("tl.simd.vcmps") == 0
    assert ops.count("tl.simd.vsts") == 1


def test_dead_guard_chunks_are_elided():
    """Whole chunks with an always-false predicate are removed: ``if i >= 128: c[i] = a[i]`` on 130 elements.

    ``i >= 128`` is always false on chunks 0/1 (lanes 0-127), yet those chunks used
    to emit ``vci + vcmps + vld + vsts`` anyway (writing no data). The plan side
    uses ``GuardFreeBounds(not(mask))`` to find the dead prefix, and
    ``chunk_lo/chunk_hi`` narrow the emission window to ``[2, 3)``, leaving only
    the tail's single ``vgather2`` + ``vsts``.
    """
    N = 130  # 2 full chunks (dead) + a 2-lane tail chunk (the only live one)

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if i >= 128:
                    c[i] = a[i]

    plan = _single_segment(k)
    assert (plan.chunk_lo, plan.chunk_hi) == (2, 3)

    stmt = emit_plan(plan)
    ops = _emitted_ops(stmt)
    # The dead chunk loop disappears entirely: no chunk loop, no full-chunk vld/vcmps/vsts.
    assert _chunk_iterations(stmt) == []
    assert ops.count("tl.simd.vld") == 0
    assert ops.count("tl.simd.vcmps") == 0
    assert ops.count("tl.simd.vgather2") == 1
    assert ops.count("tl.simd.vsts") == 1


def test_unmasked_store_keeps_every_chunk_live():
    """An unmasked store is not removed by mistake: ``chunk_lo/chunk_hi`` keep the default full window."""
    N = 130

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i]

    plan = _single_segment(k)
    assert (plan.chunk_lo, plan.chunk_hi) == (0, 3)
    stmt = emit_plan(plan)
    # The full-chunk loop still covers chunks 0/1; the tail still uses the masked gather.
    assert [(name, lo, extent) for name, lo, extent in _chunk_iterations(stmt)] == [("chunk", 0, 2)]
    assert _emitted_ops(stmt).count("tl.simd.vld") == 1


def test_lane_invariant_indirect_load_is_uniform():
    """An indirect address with Delta == 0 (``a[b[0]]``) is a lane-uniform scalar read -> ``vdup`` broadcast, not the index-vector channel."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((1,), "int32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((1,), "int32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[b[0]] * 2.0

    segment = _single_segment(k)
    assert segment.stores[0].kind == AccessClass.CONTIGUOUS
    ops = _emitted_ops(emit_plan(segment))
    assert "tl.simd.vdup" in ops
    assert "tl.simd.vgather2" not in ops and "tl.simd.vld" not in ops


def test_bool_store_is_rejected_at_build_time():
    """D11 family: a ``bool`` store is rejected at build time instead of raising ``ValueError`` at emit time."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "bool")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "bool")
            for i in T.Parallel(N):
                c[i] = a[i] > 0.0

    with pytest.raises(Unsupported, match="not supported"):
        _single_segment(k)


def test_masked_branch_merging_falls_back_to_predication():
    """Follow-up to defect D12: an ``if/else`` same-address merge where one arm already carries
    a predicate (nested else-if) is no longer rejected.

    There is no single select condition, but the two arms can still be predicated
    separately: ``c = a>0 ? 1 : (b>0 ? 2 : c)`` is emitted as two stores with
    complementary masks.
    """
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if a[i] > 0.0:
                    c[i] = 1.0
                else:
                    if b[i] > 0.0:
                        c[i] = 2.0

    plan = _single_segment(k)
    assert [store.spec.buffer.name for store in plan.stores] == ["c", "c"]
    assert all(store.mask is not None for store in plan.stores), "each arm is predicated"
    assert "tl.simd.pand" in str(emit_plan(plan)) or "tl.simd.pnot" in str(emit_plan(plan))


# ---------------------------------------------------------------------------
# Design-level audit regressions (R1-R7)
#
# Unlike the D* defects above, this group covers issues at the "unified model /
# emission contract" level: mostly **build succeeds but semantics are wrong**
# (silent wrong emission) or **build succeeds and emit crashes**. Each case pins
# the correct post-fix contract: either reject, or emit a form equivalent to the
# scalar semantics.
# ---------------------------------------------------------------------------


def test_predicate_combine_follows_the_data_element_width():
    """R1: the execution mask of predicate-to-predicate ops (pand/por/pnot) must follow the
    **data element** granularity.

    ``(100,)`` float16 -> lanes = 128, ``tail = 100``. If the user predicate
    (``vcmp``, b16) and the tail predicate (``update_mask``, b16) are combined with
    ``pand`` at a fixed b32 (64 lanes), lanes 64..127 are cleared and the tail only
    writes half. After the fix the mask width stays b16.
    """
    M = 100

    @T.prim_func
    def k(A: T.Tensor((M,), "float16"), B: T.Tensor((M,), "float16"), C: T.Tensor((M,), "float16")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float16")
            b = T.alloc_shared((M,), "float16")
            c = T.alloc_shared((M,), "float16")
            for i in T.Parallel(M):
                if a[i] > b[i]:
                    c[i] = a[i]

    segment = _single_segment(k)
    assert segment.lanes == 128 and segment.tail == 100
    stmt = emit_plan(segment)
    assert "tl.simd.pand" in _emitted_ops(stmt)
    widths = {int(arg.value) for arg in _call_args(stmt, "tl.simd.pset") if isinstance(arg, tirx.IntImm)}
    assert widths == {16}, f"predicate ops must run at the data element width (b16), got {widths}"


def test_lane_uniform_store_condition_is_broadcast_as_a_predicate():
    """R2: the condition of ``if a[0] > 0.0: c[i] = a[i]`` is uniform along the vector axis.

    A lane-uniform condition must not become a **scalar comparison** (scalar
    instructions are forbidden inside ``__simd_vf__``) and should not be used as a
    data vector either. The correct path is a vector predicate: ``vdup(a[0])`` +
    ``vcmps`` produce an all-0/all-1 mask, which is then ``pand``-ed with the tail
    mask (the scalar read of ``a[0]`` is only an operand of ``vdup``).
    """
    M = 100

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                if a[0] > 0.0:
                    c[i] = a[i]

    statement = emit_plan(_single_segment(k))  # build succeeded => emit must succeed
    ops = _emitted_ops(statement)
    assert "tl.simd.vcmps" in ops, "the uniform guard is a vector compare"
    assert "tl.simd.pand" in ops, "conjoined with the tail mask (100 < 128)"
    assert "tl.simd.update_mask" in ops  # the tail predicate is still present


def test_conditional_reduction_is_rejected():
    """R3: a lane-dependent guard around a reduction has no mask semantics.

    ``if i < 64: s[0] += a[i]`` is a cross-lane reduction; a partial predicate
    cannot be expressed by the reduction mask (it would sum over all lanes). The
    tree planner reports ``tree.predicated_reduce`` with the failing path.

    In the production pipeline P1 already marks this nest ILLEGAL (the guarded
    accumulating write has a cross-iteration dependency), so P2 never sees it;
    the case pins the planner's own defensive path.
    """
    M = 100

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), S: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            s = T.alloc_shared((1,), "float32")
            for i in T.Parallel(M):
                if i < 64:
                    s[0] = s[0] + a[i]

    with _rejects("tree.predicated_reduce"):
        _single_segment(k)


def test_repeated_old_value_inside_a_reduction_is_rejected():
    """R4: the old value appears twice in ``s[0] = s[0] + (s[0] + a[i,j])``.

    ``_strip_reduce`` strips both occurrences along the reduction spine, leaving a
    single initial value -> one copy of ``s[0]`` is lost. It is now rejected
    explicitly because the strip count must be exactly 1.
    """
    M, K = 4, 128

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), S: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            s = T.alloc_shared((1,), "float32")
            for i in T.Parallel(M):
                for j in T.serial(K):
                    s[0] = s[0] + (s[0] + a[i, j])

    with pytest.raises(Unsupported, match="appears more than once"):
        _single_segment(k)


def test_serial_loop_with_non_zero_lower_bound_is_rejected():
    """R5: the iteration domain is modelled as ``0 <= x < extent``, so the lower bound of
    ``T.serial(1, K)`` would be dropped.

    The old implementation emitted ``for j in range(K-1)`` (starting at 0), shifting
    every index by one. It is now rejected explicitly.
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(1, KS):
                    c[i, j] = a[i, j] * 2.0

    with pytest.raises(Unsupported, match="must start at 0"):
        _single_segment(k)


def test_multiple_writes_to_one_buffer_inside_a_materialized_loop_keep_source_order():
    """R6 (updated): multiple writes to the same buffer (``c[i,2*j]`` and ``c[i,2*j+1]``) in one serial body are no longer rejected.

    The two writes target disjoint addresses (even/odd columns), so P1 can prove
    that no cross-iteration alias exists; ``j`` still cannot be absorbed as a
    vector axis (two writes to one buffer at different addresses), so it becomes a
    plan-level ``Loop`` around the chunk.  The two vector stores inside it are
    emitted in source order, so a source-order bug (replaying both stores from one
    chunk loop) would be visible.
    """
    MS, KS = 4, 128

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, 2 * KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, 2 * KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, 2 * j] = a[i, j]
                    c[i, 2 * j + 1] = 2.0

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    text = str(emit_plan(build_plan(_first_parallel(k))))
    assert "for j in range(128):" in text
    scatter_at_even = text.index("c[0, j * 2]")
    scatter_at_odd = text.index("c[0, j * 2 + 1]")
    assert scatter_at_even < scatter_at_odd


@pytest.mark.parametrize("runtime_bound", [False, True])
def test_non_absorbable_loop_is_materialized_as_a_plan_loop(runtime_bound):
    """A loop the planner cannot absorb (last-write-wins) becomes a plan-level Loop.

    ``c[i] = S[i, j]`` writes one address per lane, so ``j`` is neither a vector
    axis nor a reduction; the tree is ``Loop(j, Chunk)`` and the loop wraps the
    chunk program. A lane-invariant runtime trip count is carried by the same
    node.
    """
    BR, BC = 128, 128

    if runtime_bound:

        @T.prim_func
        def k(A: T.Tensor((BR, BC), "float32"), C: T.Tensor((BR,), "float32")):
            with T.Kernel(2) as bx:
                s = T.alloc_shared((BR, BC), "float32")
                c = T.alloc_shared((BR,), "float32")
                t = T.min(BC, BC + bx)
                for i in T.Parallel(BR):
                    for j in T.serial(t):
                        c[i] = s[i, j]

    else:

        @T.prim_func
        def k(A: T.Tensor((BR, BC), "float32"), C: T.Tensor((BR,), "float32")):
            with T.Kernel(1):
                s = T.alloc_shared((BR, BC), "float32")
                c = T.alloc_shared((BR,), "float32")
                for i in T.Parallel(BR):
                    for j in T.serial(BC):
                        c[i] = s[i, j]

    program = _plan(k)
    assert len(program.segments) == 1
    assert program.root.kind == "loop" and str(program.root.loop_var) == "j"
    stmt = emit_plan(program)
    names = _loop_var_names(stmt)
    assert "j" in names
    assert any(name.startswith("chunk") for name in names), f"chunk loop missing: {names}"
    # ``j`` only reads ``s`` and writes ``c``: no cross-iteration conflict, so no
    # end-of-iteration VST->VLD barrier is inserted.
    assert "tl.simd.mem_bar" not in str(stmt)
    if runtime_bound:
        assert "t" in [str(extent) for extent in _for_extents(stmt)]


def test_lane_dependent_loop_bound_is_rejected():
    """A loop bound that reads the lane index would leave the counter scalar."""
    BR = 64

    @T.prim_func
    def k(A: T.Tensor((BR, BR), "float32"), C: T.Tensor((BR,), "float32")):
        with T.Kernel(1):
            s = T.alloc_shared((BR, BR), "float32")
            c = T.alloc_shared((BR,), "float32")
            for i in T.Parallel(BR):
                for j in T.serial(i + 1):
                    c[i] = s[i, j]

    with _rejects("tree.scalar_bound"):
        _single_segment(k)


def test_parallel_axis_reduction_with_non_uniform_pre_falls_back():
    """R7: when reducing along the **parallel axis**, ``pre``/``post`` are emitted by
    ``_emit_outer_store`` and must be lane-uniform.

    The old implementation checked this gate only on the "along the serial axis"
    branch, so ``t[i] = 1.0`` (contiguous along i) + a reduction along i passed
    build and raised ``Unsupported`` during emit. The axis is now rejected at build
    time, the planner falls back to the serial axis ``j`` (the correct reduction),
    and the diagnostics keep the fallback reason.
    """
    M, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((KS, M), "float32"), Tt: T.Tensor((M,), "float32"), S: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((KS, M), "float32")
            t = T.alloc_shared((M,), "float32")
            s = T.alloc_shared((1,), "float32")
            for i in T.Parallel(M):
                t[i] = 1.0
                for j in T.serial(KS):
                    s[0] = s[0] + a[j, i]

    program = _plan(k)
    segment = program.segments[0]
    assert segment.vector_axis.name == "j"
    assert any("-> fallback" in note for note in segment.diagnostics)
    emit_plan(program)  # build succeeded => emit must succeed


def test_multi_segment_shares_prologue_and_passes_the_accumulator_register():
    """Two kinds of inter-segment redundancy in a multi-segment program: repeated
    chunk-independent bindings, and "re-read immediately after write-back".

    Segment 0 accumulates ``s[i] += a[i,j]`` into a ``local.var`` register and
    writes it back to memory; segment 1 immediately computes ``c[i,l] = s[i]*2``.
    After the fix:

    * the full-lane mask and the stride ramp are each bound only once
      (one ``pset`` / one ``vci``);
    * segment 1's ``vld s`` passes through segment 0's accumulator register
      (``vld`` remains only for segment 0's initial load).
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] + a[i, j]
                for l in T.serial(KS):
                    c[i, l] = s[i] * 2.0

    program = _plan(k)
    assert isinstance(program, VectorProgram) and len(program.segments) == 2
    ops = _emitted_ops(emit_plan(program))
    assert ops.count("tl.simd.pset") == 1, "the full-lane mask should be bound once at program level"
    assert ops.count("tl.simd.vci") == 1, "the stride ramp for the same stride should be reused across segments"
    assert ops.count("tl.simd.vld") == 1, "segment 1's s read should pass through segment 0's accumulator register"


def test_multi_segment_carry_is_not_used_across_chunks():
    """Counterexample: when segments 0/1 are both flattened into 8 chunks (the lane base
    changes per chunk), the carry **must not** be used.

    Each segment of ``p_serial_compose`` runs its own ``chunk_s<k>`` loop; segment 1
    reading ``c`` written by segment 0 must re-``vld`` (twice) -- a carry is only
    allowed for a "single chunk, lane base always 0" segment.
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        C: T.Tensor((MS, KS), "float32"),
        D: T.Tensor((MS, KS), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            d = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    d[i, l] = c[i, l] + 1.0

    program = _plan(k)
    assert program.segments[0].chunks == 8 and program.segments[1].chunks == 8
    ops = _emitted_ops(emit_plan(program))
    assert ops.count("tl.simd.pset") == 1, "mask sharing still applies"
    assert ops.count("tl.simd.vld") == 2, "registers must not be carried across multi-chunk segments"


def test_carry_is_invalidated_by_an_intermediate_write():
    """A carry must be invalidated by an intermediate write: a re-read after
    ``s[i] = 0.0`` must not reuse the old register.

    Segment 0's reduction write-back registers the register value of ``s[i]``;
    segment 1's pre overwrite ``s[i] = 0.0`` ``note_write``-s away every carry
    entry for that buffer, so segment 2's ``vld s`` must really exist.
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] + a[i, j]
                s[i] = 0.0
                for l in T.serial(KS):
                    c[i, l] = s[i] * 2.0

    program = _plan(k)
    ops = _emitted_ops(emit_plan(program))
    # Segment 0's initial accumulator load + segment 2's real re-read after the intermediate write.
    assert ops.count("tl.simd.vld") == 2, "segment 0's register must not be reused after an intermediate write"


def test_carry_tracks_the_newest_producer():
    """Chained reduction: segment 1's initial accumulator value may pass through segment
    0's register, but the consuming segment must use segment 1's value."""
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        B: T.Tensor((MS, KS), "float32"),
        S: T.Tensor((MS,), "float32"),
        C: T.Tensor((MS, KS), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] = s[i] + a[i, j]
                for j in T.serial(KS):
                    s[i] = s[i] + b[i, j]
                for l in T.serial(KS):
                    c[i, l] = s[i] * 2.0

    program = _plan(k)
    assert len(program.segments) == 3
    stmt = emit_plan(program)
    ops = _emitted_ops(stmt)
    assert ops.count("tl.simd.vld") == 1, (
        "segment 1's initial value and segment 2's read should both be satisfied by the carry/latest register"
    )
    assert "accs1_0[0]" in str(stmt), "the consuming segment must use the register written back by segment 1 (the newest producer)"


# ---------------------------------------------------------------------------
# design A: structural tree rendering + invariant I5
# ---------------------------------------------------------------------------


def test_emitted_regions_have_no_scalar_data_ops():
    """I5: successful emissions contain no scalar tensor load/store."""
    N = 128

    @T.prim_func
    def copy_k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i]

    @T.prim_func
    def predicated_k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if a[i] > 0.0:
                    c[i] = a[i]

    M, K = 4, 128

    @T.prim_func
    def reduce_k(A: T.Tensor((M, K), "float32"), S: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            s = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                for j in T.serial(K):
                    s[i] = s[i] + a[i, j]

    for func in (copy_k, predicated_k, reduce_k):
        assert verify_no_scalar_data_ops(emit_plan(_plan(func))) == []

    # ... and the checker really flags a scalar tensor store (negative control).
    buf = tirx.decl_buffer((16,), "float32", name="buf", scope="shared.dyn")
    stmt = tirx.BufferStore(buf, tirx.FloatImm("float32", 1.0), [0])
    violations = verify_no_scalar_data_ops(stmt)
    assert len(violations) == 1 and "buf" in violations[0]


def _tree_chunk_nodes(node):
    """Chunk nodes of a plan tree in emission (source) order."""
    if node is None:
        return []
    if node.kind == "chunk":
        return [node]
    if node.kind == "seq":
        out = []
        for child in node.children:
            out.extend(_tree_chunk_nodes(child))
        return out
    if node.kind == "loop":
        return _tree_chunk_nodes(node.body)
    return []


def _contains_if(stmt):
    """Whether an emitted region still carries scalar control flow."""
    found = []

    def visit(node):
        if isinstance(node, tirx.IfThenElse):
            found.append(node)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return bool(found)


def test_tree_planner_reports_structure_errors_with_a_path():
    """``while`` / ``allocate`` / non-transparent attr are hard ``tree.*`` errors."""
    i = tirx.Var("i", "int32")
    buf = tirx.decl_buffer((16,), "float32", name="c", scope="shared.dyn")
    store = tirx.BufferStore(buf, tirx.FloatImm("float32", 0.0), [i])
    cases = [
        ("tree.while", tirx.While(tirx.IntImm("int32", 0), store)),
        ("tree.alloc", tirx.AllocBuffer(buf)),
        ("tree.opaque", tirx.AttrStmt(tirx.IntImm("int32", 1), "test_scope", tirx.IntImm("int32", 0), store)),
    ]
    for code, body in cases:
        stmt = tirx.For(i, 0, 16, tirx.ForKind.PARALLEL, body)
        with _rejects(code) as excinfo:
            build_plan(stmt)
        assert any(note.startswith("at body") for note in excinfo.value.diagnostics)


# ---------------------------------------------------------------------------
# design A: shared outer shell (one loop around sibling shell chunks)
#
# The loop bodies below are deliberately *non-absorbable overwrites*: they are
# the minimal shape that forces a plan-level ``Loop`` node.  Customer-facing
# accumulation forms under the same guards are covered by the device suite
# (``test_parallel_lane_uniform_branch_with_sibling_shells``).
# ---------------------------------------------------------------------------


def _shared_names(program) -> list[str]:
    return [str(axis.var) for axis in program.shared_shells]


def test_sibling_shells_share_one_outer_loop():
    """``for _kk { for m; for n }`` (all unclean) -> one shared ``_kk`` loop.

    The common ancestor is factored out as ``shared_shells`` instead of being
    copied into every segment. The tree is ``Loop(_kk, Seq(Loop(m, Chunk),
    Loop(n, Chunk)))`` and the shared loop is emitted exactly once. No loop needs
    a barrier: each body only reads a/b and writes s/u.
    """
    MS, KS = 64, 64

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), B: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), U: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            u = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for _kk in T.serial(KS):
                    for m in T.serial(KS):
                        s[i] = a[i, m] * 2.0
                    for n in T.serial(KS):
                        u[i] = b[i, n] + 1.0

    program = _plan(k)
    assert isinstance(program, VectorProgram)
    assert _shared_names(program) == ["_kk"], "the ancestor is shared, not copied"
    assert len(program.segments) == 2
    assert program.axes == [], "the lane axis is consumed by both segments"

    # The compatibility ``segments`` view is exactly the in-order chunk list.
    nodes = _tree_chunk_nodes(program.root)
    assert len(nodes) == len(program.segments) == 2
    assert all(node.chunk.same_as(segment) for node, segment in zip(nodes, program.segments))
    assert [node.chunk.vector_axis.name for node in nodes] == [segment.vector_axis.name for segment in program.segments]

    # Tree: Loop(_kk, Seq(Loop(m, Chunk), Loop(n, Chunk))); the header names the
    # shared chain and the rendered body is the indented plan tree (never the old
    # ``-- segment``).
    root = program.root
    assert root.kind == "loop" and str(root.loop_var) == "_kk"
    assert root.body.kind == "seq" and len(root.body.children) == 2
    assert [child.kind for child in root.body.children] == ["loop", "loop"]
    assert [str(child.loop_var) for child in root.body.children] == ["m", "n"]
    text = render_plan(program)
    assert text.startswith("program: axes=[] shells=['_kk'] segments=2")
    assert "loop _kk [0, 64)" in text and "\n  seq\n" in text
    assert "chunk #0" in text and "chunk #1" in text and "-- segment" not in text

    stmt = emit_plan(program)
    assert tirx.ForKind.PARALLEL not in _for_kinds(stmt)
    names = _loop_var_names(stmt)
    assert names.count("_kk") == 1, f"the shared loop must appear once: {names}"
    assert names.count("m") == 1 and names.count("n") == 1
    ops = _emitted_ops(stmt)
    assert "tl.simd.vmuls" in ops and "tl.simd.vadds" in ops
    # Every loop only reads a/b and writes s/u: no cross-iteration conflict, so
    # no end-of-iteration barrier is needed anywhere.
    assert ops.count("tl.simd.mem_bar") == 0


@pytest.mark.parametrize("condition", ["uniform", "lane_varying"])
def test_sibling_shells_under_a_branch_are_planned(condition):
    """``if c { for m; for n }`` keeps one loop chain per arm, guarded by masks.

    Lane-uniform and lane-varying conditions take the same structural path: the
    guard is pushed into every leaf store (``mask = c``) and no ``If`` node
    reaches the VF (bisheng only accepts vector work there). Each arm keeps its
    own loop chain in source order, so the arms are never zipped into one
    ``for m, n`` loop.
    """
    MS, KS = 64, 64

    if condition == "uniform":

        @T.prim_func
        def k(
            A: T.Tensor((MS, KS), "float32"), B: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), U: T.Tensor((MS,), "float32")
        ):
            with T.Kernel(1):
                a = T.alloc_shared((MS, KS), "float32")
                b = T.alloc_shared((MS, KS), "float32")
                s = T.alloc_shared((MS,), "float32")
                u = T.alloc_shared((MS,), "float32")
                for i in T.Parallel(MS):
                    if a[0, 0] > 0.0:
                        for m in T.serial(KS):
                            s[i] = a[i, m] * 2.0
                        for n in T.serial(KS):
                            u[i] = b[i, n] + 1.0

    else:

        @T.prim_func
        def k(
            A: T.Tensor((MS, KS), "float32"), B: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), U: T.Tensor((MS,), "float32")
        ):
            with T.Kernel(1):
                a = T.alloc_shared((MS, KS), "float32")
                b = T.alloc_shared((MS, KS), "float32")
                s = T.alloc_shared((MS,), "float32")
                u = T.alloc_shared((MS,), "float32")
                for i in T.Parallel(MS):
                    if i < MS // 2:
                        for m in T.serial(KS):
                            s[i] = a[i, m] * 2.0
                        for n in T.serial(KS):
                            u[i] = b[i, n] + 1.0

    program = _plan(k)
    assert len(program.segments) == 2
    # Every store carries the guard; the tree has no If node.
    for segment in program.segments:
        assert segment.stores
        for store in segment.stores:
            assert store.mask is not None and store.mask.recipe == "vcmps"
    root = program.root
    assert root.kind == "seq"
    assert [child.kind for child in root.children] == ["loop", "loop"]
    assert [str(child.loop_var) for child in root.children] == ["m", "n"]

    stmt = emit_plan(program)
    assert not _contains_if(stmt), "the VF contract has no scalar control flow"
    ops = _emitted_ops(stmt)
    assert ops.count("tl.simd.vcmps") == 2 and ops.count("tl.simd.vsts") == 2


def test_lane_uniform_if_with_else_predicates_both_arms():
    """A two-arm lane-uniform if predicates then with ``c`` and else with ``!c``.

    Both arms are multi-chunk, so the structural path is used. The masks are
    complementary: exactly one arm writes, matching the scalar branch.
    """
    MS, KS = 64, 64

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), B: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), U: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            u = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                if a[0, 0] > 0.0:
                    for m in T.serial(KS):
                        s[i] = a[i, m] * 2.0
                    for n in T.serial(KS):
                        u[i] = b[i, n] + 1.0
                else:
                    for p in T.serial(KS):
                        s[i] = a[i, p] * 3.0
                    for q in T.serial(KS):
                        u[i] = b[i, q] - 1.0

    program = _plan(k)
    assert len(program.segments) == 4
    recipes = [store.mask.recipe for seg in program.segments for store in seg.stores]
    assert recipes == ["vcmps"] * 4, "both arms are guarded by a vector compare"

    stmt = emit_plan(program)
    assert not _contains_if(stmt)
    body = str(stmt)
    # The then arm compares ``s[0] > 0`` (canonical ``0 < s[0]``); the else arm
    # is simplified to the complementary ``s[0] <= 0``.
    assert body.count('"lt"') == 2 and body.count('"le"') == 2
    ops = _emitted_ops(stmt)
    assert ops.count("tl.simd.vsts") == 4


def test_store_and_shell_siblings_keep_source_order():
    """N1 regression: a store that precedes a scalar loop must stay before it.

    Both the plain form (``u[i] = ...`` then ``for m``) and the same shape
    inside an ``if`` arm used to be replayed by a chunk that wrapped the loop;
    the loop is now a plan-level node emitted after the store.
    """
    MS, KS = 64, 64

    @T.prim_func
    def flat(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), U: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            u = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                u[i] = u[i] + 1.0
                for m in T.serial(KS):
                    s[i] = a[i, m] * 2.0

    @T.prim_func
    def guarded(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), U: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            u = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                if i < MS // 2:
                    u[i] = u[i] + 1.0
                    for m in T.serial(KS):
                        s[i] = a[i, m] * 2.0

    for kernel in (flat, guarded):
        program = _plan(kernel)
        root = program.root
        assert root.kind == "seq"
        assert [child.kind for child in root.children] == ["chunk", "loop"]
        assert str(root.children[1].loop_var) == "m"
        text = str(emit_plan(program))
        assert text.index("access_ptr(u[0]") < text.index("for m in range(64):")


def test_stores_under_one_guard_share_a_single_compare():
    """Two stores generated from one source ``if`` share the predicate: the
    condition is built once per chunk and P3 binds one ``vcmps`` for both.

    The guard reads ``a[0]`` -- a buffer the nest does not write.  A guard on a
    buffer the nest also writes (e.g. ``s[0]`` while storing to ``s[i]``) is a
    read/write race that P1 rejects as ILLEGAL, so it never reaches the planner.
    """
    N = 64

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), S: T.Tensor((N,), "float32"), U: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            s = T.alloc_shared((N,), "float32")
            u = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if a[0] > 0.0:
                    s[i] = a[i] * 2.0
                    u[i] = b[i] + 1.0

    program = _plan(k)
    assert len(program.segments) == 1
    segment = program.segments[0]
    assert len(segment.stores) == 2
    assert segment.stores[0].mask.same_as(segment.stores[1].mask)
    ops = _emitted_ops(emit_plan(program))
    assert ops.count("tl.simd.vcmps") == 1
    assert ops.count("tl.simd.vsts") == 2
