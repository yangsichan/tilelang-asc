"""Offline structural tests for P2 planning: access classification and body shapes.

Pins the mathematical core (``finite_difference`` / ``classify_delta``) and the
planning scenarios derived from it: if/select, reductions, multiple stores,
serial-shell absorption, multi-segment programs and diagnostics.
Hardware independent; emission-width and NZ details live in the sibling files.
"""

from __future__ import annotations


import pytest

import tilelang.ascend.language as T
import tvm
from tvm import arith, tirx

from tilelang.ascend.analysis.parallel_vectorize import (
    AccessClass,
    Unsupported,
    VectorProgram,
    build_plan,
    classify_delta,
    emit_plan,
    finite_difference,
    render_plan,
)
from testing.ascend.transform._parallel_vectorize_helpers import (
    _emitted_ops,
    _eval_lane,
    _for_kinds,
    _outer_parallel_nests,
    _over_reading_contiguous_load,
    _plan,
    _recipes,
    _rejects,
    _single_segment,
)


# ---------------------------------------------------------------------------
# the mathematical core: Δ(offset) decides everything
# ---------------------------------------------------------------------------


def test_finite_difference_classifies_contiguous():
    analyzer = arith.Analyzer()
    i = tirx.Var("i", "int32")
    assert classify_delta(finite_difference(i, i, analyzer)) == (AccessClass.CONTIGUOUS, 1)


def test_finite_difference_classifies_uniform():
    analyzer = arith.Analyzer()
    i = tirx.Var("i", "int32")
    assert classify_delta(finite_difference(i * 0 + 7, i, analyzer)) == (AccessClass.UNIFORM, 0)


def test_finite_difference_classifies_strided():
    analyzer = arith.Analyzer()
    i = tirx.Var("i", "int32")
    assert classify_delta(finite_difference(i * 2, i, analyzer)) == (AccessClass.STRIDED, 2)


def test_finite_difference_absorbs_flatten_identity():
    """``i//16*16 + i%16 == i`` after simplification, so Δ == 1 without any
    template matching (this is what makes §4 Case II a *math* result)."""
    analyzer = arith.Analyzer()
    i = tirx.Var("i", "int32")
    flattened = tirx.floordiv(i, 16) * 16 + tirx.floormod(i, 16)
    assert classify_delta(finite_difference(flattened, i, analyzer)) == (AccessClass.CONTIGUOUS, 1)


def test_finite_difference_classifies_non_constant_as_piecewise():
    """``i*i`` has Δ == 2i+1, not a constant → ``PIECEWISE`` at the Δ layer.

    The Δ layer only decides *constant vs non-constant*; the structural
    refinement (constructible index vector → ``PIECEWISE`` / ``INDIRECT`` /
    ``PERMUTE``, otherwise ``TRUE_RANDOM``) happens in
    :func:`classify_access` (improvement 4).
    """
    analyzer = arith.Analyzer()
    i = tirx.Var("i", "int32")
    assert classify_delta(finite_difference(i * i, i, analyzer))[0] == AccessClass.PIECEWISE


def test_const_int_proves_constant_beyond_literals():
    """``Δ`` need not already be a literal ``IntImm``: with an analyzer,
    ``(i//4 - i//4) + 3`` is *proved* to be the constant 3, so it classifies as
    ``STRIDED`` with stride 3 instead of falling into ``PIECEWISE``.  Without an
    analyzer the literal-only path cannot fold it."""
    analyzer = arith.Analyzer()
    i = tirx.Var("i", "int32")
    expr = (tirx.floordiv(i, 4) - tirx.floordiv(i, 4)) + 3
    assert classify_delta(expr, analyzer) == (AccessClass.STRIDED, 3)
    assert classify_delta(expr) == (AccessClass.PIECEWISE, None)


# ---------------------------------------------------------------------------
# §0 basic shapes
# ---------------------------------------------------------------------------


def test_copy():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.chunks == 2 and plan.tail == 0  # float32 -> 64 lanes, N=128
    assert plan.stores[0].kind == AccessClass.CONTIGUOUS
    assert _recipes(plan.stores[0].value) == ["vload"]

    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vld" in ops and "tl.simd.vsts" in ops


def test_scalar_broadcast():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[0]

    plan = _single_segment(k)
    assert plan.stores[0].value.recipe == "uniform_load"
    assert "tl.simd.vdup" in _emitted_ops(emit_plan(plan))


def test_fragment_nest_is_rejected_by_build_plan():
    """§0 Case III guard: a ``local.fragment`` operand has no memory form.

    P1 keeps such a nest out of P2/P3 entirely (it is reported OPAQUE, see
    ``test_ascend_vsimd_lift.py``), so the planner only needs to defend its own
    API: planning the nest directly must be refused instead of emitting an
    un-compilable ``vld``/``vsts`` pair against a V-core register tile.
    """
    N = 64

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "float32")
            c_ub = T.alloc_shared((N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                frag = T.alloc_fragment((N,), "float32")
                for i in T.Parallel(N):
                    frag[i] = a_ub[i] + 1.0
                for i in T.Parallel(N):
                    c_ub[i] = frag[i] * 2.0
            T.copy(c_ub, C)

    for nest in _outer_parallel_nests(k):
        with pytest.raises(Unsupported, match="local.fragment"):
            build_plan(nest)


def test_if_else_merges_into_select():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if a[i] == 0.0:
                    c[i] = 0.0
                else:
                    c[i] = T.exp(b[i])

    plan = _single_segment(k)
    assert plan.stores[0].mask is None  # both arms write the same location
    assert plan.stores[0].value.recipe == "vsel"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vsel" in ops and "tl.simd.vcmps" in ops and "tl.simd.vexp" in ops


def test_if_single_arm_is_predicated_store():
    P = 8

    @T.prim_func
    def k(A: T.Tensor((P, P), "float32"), C: T.Tensor((P, P), "float32")):
        with T.Kernel(1):
            c = T.alloc_shared((P, P), "float32")
            for i, j in T.Parallel(P, P):
                if i < j:
                    c[i, j] = 0.0

    plan = _single_segment(k)
    # The innermost dimension of ``(8, 8)`` has only 8 lanes (a register has 64), so P1's
    # composite contiguous axis flattens the whole nest.
    assert plan.vector_axis.name.startswith("k_")
    store = plan.stores[0]
    assert store.mask is not None
    # After flattening both i and j vary with the composite axis, so ``i < j`` becomes a
    # per-lane comparison (no longer a scalar vcmps).
    assert store.mask.recipe == "vcmp"
    ops = _emitted_ops(emit_plan(plan))
    # After flattening, 8x8 == 64 lanes is exactly one full chunk, so no tail predicate
    # is needed.
    assert "tl.simd.vcmp" in ops and "tl.simd.update_mask" not in ops


def test_statement_if_arm_load_is_guarded():
    """D16: memory accesses inside the arm of a statement-level single-arm if are lazy too.
    ``a[i-32]`` is out of bounds on lanes 0..31, so the load must carry the store
    predicate (emit a masked ``vgather2``) instead of a whole-lane ``vld a[-32]``.
    The guard is always true on chunk 1 (``i in [64,128)``) -> only chunk 0 pays for
    the ``vgather2``."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if i >= 32:
                    c[i] = a[i - 32]

    plan = _single_segment(k)
    store = plan.stores[0]
    assert store.mask is not None
    assert store.value.recipe == "vload" and store.value.predicate is not None
    assert plan.guard_boundaries == [1]  # the guard is always true from chunk 1 on
    ops = _emitted_ops(emit_plan(plan))
    assert ops.count("tl.simd.vgather2") == 1  # chunk 0: out-of-bounds risk, masked
    assert ops.count("tl.simd.vld") == 1  # chunk 1: guard always true, back to a plain vld
    assert ops.count("tl.simd.vcmps") == 1  # the predicate is computed only on guarded chunk 0


def test_statement_if_prefix_guard_is_dropped_before_boundary():
    """Mirror of D16: an always-true **prefix** guard. In ``if i < 96: c[i] = a[i+32]``,
    ``a[i+32]`` is only safe for ``i < 96`` (``i >= 96`` would read
    ``a[128..159]``); the guard is always true on chunk 0 -> chunk 0 falls back to a
    plain ``vld``, while chunk 1 still uses the masked ``vgather2``."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if i < 96:
                    c[i] = a[i + 32]

    plan = _single_segment(k)
    store = plan.stores[0]
    assert store.mask is not None
    assert store.value.recipe == "vload" and store.value.predicate is not None
    assert plan.guard_boundaries == [1]  # the guard is always true on chunk 0
    ops = _emitted_ops(emit_plan(plan))
    assert ops.count("tl.simd.vld") == 1  # chunk 0: guard always true, plain vld
    assert ops.count("tl.simd.vgather2") == 1  # chunk 1: out-of-bounds risk, masked
    assert ops.count("tl.simd.vcmps") == 1  # the predicate is computed only on guarded chunk 1


def test_guard_boundaries_cut_prefix_and_suffix_independently():
    """Prefix and suffix guards coexist in one segment: two cut points ``[1, 7]`` split
    8 chunks into three ranges.

    The suffix guard of ``c`` (``i >= 32``) is always true from chunk 1 on, and the
    prefix guard of ``d`` (``i < 480``) is always true through chunk 6; the two
    always-true ranges are complementary, so only chunk 0 and chunk 7 each need one
    masked ``vgather2``.
    """
    M = 512

    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((M,), "float32"), D: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((M,), "float32")
            d = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                if i >= 32:
                    c[i] = a[i - 32]
                if i < 480:
                    d[i] = a[i + 32]

    plan = _single_segment(k)
    assert plan.guard_boundaries == [1, 7]
    ops = _emitted_ops(emit_plan(plan))
    # chunk 0: c guarded + d always-true; chunks 1..6: both always-true; chunk 7:
    # c always-true + d guarded.
    assert ops.count("tl.simd.vgather2") == 2
    assert ops.count("tl.simd.vld") == 4
    assert ops.count("tl.simd.vcmps") == 2


def test_statement_if_else_merge_keeps_branch_laziness():
    """D16: a same-address if/else merges into a **lazy** ``if_then_else`` (not an eager
    ``Select``), the true arm's out-of-bounds read carries a conditional guard, and
    the constant else arm has no load."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if i >= 32:
                    c[i] = a[i - 32]
                else:
                    c[i] = 0.0

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vsel"
    assert value.args[1].recipe == "vload" and value.args[1].predicate is not None
    ops = _emitted_ops(emit_plan(plan))
    assert ops.count("tl.simd.vgather2") == 1
    assert ops.count("tl.simd.vld") == 1


def test_statement_if_safe_load_stays_unmasked():
    """Refinement of D16: when the arm address is **provably safe**, no guard is added and
    a plain ``vld`` is still emitted (no performance regression)."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                if a[i] != 0.0:
                    c[i] = a[i]

    plan = _single_segment(k)
    assert plan.stores[0].value.recipe == "vload" and plan.stores[0].value.predicate is None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vld" in ops and "tl.simd.vgather2" not in ops


def test_ternary():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = T.if_then_else(a[i] > b[i], a[i], b[i])

    plan = _single_segment(k)
    assert plan.stores[0].value.recipe == "vsel"
    # P0-3: both arm addresses are provably inside the buffer -> no predication is
    # needed and plain ``vld`` is still emitted.
    assert plan.stores[0].value.args[1].predicate is None
    assert plan.stores[0].value.args[2].predicate is None
    ops = _emitted_ops(emit_plan(plan))
    # a lane-varying condition compares per lane (vcmp); it is *not* broadcast.
    assert "tl.simd.vcmp" in ops
    assert "tl.simd.update_mask" not in ops
    assert "tl.simd.vgather2" not in ops


def test_lazy_conditional_with_unsafe_load_is_predicated():
    """P0-3: in ``if_then_else(i > 0, a[i-1], 0)``, ``a[i-1]`` is out of bounds on lane
    0, so the true arm's load must carry a conditional guard (a predicated
    ``vgather2``) instead of a whole-lane ``vld``."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = T.if_then_else(i > 0, a[i - 1], 0.0)

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vsel" and not value.is_uniform
    true_arm = value.args[1]
    assert true_arm.recipe == "vload" and true_arm.predicate is not None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vgather2" in ops  # guarded load channel
    assert "tl.simd.vcmps" in ops  # the condition itself (i > 0)


def test_lazy_conditional_with_unpredicable_uniform_load_is_rejected():
    """P0-3: a **lane-uniform** indirect read inside a lazy conditional arm cannot be
    predicated (there is no masked scalar read), so build must reject it explicitly
    instead of emitting an unconditional scalar read."""
    N = 128

    @T.prim_func
    def k(
        A: T.Tensor((N,), "float32"),
        D: T.Tensor((16,), "float32"),
        IDX: T.Tensor((1,), "int32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            d = T.alloc_shared((16,), "float32")
            idx = T.alloc_shared((1,), "int32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = T.if_then_else(a[i] > 0.0, d[idx[0]], 0.0)

    with pytest.raises(Unsupported, match="cannot be predicated"):
        _plan(k)


def test_lazy_conditional_with_safe_uniform_load_is_allowed():
    """Refinement of P0-3: when the scalar read address in the arm is **provably safe**, it
    need not be rejected and needs no guard either."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((16,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((16,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = T.if_then_else(a[i] > 0.0, b[0] + 1.0, 0.0)

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.args[1].is_uniform  # b[0] + 1 folds into a scalar + vdup
    assert "tl.simd.vsel" in _emitted_ops(emit_plan(plan))


def test_ternary_with_loop_invariant_condition_broadcasts_predicate():
    """§1 Case II as written in the catalogue: the condition depends only on the
    outer axis ``i_s2``, so it is uniform along the vector axis ``i_v`` → the
    whole chunk takes the same branch.  It is materialised as a vector predicate
    (broadcast operand + ``vcmps``) fed to ``vsel`` — no branch hoisting, and no
    scalar compare (a scalar instruction inside ``__simd_vf__`` is refused by
    bisheng)."""
    S, DV = 16, 128

    @T.prim_func
    def k(G: T.Tensor((S,), "float32"), D: T.Tensor((S, DV), "float32")):
        with T.Kernel(1):
            g = T.alloc_shared((S,), "float32")
            dv = T.alloc_shared((S, DV), "float32")
            for i_s2, i_v in T.Parallel(S, DV):
                dv[i_s2, i_v] = T.if_then_else(10.0 - g[i_s2] <= 0, dv[i_s2, i_v] * g[i_s2], 0.0)

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vsel" and value.detail == "uniform-cond select"
    assert value.args[0].recipe == "vcmps"  # a vector predicate, not a bool scalar
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcmps" in ops and "tl.simd.vsel" in ops
    assert "tl.simd.update_mask" not in ops, "no scalar-select mask"
    assert "tl.simd.vmuls" in ops and "tl.simd.vdup" in ops
    assert "tl.simd.vcmp" not in ops  # the condition is not compared per lane


def test_select_node_with_loop_invariant_condition_broadcasts_predicate():
    """Same path through the ``tirx.Select`` node (not ``T.if_then_else``)."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = tirx.Select(a[0] > 0.0, a[i], b[i])

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vsel" and value.detail == "uniform-cond select"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcmps" in ops and "tl.simd.vsel" in ops
    assert "tl.simd.update_mask" not in ops
    assert "tl.simd.vcmp" not in ops


def test_loop_invariant_condition_with_uniform_arm_broadcasts_and_dedups():
    """A uniform condition whose false arm is a scalar keeps working: the arm is
    materialised by ``vdup``, and the predicate is the vector one (broadcast +
    ``vcmps``)."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = T.if_then_else(a[0] > 0.0, b[i], 0.0)

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vsel" and value.args[2].is_uniform
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcmps" in ops and "tl.simd.vdup" in ops and "tl.simd.vsel" in ops
    assert "tl.simd.update_mask" not in ops


def test_uniform_select_is_a_vector_select_over_broadcast_arms():
    """When condition *and* both arms are loop-invariant, the select is still
    materialised with vector instructions: the arms are broadcast and the
    condition becomes a ``vcmps`` mask. A scalar ``?:`` (the old scalar-fold
    shape) would be an AIV-loop scalar instruction, which bisheng refuses."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = T.if_then_else(a[0] > 0.0, b[0], 2.0)

    plan = _single_segment(k)
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vsel" in ops and "tl.simd.vcmps" in ops
    assert "tl.simd.vdup" in ops and "tl.simd.vsts" in ops


# ---------------------------------------------------------------------------
# §2 operand rank mismatch / reduction
# ---------------------------------------------------------------------------


def test_operand_rank_mismatch_broadcasts_outer_load():
    M, K = 16, 128

    @T.prim_func
    def k(S: T.Tensor((K,), "float32"), C: T.Tensor((M, K), "float32")):
        with T.Kernel(1):
            g = T.alloc_shared((K,), "float32")
            d = T.alloc_shared((M, K), "float32")
            for i, j in T.Parallel(M, K):
                d[i, j] = g[i] - g[j]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "j"
    value = plan.stores[0].value
    # The SIMD layer has no vsubs: ``uniform - vector`` can only emit
    # ``vsub(vdup(uniform), vector)``. Only ``vector - scalar`` may fold into
    # ``vadds(vector, -scalar)``; the reverse fold would produce ``vector - scalar``
    # (opposite sign, P0-1).
    assert value.recipe == "vsub"
    assert value.args[0].recipe == "vbrc"
    assert value.args[0].args[0].recipe == "uniform_load"
    assert value.args[1].recipe == "vload"
    # Numeric golden: with i=3, lane j must be 3 - j (not j - 3).
    i_var = plan.axes[0].var
    got = [_eval_lane(value, lane, plan, env={i_var: 3}) for lane in range(8)]
    assert got == [3 - lane for lane in range(8)]
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vsub" in ops and "tl.simd.vdup" in ops and "tl.simd.vld" in ops
    assert "tl.simd.vadds" not in ops


def test_vector_minus_scalar_still_folds_to_vadds():
    """Reverse guard for P0-1: ``vector - scalar`` must still fold into
    ``vadds(vector, -scalar)`` (negating a scalar is scalar overhead) and must not be
    broken by the previous fix."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i] - 1.0

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vadds"
    assert value.args[0].recipe == "vload" and value.args[1].recipe == "uniform_expr"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vadds" in ops and "tl.simd.vsub" not in ops


def test_mixed_element_width_segment_is_rejected():
    """P0-2 + Stage 1: one segment shares a single chunk width (chunk stride = lanes).
    When f16 and int32 are mixed, the latter can only cover half the elements per
    chunk, silently under-writing or going out of bounds; therefore a store wider
    than the chunk must be a narrowing conversion (``vcvt`` packed store), and an
    ordinary store must still be rejected."""
    N = 128

    @T.prim_func
    def k(
        AF: T.Tensor((N,), "float16"),
        AI: T.Tensor((N,), "int32"),
        CF: T.Tensor((N,), "float16"),
        CI: T.Tensor((N,), "int32"),
    ):
        with T.Kernel(1):
            af = T.alloc_shared((N,), "float16")
            ai = T.alloc_shared((N,), "int32")
            cf = T.alloc_shared((N,), "float16")
            ci = T.alloc_shared((N,), "int32")
            for i in T.Parallel(N):
                cf[i] = af[i]
                ci[i] = ai[i]

    with pytest.raises(Unsupported, match="not a narrowing conversion"):
        _plan(k)


def test_same_width_mixed_element_types_derive_per_load_dtype():
    """P0-2: f32 / int32 have the same width and may share a segment; but each value
    node's dtype must be the element type of the **accessed buffer** (an int32 load
    must not claim ``float32x64``)."""
    N = 128

    @T.prim_func
    def k(
        AF: T.Tensor((N,), "float32"),
        AI: T.Tensor((N,), "int32"),
        CF: T.Tensor((N,), "float32"),
        CI: T.Tensor((N,), "int32"),
    ):
        with T.Kernel(1):
            af = T.alloc_shared((N,), "float32")
            ai = T.alloc_shared((N,), "int32")
            cf = T.alloc_shared((N,), "float32")
            ci = T.alloc_shared((N,), "int32")
            for i in T.Parallel(N):
                cf[i] = af[i]
                ci[i] = ai[i]

    plan = _single_segment(k)
    assert plan.lanes == 64
    assert [store.value.dtype for store in plan.stores] == ["float32x64", "int32x64"]
    ops = _emitted_ops(emit_plan(plan))
    assert ops.count("tl.simd.vld") == 2 and ops.count("tl.simd.vsts") == 2


def test_reduction_detected():
    M, K = 4, 256

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            acc = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                acc[i] = acc[i] + a[i, j]

    plan = _single_segment(k)
    assert plan.reduce is not None
    assert plan.reduce.op == "vadd" and plan.reduce.recipe == "vcadd"
    assert plan.vector_axis.name == "j"
    # the store itself is lane-uniform (dst rank < src rank).
    assert plan.stores[0].kind == AccessClass.UNIFORM
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcadd" in ops and "tl.simd.vsts" in ops


def test_reduction_max_maps_to_vcmax():
    M, K = 4, 256

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            acc = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                acc[i] = T.max(acc[i], a[i, j])

    plan = _single_segment(k)
    assert plan.reduce is not None and plan.reduce.recipe == "vcmax"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcmax" in ops
    # accumulate form (``dst = max(dst, x)``) merges the old destination value.
    assert "tl.simd.vld" in ops


def test_reduction_with_tail_fills_identity():
    """``extent % lanes != 0``: the last chunk is a partial tile.  Its inactive
    lanes read out of range, so the tail predicate must be applied.  For **Add**
    the identity is 0 and ``MODE_ZEROING`` already zeroes inactive lanes, so the
    predicate is passed straight to ``vcadd`` (no ``vsel`` fill); Max/Min still
    need an explicit identity fill."""
    M, K = 4, 32  # K < 64 lanes -> tail != 0

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            acc = T.alloc_shared((M,), "float32")
            for i, j in T.Parallel(M, K):
                acc[i] = acc[i] + a[i, j]

    plan = _single_segment(k)
    assert plan.reduce is not None and plan.tail == 32
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.update_mask" in ops  # tail predicate
    assert "tl.simd.vsel" not in ops  # Add needs no identity padding
    assert "tl.simd.vcadd" in ops  # cross-lane reduction (with the tail mask)
    assert "tl.simd.vsts" in ops  # ONEPT write-back


# ---------------------------------------------------------------------------
# §3 multiple statements / scalar temporaries
# ---------------------------------------------------------------------------


def test_multi_store():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32"), D: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            d = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i]
                d[i] = a[i] * 2.0

    plan = _single_segment(k)
    assert len(plan.stores) == 2
    assert plan.stores[0].value.recipe == "vload"
    assert plan.stores[1].value.recipe == "vmuls"
    assert _emitted_ops(emit_plan(plan)).count("tl.simd.vsts") == 2


def test_store_strided_on_the_only_axis_is_scattered():
    """§3 Case I as written in the catalogue: ``g[i, 0]`` (shape ``(BM, 2)``) keeps
    the trailing index fixed, so Δ == 2 along the only axis → ``STRIDED``.  Since
    strided stores are now first-class in the index-vector channel, the axis *is*
    a candidate and the write is emitted as ``vscatter`` (improvement 1)."""
    BM = 64

    @T.prim_func
    def k(A: T.Tensor((BM,), "float32"), C: T.Tensor((BM, 2), "float32")):
        with T.Kernel(1):
            mv = T.alloc_shared((BM,), "float32")
            g = T.alloc_shared((BM, 2), "float32")
            for i in T.Parallel(BM):
                g[i, 0] = mv[i]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.stores[0].kind == AccessClass.STRIDED
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vscatter" in ops and "tl.simd.vld" in ops


def test_multistore_with_unit_trailing_dim_is_contiguous():
    """The same multi-store shape becomes vectorizable once the trailing
    dimension is 1 (``g[i, 0]`` → ``Δ == 1``): the shape, not the code text,
    decides."""
    BM = 64

    @T.prim_func
    def k(A: T.Tensor((BM,), "float32"), B: T.Tensor((BM,), "float32"), C: T.Tensor((BM, 1), "float32"), D: T.Tensor((BM, 1), "float32")):
        with T.Kernel(1):
            mv = T.alloc_shared((BM,), "float32")
            mi = T.alloc_shared((BM,), "float32")
            g = T.alloc_shared((BM, 1), "float32")
            i2 = T.alloc_shared((BM, 1), "float32")
            for i in T.Parallel(BM):
                g[i, 0] = mv[i]
                i2[i, 0] = mi[i]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert [s.kind for s in plan.stores] == [AccessClass.CONTIGUOUS, AccessClass.CONTIGUOUS]
    assert _emitted_ops(emit_plan(plan)).count("tl.simd.vsts") == 2


def test_scalar_alias_folds_into_value():
    S, K = 16, 128

    @T.prim_func
    def k(A: T.Tensor((S, K), "float32"), C: T.Tensor((S, K), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((S, K), "float32")
            c = T.alloc_shared((S, K), "float32")
            for i, j in T.Parallel(S, K):
                acc: T.float32 = a[i, j] * 2.0
                c[i, j] = acc + 1.0

    plan = _single_segment(k)
    # ``acc`` is inlined: vadds(vmuls(vload, 2.0), 1.0)
    recipes = _recipes(plan.stores[0].value)
    assert recipes[0] == "vadds" and recipes[1] == "vmuls"


def test_floordiv_value_operator_strength_reduced():
    """§3 Case II complex scalar: ``FloorDiv`` / ``FloorMod`` used as a *value*
    operator.  ``i // 4`` is division by ``2^k``, so it strength-reduces to an
    arithmetic right shift (``vshrs``) and the float store then converts it
    (``vcvt``) — no integer ``vdiv`` needed."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i] + i // 4

    plan = _single_segment(k)
    recipes = _recipes(plan.stores[0].value)
    assert "vshrs" in recipes and "vcvt" in recipes
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vshrs" in ops and "tl.simd.vcvt" in ops


def test_floormod_value_operator_strength_reduced():
    """``i % 8`` → ``vand(i, 7)``: the two's-complement bitwise-and is exactly
    ``FloorMod`` for a power-of-two divisor, so a masked ``vdup`` constant feeds
    the existing vector-vector ``vand``."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "int32"), C: T.Tensor((N,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "int32")
            c = T.alloc_shared((N,), "int32")
            for i in T.Parallel(N):
                c[i] = a[i] + i % 8

    plan = _single_segment(k)
    recipes = _recipes(plan.stores[0].value)
    assert "vand" in recipes and "vbrc" in recipes
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vand" in ops and "tl.simd.vdup" in ops


def test_floordiv_by_non_power_of_two_is_rejected():
    """A non-power-of-two constant divisor has no exact 32-bit strength reduction
    (it needs 64-bit ``vmull`` magic numbers), so it is rejected rather than
    mis-emitted as a shift."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "int32"), C: T.Tensor((N,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "int32")
            c = T.alloc_shared((N,), "int32")
            for i in T.Parallel(N):
                c[i] = a[i] + i // 3

    with pytest.raises(Unsupported, match="non-power-of-two constant 3"):
        _plan(k)


# ---------------------------------------------------------------------------
# §4 expression subscripts / flattening / strided
# ---------------------------------------------------------------------------


def test_expression_subscript():
    # PAD is a whole 32B block (8 f32 elements): a block load needs a
    # 32B-aligned base, so a non-multiple offset (e.g. 3) is rejected by P2
    # (``plan.alignment``) and the segment falls back to another axis.
    B0, LEN, PAD = 2, 128, 8

    @T.prim_func
    def k(A: T.Tensor((B0, PAD + LEN), "float32"), C: T.Tensor((B0, LEN), "float32")):
        with T.Kernel(1):
            g = T.alloc_shared((B0, PAD + LEN), "float32")
            d = T.alloc_shared((B0, LEN), "float32")
            for b0, i_t in T.Parallel(B0, LEN):
                d[b0, i_t] = g[b0, PAD + i_t]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i_t"
    assert plan.stores[0].kind == AccessClass.CONTIGUOUS
    assert plan.stores[0].value.recipe == "vload"
    assert "tl.simd.vld" in _emitted_ops(emit_plan(plan))


def test_misaligned_stream_base_is_rejected():
    """A block ``vld``/``vsts`` needs a 32B-aligned base.

    ``a[i + 4]`` starts every chunk 4 f32 elements (16B) off the block boundary,
    and the emitter folds the whole element offset into the base pointer, so the
    instruction would decode the wrong 32B block (device report: a 400B f32 row
    stride returns 507035). The emitted form has no offset operand yet, so P2
    rejects the segment instead of miscompiling it.
    """
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N + 4,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N + 4,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i + 4]

    with _rejects("plan.alignment"):
        _plan(k)


def test_expression_subscript_reads_shared_operand_as_uniform():
    """§4 Case I (UB variant): ``Gb[0, S + i_s, 0]`` does not vary along the
    vector axis ``i_k`` → lane-uniform → a single scalar read broadcast by
    ``vdup``."""
    BS, S, DK, BH = 128, 16, 64, 8

    @T.prim_func
    def k(OUT: T.Tensor((S, DK), "float32")):
        with T.Kernel(1):
            gb = T.alloc_shared((2, BS * 8, BH), "float32")
            gs = T.alloc_shared((S, DK), "float32")
            for i_s, i_k in T.Parallel(S, DK):
                gs[i_s, i_k] = gb[0, 0 * S + i_s, 0]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i_k"
    assert plan.stores[0].kind == AccessClass.CONTIGUOUS
    assert plan.stores[0].value.recipe == "uniform_load"
    assert "tl.simd.vdup" in _emitted_ops(emit_plan(plan))


def test_highdim_flatten():
    M, LEN = 2, 128

    @T.prim_func
    def k(H: T.Tensor((M, 8, 16), "float32"), D: T.Tensor((M, 8, 16), "float32"), C: T.Tensor((M, LEN), "float32")):
        with T.Kernel(1):
            h = T.alloc_shared((M, 8, 16), "float32")
            d = T.alloc_shared((M, 8, 16), "float32")
            c = T.alloc_shared((M, LEN), "float32")
            for i_kv, _ in T.Parallel(M, LEN):
                c[i_kv, _] = h[i_kv, _ // 16, _ % 16] * d[i_kv, _ // 16, _ % 16]

    plan = _single_segment(k)
    # The 3D index flattens to ``i_kv*128 + _`` → Δ == 1 without a template match.
    assert plan.vector_axis.name == "_"
    assert plan.stores[0].kind == AccessClass.CONTIGUOUS
    assert plan.stores[0].value.recipe == "vmul"
    assert _recipes(plan.stores[0].value).count("vload") == 2


def test_block_tiled_fragment_layout_is_piecewise_gather():
    """§4 Case II block-tiled/store-permute: ``cs[i//MX, j//MY, i%MX, j%MY]`` is
    a *block* permutation of the indices, so ``Δ(offset)`` is not constant even
    after flattening.  MX=MY=4 are powers of two, so the per-lane element offset
    *is* constructible → ``PIECEWISE`` → the load is emitted as ``vgather2`` with
    the absolute offset vector (improvement 1 + 4), while the contiguous store stays a
    plain ``vsts``."""
    BM, BN, MX, MY = 16, 16, 4, 4

    @T.prim_func
    def k(CS: T.Tensor((BM // MX, BN // MY, MX, MY), "float32"), C: T.Tensor((BM, BN), "float32")):
        with T.Kernel(1):
            cs = T.alloc_shared((BM // MX, BN // MY, MX, MY), "float32")
            c = T.alloc_shared((BM, BN), "float32")
            for i, j in T.Parallel(BM, BN):
                c[i, j] = cs[i // MX, j // MY, i % MX, j % MY]

    plan = _single_segment(k)
    assert plan.stores[0].kind == AccessClass.CONTIGUOUS
    value = plan.stores[0].value
    assert value.recipe == "gather" and value.detail == AccessClass.PIECEWISE.value
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vgather2" in ops and "tl.simd.vsts" in ops


def test_strided_access():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((2 * N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((2 * N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[2 * i]

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "vload_strided" and value.detail == "stride=2"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vgather2" in ops


def test_indirect_gather_is_expressed():
    """Indirect access ``a[b[i]]`` has a ``BufferLoad`` inside its index, so the
    offset cannot be built from the loop axes alone.  It is classified
    ``INDIRECT`` and emitted as ``vgather2`` whose index register is the loaded
    per-lane ``b[i]`` vector (improvement 1)."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "int32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "int32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[b[i]]

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "gather" and value.detail == AccessClass.INDIRECT.value
    assert "tl.simd.vgather2" in _emitted_ops(emit_plan(plan))


def test_reversal_is_permute_gather():
    """``a[N - 1 - i]`` has Δ == -1 along ``i``: a bijective reversal, not a
    stride.  It is classified ``PERMUTE`` and lowered through the same per-lane
    index-vector channel (improvement 1 + 4)."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[N - 1 - i]

    plan = _single_segment(k)
    value = plan.stores[0].value
    assert value.recipe == "gather" and value.detail == AccessClass.PERMUTE.value
    # Numeric golden: every lane index in chunk 0 must be exactly 127, 126, ...
    # (P0-1: it must not be i-127).
    assert [_eval_lane(value, lane, plan) for lane in range(4)] == [127, 126, 125, 124]
    assert [_eval_lane(value, lane, plan, base=64) for lane in range(4)] == [63, 62, 61, 60]
    assert "tl.simd.vgather2" in _emitted_ops(emit_plan(plan))


def test_true_random_access_is_rejected():
    """``a[i // 3]`` has a non-constant Δ with a *non*-power-of-two divisor, so its
    per-lane element offset cannot be built in the index-vector channel (improvement 4:
    ``TRUE_RANDOM``); the vectorizer must reject rather than silently degrade."""
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i] = a[i // 3]

    with _rejects("plan.no_vector_axis"):
        _plan(k)


# ---------------------------------------------------------------------------
# Improvement 7: diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_list_every_candidate_axis_on_success():
    """A successful plan still records *every* axis' classification plus which one
    won, so a caller can see the full decision without re-deriving it (improvement 7)."""
    N = 64

    @T.prim_func
    def k(A: T.Tensor((N, N), "float32"), C: T.Tensor((N, N), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N, N), "float32")
            c = T.alloc_shared((N, N), "float32")
            for i, j in T.Parallel(N, N):
                c[i, j] = a[i, j]

    plan = _single_segment(k)
    assert any(line.startswith("axis 'i'") for line in plan.diagnostics)
    assert any(line.startswith("axis 'j'") for line in plan.diagnostics)
    assert any(line.startswith("selected vector axis") for line in plan.diagnostics)
    assert "diagnostics:" in render_plan(plan)


def test_diagnostics_are_attached_to_a_rejection():
    """When no axis works, the raised ``Unsupported`` carries the per-axis notes
    (improvement 7), not just a single flattened reason."""
    N = 64

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N,), "float32")
            for i in T.Parallel(N):
                c[i // 3] = a[i]  # the *store* index is not constructible

    with pytest.raises(Unsupported) as excinfo:
        _plan(k)
    notes = excinfo.value.diagnostics
    assert any("true-random store" in note for note in notes)
    assert any("not a power of two" in note for note in notes)


# ---------------------------------------------------------------------------
# §5 atomics / §6 complex control flow
# ---------------------------------------------------------------------------


def test_atomic_rejected():
    N = 128

    @T.prim_func
    def k(A: T.Tensor((N,), "float32"), C: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((1,), "float32")
            for i in T.Parallel(N):
                T.atomic_add(c[0], a[i])

    with pytest.raises(Unsupported, match="atomic"):
        _plan(k)


def test_nested_loop_rejected():
    """A lane-varying predicate on a shared scalar accumulator has no vector form.

    ``if a[tm, g_i] != 0: cnt[0] += 1`` increments ``cnt[0]`` once per satisfying
    ``(tm, g_i)``; vectorizing ``tm`` would make several lanes race on the same
    scalar. The tree planner rejects it at the leaf instead of emitting a single
    masked broadcast of ``cnt[0] + 1``.

    As in :func:`test_serial_shared_scalar_rejected`, this is a DSL mistake that
    P1 already hard-errors on; the case covers the planner's defensive check.
    """
    S, KD = 16, 16

    @T.prim_func
    def k(A: T.Tensor((S, KD), "float32"), C: T.Tensor((S, KD), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((S, KD), "float32")
            c = T.alloc_shared((S, KD), "float32")
            cnt = T.alloc_shared((1,), "int32")
            for tm in T.Parallel(S):
                cnt[0] = 0
                for g_i in T.serial(4):
                    if a[tm, g_i] != 0.0:
                        cnt[0] += 1
                        c[tm, g_i] = a[tm, g_i]

    with _rejects("plan.lane_varying_uniform_store"):
        _plan(k)


# ---------------------------------------------------------------------------
# §6 step 1: a directly nested, clean T.serial loop is absorbed as an extra candidate axis (S1 / S3)
# ---------------------------------------------------------------------------


def test_direct_serial_copy_absorbed():
    """S1: ``for i in P: for j in serial: c[i,j] = a[i,j]*2``. The serial body has no
    loop-carried dependence -> j is absorbed as a candidate axis; the store is
    contiguous along j -> j is chosen for vectorization (contiguous vld/vmuls/vsts)."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0

    plan = _single_segment(k)
    # ``KS=8 < lanes=64`` -> P1's composite contiguous axis flattens the whole nest into
    # one full-lane contiguous axis of 512 elements = exactly 8 full chunks. Flattening
    # **requires** the serial axis to have been absorbed into the iteration domain.
    assert plan.vector_axis.name.startswith("k_")
    assert plan.vector_axis.extent.value == MS * KS
    assert plan.stores[0].kind.value == "contiguous"
    assert plan.reduce is None
    stmt = emit_plan(plan)
    ops = _emitted_ops(stmt)
    # Full-lane contiguous: only vld/vsts, no predicated gather and no tail predicate.
    assert "tl.simd.vld" in ops and "tl.simd.vsts" in ops
    assert "tl.simd.vgather2" not in ops and "tl.simd.update_mask" not in ops
    assert not _over_reading_contiguous_load(stmt)


def test_direct_serial_predicated_absorbed():
    """S1 + predicate: an ``if`` inside the serial body lowers to a predicated store, and
    the mask varies per lane along the vector axis.

    ``KS=8 < lanes=64`` -> same as :func:`test_direct_serial_copy_absorbed`, using P1's
    composite contiguous axis; the predicate ``a[i, j] != 0`` becomes a vector compare
    over the whole lane axis after flattening (``vcmps``).
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    if a[i, j] != 0.0:
                        c[i, j] = a[i, j]

    plan = _single_segment(k)
    assert plan.vector_axis.name.startswith("k_")
    assert plan.stores[0].mask is not None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vcmps" in ops and "tl.simd.vsts" in ops
    assert "tl.simd.vgather2" not in ops


def test_flatten_is_skipped_when_a_load_still_needs_gather():
    """P1's read-side gate: flattening only rewrites indices into a decomposition of ``k``
    and **cannot eliminate** a gather.

    After flattening ``d[i, j] = g[i] - g[j]``, both reads are piecewise accesses
    (``g[k//32]`` / ``g[k%32]``) -> the composite axis does not take over, and the
    original single-axis path (``vadds`` + 1 gather, no ``vdup``) stays unchanged.
    """
    S, M = 32, 16

    @T.prim_func
    def k(G: T.Tensor((S,), "float32"), D: T.Tensor((M, S), "float32")):
        with T.Kernel(1):
            g = T.alloc_shared((S,), "float32")
            d = T.alloc_shared((M, S), "float32")
            for i in T.Parallel(M):
                for j in T.Parallel(S):
                    d[i, j] = g[i] - g[j]

    plan = _single_segment(k)
    assert [axis.name for axis in plan.axes] == ["i", "j"]
    assert plan.vector_axis.name == "j"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vgather2" in ops and "tl.simd.vsub" in ops
    assert "tl.simd.vdup" in ops  # uniform-side negation does not apply: ``g[i] - g[j]`` is scalar - vector


def test_direct_serial_reduction_absorbed():
    """S3: ``for i in P: for j in serial: s[i] += a[i,j]``.

    The inner serial axis has only 8 lanes (8/64); choosing it for a cross-lane
    reduction would waste 56 lanes. Axis scoring already normalizes by
    ``active/lanes`` -> the parallel axis i is chosen: ``s_vec += a[:, j]``, with the
    serial loop kept as a real loop (no ``vcadd`` needed)."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] += a[i, j]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vadd" in ops and "tl.simd.vgather2" in ops
    assert "tl.simd.vcadd" not in ops


def test_serial_shared_scalar_rejected():
    """S4 boundary: a lane-varying predicate on a **shared scalar** accumulator.

    ``if a[i, j] != 0: cnt[0] += 1`` increments ``cnt[0]`` once per satisfying
    lane; the per-lane condition has no vector form (a cross-lane count is not a
    mask), so the leaf is rejected.

    This is a DSL mistake: in the production pipeline P1 already hard-errors on
    the accumulating store that ignores the parallel axis (and on the non-injective
    scalar write), so P2 never sees the nest.  The case pins the planner's own
    defensive check (``build_plan`` is also a public test entry point).
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            cnt = T.alloc_shared((1,), "int32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    if a[i, j] != 0.0:
                        cnt[0] += 1
                        c[i, j] = a[i, j]

    with _rejects("plan.lane_varying_uniform_store"):
        _plan(k)


def test_serial_stencil_rejected():
    """Boundary: a cross-iteration read inside the serial body (stencil ``c[i,j-1]``) is not
    an accumulating self-read.

    P1/P2 no longer rejects it directly as a "non-absorbable shell": the shell is
    first unrolled per iteration (semantically equivalent), and the cross-address
    read is then identified as a stencil during axis selection -> still rejected, but
    with a more accurate reason."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = c[i, j - 1] + a[i, j]

    with _rejects("plan.no_vector_axis") as excinfo:
        _plan(k)
    assert "stencil" in str(excinfo.value)


# ---------------------------------------------------------------------------
# §6 step 2: multiple statements in a T.Parallel body with one clean T.serial in the middle (pre/post segments)
# ---------------------------------------------------------------------------


def test_serial_pre_statement_absorbed():
    """M2 pre-statement + serial: ``s[i]=0.0; for j in serial: s[i]+=a[i,j]``.

    Vectorize along the parallel axis i: the chunk loop wraps
    ``pre(s[i]=0) -> for j: main``, so pre runs once per chunk (not once per j
    iteration)."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                s[i] = 0.0
                for j in T.serial(KS):
                    s[i] += a[i, j]

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    assert [s.spec.buffer.name for s in plan.pre_stores] == ["s"]
    assert [s.kind for s in plan.pre_stores] == [AccessClass.CONTIGUOUS]
    assert plan.post_stores == []
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vadd" in ops and "tl.simd.vsts" in ops
    assert "tl.simd.vcadd" not in ops


def test_serial_post_statement_absorbed():
    """M3 serial + post-statement: ``for j in serial: s[i]+=a[i,j]; s[i]=s[i]*2``.

    Vectorize along the parallel axis i: the chunk loop wraps
    ``for j: main -> post``, and post runs once per chunk in the register/shared
    memory (``vmuls``) instead of going through a cross-lane reduction."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] += a[i, j]
                s[i] = s[i] * 2.0

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    assert plan.pre_stores == []
    assert [s.spec.buffer.name for s in plan.post_stores] == ["s"]
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vadd" in ops and "tl.simd.vmuls" in ops
    assert "tl.simd.vcadd" not in ops


def test_serial_sandwiched_statements_absorbed():
    """Pre + serial + post (sandwich): ``s[i]=0.0; for j: s[i]+=a[i,j]; s[i]=s[i]*2``.

    After vectorizing along the parallel axis i, the three parts still run in the
    order pre -> for j: body -> post."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            for i in T.Parallel(MS):
                s[i] = 0.0
                for j in T.serial(KS):
                    s[i] += a[i, j]
                s[i] = s[i] * 2.0

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    assert [s.spec.buffer.name for s in plan.pre_stores] == ["s"]
    assert [s.spec.buffer.name for s in plan.post_stores] == ["s"]
    # StoreRole: the plan (not the emitter) decides that the pre store feeds the
    # accumulator initial value and that the post owns the write-back.
    assert plan.pre_stores[0].role == "pre_folded"
    assert plan.stores[0].role == "post_superseded"
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vadd" in ops and "tl.simd.vmuls" in ops
    # Accumulator register promotion: the same-address pre folds into a ``vdup`` register
    # initial value, the body only updates the register each round (no per-round vld +
    # vsts), and post reads the register directly -> the whole segment has one write-back.
    assert ops.count("tl.simd.vsts") == 1
    assert "tl.simd.vld" not in ops
    text = str(emit_plan(plan))
    assert 'scope="local.var"' in text and "acc" in text


def test_serial_independent_store_group_absorbed():
    """M1 independent store group + serial: ``c[i,0]=1.0; for j: c[i,j]=a[i,j]*2``.

    Vectorize along the parallel axis i; ``c[i,0]`` is inevitably overwritten by the
    ``j=0`` round (the body does not read c) -> the pre write is dead and is removed
    at build time. The body is a strided write along i, using
    ``vgather2``/``vscatter``."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                c[i, 0] = 1.0
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0

    plan = _single_segment(k)
    assert plan.vector_axis.name == "i"
    assert plan.reduce is None
    assert plan.pre_stores == []  # the dead write was removed
    assert [s.kind for s in plan.stores] == [AccessClass.STRIDED]
    ops = _emitted_ops(emit_plan(plan))
    assert "tl.simd.vmuls" in ops and "tl.simd.vgather2" in ops and "tl.simd.vscatter" in ops


# ---------------------------------------------------------------------------
# §6 step 3: **multiple sibling** clean T.serial loops in a T.Parallel body -> VectorProgram
# ---------------------------------------------------------------------------


def test_multi_serial_independent_segments():
    """Two sibling clean serial loops (independent) -> each segment is vectorized
    independently, sharing the outer parallel axis i (wrapped once at the outermost
    level), and the two segments are emitted in source order."""
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        B: T.Tensor((MS, KS), "float32"),
        C: T.Tensor((MS, KS), "float32"),
        D: T.Tensor((MS, KS), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            d = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    d[i, l] = b[i, l] + 1.0

    program = _plan(k)
    assert isinstance(program, VectorProgram)
    # Both segments are pure elementwise -> each is flattened into a full-lane composite
    # axis (i x j / i x l, 512 elements).
    assert program.axes == []
    assert [seg.vector_axis.name for seg in program.segments] == ["k_i_j", "k_i_l"]
    assert [seg.chunks for seg in program.segments] == [8, 8]
    assert [seg.stores[0].spec.buffer.name for seg in program.segments] == ["c", "d"]
    assert all(seg.stores[0].kind == AccessClass.CONTIGUOUS for seg in program.segments)

    stmt = emit_plan(program)
    kinds = _for_kinds(stmt)
    assert tirx.ForKind.PARALLEL not in kinds
    # One full-lane chunk loop per segment; no serial-axis loop of 8/64 lanes.
    assert kinds.count(tirx.ForKind.SERIAL) == 2


def test_multi_serial_reduce_then_use():
    """Reduction segment + broadcast-use segment: both are vectorized along the parallel
    axis i (serial loops kept); the inter-segment dependence ``s[i]`` is per-lane, so
    source-order emission preserves the semantics."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    s[i] += a[i, j]
                for l in T.serial(KS):
                    c[i, l] = s[i] * 2.0

    program = _plan(k)
    assert isinstance(program, VectorProgram)
    assert len(program.segments) == 2
    assert [seg.vector_axis.name for seg in program.segments] == ["i", "i"]
    # The vector axis i already became the lane dimension: the program level must not wrap
    # `for i` again, or the whole segment would repeat 64 times.
    assert program.axes == []
    assert all(seg.reduce is None for seg in program.segments)
    stmt = emit_plan(program)
    assert "for i in range(64)" not in str(stmt)
    ops = _emitted_ops(stmt)
    assert "tl.simd.vadd" in ops and "tl.simd.vmuls" in ops
    assert "tl.simd.vcadd" not in ops


def test_multi_serial_compose_keeps_source_order():
    """Elementwise composition: a later segment reads ``c[i,l]`` written by an earlier one.
    Source-order emission guarantees "compute c before d", so the emitted instructions
    compute c before the arithmetic on d."""
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

    ops = _emitted_ops(emit_plan(_plan(k)))
    assert "tl.simd.vmuls" in ops and "tl.simd.vadds" in ops
    assert ops.index("tl.simd.vmuls") < ops.index("tl.simd.vadds")


def test_multi_serial_leading_statement_is_a_source_order_chunk():
    """A statement group before the first loop chain becomes its own chunk.

    Design A §7: the plan mirrors the IR, so ``s[i] = 0`` is ``Chunk[s]`` and the
    two loop chains are the following chunks, emitted in source order. (The
    single-chain case still folds an identity initializer into the reduction's
    ``pre`` through the whole-body fast path.)
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), S: T.Tensor((MS,), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            s = T.alloc_shared((MS,), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                s[i] = 0.0
                for j in T.serial(KS):
                    s[i] += a[i, j]
                for l in T.serial(KS):
                    c[i, l] = s[i] * 2.0

    program = _plan(k)
    assert isinstance(program, VectorProgram)
    assert [seg.stores[0].spec.buffer.name for seg in program.segments] == ["s", "s", "c"]
    assert program.segments[0].stores[0].value.recipe == "imm"
    assert [store.spec.buffer.name for store in program.segments[1].pre_stores] == []


def test_multi_serial_one_unclean_is_rejected():
    """One of the sibling serial loops has a cross-iteration read (stencil ``c[i,l-1]``)
    -> that segment cannot be absorbed.

    Per-iteration shell unrolling only happens when a shell is "non-absorbable"; after
    unrolling, the cross-address read is identified as a stencil during axis selection,
    and the whole nest is still rejected (no silent degradation), but the reason
    changes from "nested serial loop" to a specific stencil diagnostic."""
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    c[i, l] = c[i, l - 1] + a[i, l]

    with _rejects("plan.no_vector_axis") as excinfo:
        _plan(k)
    assert "stencil" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Level 0/1: unified return contract, equivalence guard, completed diagnostics (defects A/B)
# ---------------------------------------------------------------------------


def test_build_plan_always_returns_vector_program():
    """Level 1: ``build_plan`` **always** returns a :class:`VectorProgram` (1..N
    segments), so callers no longer branch on two types. In a single-segment case
    ``program.axes`` is the hoisted prefix of "parallel but not the vector axis"
    (here only the outer ``i``), while the vector axis ``j`` is strip-mined inside
    that segment."""
    M, K = 16, 128

    @T.prim_func
    def k(A: T.Tensor((M, K), "float32"), C: T.Tensor((M, K), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            c = T.alloc_shared((M, K), "float32")
            for i, j in T.Parallel(M, K):
                c[i, j] = a[i, j]

    program = _plan(k)
    assert isinstance(program, VectorProgram)
    assert len(program.segments) == 1
    assert [axis.name for axis in program.axes] == ["i"]
    assert all(axis.is_parallel for axis in program.axes)
    assert program.segments[0].vector_axis.name == "j"
    # Program-level diagnostics are **always non-empty** for one or many segments (symmetric
    # with the multi-segment path).
    assert any("single-segment program" in note for note in program.diagnostics)
    # The human-readable text of a 1-segment program has exactly the same form as the
    # "single-segment plan" (just one extra program-level summary line).
    assert render_plan(program) == render_plan(program.segments[0]) + ("\n  single-segment program: 1 segment, vector axis 'j'")
    assert not render_plan(program).startswith("program:")


def test_single_segment_program_emits_identically_to_single_plan():
    """Equivalence guard (step 2): the same IR goes through the "single-segment path"
    (``emit_plan(VectorPlan)``: the segment wraps all non-vector axes itself) and the
    "1-segment program unified path" (``emit_plan(VectorProgram)``: ``program.axes``
    are hoisted and wrapped once, then skipped inside the segment by variable
    identity); assert that the two emitted TIR trees are **structurally equivalent
    node by node** (``map_free_vars=True`` matches free variables by first occurrence,
    ignoring names).

    Covers three axis shapes; the third one, "the vector axis is the outer parallel
    axis", is the key case where ``hoisted`` must skip by **variable identity** (not
    by skipping all parallel axes) to stay equivalent -- otherwise ``j`` inside the
    segment would be wrapped again.
    """
    M, K = 64, 128

    @T.prim_func
    def multi_axis_copy(A: T.Tensor((M, K), "float32"), C: T.Tensor((M, K), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), "float32")
            c = T.alloc_shared((M, K), "float32")
            for i, j in T.Parallel(M, K):
                c[i, j] = a[i, j]

    MS, KS = 64, 8

    @T.prim_func
    def serial_absorbed(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0

    MD = 64

    @T.prim_func
    def outer_parallel_vector_axis(A: T.Tensor((MD, MD), "float32"), C: T.Tensor((MD, MD), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((MD, MD), "float32")
            c = T.alloc_shared((MD, MD), "float32")
            for i, j in T.Parallel(MD, MD):
                c[j, i] = a[i, j]  # joint scoring picks the outer axis i (making the store contiguous)

    cases = [
        ("multi-axis copy", multi_axis_copy, "j", ["i"]),
        # The innermost dimension of ``(64, 8)`` has only 8 lanes -> P1 flattens it into a
        # single axis, leaving no outer parallel axis to hoist.
        ("serial absorbed", serial_absorbed, "k_i_j", []),
        ("outer parallel vector axis", outer_parallel_vector_axis, "i", []),
    ]
    for name, func, vector_axis, outer_axes in cases:
        program = _plan(func)
        assert isinstance(program, VectorProgram), name
        assert len(program.segments) == 1, name
        assert program.segments[0].vector_axis.name == vector_axis, name
        assert [axis.name for axis in program.axes] == outer_axes, name

        direct = emit_plan(program.segments[0])  # single-segment path: the segment wraps all axes
        unified = emit_plan(program)  # unified path: program.axes hoisted and wrapped once
        assert tvm.ir.structural_equal(direct, unified, map_free_vars=True), name
        # The emitted recipe sequence and loop kinds also match item by item (a stronger
        # behavioral signal).
        assert _emitted_ops(direct) == _emitted_ops(unified), name
        assert _for_kinds(direct) == _for_kinds(unified), name


def test_multi_segment_program_diagnostics_are_populated():
    """Defect A: the multi-segment path no longer produces plans with "always-empty
    diagnostics" -- the program level has a summary, and each segment has its own
    per-axis classification and "selected vector axis" explanation, isomorphic to
    ``plan.diagnostics`` on the single-segment path."""
    MS, KS = 64, 8

    @T.prim_func
    def k(
        A: T.Tensor((MS, KS), "float32"),
        B: T.Tensor((MS, KS), "float32"),
        C: T.Tensor((MS, KS), "float32"),
        D: T.Tensor((MS, KS), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            b = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            d = T.alloc_shared((MS, KS), "float32")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    d[i, l] = b[i, l] + 1.0

    program = _plan(k)
    assert isinstance(program, VectorProgram)
    assert len(program.segments) == 2
    assert any("multi-segment program" in note for note in program.diagnostics)
    for segment in program.segments:
        assert any("axis 'i'" in note for note in segment.diagnostics)
        assert any(f"'{segment.vector_axis.name}'" in note for note in segment.diagnostics)
        assert any("segment vector axis" in note or "flattened segment" in note for note in segment.diagnostics)


def test_multi_segment_failure_reason_is_preserved():
    """Defect B: when one segment fails to build, the failure is **no longer swallowed** --
    the segment-level reason is merged into the final ``Unsupported.diagnostics``
    through ``notes``, instead of only the generic "nested serial loop" left after
    falling back to the single-segment path.

    The second segment writes a ``bool`` target (``T.simd`` has no bool vector
    store) -> that segment raises Unsupported during store planning; the whole nest
    is still rejected as before, but the reason is visible in the diagnostics.
    """
    MS, KS = 64, 8

    @T.prim_func
    def k(A: T.Tensor((MS, KS), "float32"), C: T.Tensor((MS, KS), "bool")):
        with T.Kernel(1):
            a = T.alloc_shared((MS, KS), "float32")
            c = T.alloc_shared((MS, KS), "float32")
            flags = T.alloc_shared((MS, KS), "bool")
            for i in T.Parallel(MS):
                for j in T.serial(KS):
                    c[i, j] = a[i, j] * 2.0
                for l in T.serial(KS):
                    flags[i, l] = a[i, l] > 0.0

    with _rejects("unsupported") as excinfo:
        _plan(k)
    notes = excinfo.value.diagnostics
    assert any(note.startswith("at ") for note in notes), notes
    assert "not supported" in str(excinfo.value), str(excinfo.value)
