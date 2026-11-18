"""Unified P1→P2→P3 contract tests over the per-Parallel ``tl.vsimd`` carriers.

The chain vectorizes ``T.Parallel`` loops inside a ``SIMD_VF`` and nothing else:

* P1 attaches ``tl.vsimd.region`` (``tl.vsimd.Region``) to every top-level
  Parallel;
* P2 plans each LIFTED region and attaches ``tl.vsimd.plan``
  (``tl.parallel_vectorize.VSimdRegionPlan``) to the **same loop**; a LIFTED
  region the planner cannot handle fails the compilation;
* P3 replaces every loop carrying a plan and strips both annotations
  unconditionally; a plan it cannot emit or an incompatible payload also fails
  the compilation.

Tile-ops outside a Parallel (top-level in the VF) are not part of this chain:
they keep their statements and belong to the regular tile-op lowering.
"""

import pytest
import tilelang
import tilelang.ascend.language as T
import tvm
from tilelang.ascend import transform as at

REGION_KEY = "tl.vsimd.region"
PLAN_KEY = "tl.vsimd.plan"


def _regions(mod):
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and REGION_KEY in node.annotations:
            found.append(node.annotations[REGION_KEY])

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    print(f"\nregions:\n{found}")
    return found


def _plan_pairs(mod):
    """``(For, plan)`` pairs: P2 attaches its plan to the loop it planned."""
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and PLAN_KEY in node.annotations:
            found.append((node, node.annotations[PLAN_KEY]))

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    print(f"\nplans:\n{found}")
    return found


def _plans(mod):
    return [plan for _, plan in _plan_pairs(mod)]


def _simd_ops(mod):
    ops = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            name = str(getattr(node.op, "name", ""))
            if name.startswith("tl.simd."):
                ops.append(name)

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return ops


def _if_count(mod):
    """Number of statement-level ``if`` nodes left in the body."""
    count = 0

    def visit(node):
        nonlocal count
        if isinstance(node, tvm.tirx.IfThenElse):
            count += 1

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return count


def _strip_vsimd_annotations(stmt):
    """Drop the P1/P2 annotations so a body can be compared with its source."""

    def visit(node):
        if isinstance(node, tvm.tirx.For) and (REGION_KEY in dict(node.annotations) or PLAN_KEY in dict(node.annotations)):
            annotations = {key: value for key, value in dict(node.annotations).items() if key not in (REGION_KEY, PLAN_KEY)}
            return tvm.tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                annotations,
                node.step,
            )
        return None

    return tvm.tirx.stmt_functor.ir_transform(stmt, None, visit)


def _lift(func):
    print(f"\nlift input:\n{func}")
    mod = at.AscendLiftVSimd()(tvm.IRModule({"main": func}))
    print(f"\nlift:\n{mod}")
    return mod


def _plan(mod):
    mod = at.AscendPlanVSimd()(mod)
    print(f"\nplan:\n{mod}")
    return mod


def _emit(mod):
    mod = at.AscendEmitVSimd()(mod)
    print(f"\nemit:\n{mod}")
    return mod


@T.prim_func
def _simple(A: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
    with T.Kernel(1):
        a_ub = T.alloc_shared((64,), "float32")
        c_ub = T.alloc_shared((64,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            for i in T.Parallel(64):
                c_ub[i] = a_ub[i] + 1.0
        T.copy(c_ub, C)


@T.prim_func
def _unsupported_dtype(A: T.Tensor((64,), "float64"), C: T.Tensor((64,), "float64")):
    """P1 can prove the nest independent, but P2 has no fp64 vector recipe."""
    with T.Kernel(1):
        a_ub = T.alloc_shared((64,), "float64")
        c_ub = T.alloc_shared((64,), "float64")
        T.copy(A, a_ub)
        with T.SimdVF():
            for i in T.Parallel(64):
                c_ub[i] = a_ub[i] * 2.0
        T.copy(c_ub, C)


@T.prim_func
def _two_vf(A: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
    with T.Kernel(1):
        a_ub = T.alloc_shared((64,), "float32")
        c_ub = T.alloc_shared((64,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            for i in T.Parallel(64):
                c_ub[i] = a_ub[i] + 1.0
            for i in T.Parallel(64):
                a_ub[i] = a_ub[i] * 2.0
        with T.SimdVF():
            for i in T.Parallel(64):
                c_ub[i] = c_ub[i] + a_ub[i]
        T.copy(c_ub, C)


@T.prim_func
def _partial(A: T.Tensor((96,), "float32"), D: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
    with T.Kernel(1):
        a_ub = T.alloc_shared((96,), "float32")
        d_ub = T.alloc_shared((64,), "float32")
        c_ub = T.alloc_shared((64,), "float32")
        T.copy(A, a_ub)
        T.copy(D, d_ub)
        with T.SimdVF():
            for i in T.Parallel(64):
                a_ub[i] = a_ub[i + 32]
            for i in T.Parallel(64):
                c_ub[i] = d_ub[i] + 1.0
        T.copy(c_ub, C)


@T.prim_func
def _fragment_fallback(A: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
    with T.Kernel(1):
        a_ub = T.alloc_shared((64,), "float32")
        c_ub = T.alloc_shared((64,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            frag = T.alloc_fragment((64,), "float32")
            for i in T.Parallel(64):
                frag[i] = a_ub[i] + 1.0
            for i in T.Parallel(64):
                c_ub[i] = frag[i] * 2.0
        T.copy(c_ub, C)


@T.prim_func
def _fragment_mixed(A: T.Tensor((64,), "float32"), D: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
    with T.Kernel(1):
        a_ub = T.alloc_shared((64,), "float32")
        d_ub = T.alloc_shared((64,), "float32")
        c_ub = T.alloc_shared((64,), "float32")
        T.copy(A, a_ub)
        T.copy(D, d_ub)
        with T.SimdVF():
            frag = T.alloc_fragment((64,), "float32")
            for i in T.Parallel(64):
                frag[i] = a_ub[i] + 1.0
            for i in T.Parallel(64):
                c_ub[i] = d_ub[i] * 2.0
        T.copy(c_ub, C)


@T.prim_func
def _vf_tileops(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128,), "float32")):
    """VF-level copy/reduce sit outside any Parallel: not this chain's input."""
    with T.Kernel(1):
        s_ub = T.alloc_shared((128, 128), "float32")
        m_old = T.alloc_shared((128,), "float32")
        m_new = T.alloc_shared((128,), "float32")
        T.copy(A, s_ub)
        with T.SimdVF():
            T.copy(m_old, m_new)
            T.reduce_max(s_ub, m_new, dim=1, clear=False)
            for i in T.Parallel(128):
                m_old[i] = m_new[i] + 1.0
        T.copy(m_old, C)


@T.prim_func
def _guarded(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
    with T.Kernel(1):
        a_ub = T.alloc_shared((128,), "float32")
        c_ub = T.alloc_shared((128,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            for i in T.Parallel(128):
                if a_ub[i] > 0.0:
                    c_ub[i] = a_ub[i] + 1.0
                else:
                    c_ub[i] = 0.0
        T.copy(c_ub, C)


@T.prim_func
def _guarded_single_arm(
    A: T.Tensor((128,), "float32"),
    C: T.Tensor((128,), "float32"),
    D: T.Tensor((128,), "float32"),
):
    with T.Kernel(1):
        a_ub = T.alloc_shared((128,), "float32")
        c_ub = T.alloc_shared((128,), "float32")
        d_ub = T.alloc_shared((128,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            for i in T.Parallel(128):
                c_ub[i] = a_ub[i] + 1.0
                if i < 96:
                    d_ub[i] = a_ub[i] * 2.0
        T.copy(c_ub, C)
        T.copy(d_ub, D)


@T.prim_func
def _guarded_mask_eq(
    A: T.Tensor((128,), "float32"),
    B: T.Tensor((128,), "float32"),
    C: T.Tensor((128,), "float32"),
):
    """``mask == mask`` has no ``vcmp`` form: the predicate truth table is used."""
    with T.Kernel(1):
        a_ub = T.alloc_shared((128,), "float32")
        b_ub = T.alloc_shared((128,), "float32")
        c_ub = T.alloc_shared((128,), "float32")
        T.copy(A, a_ub)
        T.copy(B, b_ub)
        with T.SimdVF():
            for i in T.Parallel(128):
                if (a_ub[i] > 0.0) == (b_ub[i] > 0.0):
                    c_ub[i] = a_ub[i]
        T.copy(c_ub, C)


@T.prim_func
def _serial_shell(A: T.Tensor((8, 64), "float32"), C: T.Tensor((8,), "float32")):
    """A scalar shell inside the Parallel: last write wins, no vector axis."""
    with T.Kernel(1):
        a_ub = T.alloc_shared((8, 64), "float32")
        c_ub = T.alloc_shared((8,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            for i in T.Parallel(8):
                for k in T.serial(4):
                    c_ub[i] = a_ub[i, k]
        T.copy(c_ub, C)


@T.prim_func
def _serial_shell_reduction(A: T.Tensor((8, 64), "float32"), C: T.Tensor((8,), "float32")):
    """A shell that accumulates along the counter is absorbed as a reduction axis."""
    with T.Kernel(1):
        a_ub = T.alloc_shared((8, 64), "float32")
        c_ub = T.alloc_shared((8,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            for i in T.Parallel(8):
                for k in T.serial(64):
                    c_ub[i] = c_ub[i] + a_ub[i, k]
        T.copy(c_ub, C)


@T.prim_func
def _parallel_under_serial_shell(A: T.Tensor((4, 128), "float32"), C: T.Tensor((128,), "float32")):
    """A ``T.serial`` shell *around* the Parallel stays in place, and its
    iterations read back what the previous one stored (D2 of the device
    report), so the emitted body must end with a VST->VLD barrier."""
    with T.Kernel(1):
        a_ub = T.alloc_shared((4, 128), "float32")
        c_ub = T.alloc_shared((128,), "float32")
        T.copy(A, a_ub)
        T.fill(c_ub, 0.0)
        with T.SimdVF():
            for k in T.serial(4):
                for i in T.Parallel(128):
                    c_ub[i] = c_ub[i] + a_ub[k, i]
        T.copy(c_ub, C)


@T.prim_func
def _parallel_under_branch(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32"), flag: T.int32):
    """A Parallel under a branch is not a top-level nest: P1 leaves it alone."""
    with T.Kernel(1):
        a_ub = T.alloc_shared((128,), "float32")
        c_ub = T.alloc_shared((128,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            if flag > 0:
                for i in T.Parallel(128):
                    c_ub[i] = a_ub[i] + 1.0
        T.copy(c_ub, C)


@T.prim_func
def _guarded_divergent(
    A: T.Tensor((128,), "float32"),
    C: T.Tensor((128,), "float32"),
    D: T.Tensor((128,), "float32"),
):
    with T.Kernel(1):
        a_ub = T.alloc_shared((128,), "float32")
        c_ub = T.alloc_shared((128,), "float32")
        d_ub = T.alloc_shared((128,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            for i in T.Parallel(128):
                if a_ub[i] > 0.0:
                    c_ub[i] = a_ub[i]
                else:
                    d_ub[i] = 1.0
        T.copy(c_ub, C)
        T.copy(d_ub, D)


@T.prim_func
def _guarded_same_buffer_disjoint_arms(
    A: T.Tensor((128,), "float32"),
    C: T.Tensor((256,), "float32"),
):
    """Both arms write the same buffer, but to disjoint parity classes."""
    with T.Kernel(1):
        a_ub = T.alloc_shared((128,), "float32")
        c_ub = T.alloc_shared((256,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            for i in T.Parallel(128):
                if a_ub[i] > 0.0:
                    c_ub[2 * i] = 1.0
                else:
                    c_ub[2 * i + 1] = 2.0
        T.copy(c_ub, C)


@T.prim_func
def _guarded_same_buffer_overlapping_arms(
    A: T.Tensor((128,), "float32"),
    C: T.Tensor((256,), "float32"),
):
    """``2 * i`` and ``i`` intersect across lanes: a real WAW, not just per-lane."""
    with T.Kernel(1):
        a_ub = T.alloc_shared((128,), "float32")
        c_ub = T.alloc_shared((256,), "float32")
        T.copy(A, a_ub)
        with T.SimdVF():
            for i in T.Parallel(128):
                if a_ub[i] > 0.0:
                    c_ub[2 * i] = 1.0
                else:
                    c_ub[i] = 2.0
        T.copy(c_ub, C)


_TAIL_N, _TAIL_BN = 1000, 256


@T.prim_func
def _tail(
    A: T.Tensor((_TAIL_N + _TAIL_BN,), "float32"),
    C: T.Tensor((_TAIL_N + _TAIL_BN,), "float32"),
):
    with T.Kernel(T.ceildiv(_TAIL_N, _TAIL_BN)) as bx:
        a_ub = T.alloc_shared((_TAIL_N + _TAIL_BN,), "float32")
        c_ub = T.alloc_shared((_TAIL_N + _TAIL_BN,), "float32")
        T.copy(A, a_ub)
        t = T.min(_TAIL_N - bx * _TAIL_BN, _TAIL_BN)
        with T.SimdVF():
            T.attr(t > 0, "tl.assume", "tail positive")
            for i in T.Parallel(t):
                c_ub[i] = a_ub[i] + 1.0
        T.copy(c_ub, C)


# ---------------------------------------------------------------------------
# carrier contract
# ---------------------------------------------------------------------------


def test_plan_is_attached_to_the_parallel_it_planned():
    """P2 writes the plan onto the same loop that carries the P1 region."""
    lifted = _lift(_simple)
    regions = _regions(lifted)
    assert len(regions) == 1 and regions[0].status == "LIFTED"

    planned = _plan(lifted)
    pairs = _plan_pairs(planned)
    assert len(pairs) == 1
    loop, plan = pairs[0]
    assert loop.kind == tvm.tirx.ForKind.PARALLEL
    assert REGION_KEY in dict(loop.annotations), "the P1 region stays for P3"
    assert int(plan.version) == 1
    assert (int(plan.vf_index), int(plan.region_index)) == (0, 0)
    assert len(plan.program.segments) == 1
    assert "tl.simd." not in str(planned["main"]), "P2 never emits"

    emitted = _emit(planned)
    body = str(emitted["main"].body)
    assert "T.parallel" not in body
    assert "tl.simd.vld" in body and "tl.simd.vsts" in body
    assert "tl.vsimd.region" not in str(emitted["main"])
    assert "tl.vsimd.plan" not in str(emitted["main"])


def test_lift_and_plan_are_metadata_only():
    lifted = _lift(_simple)
    planned = _plan(lifted)
    tvm.ir.assert_structural_equal(_strip_vsimd_annotations(lifted["main"].body), _simple.body, map_free_vars=True)
    tvm.ir.assert_structural_equal(_strip_vsimd_annotations(planned["main"].body), _simple.body, map_free_vars=True)


def test_printed_annotations_show_their_content():
    """``print(mod)`` renders both carriers inline instead of a metadata blob."""
    text = str(_plan(_lift(_simple)))
    assert 'metadata["tl.vsimd.Region"]' not in text
    assert '"tl.vsimd.region": {"status": "LIFTED"' in text
    assert '"axes": ["i in [0, 64)"]' in text
    assert '"accesses": ["W c_ub[i] = a_ub[i] + T.float32(1.0)"' in text
    assert '"tl.vsimd.plan": {"version": 1' in text
    assert '"program": {"segments": [' in text


def test_top_level_tileops_are_left_to_the_tileop_lowering():
    """Only the Parallel is planned; VF-level copy/reduce keep their statements."""
    lifted = _lift(_vf_tileops)
    assert len(_regions(lifted)) == 1, "just the Parallel gets a region"

    planned = _plan(lifted)
    assert len(_plans(planned)) == 1, "just the Parallel gets a plan"
    body = str(planned["main"].body)
    assert "T.ascend_copy(T.region(m_old" in body
    assert "T.reduce(" in body

    emitted = _emit(planned)
    body = str(emitted["main"].body)
    assert "T.parallel" not in body, "the Parallel is vectorized"
    assert "tl.simd.vadds" in body
    assert "T.ascend_copy(T.region(m_old" in body, "the VF copy is untouched"
    assert "T.reduce(" in body, "the VF reduce is untouched"
    assert "tl.vsimd.region" not in str(emitted["main"])
    assert "tl.vsimd.plan" not in str(emitted["main"])


# ---------------------------------------------------------------------------
# guarded Parallel bodies
# ---------------------------------------------------------------------------


def test_guarded_same_address_arms_become_a_select():
    lifted = _lift(_guarded)
    assert [op.kind for op in _regions(lifted)[0].ops] == ["if"]

    planned = _plan(lifted)
    pairs = _plan_pairs(planned)
    assert len(pairs) == 1
    plan = pairs[0][1].program
    assert plan.segments[0].stores[0].mask is None, "select, not predication"

    emitted = _emit(planned)
    body = str(emitted["main"].body)
    assert "T.parallel" not in body
    assert _if_count(emitted) == 0
    assert "tl.simd.vsel" in body and "tl.simd.vcmps" in body


def test_guarded_single_arm_with_a_sibling_store_is_predicated():
    lifted = _lift(_guarded_single_arm)
    assert [op.kind for op in _regions(lifted)[0].ops] == ["store", "if"]

    planned = _plan(lifted)
    plan = _plan_pairs(planned)[0][1].program
    stores = [store for segment in plan.segments for store in segment.stores]
    assert [store.spec.buffer.name for store in stores] == ["c_ub", "d_ub"]
    assert stores[0].mask is None
    assert stores[1].mask is not None

    emitted = _emit(planned)
    body = str(emitted["main"].body)
    assert "T.parallel" not in body
    assert _if_count(emitted) == 0
    assert "tl.simd.vadds" in body and "tl.simd.vcmps" in body


def test_guarded_mask_equality_uses_the_predicate_truth_table():
    """``(a > 0) == (b > 0)`` is a mask EQ: ``pnot(pxor(a, b))``."""
    lifted = _lift(_guarded_mask_eq)
    assert [op.kind for op in _regions(lifted)[0].ops] == ["if"]

    planned = _plan(lifted)
    pairs = _plan_pairs(planned)
    assert len(pairs) == 1
    stores = [store for segment in pairs[0][1].program.segments for store in segment.stores]
    assert len(stores) == 1 and stores[0].mask is not None

    emitted = _emit(planned)
    body = str(emitted["main"].body)
    assert "T.parallel" not in body
    assert "tl.simd.pxor" in body and "tl.simd.pnot" in body
    assert "tl.simd.vcmp(" not in body, "no vector compare between two masks"


def test_serial_shell_is_unrolled_iteration_by_iteration():
    """A shell whose writes do not vary along the counter keeps its semantics."""
    lifted = _lift(_serial_shell)
    region = _regions(lifted)[0]
    assert [op.kind for op in region.ops] == ["serial"]
    assert [axis.var.name for axis in region.ops[0].axes] == ["k"], "the shell counter is not a region axis"
    assert [axis.var.name for axis in region.axes] == ["i"]

    planned = _plan(lifted)
    assert len(_plans(planned)) == 1
    emitted = _emit(planned)
    body = str(emitted["main"].body)
    assert "T.parallel" not in body
    assert "T.serial" not in body, "the shell is unrolled at plan time"
    assert body.count("tl.simd.vsts") == 4, "one predicated-free store per iteration"


def test_serial_shell_reduction_is_absorbed_as_an_axis():
    """An accumulating shell is a reduction along the counter, not an unroll."""
    lifted = _lift(_serial_shell_reduction)
    assert [op.kind for op in _regions(lifted)[0].ops] == ["serial"]

    planned = _plan(lifted)
    assert len(_plans(planned)) == 1
    emitted = _emit(planned)
    body = str(emitted["main"].body)
    assert "T.parallel" not in body
    assert "T.serial" not in body
    assert "tl.simd.vsts" in body
    assert "tl.simd.mem_bar" not in body, "no scalar shell, no barrier"


def test_parallel_under_a_serial_shell_closes_each_iteration():
    """The enclosing serial shell reads back its own store every iteration."""
    lifted = _lift(_parallel_under_serial_shell)
    assert [region.status for region in _regions(lifted)] == ["LIFTED"]

    emitted = _emit(_plan(lifted))
    body = str(emitted["main"].body)
    assert "T.parallel" not in body
    assert "for k in range(4):" in body, "the outer shell stays a serial loop"
    assert body.index("for k in range(4):") < body.index("tl.simd.mem_bar")
    assert body.count("tl.simd.mem_bar") == 1
    assert "VST_VLD" in body


def test_parallel_under_a_branch_is_not_lifted():
    """Not a top-level nest: P1 attaches no region and logs why."""
    lifted = _lift(_parallel_under_branch)
    assert _regions(lifted) == []
    planned = _plan(lifted)
    assert _plans(planned) == []
    emitted = _emit(planned)
    assert "T.parallel" in str(emitted["main"].body)


def test_guarded_divergent_arms_become_predicated_stores():
    """Different write targets per arm: each arm keeps its own predicated store."""
    lifted = _lift(_guarded_divergent)
    assert [region.status for region in _regions(lifted)] == ["LIFTED"]

    planned = _plan(lifted)
    pairs = _plan_pairs(planned)
    assert len(pairs) == 1, "divergent arms are plannable"
    stores = [store for segment in pairs[0][1].program.segments for store in segment.stores]
    assert [store.spec.buffer.name for store in stores] == ["c_ub", "d_ub"]
    assert all(store.mask is not None for store in stores), "both arms are predicated"

    emitted = _emit(planned)
    body = str(emitted["main"].body)
    assert "T.parallel" not in body
    assert _if_count(emitted) == 0
    assert "tl.simd.vcmps" in body and "tl.simd.pnot" in body
    assert _regions(emitted) == []
    assert _plans(emitted) == []


def test_guarded_same_buffer_disjoint_arms_are_vectorized():
    """``c[2i]`` / ``c[2i+1]`` never collide: guard-aware legality lifts the nest."""
    lifted = _lift(_guarded_same_buffer_disjoint_arms)
    assert [region.status for region in _regions(lifted)] == ["LIFTED"]

    planned = _plan(lifted)
    pairs = _plan_pairs(planned)
    assert len(pairs) == 1
    stores = [store for segment in pairs[0][1].program.segments for store in segment.stores]
    assert [store.spec.buffer.name for store in stores] == ["c_ub", "c_ub"]
    assert all(store.mask is not None for store in stores), "both arms are predicated"

    emitted = _emit(planned)
    body = str(emitted["main"].body)
    assert "T.parallel" not in body
    assert "tl.simd.vscatter" in body and "tl.simd.pnot" in body


def test_guarded_same_buffer_overlapping_arms_keep_the_original_loop():
    """``c[2i]`` and ``c[i]`` intersect across lanes: the nest must stay scalar."""
    lifted = _lift(_guarded_same_buffer_overlapping_arms)
    region = _regions(lifted)[0]
    assert region.status == "OPAQUE"
    assert region.reason_code == "ILLEGAL"

    planned = _plan(lifted)
    assert _plans(planned) == []

    emitted = _emit(planned)
    body = str(emitted["main"].body)
    assert "T.parallel" in body, "P3 leaves the rejected loop untouched"
    assert "tl.simd.vscatter" not in body


def test_symbolic_tail_context_reaches_the_planner():
    lifted = _lift(_tail)
    region = _regions(lifted)[0]
    assert region.status == "LIFTED"
    assert [str(binding.var) for binding in region.bindings] == ["t"]
    assert [str(condition) for condition in region.assumptions] == ["t > 0"]
    assert [str(axis.extent) for axis in region.axes] == ["t"]

    planned = _plan(lifted)
    assert len(_plans(planned)) == 1

    emitted = _emit(planned)
    body = str(emitted["main"].body)
    assert "T.parallel" not in body
    assert "update_mask" in body and "T.min(" in body


# ---------------------------------------------------------------------------
# independence and degradation
# ---------------------------------------------------------------------------


def test_multiple_vfs_and_parallels_are_planned_independently():
    lifted = _lift(_two_vf)
    assert [(int(region.vf_index), region.status) for region in _regions(lifted)] == [
        (0, "LIFTED"),
        (0, "LIFTED"),
        (1, "LIFTED"),
    ]

    planned = _plan(lifted)
    assert [(int(plan.vf_index), int(plan.region_index)) for plan in _plans(planned)] == [(0, 0), (0, 1), (1, 0)]
    assert len(_plan_pairs(planned)) == 3, "one plan per loop, on that loop"

    emitted = _emit(planned)
    body = str(emitted["main"].body)
    assert "T.parallel" not in body
    assert "tl.simd.vadds" in body and "tl.simd.vmuls" in body
    assert _regions(emitted) == [] and _plans(emitted) == []


def test_partial_lift_plans_only_the_lifted_parallel():
    lifted = _lift(_partial)
    assert [region.status for region in _regions(lifted)] == ["OPAQUE", "LIFTED"]

    planned = _plan(lifted)
    assert len(_plans(planned)) == 1

    emitted = _emit(planned)
    body = str(emitted["main"].body)
    assert "tl.simd.vadds" in body, "the lifted Parallel is vectorized"
    assert "T.parallel" in body, "the opaque Parallel keeps its original form"
    assert _regions(emitted) == []
    assert _plans(emitted) == []


def test_lifted_region_p2_cannot_plan_fails():
    """P1 approved the region; leaving it scalar inside the VF is a hard error."""
    lifted = _lift(_unsupported_dtype)
    assert [region.status for region in _regions(lifted)] == ["LIFTED"]

    with pytest.raises(tvm.error.InternalError, match="P2 cannot vectorize"):
        _plan(lifted)


def test_incompatible_plan_payload_fails():
    """P3 must not silently ignore a plan payload it cannot trust."""
    planned = _plan(_lift(_simple))

    def rebuild(node):
        if isinstance(node, tvm.tirx.For) and PLAN_KEY in dict(node.annotations):
            return tvm.tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                {PLAN_KEY: tvm.tirx.IntImm("int64", 0)},
                node.step,
            )
        return None

    mutated = tvm.tirx.stmt_functor.ir_transform(planned["main"].body, None, rebuild)
    func = planned["main"].with_body(mutated)
    with pytest.raises(tvm.error.InternalError, match="malformed tl.vsimd.plan"):
        _emit(tvm.IRModule({"main": func}))


def test_fragment_access_is_not_vectorized():
    lifted = _lift(_fragment_fallback)
    regions = _regions(lifted)
    assert [region.status for region in regions] == ["OPAQUE", "OPAQUE"]
    assert all(region.reason_code == "UNSUPPORTED" for region in regions)
    assert all("fragment" in region.reason for region in regions)

    planned = _plan(lifted)
    assert _plans(planned) == []

    emitted = _emit(planned)
    assert _simd_ops(emitted) == []
    assert "T.parallel" in str(emitted["main"].body)
    assert _regions(emitted) == []
    assert _plans(emitted) == []


def test_fragment_rejection_is_per_parallel():
    lifted = _lift(_fragment_mixed)
    regions = _regions(lifted)
    assert [region.status for region in regions] == ["OPAQUE", "LIFTED"]
    assert regions[1].reason_code == ""

    planned = _plan(lifted)
    pairs = _plan_pairs(planned)
    assert len(pairs) == 1, "only the fragment-free Parallel is planned"
    loop, plan = pairs[0]
    assert plan.program.segments[0].stores[0].spec.buffer.name == "c_ub"
    # The planned loop is the second one (region_index 1).
    assert int(plan.region_index) == 1
    assert loop is not None

    emitted = _emit(planned)
    assert _simd_ops(emitted), "the fragment-free Parallel is vectorized"
    assert "T.parallel" in str(emitted["main"].body), "the fragment nest stays"
    assert _regions(emitted) == []
    assert _plans(emitted) == []


def test_unplanned_annotated_loop_is_not_emitted():
    """If a pass drops the plan annotation, P3 emits nothing for that loop."""
    planned = _plan(_lift(_simple))

    def rebuild(node):
        if isinstance(node, tvm.tirx.For) and PLAN_KEY in dict(node.annotations):
            return tvm.tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
            )
        return None

    mutated = tvm.tirx.stmt_functor.ir_transform(planned["main"].body, None, rebuild)
    func = planned["main"].with_body(mutated)
    emitted = _emit(tvm.IRModule({"main": func}))
    assert _simd_ops(emitted) == []
    assert "T.parallel" in str(emitted["main"].body)
    assert _regions(emitted) == []
    assert _plans(emitted) == []


def test_p3_without_p2_only_cleans_annotations():
    lifted = _lift(_simple)
    emitted = _emit(lifted)
    assert "T.parallel" in str(emitted["main"].body)
    assert _simd_ops(emitted) == []
    assert _regions(emitted) == []
    assert _plans(emitted) == []

    # P2/P3 on an already emitted module is a no-op (idempotent chain).
    again = _emit(_plan(emitted))
    tvm.ir.assert_structural_equal(again["main"], emitted["main"])


# ---------------------------------------------------------------------------
# pipeline wiring
# ---------------------------------------------------------------------------


def test_pipeline_uses_the_region_path_by_default():
    with tvm.target.Target("ascend"), tilelang.transform.PassContext():
        out = tilelang.lower(_simple)
    assert "simd_inst::" in out.kernel_source
    assert "tl.vsimd.region" not in out.kernel_source
    assert "tl.vsimd.plan" not in out.kernel_source


def test_p1_disabled_leaves_the_original_loop():
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ASCEND_SIMD_LIFT: False}):
        lifted = _lift(_simple)
        assert _regions(lifted) == []
        planned = _plan(lifted)
        assert _plans(planned) == []
        emitted = _emit(planned)
    assert _simd_ops(emitted) == []
    tvm.ir.assert_structural_equal(emitted["main"].body, _simple.body, map_free_vars=True)


if __name__ == "__main__":
    tilelang.testing.main()
