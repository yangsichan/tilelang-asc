"""Shared helpers for the offline parallel_vectorize P2/P3 tests.

Extracted verbatim from the former single-file suite; the test cases themselves
live in ``test_ascend_parallel_vectorize_*.py``.
"""

from __future__ import annotations

import contextlib
import os

import pytest

import tilelang
import tilelang.ascend.language as T
import tvm
from tvm import arith, tirx

from tilelang.ascend import transform as at
from tilelang.ascend.analysis.parallel_vectorize import (
    Unsupported,
    VectorProgram,
    build_plan,
    emit_plan,
    finite_difference,
    render_plan,
)
from tilelang.layout import (
    make_ascend_compact_nz_layout,
)


# ---------------------------------------------------------------------------
# IR dump switch
# ---------------------------------------------------------------------------
#
# ``TL_PARALLEL_VECTORIZE_DUMP_IR=1 pytest -s <this file>`` prints, for every case that
# calls the helpers below, the plan render and the vectorized IR. Off by default
# so the normal suite stays silent.
_DUMP_ENV = "TL_PARALLEL_VECTORIZE_DUMP_IR"


def _dump_enabled() -> bool:
    return os.environ.get(_DUMP_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _dump_plan_ir(label: str, plan) -> None:
    """Print the rendered plan plus the vectorized IR (gated by ``TL_PARALLEL_VECTORIZE_DUMP_IR``)."""
    if not _dump_enabled():
        return
    print(f"\n===== {label}: plan =====")
    print(render_plan(plan))
    print(f"===== {label}: vectorized IR =====")
    print(emit_plan(plan))


def _dump_stmt(label: str, stmt) -> None:
    """Print one IR fragment (pass output / report), gated by the same switch."""
    if not _dump_enabled():
        return
    print(f"\n===== {label} =====")
    print(stmt)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _first_parallel(func) -> tirx.For:
    """Return the outermost ``T.Parallel`` loop of a lowered prim func.

    ``post_order_visit`` yields the innermost nest first, so the outermost one is
    the *last* collected: ``build_plan`` needs the head of the nest to recover
    every axis (``_split_nest`` walks all consecutive ``T.Parallel`` loops).
    """
    found: list = []

    def visit(node):
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL:
            found.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    assert found, "the kernel has no T.Parallel nest"
    return found[-1]


def _for_kinds(stmt) -> list:
    kinds: list = []

    def visit(node):
        if isinstance(node, tirx.For):
            kinds.append(node.kind)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return kinds


def _simd_masks(stmt, op_name: str) -> list:
    """The mask argument (3rd operand) of every ``op_name`` call, as text."""
    masks: list = []

    def visit(node):
        if isinstance(node, tirx.Call) and str(getattr(node.op, "name", "")) == op_name:
            masks.append(str(node.args[2]))

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return masks


def _loop_var_names(stmt) -> list:
    names: list = []

    def visit(node):
        if isinstance(node, tirx.For):
            names.append(node.loop_var.name)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return names


def _plan(func, layouts=None):
    return build_plan(_first_parallel(func), layouts=layouts)


@contextlib.contextmanager
def _rejects(code: str):
    """Assert the stable rejection code (the diagnostic text is free to change)."""
    with pytest.raises(Unsupported) as excinfo:
        yield excinfo
    assert excinfo.value.code == code


def _emitted_text(func, layouts=None) -> str:
    """Vectorized IR text (used by the stage-1/stage-2 store-shape assertions)."""
    return str(emit_plan(_plan(func, layouts)))


def _count_parallel_nests(stmt) -> int:
    count = 0

    def visit(node):
        nonlocal count
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL:
            count += 1

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return count


def _buffer(func, name: str) -> tirx.Buffer:
    found: dict = {}

    def visit(node):
        if isinstance(node, tirx.BufferLoad):
            found[node.buffer.name] = node.buffer

    tirx.stmt_functor.post_order_visit(func.body, visit)
    assert name in found, f"buffer {name!r} is not accessed by the kernel"
    return found[name]


def _nz(func, factory=make_ascend_compact_nz_layout, name: str = "c_ub"):
    """``{Buffer: Layout}`` for an explicit NZ layout (manual equivalent of the P1 region handoff)."""
    buffer = _buffer(func, name)
    return {buffer: factory(buffer)}


def _single_segment(func):
    """``build_plan`` always returns a :class:`VectorProgram` (level 1). Single-segment
    cases use this helper to get back the only :class:`VectorPlan`; it also asserts the
    contract itself: exactly one segment, and ``program.axes`` only contains the lifted
    parallel-axis prefix (see :func:`_outer_axes`)."""
    program = build_plan(_first_parallel(func))
    assert isinstance(program, VectorProgram), "build_plan must always return a VectorProgram"
    assert len(program.segments) == 1, f"expected a single segment, got {len(program.segments)}"
    return program.segments[0]


def _emitted_ops(stmt) -> list:
    """Extract the ordered ``tl.simd.*`` op names emitted into *stmt*."""
    ops: list = []

    def visit(node):
        if isinstance(node, tirx.Call):
            name = str(getattr(node.op, "name", ""))
            if name.startswith("tl.simd."):
                ops.append(name)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return ops


def _call_args(stmt, op_name: str) -> list:
    """Collect the arguments of every ``op_name`` intrinsic call in *stmt*."""
    found: list = []

    def visit(node):
        if isinstance(node, tirx.Call) and str(getattr(node.op, "name", "")) == op_name:
            found.extend(node.args)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return found


def _recipes(node) -> list:
    """Flatten a :class:`ValueNode` DAG into its recipe names."""
    out = [node.recipe]
    for arg in node.args:
        out.extend(_recipes(arg))
    return out


def _eval_lane(node, lane: int, plan, env: dict | None = None, base: int = 0):
    """Evaluate a :class:`ValueNode` DAG on a single lane (for numeric goldens).

    The memory model is the identity: a load returns its element offset. This is
    enough to pin down "index/value arithmetic" bugs (sign flips, offset
    misalignment) that structural-only assertions cannot see.

    ``env`` assigns non-vector axis variables (e.g. ``{i: 3}``); ``base`` is the
    lane base address of the chunk.
    """

    def offset(access, lane_value):
        mapping = dict(env or {})
        mapping[plan.vector_axis.var] = tirx.IntImm("int32", int(lane_value))
        simplified = arith.Analyzer().simplify(tirx.stmt_functor.substitute(access.offset, mapping))
        assert isinstance(simplified, (tirx.IntImm, tirx.FloatImm)), f"offset did not fold: {simplified}"
        return int(simplified.value)

    def scalar_value(expr):
        if isinstance(expr, (int, float)):
            return int(expr)
        simplified = arith.Analyzer().simplify(expr)
        if isinstance(simplified, (tirx.IntImm, tirx.FloatImm)):
            return int(simplified.value)
        simplified = arith.Analyzer().simplify(tirx.stmt_functor.substitute(simplified, dict(env or {})))
        assert isinstance(simplified, (tirx.IntImm, tirx.FloatImm)), f"scalar did not fold: {simplified}"
        return int(simplified.value)

    def ev(n):
        recipe = n.recipe
        if recipe == "lane_index":
            return base + lane
        if recipe in ("imm", "uniform_expr"):
            return scalar_value(n.scalar)
        if recipe == "uniform_load":
            return offset(n.access, 0)  # invariant along the vector axis: substitute 0
        if recipe in ("vload", "vload_strided"):
            return offset(n.access, base + lane)
        if recipe == "gather":
            return ev(n.args[0]) if n.args else offset(n.access, base + lane)
        if recipe == "vbrc":
            return ev(n.args[0])
        if recipe == "vadds":
            return ev(n.args[0]) + ev(n.args[1])
        if recipe == "vadd":
            return ev(n.args[0]) + ev(n.args[1])
        if recipe == "vsub":
            return ev(n.args[0]) - ev(n.args[1])
        if recipe in ("vmul", "vmuls"):
            return ev(n.args[0]) * ev(n.args[1])
        raise AssertionError(f"no lane evaluator for recipe '{recipe}'")

    return ev(node)


def _for_extents(stmt) -> list:
    """Collect the extent of every ``For`` (to check symbolic extents are not silently truncated)."""
    extents: list = []

    def visit(node):
        if isinstance(node, tirx.For):
            extents.append(node.extent)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return extents


def _negative_immediate_muls(stmt) -> bool:
    """Whether ``vmuls(<uint32 ramp>, <negative constant>)`` exists -- an unsigned ramp
    times a negative stride is meaningless (defect D8)."""
    bad: list = []

    def visit(node):
        if (
            isinstance(node, tirx.Call)
            and str(getattr(node.op, "name", "")) == "tl.simd.vmuls"
            and len(node.args) >= 2
            and isinstance(node.args[1], tirx.IntImm)
            and int(node.args[1].value) < 0
        ):
            bad.append(True)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return bool(bad)


def _loop_maxima(stmt) -> dict:
    """Collect the maximum value of every loop variable (constant ranges; used to check
    whether an access span goes out of bounds)."""
    maxima: dict = {}

    def visit(node):
        if isinstance(node, tirx.For) and isinstance(node.min, tirx.IntImm) and isinstance(node.extent, tirx.IntImm):
            lo, extent = int(node.min.value), int(node.extent.value)
            maxima[node.loop_var] = lo + extent - 1 if extent > 0 else lo

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return maxima


def _over_reading_contiguous_load(stmt) -> bool:
    """Whether a contiguous ``vld`` access_ptr footprint exceeds the source buffer's
    allocation.

    There is only one criterion: ``start + extent > number of elements in the
    buffer``. For a multi-dimensional buffer, a whole-lane read crossing the
    innermost dimension is not an error by itself -- as long as it stays in the
    same compact allocation (exactly the shape P1 produces after flattening:
    reading 8 rows of ``(64, 8)`` at once), the cross-row read still lands inside
    the allocation. Loop variables are substituted at their **maximum** (only for
    indices whose ``Delta`` along that variable is a non-negative constant, so the
    result is a true upper bound).
    """
    analyzer = arith.Analyzer()
    maxima = _loop_maxima(stmt)
    bad: list = []

    def max_start(indices, strides):
        total = 0
        for index, stride in zip(indices, strides):
            expr = index
            for var, hi in maxima.items():
                delta = finite_difference(expr, var, analyzer)
                if isinstance(delta, tirx.IntImm) and int(delta.value) >= 0:
                    expr = tirx.stmt_functor.substitute(expr, {var: tirx.IntImm("int32", hi)})
                    expr = analyzer.simplify(expr)
            if not isinstance(expr, tirx.IntImm):
                return None
            total += int(expr.value) * stride
        return total

    def visit(node):
        if not (isinstance(node, tirx.Call) and str(getattr(node.op, "name", "")) == "tl.simd.vld"):
            return
        addr = node.args[0]
        if not (isinstance(addr, tirx.Call) and str(getattr(addr.op, "name", "")) == "tl.access_ptr"):
            return
        inner, extent = addr.args[0], addr.args[1]
        if not (isinstance(inner, tirx.BufferLoad) and isinstance(extent, tirx.IntImm)):
            return
        shape = inner.buffer.shape
        if not shape or not all(isinstance(dim, tirx.IntImm) for dim in shape):
            return
        strides, numel = [1] * len(shape), 1
        for k in range(len(shape) - 1, -1, -1):
            strides[k] = numel
            numel *= int(shape[k].value)
        start = max_start(inner.indices, strides)
        if start is not None and start + int(extent.value) > numel:
            bad.append(True)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return bool(bad)


def _alloc_buffers(stmt) -> list:
    """Collect the ``AllocBuffer`` statements of the emitted TIR (mutable register
    accumulators for reductions)."""
    found: list = []

    def visit(node):
        if isinstance(node, tirx.AllocBuffer):
            found.append(node.buffer)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return found


def _chunk_iterations(stmt) -> list[tuple]:
    """Collect ``(loop_var name, start, iteration count)`` for every chunk loop
    (constant ranges)."""
    found: list = []

    def visit(node):
        if isinstance(node, tirx.For) and "chunk" in node.loop_var.name:
            lo = int(node.min.value) if isinstance(node.min, tirx.IntImm) else None
            extent = int(node.extent.value) if isinstance(node.extent, tirx.IntImm) else None
            found.append((node.loop_var.name, lo, extent))

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return found


def _one_dimensional_copy(M):
    @T.prim_func
    def k(A: T.Tensor((M,), "float32"), C: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M,), "float32")
            c = T.alloc_shared((M,), "float32")
            for i in T.Parallel(M):
                c[i] = a[i]

    return k


_WIDE = 128


def _mixed_width_kernel():
    """fp32 store + bf16 narrowing store (a mixed-width segment shaped like softmax)."""

    @T.prim_func
    def k(A: T.Tensor((_WIDE, _WIDE), "float32"), B: T.Tensor((_WIDE, _WIDE), "float32"), C: T.Tensor((_WIDE, _WIDE), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            b_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            c_ub = T.alloc_shared((_WIDE, _WIDE), "bfloat16")
            T.copy(A, a_ub)
            for i, j in T.Parallel(_WIDE, _WIDE):
                b_ub[i, j] = T.exp(a_ub[i, j])
                c_ub[i, j] = T.Cast("bfloat16", b_ub[i, j])
            T.copy(b_ub, B)
            T.copy(c_ub, C)

    return k


def _cast_kernel(n, src_dtype, dst_dtype):
    @T.prim_func
    def k(A: T.Tensor((n,), src_dtype), C: T.Tensor((n,), dst_dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), src_dtype)
            c_ub = T.alloc_shared((n,), dst_dtype)
            T.copy(A, a_ub)
            for i in T.Parallel(n):
                c_ub[i] = T.Cast(dst_dtype, a_ub[i])
            T.copy(c_ub, C)

    return k


def _mixed_recipe_kernel():
    m, n = 8, 128

    @T.prim_func
    def k(A: T.Tensor((m, n), "float32"), S: T.Tensor((m, n), "float32"), C: T.Tensor((m, n), "bfloat16"), D: T.Tensor((m,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((m, n), "float32")
            s_ub = T.alloc_shared((m, n), "float32")
            c_ub = T.alloc_shared((m, n), "bfloat16")
            d_ub = T.alloc_shared((m,), "float32")
            T.copy(A, a_ub)
            T.copy(S, s_ub)
            with T.SimdVF():
                for i, j in T.Parallel(m, n):
                    c_ub[i, j] = T.Cast("bfloat16", a_ub[i, j] * 2.0)
                    for kk in T.serial(n):
                        d_ub[i] += s_ub[i, kk]
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    return k


def _declared_extents(stmt):
    """``(buffer name, declared extent)`` of every emitted ``T.access_ptr``."""
    found = []

    def visit(node):
        if isinstance(node, tirx.Call) and str(getattr(node.op, "name", "")) == "tl.access_ptr":
            base = node.args[0]
            buffer = getattr(base, "buffer", None)
            extent = node.args[1]
            value = int(extent.value) if isinstance(extent, tirx.IntImm) else str(extent)
            found.append((str(getattr(buffer, "name", "?")), value))

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return found


def _nz_2d_kernel():
    """``c_ub[i, j] = exp(a_ub[i, j])``: one fp32 store whose source is ND."""

    @T.prim_func
    def k(A: T.Tensor((_WIDE, _WIDE), "float32"), C: T.Tensor((_WIDE, _WIDE), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            c_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            T.copy(A, a_ub)
            for i, j in T.Parallel(_WIDE, _WIDE):
                c_ub[i, j] = T.exp(a_ub[i, j])
            T.copy(c_ub, C)

    return k


def _p1_items(func) -> list:
    """Typed ``tl.vsimd.region`` annotations of a P1-lifted function (per nest)."""
    return _region_annotations(_run_lift(func).body)


def _region_annotations(stmt) -> list:
    """Typed ``tl.vsimd.region`` annotations of every annotated For."""
    found: list = []

    def visit(node):
        if isinstance(node, tirx.For) and "tl.vsimd.region" in node.annotations:
            found.append(node.annotations["tl.vsimd.region"])

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return found


def _outer_parallel_nests(func) -> list:
    """Outermost ``T.Parallel`` nests in source order (one per P1 nest)."""

    nests: list = []

    def walk(stmt):
        if isinstance(stmt, tirx.For):
            if stmt.kind == tirx.ForKind.PARALLEL:
                nests.append(stmt)
                return
            walk(stmt.body)
            return
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                walk(child)
            return
        if isinstance(stmt, tirx.SBlockRealize):
            walk(stmt.block)
            return
        body = getattr(stmt, "body", None)
        if body is not None:
            walk(body)

    walk(func.body)
    return nests


def _stored_buffers(stmt) -> list:
    names: list = []

    def visit(node):
        if isinstance(node, tirx.BufferStore):
            names.append(node.buffer.name)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return names


def _assert_p1_rejected(func, status: str, reason_match: str):
    regions = _p1_items(func)
    assert len(regions) == 1
    region = regions[0]
    assert region.status == status, (region.status, region.reason)
    assert reason_match in region.reason


def _module_entry(func):
    """Wrap *func* in a module and return ``(module, entry_name)``."""
    mod = tvm.IRModule.from_expr(func)
    names = [gv.name_hint for gv in mod.get_global_vars()]
    assert len(names) == 1, names
    return mod, names[0]


def _run_lift(func):
    """P1: attach ``tl.vsimd.region`` to every top-level Parallel."""
    mod, name = _module_entry(func)
    return at.AscendLiftVSimd()(mod)[name]


def _run_plan(func):
    """P2: plan the P1 groups (body unchanged)."""
    mod, name = _module_entry(func)
    return at.AscendPlanVSimd()(mod)[name]


def _run_new_path(func):
    """P1 -> P2 -> P3 on one PrimFunc."""
    mod, name = _module_entry(func)
    mod = at.AscendLiftVSimd()(mod)
    mod = at.AscendPlanVSimd()(mod)
    return at.AscendEmitVSimd()(mod)[name]


def _annotations(stmt, key: str) -> list:
    """Typed ``key`` annotations of every SBlock in *stmt*."""
    found: list = []

    def visit(node):
        if isinstance(node, tirx.SBlock) and key in node.annotations:
            found.append(node.annotations[key])

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return found


def _region_plans(stmt) -> list:
    """Typed ``tl.vsimd.plan`` annotations: P2 attaches them to the loop."""
    found: list = []

    def visit(node):
        if isinstance(node, tirx.For) and "tl.vsimd.plan" in node.annotations:
            found.append(node.annotations["tl.vsimd.plan"])

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return found


def _lower_source(func, config=None) -> str:
    """Lower under the given pass context (default: no override) to device source."""
    with tvm.target.Target("ascend"), tilelang.transform.PassContext(config=config or {}):
        return tilelang.lower(func, target="ascend").kernel_source


def _p1_kernel_1d():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    return main


def _p1_kernel_extent(extent: int):
    @T.prim_func
    def main(A: T.Tensor((extent,), "float32"), C: T.Tensor((extent,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((extent,), "float32")
            c_ub = T.alloc_shared((extent,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(extent):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    return main
