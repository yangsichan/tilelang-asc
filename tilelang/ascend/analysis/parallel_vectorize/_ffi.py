"""FFI bindings and value types of the parallel_vectorize P2/P3 implementation.

The C++ port (``src/ascend/transform/parallel_vectorize``) registers its plan objects as
TVM FFI objects whose reflected field names match the Python dataclasses, so
tests (and the emit layer) read ``program.segments[0].stores[0].value.recipe``
without any re-construction. Only two derived members need a Python-side helper:

* :attr:`StorePlan.kind` — the C++ field is ``kind_value`` (a string), because a
  reflected field would shadow a Python property of the same name;
* :meth:`AccessInfo.axis_class` — a method, forwarded to the C++ side.

Rejections arrive as
``{"ok": False, "code": ..., "message": ..., "diagnostics": [...]}`` and are
re-raised as :class:`Unsupported` carrying the same reason code, text and
diagnostic order (D3). Assert on :attr:`Unsupported.code`; the message text is
free to change.

The two value types below are the only Python-owned definitions: an enum and an
exception cannot cross the FFI boundary as objects, so they mirror the C++
``AccessClass`` strings and the ``parallel_vectorize::Unsupported`` payload.
"""

from __future__ import annotations

import enum

import tvm_ffi
from tvm.ir.base import Node

from tilelang.transform import _ffi_api


class AccessClass(str, enum.Enum):
    """Memory access pattern along one axis.

    A constant Delta is classified solely by ``classify_delta``; a **non-constant
    Delta** is refined structurally by ``classify_access`` (improvement 4), which
    splits the previously vague ``NON_AFFINE`` label into three distinct ones:

    * ``PIECEWISE`` -- Delta is non-constant, but the element offset can be
      materialized in the per-lane index-vector channel (affine + power-of-two
      ``FloorDiv`` / ``FloorMod`` + integer arithmetic) -> ``vgather2`` /
      ``vscatter``;
    * ``PERMUTE`` -- Delta == -1, i.e. a **bijective reverse order** along this
      axis (a permutation rather than a stride) -> also goes through the
      index-vector channel;
    * ``TRUE_RANDOM`` -- Delta is non-constant and the index vector **cannot** be
      materialized (e.g. non-power-of-two constant division, data-dependent
      indices) -> rejected explicitly, never silently downgraded.
    """

    CONTIGUOUS = "contiguous"  # Delta == 1
    UNIFORM = "uniform"  # Delta == 0
    STRIDED = "strided"  # Delta == k, |k| > 1 (constant-stride access)
    PERMUTE = "permute"  # Delta == -1 (bijective reverse order, a permutation)
    PIECEWISE = "piecewise"  # non-constant Delta, materializable in the index-vector channel
    INDIRECT = "indirect"  # the index itself contains a BufferLoad (gather/scatter)
    TRUE_RANDOM = "true_random"  # non-constant and non-materializable Delta -> reject


class Unsupported(Exception):
    """Raised when a nest cannot be vectorized; carries the diagnostic reason(s).

    ``code`` is the stable rejection category (e.g. ``body.nested_loop``,
    ``plan.no_vector_axis``); tests should assert on it rather than on the
    English message. Unclassified rejections use ``"unsupported"``.
    ``diagnostics`` collects **every blocker** seen while planning (the
    classification summary and failure reason of each candidate axis), so the
    caller sees all impediments at once instead of only the first failure
    (improvement 7). ``build_plan`` fills it; ``str(err)`` remains a one-line
    summary.
    """

    def __init__(
        self,
        message: str,
        *,
        diagnostics: list[str] | None = None,
        code: str = "unsupported",
    ):
        super().__init__(message)
        self.diagnostics: list[str] = list(diagnostics or [])
        self.code: str = code


# ---------------------------------------------------------------------------
# FFI object types (type keys registered by the C++ port)
# ---------------------------------------------------------------------------


@tvm_ffi.register_object("tl.parallel_vectorize.Axis")
class Axis(Node):
    """One iteration axis (``0 <= var < extent``)."""


@tvm_ffi.register_object("tl.parallel_vectorize.AccessInfo")
class AccessInfo(Node):
    """One buffer access plus its per-axis classification."""

    def axis_class(self, axis):
        """Return ``(AccessClass, stride)`` for ``axis`` (``stride`` may be None)."""
        kind, stride = _ffi_api.AscendParallelVectorizeAxisClass(self, axis)
        return (AccessClass(kind), stride)


@tvm_ffi.register_object("tl.parallel_vectorize.StoreSpec")
class StoreSpec(Node):
    """Where to write, what to write, and an optional predicate."""


@tvm_ffi.register_object("tl.parallel_vectorize.ValueNode")
class ValueNode(Node):
    """A value DAG node (instruction-selection agnostic recipe)."""


@tvm_ffi.register_object("tl.parallel_vectorize.ReduceInfo")
class ReduceInfo(Node):
    """Reduction description of one store."""


@tvm_ffi.register_object("tl.parallel_vectorize.StorePlan")
class StorePlan(Node):
    """Full plan of one write."""

    @property
    def kind(self):
        # ``kind_value`` is the reflected string; ``kind`` is the enum the tests
        # and the emitter compare against.
        return AccessClass(self.kind_value)


@tvm_ffi.register_object("tl.parallel_vectorize.VectorPlan")
class VectorPlan(Node):
    """Segment-level plan (one body + optional pre/post siblings)."""


@tvm_ffi.register_object("tl.parallel_vectorize.PlanNode")
class PlanNode(Node):
    """One node of the structural plan tree (design A).

    ``kind`` is one of ``seq`` / ``loop`` / ``chunk``.  ``chunk`` nodes carry a
    :class:`VectorPlan`; structure nodes only carry control flow.  A
    lane-uniform ``if`` is not a node: its arms are emitted as predicated
    chunks (store masks), because the VF contract only allows vector work.
    """

    @property
    def kind(self):
        return self.kind_value


@tvm_ffi.register_object("tl.parallel_vectorize.VectorProgram")
class VectorProgram(Node):
    """``build_plan`` result: 1..N segments (``root`` is the plan tree)."""


# ---------------------------------------------------------------------------
# API dispatch helpers
# ---------------------------------------------------------------------------


def _raise(result) -> None:
    raise Unsupported(
        result["message"],
        diagnostics=list(result["diagnostics"]),
        code=result["code"],
    )


def build_plan(stmt, lanes=None, layouts=None):
    """C++ ``build_plan``: ``T.Parallel`` nest -> :class:`VectorProgram`.

    ``layouts`` is the explicit ``{Buffer: Layout}`` map of the nest's
    destination buffers (the P1 region hand-off); a buffer without an entry
    keeps the default ND store, an Ascend NZ entry scatters 32B blocks.

    A ``local.fragment`` access is rejected: fragment vectorization is not
    supported, and planning it would emit an un-compilable ``vld``/``vsts``
    pair. P1 keeps such nests out of the vectorization path, so this only
    guards direct planner callers.
    """
    result = _ffi_api.AscendParallelVectorizeBuildPlan(stmt, lanes, layouts)
    if not result["ok"]:
        _raise(result)
    return result["program"]


def emit_plan(plan):
    """C++ ``emit_plan``: plan -> ``T.simd.*`` TIR."""
    result = _ffi_api.AscendParallelVectorizeEmitPlan(plan)
    if not result["ok"]:
        _raise(result)
    return result["stmt"]


def render_plan(plan):
    """C++ ``render_plan``: plan -> human-readable text."""
    result = _ffi_api.AscendParallelVectorizeRenderPlan(plan)
    if not result["ok"]:
        _raise(result)
    return result["text"]


def verify_no_scalar_data_ops(stmt):
    """Invariant I5: list scalar data instructions in an emitted region.

    An empty list means the region only contains vector data instructions;
    scalar values appear only in loop bounds, conditions, addresses, chunk
    indices and ``Bind`` values.
    """
    result = _ffi_api.AscendParallelVectorizeVerifyNoScalarDataOps(stmt)
    return list(result["violations"])


def finite_difference(offset, var, analyzer):
    """C++ ``finite_difference``.

    The Python analyzer object is not transferable over FFI, so the C++ side
    creates a fresh :class:`arith.Analyzer` (identical for every caller that does
    not pre-bind variables; the parity harness covers this).
    """
    return _ffi_api.AscendParallelVectorizeFiniteDifference(offset, var, analyzer is not None)


def classify_delta(delta, analyzer=None):
    """C++ ``classify_delta``: ``(AccessClass, stride_or_None)``."""
    kind, stride = _ffi_api.AscendParallelVectorizeClassifyDelta(delta, analyzer is not None)
    return (AccessClass(kind), stride)
