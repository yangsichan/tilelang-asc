"""Offline structural tests for NZ-layout stores (``vsstb`` + pack2).

The target buffer carries an explicit Ascend NZ layout, so the planner
classifies the store as ``layout_kind == "nz"`` and the emitter scatters it
with a derived block stride.
"""

from __future__ import annotations


import pytest

import tilelang.ascend.language as T

from tilelang.ascend.analysis.parallel_vectorize import (
    Unsupported,
    emit_plan,
)
from tilelang.layout import (
    make_ascend_compact_nz_layout,
    make_ascend_major_mn_layout,
    make_ascend_nz_layout,
)
from testing.ascend.transform._parallel_vectorize_helpers import (
    _WIDE,
    _count_parallel_nests,
    _dump_plan_ir,
    _emitted_text,
    _nz,
    _nz_2d_kernel,
    _p1_items,
    _plan,
    _rejects,
)


# ===========================================================================
# Stage 2: the destination buffer has an explicit NZ layout (direct register store + half-register pairing)
# ===========================================================================
#
# Writing a buffer annotated as Ascend NZ (compact ``[col/C0, row, col%C0]`` or
# canonical MajorK) cannot use a contiguous store: the planner classifies that
# store as ``layout_kind == "nz"``, derives the block stride from the layout's
# forward mapping, and the emitter scatters it with ``tl.simd.vsstb``.
#
# A narrowing store (``f32 -> bf16``) only fills half of its register, and
# ``vsstb`` has no PK variant, so stage 2b widens it first: the two halves of a
# chunk pair are de-interleaved by one ``vld2(DINTLV_B32)``, narrowed separately
# into the even/odd 16-bit slots of the destination register
# (``vcvt PART_EVEN/PART_ODD``), merged into a full register with ``vor``, and
# then a single ``vsstb`` scatters the whole row; a contiguous destination in the
# same segment is restored with ``vintlv`` at the write-back. The block stride is
# computed, not hard-coded.


def test_compact_nz_store_plan_emit_and_lowering():
    """Classify the NZ store, emit vsstb and replace the nest in one compile."""
    kernel = _nz_2d_kernel()
    layouts = _nz(kernel)
    plan = _plan(kernel, layouts)
    segment = plan.segments[0]
    store = segment.stores[0]

    assert segment.lanes == 64
    assert store.layout_kind == "nz"
    # Compact NZ is [col/C0, row, col%C0]; two neighbouring 32B blocks are
    # ``rows`` blocks apart, i.e. 128 * 8 fp32 elements / C0 = 128 blocks.
    assert store.nz_block_stride == _WIDE

    _dump_plan_ir("compact NZ store", plan)
    stmt = emit_plan(plan)
    text = str(stmt)

    assert "tl.simd.vsstb" in text
    # (block_stride << 16): the low 16 bits are the POST_UPDATE advance; a plain direct store always has 0.
    assert str(_WIDE << 16) in text
    # The register is scattered, never written contiguously.
    assert "tl.simd.vsts" not in text
    assert _count_parallel_nests(stmt) == 0


def test_canonical_nz_store_uses_the_same_block_stride():
    kernel = _nz_2d_kernel()
    layouts = _nz(kernel, make_ascend_nz_layout)
    store = _plan(kernel, layouts).segments[0].stores[0]

    assert store.layout_kind == "nz"
    assert store.nz_block_stride == _WIDE
    assert "tl.simd.vsstb" in _emitted_text(kernel, layouts)


def test_nd_destination_keeps_the_contiguous_store():
    """Zero-diff guard: no explicit layout ⇒ the historical ND form."""

    kernel = _nz_2d_kernel()
    store = _plan(kernel).segments[0].stores[0]

    assert store.layout_kind == "nd"
    assert store.nz_block_stride is None

    text = _emitted_text(_nz_2d_kernel())
    assert '"NORM_B32"' in text
    assert "vsstb" not in text


def test_p1_hand_off_exposes_the_annotated_layouts():
    """P1 hands the annotated buffer names over to P2 with the region."""

    @T.prim_func
    def k(A: T.Tensor((_WIDE, _WIDE), "float32"), C: T.Tensor((_WIDE, _WIDE), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            c_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            T.copy(A, a_ub)
            T.annotate_layout({c_ub: make_ascend_compact_nz_layout(c_ub)})
            with T.SimdVF():
                for i, j in T.Parallel(_WIDE, _WIDE):
                    c_ub[i, j] = T.exp(a_ub[i, j])
            T.copy(c_ub, C)

    items = _p1_items(k)
    lifted = [item for item in items if item.status == "LIFTED"]
    assert lifted, "the nest must be lifted"
    assert [capture.buffer.name for capture in lifted[0].captures if capture.layout is not None] == ["c_ub"]


def test_explicit_non_nz_layout_is_rejected():
    kernel = _nz_2d_kernel()
    layouts = _nz(kernel, make_ascend_major_mn_layout)

    with pytest.raises(Unsupported, match="explicit non-NZ layout"):
        _plan(kernel, layouts)


def test_half_register_nz_store_is_paired_into_a_full_register():
    """Stage 2b: narrowing + NZ use the paired form (vld2 + part0/1 + vor + one vsstb)."""

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

    store = _plan(k, _nz(k)).segments[0].stores[1]
    assert store.store_lane_mode == "pk_half"
    assert store.layout_kind == "nz"
    assert store.nz_block_stride == _WIDE

    _dump_plan_ir("narrow + NZ (pack2)", _plan(k, _nz(k)))
    text = _emitted_text(k, _nz(k))
    # The f32 values of two chunks are de-interleaved by one vld2 and then written
    # into the even/odd 16-bit slots of the destination register.
    assert text.count("tl.simd.vld2(") == 1
    assert "DINTLV_B32" in text
    assert "tl.simd.vdintlv" not in text, "de-interleaving is already folded into vld2; do not emit it again"
    assert "PART_EVEN" in text
    assert "PART_ODD" in text
    assert "tl.simd.vor" in text
    assert "uint16x128" in text
    # A contiguous ND destination in the same segment is restored into two contiguous
    # halves at write-back: vintlv + two vsts.
    assert "tl.simd.vintlv" in text
    assert text.count("tl.simd.vsts") == 2
    # After merging it is a full register: one vsstb scatters the whole row, so the
    # half-register packed store is no longer needed.
    assert text.count("tl.simd.vsstb") == 1
    assert str(_WIDE << 16) in text
    assert '"PK_B32"' not in text


def test_paired_store_alone_loads_both_chunks():
    """A single narrowing + NZ store: one vld2 -> two parts -> one vsstb (DSA recipe)."""

    @T.prim_func
    def k(A: T.Tensor((_WIDE, _WIDE), "float32"), C: T.Tensor((_WIDE, _WIDE), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            c_ub = T.alloc_shared((_WIDE, _WIDE), "bfloat16")
            T.copy(A, a_ub)
            for i, j in T.Parallel(_WIDE, _WIDE):
                c_ub[i, j] = T.Cast("bfloat16", a_ub[i, j])
            T.copy(c_ub, C)

    text = _emitted_text(k, _nz(k))
    assert text.count("tl.simd.vld2(") == 1
    assert "tl.simd.vld(" not in text
    assert "tl.simd.vdintlv" not in text
    assert text.count("tl.simd.vsstb") == 1


def test_paired_store_loops_over_chunk_pairs():
    """4 chunks = 2 pairs: each iteration advances two chunks, and the loop emits one vsstb."""

    rows, wide = 64, 256

    @T.prim_func
    def k(A: T.Tensor((rows, wide), "float32"), C: T.Tensor((rows, wide), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, wide), "float32")
            c_ub = T.alloc_shared((rows, wide), "bfloat16")
            T.copy(A, a_ub)
            for i, j in T.Parallel(rows, wide):
                c_ub[i, j] = T.Cast("bfloat16", a_ub[i, j])
            T.copy(c_ub, C)

    segment = _plan(k, _nz(k)).segments[0]
    assert segment.lanes == 64
    assert segment.chunks == 4

    text = _emitted_text(k, _nz(k))
    # One iteration per chunk pair: chunk index 2*pair, stride 2 * lanes.
    assert "T.grid(%d, 2)" % rows in text
    assert "chunk_s0 * 128" in text
    assert text.count("tl.simd.vsstb") == 1
    assert str(rows << 16) in text


def test_paired_store_needs_whole_chunk_pairs():
    """64 columns = 1 chunk, which cannot form a chunk pair."""

    m = 64

    @T.prim_func
    def k(A: T.Tensor((_WIDE, m), "float32"), C: T.Tensor((_WIDE, m), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, m), "float32")
            c_ub = T.alloc_shared((_WIDE, m), "bfloat16")
            T.copy(A, a_ub)
            for i, j in T.Parallel(_WIDE, m):
                c_ub[i, j] = T.Cast("bfloat16", a_ub[i, j])
            T.copy(c_ub, C)

    with pytest.raises(Unsupported, match="whole chunk pairs"):
        _plan(k, _nz(k))


def test_paired_store_must_be_the_last_store():
    """The merged register exists only after pairing, so no statement may follow the packed store."""

    @T.prim_func
    def k(
        A: T.Tensor((_WIDE, _WIDE), "float32"),
        B: T.Tensor((_WIDE, _WIDE), "float32"),
        C: T.Tensor((_WIDE, _WIDE), "bfloat16"),
        D: T.Tensor((_WIDE, _WIDE), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            b_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            c_ub = T.alloc_shared((_WIDE, _WIDE), "bfloat16")
            d_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            T.copy(A, a_ub)
            for i, j in T.Parallel(_WIDE, _WIDE):
                b_ub[i, j] = T.exp(a_ub[i, j])
                c_ub[i, j] = T.Cast("bfloat16", b_ub[i, j])
                d_ub[i, j] = b_ub[i, j] * T.float32(2.0)
            T.copy(b_ub, B)
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    with pytest.raises(Unsupported, match="must be the last store"):
        _plan(k, _nz(k))


def test_nz_store_with_predicate_is_rejected():
    @T.prim_func
    def k(A: T.Tensor((_WIDE, _WIDE), "float32"), C: T.Tensor((_WIDE, _WIDE), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            c_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            T.copy(A, a_ub)
            for i, j in T.Parallel(_WIDE, _WIDE):
                if j < _WIDE - 8:
                    c_ub[i, j] = T.exp(a_ub[i, j])
            T.copy(c_ub, C)

    with _rejects("body.statement_predicate"):
        _plan(k, _nz(k))


def test_nz_store_with_partial_column_chunk_is_rejected():
    """96 columns is not a whole number of 64-element chunks."""

    m = 96

    @T.prim_func
    def k(A: T.Tensor((m, m), "float32"), C: T.Tensor((m, m), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((m, m), "float32")
            c_ub = T.alloc_shared((m, m), "float32")
            T.copy(A, a_ub)
            for i, j in T.Parallel(m, m):
                c_ub[i, j] = T.exp(a_ub[i, j])
            T.copy(c_ub, C)

    with _rejects("plan.partial_chunks"):
        _plan(k, _nz(k))
