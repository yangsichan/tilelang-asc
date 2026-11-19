"""Offline structural tests for P3 emission across element widths.

Mixed-width segments (narrowing cast → half register + packed store), the
same-width signedness reinterpret path and the widening (``UNPK``) recipes.
"""

from __future__ import annotations


import pytest

import tilelang.ascend.language as T

from tilelang.ascend.analysis.parallel_vectorize import (
    emit_plan,
)
from testing.ascend.transform._parallel_vectorize_helpers import (
    _WIDE,
    _cast_kernel,
    _count_parallel_nests,
    _declared_extents,
    _dump_plan_ir,
    _emitted_text,
    _mixed_recipe_kernel,
    _mixed_width_kernel,
    _nz,
    _plan,
)


# ===========================================================================
# Stage 1: mixed element widths within one segment (narrowing cast -> half register + PK packed store)
# ===========================================================================
#
# A narrowing ``float32 -> bfloat16`` ``Cast`` turns only 64 f32 lanes into 64
# valid bf16 values (a bf16 register has 128 lanes), so the segment width is the
# minimum actually produced by the stores (``plan.lanes``), and the fraction of
# each store's own register that is written is recorded in ``store_lane_mode``;
# the wider side uses a packed store (``PK_B*``), matching the board-validated
# cast kernel exactly. Hardware-independent: only plan fields and the emitted
# structure are asserted.


def test_mixed_width_segment_plan_and_emit():
    """Plan width/mode, packed store form and whole-nest replacement in one compile."""
    kernel = _mixed_width_kernel()
    plan = _plan(kernel)
    assert len(plan.segments) == 1
    segment = plan.segments[0]

    # 16384 logical elements: a float32 register covers 64 of them per chunk.
    assert segment.lanes == 64
    assert segment.chunks == 2  # vector axis j is 128 wide
    assert segment.tail == 0

    fp32_store, bf16_store = segment.stores
    assert fp32_store.store_lanes == 64
    assert fp32_store.store_lane_mode == "full"
    assert fp32_store.store_pack_bits is None

    assert bf16_store.store_lanes == 128
    assert bf16_store.store_lane_mode == "pk_half"
    # The packed store suffix is the width of the conversion source register.
    assert bf16_store.store_pack_bits == 32
    assert bf16_store.value.recipe == "vcvt"

    _dump_plan_ir("mixed width", plan)
    stmt = emit_plan(plan)
    text = str(stmt)

    # Full-width store keeps the historical contiguous form.
    assert '"NORM_B32"' in text
    # The wider store writes exactly 64 bfloat16 elements through the packed form.
    assert '"PK_B32"' in text
    assert '"NORM_B16"' not in text, "the bf16 store must not write its full register"
    # The narrowing conversion selects the even 16-bit slots (``simd.vcvt(part=0)``,
    # the verified cast-kernel recipe); the packed store then writes those lanes.
    assert "PART_EVEN" in text
    # store -> load forwarding: the cast reads the freshly computed register
    # instead of reloading UB (no VST->VLD barrier needed).
    assert text.count("tl.simd.vld") == 1
    # The whole nest is replaced by the vector form.
    assert _count_parallel_nests(stmt) == 0
    assert "tl.simd.vsts" in text


def test_narrowing_cast_store_alone_packs_half_register():
    """A lone ``f32 -> bf16`` store is the same shape: 64 of 128 lanes."""

    @T.prim_func
    def k(A: T.Tensor((_WIDE, _WIDE), "float32"), C: T.Tensor((_WIDE, _WIDE), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            c_ub = T.alloc_shared((_WIDE, _WIDE), "bfloat16")
            T.copy(A, a_ub)
            for i, j in T.Parallel(_WIDE, _WIDE):
                c_ub[i, j] = T.Cast("bfloat16", a_ub[i, j])
            T.copy(c_ub, C)

    segment = _plan(k).segments[0]
    assert segment.lanes == 64
    assert segment.stores[0].store_lane_mode == "pk_half"
    assert segment.stores[0].store_pack_bits == 32

    text = _emitted_text(k)
    assert '"PK_B32"' in text
    assert "PART_EVEN" in text, "narrowing needs the half-register part selector"


def test_same_width_segment_keeps_full_mode():
    """Zero-diff guard: an ordinary fp32 nest still uses the full store form."""

    @T.prim_func
    def k(A: T.Tensor((_WIDE, _WIDE), "float32"), B: T.Tensor((_WIDE, _WIDE), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            b_ub = T.alloc_shared((_WIDE, _WIDE), "float32")
            T.copy(A, a_ub)
            for i, j in T.Parallel(_WIDE, _WIDE):
                b_ub[i, j] = T.exp(a_ub[i, j])
            T.copy(b_ub, B)

    segment = _plan(k).segments[0]
    assert segment.lanes == 64
    assert segment.stores[0].store_lane_mode == "full"
    assert segment.stores[0].store_pack_bits is None

    text = _emitted_text(k)
    assert '"NORM_B32"' in text
    assert "PK_B" not in text


# ===========================================================================
# Stage 1 (continued): same-width signedness reinterpret and widening conversion (UNPK load)
# ===========================================================================
#
# Same-width integer signedness changes (i32<->u32, ...) are bit-identical and
# must not go through ``vcvt`` (that is a numeric conversion); they are a
# register reinterpret (the PTO counterpart of ``vinterpret_cast``): P2 emits
# the ``reinterpret`` recipe and P3 emits ``tirx.reinterpret``, producing no
# instruction at all.
#
# When widening (a larger bit width), the chunk is only as wide as the store
# register while the source register is wider, so the load must spread the
# valid elements into the even/0-th sub-slots of the wider register
# (``UNPK_B16`` / ``UNPK4_B8``), and ``vcvt`` then reads them with
# ``PART_EVEN`` / ``PART_P0`` -- matching the board-level cast kernel recipe.


def test_same_width_signedness_cast_is_a_register_reinterpret():
    """``i32 -> u32`` is bit-identical: a reinterpret, never a ``vcvt``."""
    segment = _plan(_cast_kernel(128, "int32", "uint32")).segments[0]
    assert segment.stores[0].value.recipe == "reinterpret"
    assert segment.stores[0].value.detail == "uint32"

    text = _emitted_text(_cast_kernel(128, "int32", "uint32"))
    assert "T.reinterpret" in text
    assert "tl.simd.vcvt" not in text, "a same-width sign change is not a value conversion"


@pytest.mark.parametrize(
    "src_dtype,dst_dtype,dist,part",
    [
        ("float16", "float32", "UNPK_B16", "PART_EVEN"),
        ("bfloat16", "float32", "UNPK_B16", "PART_EVEN"),
        ("int16", "int32", "UNPK_B16", "PART_EVEN"),
        ("int16", "uint32", "UNPK_B16", "PART_EVEN"),
        ("uint16", "uint32", "UNPK_B16", "PART_EVEN"),
        ("float8_e4m3fn", "float32", "UNPK4_B8", "PART_P0"),
        ("int8", "int32", "UNPK4_B8", "PART_P0"),
        ("uint8", "uint32", "UNPK4_B8", "PART_P0"),
    ],
)
def test_widening_cast_loads_with_unpack(src_dtype, dst_dtype, dist, part):
    segment = _plan(_cast_kernel(128, src_dtype, dst_dtype)).segments[0]
    assert segment.lanes == 64, "the chunk is the store register width"
    assert segment.stores[0].value.recipe == "vcvt"

    text = _emitted_text(_cast_kernel(128, src_dtype, dst_dtype))
    assert f'"{dist}"' in text, "the widening load must spread into wider containers"
    assert f'"{part}"' in text, "the conversion reads the spread sub-slots"


@pytest.mark.parametrize(
    "src_dtype,dst_dtype,part",
    [
        ("int8", "int16", "PART_EVEN"),
        ("uint8", "uint16", "PART_EVEN"),
        ("int8", "float16", "PART_EVEN"),
        ("uint8", "float16", "PART_EVEN"),
    ],
)
def test_widening_from_eight_to_sixteen_bits_is_supported(src_dtype, dst_dtype, part):
    """``i8/u8 -> i16/f16`` is a two-way widening: ``UNPK_B8`` + ``PART_EVEN``."""
    segment = _plan(_cast_kernel(128, src_dtype, dst_dtype)).segments[0]
    assert segment.lanes == 128, "the chunk is the 16-bit store register width"
    text = _emitted_text(_cast_kernel(128, src_dtype, dst_dtype))
    assert '"UNPK_B8"' in text, "the byte load must spread into 16-bit containers"
    assert f'"{part}"' in text


@pytest.mark.parametrize("src_dtype,dst_dtype", [("int8", "int16"), ("uint8", "uint16")])
def test_widening_from_eight_to_sixteen_bits_with_a_tail_uses_the_byte_gather(src_dtype, dst_dtype):
    """A tail cannot be applied after the unpack; the widen gather covers it.

    ``vgather2`` zero-extends an 8-bit source (CANN intrinsic contract), so the
    signed pair still needs the ``PART_EVEN`` sign extension while the unsigned
    pair is already the converted value.
    """
    n = 100
    segment = _plan(_cast_kernel(n, src_dtype, dst_dtype)).segments[0]
    assert segment.lanes == 128 and segment.tail == n
    text = _emitted_text(_cast_kernel(n, src_dtype, dst_dtype))
    assert "tl.simd.vgather2" in text
    assert '"UNPK_B8"' not in text, "the masked load is the widening gather"
    if src_dtype == "int8":
        assert "tl.simd.vcvt" in text, "a signed byte needs the sign extension"
        assert '"PART_EVEN"' in text
    else:
        assert "tl.simd.vcvt" not in text, "zero-extension already is the u8 -> u16 value"


def test_eight_bit_tail_uses_the_byte_gather_and_pack_store():
    """An 8-bit tail is gathered into 16-bit lanes and narrowed by ``PK_B16``."""
    n = 100
    segment = _plan(_cast_kernel(n, "uint8", "uint8")).segments[0]
    assert segment.lanes == 256 and segment.tail == n

    text = _emitted_text(_cast_kernel(n, "uint8", "uint8"))
    assert "tl.simd.vgather2" in text, "a byte tail load is a widening gather"
    assert '"PK_B16"' in text, "the 16-bit lanes are written back as packed bytes"
    assert '"NORM_B8"' not in text, "a byte NORM store would interleave 16-bit lanes"

    # The packed store writes at most half a register, which is also the
    # declared access footprint.
    for name, extent in _declared_extents(emit_plan(_plan(_cast_kernel(n, "uint8", "uint8")))):
        if name == "c_ub":
            assert extent == 128


def test_mixed_recipe_nest_is_split_per_statement():
    """Elementwise + NZ + row reduction have no common axis: P2 splits them.

    The single segment (NZ store as a pre sibling of the reduction loop) can
    only be refused by the emitter; the source-order split gives the NZ store
    its own paired-form segment and the reduction its own.
    """
    kernel = _mixed_recipe_kernel()
    layouts = _nz(kernel, name="c_ub")
    program = _plan(kernel, layouts)

    assert len(program.segments) == 2
    nz_segment, reduce_segment = program.segments
    assert nz_segment.stores[0].spec.buffer.name == "c_ub"
    assert nz_segment.stores[0].layout_kind == "nz"
    assert reduce_segment.stores[0].spec.buffer.name == "d_ub"

    text = str(emit_plan(program))
    assert "vsstb" in text, "the NZ segment is emitted in its paired form"
    assert "vcadd" in text or "vadd" in text, "the reduction segment is vectorized"


def test_strided_access_declares_the_spanned_envelope():
    """A stride-2 ramp touches ``(lanes - 1) * stride + 1 = 127`` elements."""
    n = 128

    @T.prim_func
    def load_k(A: T.Tensor((2 * n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((2 * n,), "float32")
            c = T.alloc_shared((n,), "float32")
            T.copy(A, a)
            for i in T.Parallel(n):
                c[i] = a[2 * i]
            T.copy(c, C)

    @T.prim_func
    def store_k(A: T.Tensor((n,), "float32"), C: T.Tensor((2 * n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float32")
            c = T.alloc_shared((2 * n,), "float32")
            T.copy(A, a)
            for i in T.Parallel(n):
                c[2 * i] = a[i]
            T.copy(c, C)

    load_extents = _declared_extents(emit_plan(_plan(load_k)))
    assert ("a", 127) in load_extents, load_extents
    store_extents = _declared_extents(emit_plan(_plan(store_k)))
    assert ("c", 127) in store_extents, store_extents


def test_indexed_access_declares_the_allocation_owner():
    """Runtime offsets: a gather/scatter declares the whole owner (may-access).

    The load side uses an indirect (runtime) index.  The store side uses a
    reversal instead of an indirect store: a runtime store index is
    data-dependent, so P1 reports the nest ``UNPROVEN`` and never plans it; a
    permutation store is injective, P1 lifts it, and it exercises the same
    index-vector ``OwnerElements`` footprint.
    """
    n = 128

    @T.prim_func
    def load_k(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "int32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float32")
            b = T.alloc_shared((n,), "int32")
            c = T.alloc_shared((n,), "float32")
            T.copy(A, a)
            T.copy(B, b)
            for i in T.Parallel(n):
                c[i] = a[b[i]]
            T.copy(c, C)

    @T.prim_func
    def store_k(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float32")
            c = T.alloc_shared((n,), "float32")
            T.copy(A, a)
            for i in T.Parallel(n):
                c[n - 1 - i] = a[i]
            T.copy(c, C)

    assert ("a", 128) in _declared_extents(emit_plan(_plan(load_k)))
    assert ("c", 128) in _declared_extents(emit_plan(_plan(store_k)))
