"""Device regression for the P1->P2->P3 ``T.Parallel`` lowering: guards.

Guarded stores: single-arm predication (also combined with a tail), lane-uniform
``update_mask``, same-address ``vsel``, nested ``else if``, divergent two-arm
stores, ``and``/``or``/``not`` predicates, mask ``==``/``!=``, and two arms
writing disjoint addresses of the *same* buffer (guard-aware legality).
"""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T

from testing.ascend.language.simdvf._simdvf_parallel_helpers import (
    F32,
    INF,
    N,
    NAN,
    _assert_close,
    _compile,
    _cycle,
    _require_npu_runtime,
)


@pytest.fixture(autouse=True)
def _npu_runtime():
    _require_npu_runtime()


# ---------------------------------------------------------------------------
# guarded stores: single arm, select, divergent arms and masks
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_predicated_single_arm_store():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(C, c_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if a_ub[i] > 0.0:
                        c_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main, out_idx=None)

    # NaN / -0 / +0 / infinities around the boundary (8 values, so it divides 128).
    a = torch.tensor(
        [NAN, -1.0, -0.0, 0.0, 1.0, -2.0, INF, -INF] * (n // 8),
        dtype=torch.float32,
    )
    c_in = torch.arange(n, dtype=torch.float32) * 0.5
    c_npu = c_in.npu()
    kernel(a.npu(), c_npu)
    assert "vcmps_gt" in source, "the guard must be a vector compare"
    # The store is masked, so inactive lanes keep the old destination value.
    expected = torch.where(a > 0.0, a * 2.0, c_in)
    _assert_close(c_npu, expected)


@tilelang.testing.requires_ascend
def test_parallel_tail_with_a_statement_predicate():
    """The tail mask and a statement predicate are AND-ed into one store mask.

    The whole chunks use ``vcmps`` as the store mask directly; the tail chunk
    loads through a masked gather and combines the two masks with ``pand``.
    """
    n = 300  # 4 whole float32 chunks (64 lanes) + a 44-lane tail

    @T.prim_func
    def main(A: T.Tensor((n,), F32), CIN: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(CIN, c_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if a_ub[i] > 0.0:
                        c_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcmps_gt" in source
    assert "update_mask" in source
    assert "simd_inst::vgather2(" in source, "the tail chunk is a masked gather load"
    assert "simd_inst::pand(" in source, "statement predicate AND-ed with the tail mask"

    a = _cycle([-3.0, -0.0, 0.0, 1.0, 2.5, -1.5, 4.0, -2.0], n)
    c_in = (torch.arange(n, dtype=torch.float32) % 7) - 3.0
    expected = torch.where(a > 0.0, a * 2.0, c_in)
    _assert_close(kernel(a.npu(), c_in.npu()), expected)


@tilelang.testing.requires_ascend
def test_parallel_lane_uniform_condition_broadcasts_the_mask():
    @T.prim_func
    def main(A: T.Tensor((N,), F32), B: T.Tensor((1,), F32), C: T.Tensor((N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            b_ub = T.alloc_shared((1,), F32)
            c_ub = T.alloc_shared((N,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(C, c_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    if b_ub[0] > 0.0:
                        c_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main, out_idx=None)
    # A lane-uniform guard must not become a scalar compare/select inside
    # ``__simd_vf__`` (bisheng: "Unsupported scalar instruction in AIV loop").
    # It is a vector predicate instead: broadcast one operand and compare it.
    # The comparison is canonicalised (literal first), so ``b[0] > 0.0`` arrives
    # as ``0.0 < b[0]`` and lowers to ``vcmps_lt(vdup(0.0f), b[0])``: assert the
    # instruction family, not the authored direction.
    assert "simd_inst::vdup<" in source, "the scalar guard is broadcast, not scalarized"
    assert "simd_inst::vcmps_" in source, "the guard becomes a vector mask"
    assert "update_mask" not in source, "no scalar-select mask inside the VF"

    a = (torch.arange(N, dtype=torch.float32) % 5) - 2.0
    c_in = torch.full((N,), -7.0, dtype=torch.float32)
    positive = c_in.clone().npu()
    kernel(a.npu(), torch.ones(1, dtype=torch.float32).npu(), positive)
    _assert_close(positive, a * 2.0)

    negative = c_in.clone().npu()
    kernel(a.npu(), torch.full((1,), -1.0, dtype=torch.float32).npu(), negative)
    _assert_close(negative, c_in)


@tilelang.testing.requires_ascend
def test_parallel_two_arm_same_address_becomes_a_select():
    @T.prim_func
    def main(A: T.Tensor((N,), F32), C: T.Tensor((N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            c_ub = T.alloc_shared((N,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    if a_ub[i] > 0.0:
                        c_ub[i] = a_ub[i] * 2.0
                    else:
                        c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vsel(" in source, "both arms share an address: one register select"
    assert "simd_inst::vcmps_gt" in source

    a = _cycle([-3.0, -0.0, 0.0, 1.0, 2.5, -1.5], N)
    _assert_close(kernel(a.npu()), torch.where(a > 0.0, a * 2.0, a + 1.0))


@tilelang.testing.requires_ascend
def test_parallel_nested_else_if():
    @T.prim_func
    def main(A: T.Tensor((N,), F32), C: T.Tensor((N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            c_ub = T.alloc_shared((N,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    if a_ub[i] > 0.0:
                        c_ub[i] = 1.0
                    else:
                        if a_ub[i] < -1.0:
                            c_ub[i] = 2.0
                        else:
                            c_ub[i] = 3.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert source.count("simd_inst::vsel(") == 2, "the nested else-if is two selects"

    a = _cycle([-3.0, -1.5, -1.0, 0.0, 1.0], N)
    expected = torch.where(a > 0.0, 1.0, torch.where(a < -1.0, 2.0, 3.0))
    _assert_close(kernel(a.npu()), expected)


@tilelang.testing.requires_ascend
def test_parallel_divergent_two_arm_store():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), C: T.Tensor((n,), F32), D: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            d_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(C, c_ub)
            T.copy(D, d_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if a_ub[i] > 0.0:
                        c_ub[i] = a_ub[i]
                    else:
                        d_ub[i] = 1.0
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    kernel, source = _compile(main, out_idx=None)
    assert "simd_inst::pnot(" in source, "each arm keeps its own complementary mask"

    a = torch.tensor([NAN, -1.0, 0.0, 1.0, 2.0, -3.0, 4.0, -4.0] * (n // 8), dtype=torch.float32)
    c_in = torch.full((n,), -7.0, dtype=torch.float32)
    d_in = torch.full((n,), -9.0, dtype=torch.float32)
    c_npu, d_npu = c_in.npu(), d_in.npu()
    kernel(a.npu(), c_npu, d_npu)

    branch = a > 0.0
    _assert_close(c_npu, torch.where(branch, a, c_in))
    _assert_close(d_npu, torch.where(branch, d_in, torch.ones_like(d_in)))


@tilelang.testing.requires_ascend
def test_parallel_condition_and_or_not():
    @T.prim_func
    def main(
        A: T.Tensor((N,), F32),
        B: T.Tensor((N,), F32),
        C1: T.Tensor((N,), F32),
        C2: T.Tensor((N,), F32),
        C3: T.Tensor((N,), F32),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            b_ub = T.alloc_shared((N,), F32)
            c1_ub = T.alloc_shared((N,), F32)
            c2_ub = T.alloc_shared((N,), F32)
            c3_ub = T.alloc_shared((N,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.fill(c1_ub, 0.0)
            T.fill(c2_ub, 0.0)
            T.fill(c3_ub, 0.0)
            with T.SimdVF():
                for i in T.Parallel(N):
                    if (a_ub[i] > 0.0) and (b_ub[i] > 0.0):
                        c1_ub[i] = 1.0
                    if (a_ub[i] > 0.0) or (b_ub[i] > 0.0):
                        c2_ub[i] = 2.0
                    if not (a_ub[i] > 0.0):
                        c3_ub[i] = 3.0
            T.copy(c1_ub, C1)
            T.copy(c2_ub, C2)
            T.copy(c3_ub, C3)

    kernel, source = _compile(main, out_idx=[-3, -2, -1])
    assert "simd_inst::pand(" in source
    assert "simd_inst::por(" in source
    # ``not (a > 0)`` is folded into ``a <= 0`` by ``Simplify`` before lowering,
    # so the complementary mask shows up as a comparison instead of a ``pnot``.
    assert "simd_inst::pnot(" in source or "simd_inst::vcmps_le(" in source

    a = _cycle([-1.0, -0.0, 0.0, 1.0, 2.0, -3.0], N)
    b = _cycle([1.0, 2.0, -1.0, -2.0, 3.0, 0.0], N)
    c1, c2, c3 = kernel(a.npu(), b.npu())
    pa, pb = a > 0.0, b > 0.0
    _assert_close(c1, torch.where(pa & pb, 1.0, torch.zeros_like(a)))
    _assert_close(c2, torch.where(pa | pb, 2.0, torch.zeros_like(a)))
    _assert_close(c3, torch.where(~pa, 3.0, torch.zeros_like(a)))


def _mask_eq_kernel(n, op):
    @T.prim_func
    def main(A: T.Tensor((n,), F32), B: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            b_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(C, c_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if op == "eq":
                        if (a_ub[i] > 0.0) == (b_ub[i] > 0.0):
                            c_ub[i] = 1.0
                    else:
                        if (a_ub[i] > 0.0) != (b_ub[i] > 0.0):
                            c_ub[i] = 1.0
            T.copy(c_ub, C)

    return main


@pytest.mark.parametrize("op", ["eq", "ne"])
@tilelang.testing.requires_ascend
def test_parallel_mask_equality(op):
    n = 128
    kernel, source = _compile(_mask_eq_kernel(n, op), out_idx=None)
    assert "simd_inst::pxor(" in source, "mask ==/!= is emitted through the predicate truth table"
    assert "simd_inst::vcmp(" not in source

    a = torch.tensor([NAN, 1.0, -1.0, 0.0, 2.0, -0.0, NAN, 3.0] * (n // 8), dtype=torch.float32)
    b = torch.tensor([1.0, NAN, 1.0, -0.0, -2.0, 0.0, NAN, 0.0] * (n // 8), dtype=torch.float32)
    c_in = torch.full((n,), -5.0, dtype=torch.float32)
    c_npu = c_in.npu()
    kernel(a.npu(), b.npu(), c_npu)

    same = (a > 0.0) == (b > 0.0)
    if op == "ne":
        same = ~same
    _assert_close(c_npu, torch.where(same, torch.ones_like(c_in), c_in))


@tilelang.testing.requires_ascend
def test_parallel_same_buffer_disjoint_arm_addresses():
    """``c[2i]`` / ``c[2i+1]`` never collide, so the guard-aware prover lifts it."""

    @T.prim_func
    def main(A: T.Tensor((N,), F32), C: T.Tensor((2 * N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            c_ub = T.alloc_shared((2 * N,), F32)
            T.copy(A, a_ub)
            T.copy(C, c_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    if a_ub[i] > 0.0:
                        c_ub[2 * i] = 1.0
                    else:
                        c_ub[2 * i + 1] = 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main, out_idx=None)
    assert "simd_inst::vscatter(" in source
    assert "simd_inst::pnot(" in source, "the else arm uses the complementary mask"

    a = _cycle([-1.0, 0.0, 1.0, -2.0, 3.0, -0.0], N)
    c_in = torch.full((2 * N,), -9.0, dtype=torch.float32)
    c_npu = c_in.clone().npu()
    kernel(a.npu(), c_npu)

    expected = c_in.clone()
    positive = a > 0.0
    expected[0::2][positive] = 1.0
    expected[1::2][~positive] = 2.0
    _assert_close(c_npu, expected)


# ---------------------------------------------------------------------------
# structural plan (design A): a lane-uniform branch with sibling element-wise loops
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_lane_uniform_branch_with_sibling_elementwise_loops():
    """``if flag[0] > 0 { for m: s[i,m] = a[i,m]*2; for n: u[i,n] = b[i,n]+1 }``.

    A lane-uniform ``if`` around two sibling element-wise row transforms: the
    branch must become a store mask, never scalar control flow inside
    ``__simd_vf__`` (bisheng/AIV rejects it and InsertSync only tracks
    ``PIPE_V``).  ``m``/``n`` are contiguous and their stores index the loop
    variable, so both serial loops are absorbed as the vector axes: the VF keeps
    only the ``i`` loop and runs 64-lane chunks with ``mask = flag[0] > 0``.
    The flag is run both set and clear, so the masked-off arm is checked too.

    A row *reduction* cannot be expressed this way: it needs a ``T.Parallel``
    axis (``s[i] += a[i,m]`` cannot use ``m`` as the lane axis), but P1 only
    gives the top-level rectangular nest a region, so a ``T.Parallel`` under an
    ``if`` stays OPAQUE and a guarded cross-lane reduction is unsupported.  This
    case therefore covers the element-wise shape the pipeline supports.
    """
    rows, k = 8, 64

    @T.prim_func
    def main(
        FLAG: T.Tensor((1,), F32),
        A: T.Tensor((rows, k), F32),
        B: T.Tensor((rows, k), F32),
        S: T.Tensor((rows, k), F32),
        U: T.Tensor((rows, k), F32),
    ):
        with T.Kernel(1):
            flag_ub = T.alloc_shared((1,), F32)
            a_ub = T.alloc_shared((rows, k), F32)
            b_ub = T.alloc_shared((rows, k), F32)
            s_ub = T.alloc_shared((rows, k), F32)
            u_ub = T.alloc_shared((rows, k), F32)
            T.copy(FLAG, flag_ub)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.fill(s_ub, -1.0)
            T.fill(u_ub, -2.0)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    if flag_ub[0] > 0.0:
                        for m in T.serial(k):
                            s_ub[i, m] = a_ub[i, m] * 2.0
                        for n in T.serial(k):
                            u_ub[i, n] = b_ub[i, n] + 1.0
            T.copy(s_ub, S)
            T.copy(u_ub, U)

    kernel, source = _compile(main, out_idx=None)
    assert "if (" not in source, "the VF contract has no scalar control flow"
    assert "simd_inst::vcmps" in source, "the guard becomes a vector store mask"
    assert source.count("simd_inst::vsts_norm(") == 2, "one store per sibling transform"
    assert "for (int32_t m" not in source, "the m loop is absorbed as the vector axis"
    assert "for (int32_t n" not in source, "the n loop is absorbed as the vector axis"
    assert "simd_inst::mem_bar(" not in source, "element-wise chunks need no VST->VLD barrier"
    assert "vcadd(" not in source, "no cross-lane reduction is involved"

    a = ((torch.arange(rows * k, dtype=torch.int32).reshape(rows, k) % 7) - 3).to(torch.float32)
    b = (torch.arange(rows * k, dtype=torch.int32).reshape(rows, k) % 5).to(torch.float32)

    s_on = torch.full((rows, k), -1.0, dtype=torch.float32).npu()
    u_on = torch.full((rows, k), -2.0, dtype=torch.float32).npu()
    kernel(torch.tensor([1.0]).npu(), a.npu(), b.npu(), s_on, u_on)
    _assert_close(s_on, a * 2.0)
    _assert_close(u_on, b + 1.0)

    # The dead arm must leave both destinations untouched.
    s_off = torch.full((rows, k), -1.0, dtype=torch.float32).npu()
    u_off = torch.full((rows, k), -2.0, dtype=torch.float32).npu()
    kernel(torch.tensor([-1.0]).npu(), a.npu(), b.npu(), s_off, u_off)
    _assert_close(s_off, torch.full((rows, k), -1.0))
    _assert_close(u_off, torch.full((rows, k), -2.0))
