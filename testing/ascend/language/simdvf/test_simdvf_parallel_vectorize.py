"""Device regression for the P1->P2->P3 ``T.Parallel`` lowering: core shapes.

Every case is a ``T.SimdVF`` kernel whose ``T.Parallel`` nest is lifted by P1,
planned by P2 and emitted by P3.  The test first asserts that the generated CCE
really contains the expected vector intrinsic (so a silent scalar lowering
cannot pass), then runs the kernel on an Ascend NPU and
compares it with a torch golden computed from the *scalar* semantics of the
source program.

Covers 1-D elementwise kernels (whole chunks, in-place read-modify-write, two
VFs in one kernel, ``T.max``/``T.min`` NaN and ``+-0`` contract, tail chunks,
8-bit whole chunks / byte tails / ``i8 <-> u8`` reinterpret), N-D nests
(two/three-axis collapse, outer-axis broadcast, flattened ``i // N, i % N``
indexing) and index transformations (stride-2 loads and stores, reversal,
indirect runtime gather, transposed scatter).
"""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T

from testing.ascend.language.simdvf._simdvf_parallel_helpers import (
    F32,
    INF,
    M,
    N,
    NAN,
    _assert_close,
    _compile,
    _require_npu_runtime,
)


@pytest.fixture(autouse=True)
def _npu_runtime():
    _require_npu_runtime()


# ---------------------------------------------------------------------------
# 1-D elementwise: whole chunks, tails, multiple VFs and 8-bit stores
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_multiple_independent_stores():
    @T.prim_func
    def main(A: T.Tensor((N,), F32), C: T.Tensor((N,), F32), D: T.Tensor((N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            c_ub = T.alloc_shared((N,), F32)
            d_ub = T.alloc_shared((N,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    c_ub[i] = a_ub[i] + 1.0
                    d_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    kernel, source = _compile(main, out_idx=[-2, -1])
    assert "simd_inst::vadds(" in source and "simd_inst::vmuls(" in source
    assert source.count("simd_inst::vsts_norm(") == 2

    a = (torch.arange(N, dtype=torch.float32) % 17) - 8.0
    c_npu, d_npu = kernel(a.npu())
    _assert_close(c_npu, a + 1.0)
    _assert_close(d_npu, a * 2.0)


@tilelang.testing.requires_ascend
def test_parallel_in_place_update_without_a_shell():
    """``c[i] = c[i] + a[i]``: one vector pass over the buffer it reads.

    The chunk program is not wrapped in a scalar shell, so the emitter must not
    add a VST->VLD barrier: the chunks are independent and each ``vsts`` closes
    its own value chain.
    """
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), CIN: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(CIN, c_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = c_ub[i] + a_ub[i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vlds_norm<" in source
    assert "simd_inst::vadd(" in source
    assert "simd_inst::vsts_norm(" in source
    assert "simd_inst::mem_bar(" not in source, "no scalar shell, no barrier"

    a = (torch.arange(n, dtype=torch.float32) % 13) - 6.0
    c_in = (torch.arange(n, dtype=torch.float32) % 5) - 2.0
    _assert_close(kernel(a.npu(), c_in.npu()), c_in + a)


# Every row is one operand pair; the mixed NaN/+-0 rows are the contract.
_MAXMIN_A = [
    NAN,
    1.0,
    NAN,
    1.0,  # NaN left / right / both
    0.0,
    -0.0,
    0.0,
    -0.0,  # +-0 in both orders
    INF,
    -INF,
    1.0,
    -1.0,  # infinities and ordinary values
    0.0,
    1.0,
    NAN,
    NAN,
]
_MAXMIN_B = [
    1.0,
    NAN,
    NAN,
    1.0,
    -0.0,
    0.0,
    0.0,
    -0.0,
    -INF,
    INF,
    1.0,
    -1.0,
    0.0,
    0.0,
    2.0,
    NAN,
]


def _maxmin_kernel(n, kind):
    @T.prim_func
    def main(A: T.Tensor((n,), F32), B: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            b_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if kind == "max":
                        c_ub[i] = T.max(a_ub[i], b_ub[i])
                    else:
                        c_ub[i] = T.min(a_ub[i], b_ub[i])
            T.copy(c_ub, C)

    return main


@pytest.mark.parametrize("kind", ["max", "min"])
@tilelang.testing.requires_ascend
def test_parallel_max_min_nan_and_signed_zero(kind):
    n = 128
    a = torch.tensor(_MAXMIN_A * (n // len(_MAXMIN_A)), dtype=torch.float32)
    b = torch.tensor(_MAXMIN_B * (n // len(_MAXMIN_B)), dtype=torch.float32)

    kernel, source = _compile(_maxmin_kernel(n, kind))
    assert f"simd_inst::v{kind}(" in source

    actual = kernel(a.npu(), b.npu()).cpu()
    # Pinned contract: the CANN ``asc_max`` / ``asc_min`` instructions, not the
    # scalar ``a > b ? a : b`` ternary.  A NaN operand propagates; the sign of a
    # zero result only survives when *both* operands carry it (``max`` keeps
    # ``-0`` only for two ``-0``; ``min`` keeps ``+0`` only for two ``+0``).
    nan = torch.isnan(a) | torch.isnan(b)
    both_zero = (a == 0) & (b == 0)
    expected = torch.full_like(a, float("nan"))
    if kind == "max":
        expected[~nan] = torch.maximum(a[~nan], b[~nan])
        keep_neg = both_zero & torch.signbit(a) & torch.signbit(b)
        expected[both_zero & ~keep_neg] = 0.0
        expected[keep_neg] = -0.0
    else:
        expected[~nan] = torch.minimum(a[~nan], b[~nan])
        keep_pos = both_zero & ~torch.signbit(a) & ~torch.signbit(b)
        expected[both_zero & ~keep_pos] = -0.0
        expected[keep_pos] = 0.0

    _assert_close(actual, expected)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros])), (
        "the sign of +-0 must follow the CANN max/min contract"
    )


@tilelang.testing.requires_ascend
def test_parallel_elementwise_tail():
    n = 1000  # 15 full float32 chunks (64 lanes) + a 40-lane tail

    @T.prim_func
    def main(A: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "update_mask_b32" in source, "the tail chunk must be predicated"

    a = torch.arange(n, dtype=torch.float32)
    _assert_close(kernel(a.npu()), a * 2.0)


@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@tilelang.testing.requires_ascend
def test_parallel_eight_bit_full_chunk(dtype):
    n = 256
    torch_dtype = getattr(torch, dtype)

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), dtype)
            c_ub = T.alloc_shared((n,), dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vlds_norm<" in source
    assert "simd_inst::vsts_norm(" in source
    assert "simd_inst::vgather2(" not in source, "a whole byte register uses the NORM path"

    if dtype == "uint8":
        a = (torch.arange(n, dtype=torch.int32) % 256).to(torch_dtype)
    else:
        a = (torch.arange(n, dtype=torch.int32) % 251 - 125).to(torch_dtype)
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.uint8), a.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@tilelang.testing.requires_ascend
def test_parallel_eight_bit_elementwise_tail(dtype):
    """An 8-bit tail is gathered into 16-bit lanes and written with ``PK_B16``.

    A ``NORM_B8`` store would reinterpret those 16-bit lanes as interleaved
    bytes and overrun the buffer (the row is shorter than one byte register).
    """
    n = 100  # one 256-byte chunk, so the whole vector is a 100-lane tail
    torch_dtype = getattr(torch, dtype)

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), dtype)
            c_ub = T.alloc_shared((n,), dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2(" in source, "the byte tail load is a widening gather"
    assert "vsts_pack_b16" in source, "the widened value is narrowed back by the pack store"

    if dtype == "uint8":
        a = (torch.arange(n, dtype=torch.int32) % 256).to(torch_dtype)
    else:
        a = (torch.arange(n, dtype=torch.int32) % 251 - 125).to(torch_dtype)
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.uint8), a.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("src_dtype", ["uint8", "int8"])
@tilelang.testing.requires_ascend
def test_parallel_eight_bit_same_width_reinterpret_tail(src_dtype):
    """``i8 <-> u8`` keeps the byte gather: bit identity through the pack store."""
    n = 100
    dst_dtype = "int8" if src_dtype == "uint8" else "uint8"
    src_torch = getattr(torch, src_dtype)

    @T.prim_func
    def main(A: T.Tensor((n,), src_dtype), C: T.Tensor((n,), dst_dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), src_dtype)
            c_ub = T.alloc_shared((n,), dst_dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast(dst_dtype, a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2(" in source
    assert "vsts_pack_b16" in source

    a = (torch.arange(n, dtype=torch.int32) % 256).to(src_torch)
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.uint8), a.view(torch.uint8), rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_parallel_two_vfs_in_one_kernel():
    """Two ``T.SimdVF`` blocks in one kernel are planned independently.

    The first VF rewrites ``a_ub`` in place, the second consumes it, so the test
    also pins that the two emitted programs run in source order.
    """
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), B: T.Tensor((n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            b_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    a_ub[i] = a_ub[i] * 2.0
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + b_ub[i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vmuls(" in source, "the first VF scales in place"
    assert "simd_inst::vadd(" in source, "the second VF adds the scaled value"
    assert source.count("simd_inst::vsts_norm(") == 2, "one store per VF (its 2-chunk loop)"

    a = (torch.arange(n, dtype=torch.float32) % 9) - 4.0
    b = (torch.arange(n, dtype=torch.float32) % 5) - 2.0
    _assert_close(kernel(a.npu(), b.npu()), a * 2.0 + b)


# ---------------------------------------------------------------------------
# N-D nests: collapse, broadcast and flattened indexing
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_two_axis_elementwise():
    @T.prim_func
    def main(A: T.Tensor((M, N), F32), C: T.Tensor((M, N), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            c_ub = T.alloc_shared((M, N), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[i, j] = a_ub[i, j] * 2.0 + 1.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vmuls(" in source and "simd_inst::vadds(" in source
    assert "simd_inst::vgather2(" not in source, "both axes are affine and contiguous"

    a = (torch.arange(M * N, dtype=torch.float32) % 11) - 5.0
    _assert_close(kernel(a.reshape(M, N).npu()), (a * 2.0 + 1.0).reshape(M, N))


@tilelang.testing.requires_ascend
def test_parallel_three_axis_collapse():
    """The P1 "collapse" is a region collapse, not a full flattening.

    ``T.Parallel(2, 4, 64)`` becomes one region with ``axes = [i, j, k]``; the
    vector axis is the innermost one, so the emitted program keeps ``i`` and
    ``j`` as serial loops and chunks only the 64 lanes of ``k``. Flattening all
    512 elements into 8 chunks is not what the name promises, and not what the
    planner does.
    """
    shape = (2, 4, 64)

    @T.prim_func
    def main(A: T.Tensor(shape, F32), C: T.Tensor(shape, F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared(shape, F32)
            c_ub = T.alloc_shared(shape, F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j, k in T.Parallel(2, 4, 64):
                    c_ub[i, j, k] = a_ub[i, j, k] + 1.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vadds(" in source
    assert "update_mask" not in source, "2 * 4 * 64 is a whole number of chunks"

    a = torch.arange(2 * 4 * 64, dtype=torch.float32).reshape(shape)
    _assert_close(kernel(a.npu()), a + 1.0)


@tilelang.testing.requires_ascend
def test_parallel_outer_axis_broadcast():
    @T.prim_func
    def main(A: T.Tensor((M, N), F32), B: T.Tensor((M,), F32), C: T.Tensor((M, N), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            b_ub = T.alloc_shared((M,), F32)
            c_ub = T.alloc_shared((M, N), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[i, j] = a_ub[i, j] + b_ub[i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vadds(" in source, "the outer operand is uniform inside a chunk"

    a = (torch.arange(M * N, dtype=torch.float32) % 13) - 6.0
    b = torch.arange(M, dtype=torch.float32) * 0.5
    _assert_close(kernel(a.reshape(M, N).npu(), b.npu()), a.reshape(M, N) + b[:, None])


@tilelang.testing.requires_ascend
def test_parallel_flattened_divmod_index():
    @T.prim_func
    def main(A: T.Tensor((M, N), F32), C: T.Tensor((M * N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            c_ub = T.alloc_shared((M * N,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(M * N):
                    c_ub[i] = a_ub[i // N, i % N] + 1.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vadds(" in source

    a = (torch.arange(M * N, dtype=torch.float32) % 7) - 3.0
    _assert_close(kernel(a.reshape(M, N).npu()), a + 1.0)


# ---------------------------------------------------------------------------
# index transformations: stride, permute, gather and scatter
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_strided_load():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((2 * n,), F32), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2 * n,), F32)
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[2 * i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2(" in source, "a stride-2 load uses the per-lane index vector"

    a = torch.arange(2 * n, dtype=torch.float32)
    _assert_close(kernel(a.npu()), a[0::2])


@tilelang.testing.requires_ascend
def test_parallel_strided_store():
    """``c[2i] = a[i] * 2`` scatters with a per-lane index vector.

    Only the even destinations are written, so every odd lane must keep the
    value it had before the kernel.
    """
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), CIN: T.Tensor((2 * n,), F32), C: T.Tensor((2 * n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((2 * n,), F32)
            T.copy(A, a_ub)
            T.copy(CIN, c_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[2 * i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vscatter(" in source, "a stride-2 store is a per-lane scatter"
    assert "simd_inst::vsts_norm(" not in source

    a = (torch.arange(n, dtype=torch.float32) % 11) - 5.0
    c_in = torch.full((2 * n,), -9.0, dtype=torch.float32)
    expected = c_in.clone()
    expected[0::2] = a * 2.0
    _assert_close(kernel(a.npu(), c_in.npu()), expected)


@tilelang.testing.requires_ascend
def test_parallel_reversal_is_a_permute_gather():
    @T.prim_func
    def main(A: T.Tensor((N,), F32), C: T.Tensor((N,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), F32)
            c_ub = T.alloc_shared((N,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(N):
                    c_ub[i] = a_ub[N - 1 - i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2(" in source, "the reversed index is a permutation gather"

    a = torch.arange(N, dtype=torch.float32)
    _assert_close(kernel(a.npu()), a.flip(0))


@tilelang.testing.requires_ascend
def test_parallel_indirect_gather():
    """A runtime index declares the whole owner; the values must still be exact."""
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), B: T.Tensor((n,), "int32"), C: T.Tensor((n,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            b_ub = T.alloc_shared((n,), "int32")
            c_ub = T.alloc_shared((n,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[b_ub[i]]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2" in source

    a = torch.arange(n, dtype=torch.float32)
    b = torch.randperm(n).to(torch.int32)
    _assert_close(kernel(a.npu(), b.npu()), a[b.long()])


@tilelang.testing.requires_ascend
def test_parallel_transposed_store():
    @T.prim_func
    def main(A: T.Tensor((M, N), F32), C: T.Tensor((N, M), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            c_ub = T.alloc_shared((N, M), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[j, i] = a_ub[i, j]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vscatter(" in source, "the transposed store is a per-lane scatter"

    a = (torch.arange(M * N, dtype=torch.float32) % 9) - 4.0
    _assert_close(kernel(a.reshape(M, N).npu()), a.reshape(M, N).t())
