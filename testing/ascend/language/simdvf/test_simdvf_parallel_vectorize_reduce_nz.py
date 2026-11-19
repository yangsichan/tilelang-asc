"""Device regression for the P1->P2->P3 ``T.Parallel`` lowering: reduce/cast/NZ.

Reductions and serial shells (row ``max``/``min``, masked-tail reduction, shell
reductions, unrolled / kept / runtime-bound shells, a serial shell around the
nest with the VST->VLD barrier, sibling reduction shells), cast stores
(narrowing, same-width signedness reinterpret, widening ``UNPK_*`` including
the 8-bit tail and the full ``vcvt`` pair matrix), and NZ layout / pack2
(annotated compact-NZ stores, elementwise-NZ + row-reduction pair, paired
``vld_x2`` + ``vor`` + ``vsstb``).
"""

import math

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T
from tilelang.layout import make_ascend_compact_nz_layout

from testing.ascend.language.simdvf._simdvf_parallel_helpers import (
    F32,
    M,
    N,
    _assert_close,
    _compile,
    _nz_physical,
    _require_npu_runtime,
)


@pytest.fixture(autouse=True)
def _npu_runtime():
    _require_npu_runtime()


F32_EPS = 2.0**-24


def _assert_tree_sum_close(actual, values):
    """Compare a vector sum against the exact (float64) sum with a real bound.

    ``vcadd`` sums one register as a balanced tree (depth ``ceil(log2(chunk))``)
    and the chunk results are combined with one add per chunk, so the
    worst-case relative error with respect to ``sum(|x|)`` is ``(depth + 1) *
    eps``. The same bound covers the source accumulator's initialization and
    the final write-back; the reference is computed in float64 so it adds no
    rounding of its own. The max abs / rel error and ULP are printed so a
    regression shows the actual error magnitude.
    """
    actual = actual.cpu().double()
    exact = values.double().sum(dim=1)
    abs_sum = values.double().abs().sum(dim=1)
    depth = math.ceil(math.log2(values.shape[1])) if values.shape[1] > 1 else 0
    bound = (depth + 1) * F32_EPS * abs_sum
    error = (actual - exact).abs()
    max_abs = error.max().item()
    max_rel = (error / exact.abs().clamp_min(torch.finfo(torch.float64).tiny)).max().item()
    ulp = torch.pow(2.0, torch.floor(torch.log2(exact.abs().clamp_min(2.0**-126))) - 23)
    max_ulp = (error / ulp).max().item()
    print(f"tree sum: max_abs={max_abs:.6e} max_rel={max_rel:.6e} max_ulp={max_ulp:.2f}")
    assert bool(torch.all(error <= bound)), (
        f"tree-sum error {max_abs:.6e} exceeds the bound {bound.max().item():.6e} (depth={depth}, eps={F32_EPS:.3e})"
    )


# ---------------------------------------------------------------------------
# reductions and serial shells
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["max", "min"])
@tilelang.testing.requires_ascend
def test_parallel_row_reduction(kind):
    rows, k = 8, 128
    identity = -3.4028234663852886e38 if kind == "max" else 3.4028234663852886e38

    @T.prim_func
    def main(A: T.Tensor((rows, k), F32), D: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, k), F32)
            d_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            T.fill(d_ub, identity)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for j in T.serial(k):
                        if kind == "max":
                            d_ub[i] = T.max(d_ub[i], a_ub[i, j])
                        else:
                            d_ub[i] = T.min(d_ub[i], a_ub[i, j])
            T.copy(d_ub, D)

    kernel, source = _compile(main)
    intrinsic = "simd_inst::vcmax(" if kind == "max" else "simd_inst::vcmin("
    assert intrinsic in source, "the row accumulator is a cross-lane reduction"

    a = (torch.arange(rows * k, dtype=torch.float32) % 37) - 18.0
    expected = a.reshape(rows, k).max(dim=1).values if kind == "max" else a.reshape(rows, k).min(dim=1).values
    _assert_close(kernel(a.reshape(rows, k).npu()), expected)


@tilelang.testing.requires_ascend
def test_parallel_row_reduction_with_a_tail():
    """A row that is not a whole number of chunks still reduces in registers.

    The whole chunk is a ``vcadd``; the tail chunk is a masked gather followed
    by a ``vcadd`` under the tail mask, and both partial sums are combined on
    lane 0 before the single ``ONEPT`` store.

    The row stride is kept 32B-aligned (104 f32 = 416B = 13 blocks): block
    ``vlds_norm``/``vsts_norm`` instructions need a 32B-aligned base, and a
    row stride that is not a multiple of 32B (e.g. 100 f32 = 400B) is rejected
    by P2 (``plan.alignment``).
    """
    rows, k = 8, 104  # 64 + 40 lanes, 416B per row

    @T.prim_func
    def main(A: T.Tensor((rows, k), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, k), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            T.fill(c_ub, 0.0)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for j in T.serial(k):
                        c_ub[i] = c_ub[i] + a_ub[i, j]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert source.count("simd_inst::vcadd(") == 2, "one whole-chunk and one tail reduction"
    assert "update_mask" in source, "the tail chunk reduces under a partial mask"
    assert "simd_inst::vgather2(" in source, "the tail load is a masked gather"

    # Float sum: vcadd changes the accumulation order, so the golden is the
    # exact float64 sum and the assertion uses the tree-reduction error bound.
    a = torch.randn(rows, k, dtype=torch.float32)
    _assert_tree_sum_close(kernel(a.npu()), a)


@tilelang.testing.requires_ascend
def test_parallel_serial_shell_reduction():
    rows, k = 8, 64

    @T.prim_func
    def main(A: T.Tensor((rows, k), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, k), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            T.fill(c_ub, 0.0)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for j in T.serial(k):
                        c_ub[i] = c_ub[i] + a_ub[i, j]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcadd(" in source, "the shell accumulator is a cross-lane reduction"

    # Float sum: the register tree is not the source's sequential order.
    a = torch.randn(rows, k, dtype=torch.float32)
    _assert_tree_sum_close(kernel(a.npu()), a)


@tilelang.testing.requires_ascend
def test_parallel_serial_shell_last_write_wins():
    rows, steps = 64, 4

    @T.prim_func
    def main(A: T.Tensor((rows, steps), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, steps), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for k in T.serial(steps):
                        c_ub[i] = a_ub[i, k]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert source.count("simd_inst::vsts_norm(") == steps, "the shell is unrolled iteration by iteration"

    a = torch.arange(rows * steps, dtype=torch.float32).reshape(rows, steps)
    _assert_close(kernel(a.npu()), a[:, -1])


@tilelang.testing.requires_ascend
def test_parallel_serial_shell_kept_as_a_scalar_loop():
    """A shell that cannot be absorbed stays a real loop around the vector work.

    ``c[i] = a[i, k]`` writes one address per lane, so ``k`` is neither a vector
    axis nor a reduction; the plan keeps it as a scalar shell instead of falling
    back (and does not unroll it, unlike the constant short shell above).
    """
    rows, steps = 64, 16

    @T.prim_func
    def main(A: T.Tensor((rows, steps), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, steps), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for k in T.serial(steps):
                        c_ub[i] = a_ub[i, k]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    # One store per chunk inside the shell loop: 16 unrolled stores would mean
    # the shell was replicated instead of kept as a loop.
    assert source.count("simd_inst::vsts_norm(") == 1, source
    assert "simd_inst::vgather2(" in source

    a = torch.arange(rows * steps, dtype=torch.float32).reshape(rows, steps)
    _assert_close(kernel(a.npu()), a[:, -1])


@tilelang.testing.requires_ascend
def test_parallel_runtime_shell_is_kept_as_a_scalar_loop():
    rows, steps = 8, 64

    @T.prim_func
    def main(A: T.Tensor((rows, steps), F32), C: T.Tensor((rows,), F32), t: T.int32):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, steps), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for k in T.serial(t):
                        c_ub[i] = a_ub[i, k]
            T.copy(c_ub, C)

    kernel, source = _compile(main, out_idx=None)
    assert "simd_inst::vgather2(" in source, "the lane address stays a gather"
    # One store per chunk inside the shell loop: replicating the shell would
    # multiply the store count by the trip count.
    assert source.count("simd_inst::vsts") == 1

    a = torch.arange(rows * steps, dtype=torch.float32).reshape(rows, steps)
    c_npu = torch.full((rows,), -1.0, dtype=torch.float32).npu()
    kernel(a.npu(), c_npu, steps)
    _assert_close(c_npu, a[:, -1])


@tilelang.testing.requires_ascend
def test_parallel_under_a_serial_shell():
    """A ``T.Parallel`` nested under a serial loop inside the VF is vectorized.

    The serial loop stays scalar and keeps its counter; the vector code replaces
    only the Parallel body, so the address may reference ``k``.
    """
    rows, steps = 128, 4

    @T.prim_func
    def main(A: T.Tensor((steps, rows), F32), C: T.Tensor((rows,), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((steps, rows), F32)
            c_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            T.fill(c_ub, 0.0)
            with T.SimdVF():
                for k in T.serial(steps):
                    for i in T.Parallel(rows):
                        c_ub[i] = c_ub[i] + a_ub[k, i]
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vadd" in source
    assert "simd_inst::vlds_norm<" in source

    a = (torch.arange(steps * rows, dtype=torch.int32).reshape(steps, rows) % 5).to(torch.float32)
    _assert_close(kernel(a.npu()), a.sum(dim=0))


@tilelang.testing.requires_ascend
def test_parallel_sibling_reduction_shells_share_one_region():
    """Two sibling reduction shells stay in one region and both are vectorized.

    P1 models the original outermost ``T.Parallel`` as one region (no fission);
    P2's tree planner materializes each shell independently, so the generated
    code still contains one cross-lane reduction per shell.
    """
    rows, k = 8, 64

    @T.prim_func
    def main(
        A: T.Tensor((rows, k), F32),
        B: T.Tensor((rows, k), F32),
        C: T.Tensor((rows,), F32),
        D: T.Tensor((rows,), F32),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((rows, k), F32)
            b_ub = T.alloc_shared((rows, k), F32)
            c_ub = T.alloc_shared((rows,), F32)
            d_ub = T.alloc_shared((rows,), F32)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.fill(c_ub, 0.0)
            T.fill(d_ub, 0.0)
            with T.SimdVF():
                for i in T.Parallel(rows):
                    for j in T.serial(k):
                        c_ub[i] = c_ub[i] + a_ub[i, j]
                    for j in T.serial(k):
                        d_ub[i] = d_ub[i] + b_ub[i, j]
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    kernel, source = _compile(main, out_idx=[-2, -1])
    assert source.count("simd_inst::vcadd(") == 2, "one reduction per sibling shell"

    a = torch.randn(rows, k, dtype=torch.float32)
    b = torch.randn(rows, k, dtype=torch.float32)
    c_npu, d_npu = kernel(a.npu(), b.npu())
    _assert_tree_sum_close(c_npu, a)
    _assert_tree_sum_close(d_npu, b)


# ---------------------------------------------------------------------------
# cast stores: narrowing, reinterpret, widening and the pair matrix
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_narrowing_cast_store():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), F32), C: T.Tensor((n,), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), F32)
            c_ub = T.alloc_shared((n,), "bfloat16")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast("bfloat16", a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcvt" in source and "PART_EVEN" in source
    assert "vsts_pack_b32" in source, "64 bf16 results are written through the packed form"

    # Exact in both float32 and bfloat16: no rounding-mode dependence.
    a = (torch.arange(n, dtype=torch.float32) % 8) - 4.0
    _assert_close(kernel(a.npu()), a.to(torch.bfloat16))


@pytest.mark.parametrize("src_dtype,pack", [("int32", "vsts_pack_quarter"), ("int16", "vsts_pack_b16")])
@tilelang.testing.requires_ascend
def test_parallel_integer_narrowing_cast_store(src_dtype, pack):
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), src_dtype), C: T.Tensor((n,), "uint8")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), src_dtype)
            c_ub = T.alloc_shared((n,), "uint8")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast("uint8", a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcvt" in source and pack in source

    # Every value is inside the uint8 range, so the expected result is the same
    # whether the narrowing saturates or wraps; only the conversion is checked.
    a = (torch.arange(n, dtype=torch.int32) % 256).to(getattr(torch, src_dtype))
    golden = a.to(torch.uint8)
    _assert_close(kernel(a.npu()), golden)


@tilelang.testing.requires_ascend
def test_parallel_fp16_to_fp8_store():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), "float16"), C: T.Tensor((n,), "float8_e4m3fn")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float16")
            c_ub = T.alloc_shared((n,), "float8_e4m3fn")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast("float8_e4m3fn", a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcvt" in source and "vsts_pack_quarter" in source

    # All values are exactly representable in e4m3 (3 mantissa bits).
    values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 8.0]
    a = torch.tensor(values * (n // len(values)), dtype=torch.float16)
    golden = a.to(torch.float8_e4m3fn)
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.uint8), golden.view(torch.uint8), rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_parallel_same_width_signedness_reinterpret():
    """``i32 -> u32`` is bit-identical: a reinterpret, never a ``vcvt``."""
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), "int32"), C: T.Tensor((n,), "uint32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "int32")
            c_ub = T.alloc_shared((n,), "uint32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast("uint32", a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vcvt" not in source, "a same-width sign change costs no instruction"
    assert "simd_inst::vld" in source and "simd_inst::vsts" in source

    # Every bit pattern must survive, including the sign bit and INT32_MIN.
    a = torch.tensor([-2147483648, -1, -12345, 0, 1, 12345, 2147483647, -7] * (n // 8), dtype=torch.int32)
    actual = kernel(a.npu()).cpu()
    torch.testing.assert_close(actual.view(torch.int32), a, rtol=0, atol=0)


_WIDENING_CASES = (
    ("float16", "float32", "vlds_unpack", "PART_EVEN"),
    ("int16", "int32", "vlds_unpack", "PART_EVEN"),
    ("int8", "int32", "vlds_unpack4", "PART_P0"),
    ("int8", "int16", "vlds_unpack", "PART_EVEN"),
    ("uint8", "uint16", "vlds_unpack", "PART_EVEN"),
    ("uint8", "float16", "vlds_unpack", "PART_EVEN"),
    ("int8", "float16", "vlds_unpack", "PART_EVEN"),
    ("float8_e4m3fn", "float32", "vlds_unpack4", "PART_P0"),
)

_FP8_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 8.0]


def _cast_kernel(n, src_dtype, dst_dtype):
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

    return main


@pytest.mark.parametrize(
    "src_dtype,dst_dtype,load,part",
    _WIDENING_CASES,
    ids=[case[0] + "->" + case[1] for case in _WIDENING_CASES],
)
@tilelang.testing.requires_ascend
def test_parallel_widening_cast_store(src_dtype, dst_dtype, load, part):
    """A widening store reads its source through the ``UNPK_*`` load layout."""
    n = 128
    kernel, source = _compile(_cast_kernel(n, src_dtype, dst_dtype))
    assert f"simd_inst::{load}" in source, "the widening load must spread the source elements"
    assert part in source, "the conversion reads the spread sub-slots"

    if src_dtype == "float8_e4m3fn":
        a = torch.tensor(_FP8_VALUES * (n // len(_FP8_VALUES)), dtype=torch.float32).npu()
        a = a.to(torch.float8_e4m3fn)
        golden = a.cpu().to(torch.float32)
        _assert_close(kernel(a), golden)
        return

    if src_dtype == "float16":
        # 0..127 are exact in float16, so the widening result is bit-exact too.
        a = torch.arange(n, dtype=torch.float16)
        golden = a.to(torch.float32)
    elif src_dtype == "int16":
        a = (torch.arange(n, dtype=torch.int32) % 401 - 200).to(torch.int16)
        golden = a.to(getattr(torch, dst_dtype))
    elif src_dtype == "uint8":
        a = (torch.arange(n, dtype=torch.int32) % 256).to(torch.uint8)
        golden = a.to(getattr(torch, dst_dtype))
    else:
        a = (torch.arange(n, dtype=torch.int32) % 251 - 125).to(torch.int8)
        golden = a.to(getattr(torch, dst_dtype))
    _assert_close(kernel(a.npu()), golden)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@tilelang.testing.requires_ascend
def test_parallel_eight_bit_widening_tail(dtype):
    """``i8 -> i16`` over a byte tail: the gather zero-extends, ``vcvt`` signs.

    A tail wider than half a register has no ``UNPK`` form, so the masked
    ``vgather2`` widens each byte into its own 16-bit lane. That widening is
    zero-extension (CANN intrinsic contract), which is already the value for
    ``uint8`` but only half of it for ``int8``; the signed pair therefore adds
    the same ``vcvt(PART_EVEN)`` the full-chunk recipe uses.
    """
    n = 100
    dst_dtype = "int16" if dtype == "int8" else "uint16"

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), C: T.Tensor((n,), dst_dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), dtype)
            c_ub = T.alloc_shared((n,), dst_dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = T.Cast(dst_dtype, a_ub[i])
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vgather2(" in source, "the byte tail is widened by the masked gather"
    assert "simd_inst::vsts_norm(" in source
    if dtype == "int8":
        assert "simd_inst::vcvt" in source, "a signed byte needs the PART_EVEN sign extension"
        assert "PART_EVEN" in source
    else:
        assert "simd_inst::vcvt" not in source, "zero-extension is already the uint8 -> uint16 value"

    a = (torch.arange(n, dtype=torch.int32) % 251 - 125).to(getattr(torch, dtype))
    _assert_close(kernel(a.npu()), a.to(getattr(torch, dst_dtype)))


_VALUES = [0, 1, 2, 3, 4, 6, 8, -1, -2, -3, -4, -6, -8]
_VALUES_UNSIGNED = [0, 1, 2, 3, 4, 6, 8, 9, 10, 12, 16, 24, 32]


def _cast_values(src_dtype, dst_dtype):
    """Exact, in-range values for the pair (no rounding or saturation involved)."""
    if src_dtype.startswith("uint") or dst_dtype.startswith("uint"):
        return _VALUES_UNSIGNED
    return _VALUES


# (src, dst, vcvt_expected)
_CAST_CASES = [
    ("float32", "int32", True),
    ("float32", "int16", True),
    ("float32", "float16", True),
    ("float32", "float8_e4m3fn", True),
    ("float16", "int32", True),
    ("float16", "int16", True),
    ("float16", "int8", True),
    ("float16", "uint8", True),
    ("float16", "bfloat16", True),
    ("bfloat16", "int32", True),
    ("bfloat16", "float16", True),
    ("bfloat16", "float32", True),
    ("bfloat16", "float8_e4m3fn", True),
    ("int32", "float32", True),
    ("int32", "int16", True),
    ("int32", "uint16", True),
    ("uint32", "int16", True),
    ("uint32", "uint16", True),
    ("uint32", "uint8", True),
    ("int16", "float32", True),
    ("int16", "float16", True),
    ("int16", "uint32", True),
    ("uint16", "uint32", True),
    ("uint16", "uint8", True),
    ("int16", "uint16", False),  # same width: a register reinterpret
    ("uint16", "int16", False),
]


@pytest.mark.parametrize(
    "src_dtype,dst_dtype,has_vcvt",
    _CAST_CASES,
    ids=[f"{src}->{dst}" for src, dst, _ in _CAST_CASES],
)
@tilelang.testing.requires_ascend
def test_parallel_cast_pair(src_dtype, dst_dtype, has_vcvt):
    n = 128

    kernel, source = _compile(_cast_kernel(n, src_dtype, dst_dtype))
    if has_vcvt:
        assert "simd_inst::vcvt<" in source, "the conversion must be a vector vcvt"
    else:
        assert "simd_inst::vcvt" not in source, "a same-width sign change costs no instruction"
        assert "simd_inst::vld" in source and "simd_inst::vsts" in source

    values = _cast_values(src_dtype, dst_dtype)
    a = torch.tensor([values[i % len(values)] for i in range(n)], dtype=getattr(torch, src_dtype))
    golden = a.to(getattr(torch, dst_dtype))
    actual = kernel(a.npu()).cpu()
    if src_dtype in ("int16", "uint16") and dst_dtype in ("int16", "uint16"):
        # Same width: every bit pattern must survive.
        torch.testing.assert_close(actual.view(torch.int16), a.view(torch.int16), rtol=0, atol=0)
    else:
        _assert_close(actual, golden)


@tilelang.testing.requires_ascend
def test_parallel_bf16_to_f16_overflow_is_inf():
    """``bf16 -> f16`` overflow becomes ``+-inf``: this pair does not saturate.

    Both formats are 16-bit, but bf16 keeps f32's exponent (max ~3.4e38) while
    f16 stops at 65504. ``simd.py`` lists saturation only for ``float32 ->
    float16/bfloat16``, the fp8 pairs and the integer pairs; ``bfloat16 ->
    float16`` keeps the IEEE overflow result, which the device confirms.
    bf16's mantissa is shorter than f16's, so rounding never changes a value
    either — in-range inputs must be bit-exact.
    """
    n = 128

    kernel, source = _compile(_cast_kernel(n, "bfloat16", "float16"))
    assert "simd_inst::vcvt<" in source, "the conversion must be a vector vcvt"

    # 1e5 / 2^17 are bf16-representable and far above the f16 maximum; 65504 is
    # the maximum itself (it rounds up to 65536 in bf16, so it overflows too).
    values = [1e5, -1e5, 131072.0, -131072.0, 65504.0, 0.5, 3.0, -2.0]
    a = torch.tensor(values * (n // len(values)), dtype=torch.bfloat16)
    golden = a.to(torch.float16)  # torch also maps overflow to +-inf
    _assert_close(kernel(a.npu()).cpu(), golden)


# ---------------------------------------------------------------------------
# NZ layout, pack2 and the rejected shapes
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
def test_parallel_compact_nz_store():
    @T.prim_func
    def main(A: T.Tensor((M, N), F32), C: T.Tensor((M, N), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            c_ub = T.alloc_shared((M, N), F32)
            T.annotate_layout({c_ub: make_ascend_compact_nz_layout(c_ub)})
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[i, j] = a_ub[i, j] * 2.0
            T.copy(c_ub, C)

    kernel, source = _compile(main)
    assert "simd_inst::vsstb(" in source, "the NZ row is a block-strided scatter"
    assert "simd_inst::vsts_norm(" not in source

    a = (torch.arange(M * N, dtype=torch.float32) % 500) - 250.0
    golden = (a * 2.0).reshape(M, N)
    _assert_close(kernel(a.reshape(M, N).npu()), _nz_physical(golden, 8))


@tilelang.testing.requires_ascend
def test_parallel_elementwise_nz_and_row_reduction_are_planned_independently():
    """Elementwise + NZ and a row reduction in one VF, each in its own nest.

    The two recipes have no common vector axis. Keeping them in one
    ``T.Parallel`` nest would make the reduction ignore an axis; P1 reports that
    nest as OPAQUE/ILLEGAL and leaves the original IR (scalar code inside a
    ``T.SimdVF`` is rejected by ccec), so the program uses one nest per recipe.
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, N), F32),
        S: T.Tensor((M, N), F32),
        C: T.Tensor((M, N), "bfloat16"),
        D: T.Tensor((M,), F32),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            s_ub = T.alloc_shared((M, N), F32)
            c_ub = T.alloc_shared((M, N), "bfloat16")
            d_ub = T.alloc_shared((M,), F32)
            T.annotate_layout({c_ub: make_ascend_compact_nz_layout(c_ub)})
            T.copy(A, a_ub)
            T.copy(S, s_ub)
            T.fill(d_ub, 0.0)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    c_ub[i, j] = T.Cast("bfloat16", a_ub[i, j] * 2.0)
                for i in T.Parallel(M):
                    for k in T.serial(N):
                        d_ub[i] += s_ub[i, k]
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    kernel, source = _compile(main, out_idx=[-2, -1])
    assert "simd_inst::vld_x2<" in source, "the narrowed NZ store uses the paired form"
    assert "simd_inst::vsstb(" in source
    assert "simd_inst::vcadd(" in source, "the row reduction has its own nest"

    a = (torch.arange(M * N, dtype=torch.int32) % 64).to(torch.float32)
    s = torch.randn(M, N, dtype=torch.float32)
    c_npu, d_npu = kernel(a.reshape(M, N).npu(), s.reshape(M, N).npu())

    golden = (a * 2.0).to(torch.bfloat16).reshape(M, N)
    _assert_close(c_npu, _nz_physical(golden, 16))
    _assert_tree_sum_close(d_npu, s)


@tilelang.testing.requires_ascend
def test_parallel_paired_nz_store_with_an_nd_sibling():
    """A narrowed NZ store next to an ND store uses the pack2 interchange.

    The two f32 chunk halves are de-interleaved by one ``vld2``, the ND half is
    re-interleaved by ``vintlv`` before its contiguous stores, and the bf16 NZ
    half is merged by ``vor`` into one full-register ``vsstb``.
    """

    @T.prim_func
    def main(A: T.Tensor((M, N), F32), C: T.Tensor((M, N), "bfloat16"), D: T.Tensor((M, N), F32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), F32)
            c_ub = T.alloc_shared((M, N), "bfloat16")
            d_ub = T.alloc_shared((M, N), F32)
            T.annotate_layout({c_ub: make_ascend_compact_nz_layout(c_ub)})
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(M, N):
                    d_ub[i, j] = a_ub[i, j] * 2.0
                    c_ub[i, j] = T.Cast("bfloat16", a_ub[i, j])
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    kernel, source = _compile(main, out_idx=None)
    assert "simd_inst::vld_x2<" in source, "one vld2 supplies both chunk halves"
    assert "simd_inst::vintlv(" in source, "the ND half is restored to chunk order"
    assert "simd_inst::vor(" in source, "the two bf16 parts merge into a full register"
    assert "simd_inst::vsstb(" in source
    assert source.count("simd_inst::vsts_norm(") == 2, "the ND half is two contiguous stores"

    a = (torch.arange(M * N, dtype=torch.int32) % 64).to(torch.float32)
    c_npu = torch.zeros((M, N), dtype=torch.bfloat16).npu()
    d_npu = torch.zeros((M, N), dtype=torch.float32).npu()
    kernel(a.reshape(M, N).npu(), c_npu, d_npu)

    _assert_close(c_npu, _nz_physical(a.to(torch.bfloat16).reshape(M, N), 16))
    _assert_close(d_npu, (a * 2.0).reshape(M, N))


if __name__ == "__main__":
    tilelang.testing.main()
