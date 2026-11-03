"""Ascend950 FP8/FP4 ND-to-NZ scatter and UB-to-L1 GEMM correctness."""

import pytest
import torch
import tilelang
import tilelang.testing
from tilelang.ascend import language as T
from tilelang import tvm
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_nz_layout

pytest.importorskip("torch_npu")


def _scatter_call(src, dst, rows, cols, c_type="fp8_e4_t"):
    return tvm.tirx.Call(
        "void",
        tvm.ir.Op.get("tl.ascend_nd2nz_scatter"),
        [src, dst, T.int32(rows), T.int32(cols), T.int32(rows + 1), tvm.tirx.StringImm(c_type), tvm.tirx.StringImm(c_type)],
    )


def _make_gemm_kernel(dtype, cols, route, operand):
    rows = 32
    sids = 2 if route == "dual_copy" else 1
    part_rows = rows // sids

    @T.prim_func
    def kernel(a: T.Tensor((2, rows, cols), dtype), b: T.Tensor((2, rows, cols), dtype), out: T.Tensor((2, rows, rows), "float32")):
        with T.MixedKernel(1, sids=sids):
            nd = T.alloc_shared((part_rows, cols), dtype)
            a1 = T.alloc_l1((rows, cols), dtype)
            b1 = T.alloc_l1((rows, cols), dtype)
            a0 = T.alloc_l0a((rows, cols), dtype)
            b0 = T.alloc_l0b((rows, cols), dtype)
            c0 = T.alloc_l0c((rows, rows), "float32")
            for stage in T.Pipelined(2):
                if operand == "A":
                    T.copy(b[stage, :, :], b1)
                    if route == "dual_copy":
                        T.dual_copy(a[stage, :, :], nd)
                        T.dual_copy(nd, a1)
                    else:
                        T.copy(a[stage, :, :], nd)
                        T.copy(nd, a1)
                else:
                    T.copy(a[stage, :, :], a1)
                    if route == "dual_copy":
                        T.dual_copy(b[stage, :, :], nd)
                        T.dual_copy(nd, b1)
                    else:
                        T.copy(b[stage, :, :], nd)
                        T.copy(nd, b1)
                T.copy(a1, a0)
                T.copy(b1, b0)
                T.gemm(a0, b0, c0, transpose_B=True, clear_accum=True)
                T.copy(c0, out[stage, :, :])

    return kernel


def _make_scatter_kernel(rows, cols, inside_vf, dtype="float8_e4m3fn"):
    c_type = "float4_e2m1x2_t" if dtype == "float4_e2m1fn" else "fp8_e4_t"

    @T.prim_func
    def main(input_gm: T.Tensor((rows, cols), dtype), output_gm: T.Tensor((rows + 1, cols), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((rows, cols), dtype)
            nz = T.alloc_shared(((rows + 1) * cols,), dtype)
            T.copy(input_gm, src)
            if inside_vf:
                with T.SimdVF():
                    T.evaluate(
                        _scatter_call(
                            T.access_ptr(src[0, 0], "r", extent=rows * cols),
                            T.access_ptr(nz[0], "w", extent=(rows + 1) * cols),
                            rows,
                            cols,
                            c_type,
                        )
                    )
            else:
                T.evaluate(
                    _scatter_call(
                        T.access_ptr(src[0, 0], "r", extent=rows * cols),
                        T.access_ptr(nz[0], "w", extent=(rows + 1) * cols),
                        rows,
                        cols,
                        c_type,
                    )
                )
            T.copy(nz, output_gm)

    return main


# Verify automatic copy/dual_copy reordering supplies either GEMM operand.
@tilelang.testing.requires_ascend
@pytest.mark.parametrize("cols", [32, 96, 256, 288])
@pytest.mark.parametrize("route", ["copy", "dual_copy"])
@pytest.mark.parametrize("operand", ["A", "B"])
def test_fp8_nd2nz_ub_to_l1_gemm(cols, route, operand):
    dtype = "float8_e4m3fn"
    kernel = tilelang.compile(_make_gemm_kernel(dtype, cols, route, operand), target="ascend", execution_backend="tvm_ffi")
    torch.manual_seed(31)
    a = torch.randint(-4, 5, (2, 32, cols)).to(getattr(torch, dtype))
    b = torch.randint(-4, 5, (2, 32, cols)).to(getattr(torch, dtype))
    out = torch.full((2, 32, 32), float("nan"), device="npu")
    kernel(a.npu(), b.npu(), out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), a.float() @ b.float().transpose(-1, -2), rtol=0, atol=0)


# Preserve all FP8 bit patterns across full vectors/tails, inside and outside SIMD VF.
@tilelang.testing.requires_ascend
@pytest.mark.parametrize("cols", [*range(32, 513, 32), 544, 608])
@pytest.mark.parametrize("inside_vf", [False, True])
def test_fp8_scatter_intrinsic_preserves_bits(cols, inside_vf):
    rows = 32
    bits = torch.randint(0, 256, (rows * cols,), dtype=torch.uint8, generator=torch.Generator().manual_seed(31))
    bits[:256] = torch.arange(256, dtype=torch.uint8)
    bits = bits.reshape(rows, cols)
    kernel = tilelang.compile(_make_scatter_kernel(rows, cols, inside_vf), target="ascend", out_idx=-1, execution_backend="tvm_ffi")
    output = kernel(bits.view(torch.float8_e4m3fn).npu()).cpu().view(torch.uint8)
    unpacked = output.reshape(cols // 32, rows + 1, 32)[:, :rows, :].permute(1, 0, 2).reshape(rows, cols)
    torch.testing.assert_close(unpacked, bits, rtol=0, atol=0)


def _make_prepacked_nz_kernel(compact):
    rows, cols = 32, 128
    storage_rows = rows + 1 if compact else rows
    layout = make_ascend_compact_nz_layout if compact else make_ascend_nz_layout
    dtype = "float8_e4m3fn"

    @T.prim_func
    def main(
        packed: T.Tensor((storage_rows, cols), dtype), identity: T.Tensor((cols, cols), dtype), out: T.Tensor((rows, cols), "float32")
    ):
        with T.MixedKernel(1, sids=1):
            nz = T.alloc_shared((storage_rows, cols), dtype)
            a_l1 = T.alloc_l1((rows, cols), dtype)
            b_l1 = T.alloc_l1((cols, cols), dtype)
            acc = T.alloc_l0c((rows, cols), "float32")
            T.annotate_layout({nz: layout(nz)})
            T.copy(packed, nz)
            T.copy(nz[:rows, :], a_l1)
            T.copy(identity, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    return main


# Prepacked NZ bytes reach GEMM through post-copy without another scatter.
@tilelang.testing.requires_ascend
@pytest.mark.parametrize("compact", [False, True])
def test_fp8_prepacked_nz_ub_to_l1(compact):
    rows, cols, c0 = 32, 128, 32
    storage_rows = rows + 1 if compact else rows
    x = (torch.arange(rows * cols).reshape(rows, cols) % 9 - 4).float()
    packed = torch.zeros((cols // c0, storage_rows, c0))
    packed[:, :rows, :] = x.reshape(rows, cols // c0, c0).permute(1, 0, 2)
    kernel = tilelang.compile(_make_prepacked_nz_kernel(compact), target="ascend", out_idx=-1, execution_backend="tvm_ffi")
    source = kernel.get_kernel_source()
    assert "ascend_nd2nz_scatter<" not in source
    assert "ascend_nd2nz_scatter_callee<" not in source
    assert "asc_copy_ub2l1" in source
    actual = kernel(packed.reshape(storage_rows, cols).to(torch.float8_e4m3fn).npu(), torch.eye(cols).to(torch.float8_e4m3fn).npu())
    torch.testing.assert_close(actual.cpu(), x, rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("cols,inside_vf", [(128, False), (576, True)])
def test_fp4_scatter_preserves_packed_bytes(cols, inside_vf):
    rows = 32
    dtype = torch.float4_e2m1fn_x2
    bits = (torch.arange(rows * cols // 2) % 256).to(torch.uint8).reshape(rows, cols // 2)
    kernel = tilelang.compile(_make_scatter_kernel(rows, cols, inside_vf, "float4_e2m1fn"), target="ascend", execution_backend="tvm_ffi")
    output = torch.empty((rows + 1, cols // 2), dtype=torch.uint8, device="npu")
    kernel(bits.view(dtype).npu(), output.view(dtype))
    actual = output.cpu().reshape(cols // 64, rows + 1, 32)[:, :rows].permute(1, 0, 2).reshape(rows, cols // 2)
    torch.testing.assert_close(actual, bits, rtol=0, atol=0)


def _pack_fp4_small_integers(x):
    # E2M1 codes for exact nonnegative integers 0, 1, 2, 3.
    codes = torch.tensor([0, 2, 4, 5], dtype=torch.uint8)[x.long()]
    return (codes[..., ::2] | (codes[..., 1::2] << 4)).view(torch.float4_e2m1fn_x2)


def _make_fp4_copy_kernel(route):
    rows, cols = 32, 128
    sids = 2 if route == "dual_copy" else 1
    storage_rows = rows + 17 if route == "prepacked" else rows
    ub_rows = storage_rows if route == "prepacked" else rows // sids
    dtype = "float4_e2m1fn"

    @T.prim_func
    def main(
        x: T.Tensor((storage_rows, cols), dtype),
        identity: T.Tensor((cols, cols), dtype),
        sa: T.Tensor((cols // 64, rows), "uint16"),
        sb: T.Tensor((cols // 64, cols), "uint16"),
        out: T.Tensor((rows, cols), "float32"),
    ):
        with T.MixedKernel(1, sids=sids):
            ub = T.alloc_shared((ub_rows, cols), dtype)
            a = T.alloc_l1((rows, cols), dtype)
            b = T.alloc_l1((cols, cols), dtype)
            sfa = T.alloc_l1((rows, cols // 64), "uint16")
            sfb = T.alloc_l1((cols, cols // 64), "uint16")
            acc = T.alloc_l0c((rows, cols), "float32")
            if route == "prepacked":
                T.annotate_layout({ub: make_ascend_compact_nz_layout(ub)})
                T.copy(x, ub)
                T.copy(ub[16 : 16 + rows, :], a)
            elif route == "dual_copy":
                T.dual_copy(x, ub)
                T.dual_copy(ub, a)
            else:
                T.copy(x, ub)
                T.copy(ub, a)
            T.copy(identity, b)
            T.copy(sa, sfa, transpose=True)
            T.copy(sb, sfb, transpose=True)
            T.gemm_blockscaled(a, b, acc, sfa, sfb, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    return main


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("route", ["copy", "dual_copy", "prepacked"])
def test_fp4_ub_to_l1_gemm(route):
    rows, cols = 32, 128
    x = torch.randint(0, 4, (rows, cols), generator=torch.Generator().manual_seed(31)).float()
    if route == "prepacked":
        packed = torch.zeros((cols // 64, rows + 17, 64))
        packed[:, 16 : 16 + rows] = x.reshape(rows, cols // 64, 64).permute(1, 0, 2)
        data = packed.reshape(rows + 17, cols)
    else:
        data = x
    kernel = tilelang.compile(_make_fp4_copy_kernel(route), target="ascend", out_idx=-1, execution_backend="tvm_ffi")
    # E8M0 exponent 127 encodes scale=1; pack two scales per uint16.
    sa = torch.full((cols // 64, rows), 0x7F7F, dtype=torch.uint16)
    sb = torch.full((cols // 64, cols), 0x7F7F, dtype=torch.uint16)
    actual = kernel(_pack_fp4_small_integers(data).npu(), _pack_fp4_small_integers(torch.eye(cols)).npu(), sa.npu(), sb.npu())
    torch.testing.assert_close(actual.cpu(), x, rtol=0, atol=0)
