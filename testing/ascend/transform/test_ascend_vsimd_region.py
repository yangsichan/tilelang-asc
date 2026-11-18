"""Structural tests for the region lift/dump/expand oracle."""

import tilelang
import tilelang.ascend.language as T
import tvm
from tvm import tirx
from tilelang.ascend import transform as at
from tilelang.ascend import vsimd_ir as ir
from tilelang.layout import Layout
from testing.ascend.transform._vsimd_report_helpers import no_vf_kernel


def _report(func):
    return str(at.AscendAnalyzeVSimdRegionReport(func))


def _lifted_regions(func):
    """Return (region, For) pairs of a lifted function, in body order."""
    mod = at.AscendLiftVSimd()(tvm.IRModule({"main": func}))
    pairs = []

    def visit(node):
        if isinstance(node, tirx.For) and "tl.vsimd.region" in dict(node.annotations):
            pairs.append((node.annotations["tl.vsimd.region"], node))

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return pairs


def test_simple_region_lifted():
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

    report = _report(main)
    assert "status=LIFTED" in report
    assert "axes: i in [0, 64)" in report
    assert "captures:" in report
    assert "W c_ub[i] = a_ub[i]" in report
    assert "expanded:" in report
    assert "for i in range(64)" in report
    assert "verify: OK" in report


def test_2d_lifted():
    m, n = 8, 64

    @T.prim_func
    def main(A: T.Tensor((m, n), "float32"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((m, n), "float32")
            c_ub = T.alloc_shared((m, n), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(m, n):
                    c_ub[i, j] = a_ub[i, j] + 1.0
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=LIFTED" in report
    assert "axes: i in [0, 8) j in [0, 64)" in report
    assert "W c_ub[i, j] = a_ub[i, j]" in report
    assert "W c_ub[i, j] = a_ub[i, j] + T.float32(1.0) dep={0,1}" in report
    assert "T.grid(8, 64)" in report
    assert "verify: OK" in report


def test_2d_broadcast_dep_axes():
    m, n = 8, 64

    @T.prim_func
    def main(
        A: T.Tensor((m, n), "float32"),
        Bias: T.Tensor((n,), "float32"),
        C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((m, n), "float32")
            bias_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((m, n), "float32")
            T.copy(A, a_ub)
            T.copy(Bias, bias_ub)
            with T.SimdVF():
                for i, j in T.Parallel(m, n):
                    c_ub[i, j] = a_ub[i, j] + bias_ub[j]
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=LIFTED" in report
    assert "R bias_ub[j] dep={1}" in report
    assert "R a_ub[i, j] dep={0,1}" in report
    assert "verify: OK" in report


def test_2d_transpose_map():
    m, n = 8, 64

    @T.prim_func
    def main(A: T.Tensor((n, m), "float32"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, m), "float32")
            c_ub = T.alloc_shared((m, n), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(m, n):
                    c_ub[i, j] = a_ub[j, i]
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=LIFTED" in report
    assert "R a_ub[j, i] dep={0,1}" in report
    assert "verify: OK" in report


def test_illegal_memory_is_opaque():
    n = 256

    @T.prim_func
    def main(A: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n - 32):
                    a_ub[i] = a_ub[i + 32]

    report = _report(main)
    assert "status=OPAQUE" in report
    assert "memory legality is ILLEGAL" in report


def test_alloc_var_opaque():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                v = T.alloc_var("float32", init=0.0)
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + v
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=OPAQUE" in report
    assert "local.var" in report


def test_opaque_evaluate_is_opaque():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    T.evaluate(T.call_extern("float32", "opaque_side_effect", a_ub[i]))

    report = _report(main)
    assert "status=OPAQUE" in report
    assert "opaque Evaluate" in report


def test_if_region_lifted_and_matches_original():
    """A guarded body is reconstructed statement by statement."""
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if i >= 32:
                        c_ub[i] = a_ub[i - 32]
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=LIFTED" in report
    assert "if i" in report
    assert "verify: OK" in report

    pairs = _lifted_regions(main)
    assert len(pairs) == 1
    region, for_node = pairs[0]
    assert [op.kind for op in region.ops] == ["if"]
    assert ir.check_region_against(region, for_node) == ""


def test_tileop_region_is_opaque():
    """A copy/reduce inside ``T.Parallel`` is not modelled by P1.

    The front-end semantic check rejects that program before P1 runs; the
    region report keeps a per-nest diagnostic instead of pretending to describe
    the tile-op.
    """
    m, j, k = 8, 8, 64

    @T.prim_func
    def main(A: T.Tensor((m, j, k), "float32"), C: T.Tensor((m, k), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((m, j, k), "float32")
            c_ub = T.alloc_shared((m, k), "float32")
            T.copy(A, s_ub)
            with T.SimdVF():
                for i in T.Parallel(m):
                    T.copy(s_ub[i, 0, :], c_ub[i, :])
                    T.reduce_sum(s_ub[i, :, :], c_ub[i, :], dim=1, clear=False)
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=OPAQUE" in report
    assert "tl.tileop.ascend_copy" in report
    pairs = _lifted_regions(main)
    assert [region.status for region, _ in pairs] == ["OPAQUE"]


def test_tail_region_lifted():
    n, bn = 1000, 256

    @T.prim_func
    def main(A: T.Tensor((n + bn,), "float32")):
        with T.Kernel(T.ceildiv(n, bn)) as bx:
            a_ub = T.alloc_shared((n + bn,), "float32")
            T.copy(A, a_ub)
            tail = T.min(n - bx * bn, bn)
            with T.SimdVF():
                for i in T.Parallel(tail):
                    a_ub[i] = a_ub[i + bn]

    report = _report(main)
    assert "status=LIFTED" in report
    assert "axes: i in" in report


def test_dynamic_shape_symbols_lifted():
    e = T.dynamic("E")

    @T.prim_func
    def main(A: T.Tensor((e,), "float32"), C: T.Tensor((e,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((e,), "float32")
            c_ub = T.alloc_shared((e,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(e):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=LIFTED" in report
    assert "shape_syms:" in report
    shape_syms = [line.strip() for line in report.splitlines() if line.strip().startswith("shape_syms:")]
    assert shape_syms == ["shape_syms: E"]


def test_versioned_outer_capture():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            tmp = T.alloc_shared((2, 64), "float32")
            acc = T.alloc_shared((2, 64), "float32")
            for w in T.serial(4):
                with T.SimdVF():
                    for i in T.Parallel(64):
                        tmp[w % 2, i] = tmp[w % 2, i] + acc[w % 2, i]

    report = _report(main)
    assert "status=LIFTED" in report
    assert "outer:" in report
    assert "w[outer_loop]" in report


def test_view_region_lifted():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            base = T.alloc_shared((n,), "float32")
            T.copy(A, base)
            with T.SimdVF():
                for i in T.Parallel(n // 2):
                    v = T.view(base, shape=(n // 2, 2), dtype="float32")
                    v[i, 0] = base[2 * i] + 1.0
            T.copy(base, C)

    report = _report(main)
    assert "status=LIFTED" in report
    assert "captures:" in report
    assert "local_views:" in report
    assert "v -> base" in report
    assert "verify: OK" in report
    assert "decl_buffer" in report


def test_layout_preserved():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.annotate_layout({a_ub: Layout([n], lambda i: [i // 64, i % 64])})
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=LIFTED" in report
    assert "layouts:" in report
    assert "a_ub" in report
    assert "verify: OK" in report


def test_2d_flattened_index():
    m, n = 8, 64

    @T.prim_func
    def main(A: T.Tensor((m * n,), "float32"), C: T.Tensor((m * n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((m * n,), "float32")
            c_ub = T.alloc_shared((m * n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(m, n):
                    c_ub[i * n + j] = a_ub[i * n + j] + 1.0
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=LIFTED" in report
    assert "W c_ub[i * 64 + j]" in report
    assert "dep={0,1}" in report
    assert "T.grid(8, 64)" in report
    assert "verify: OK" in report


def test_inner_bind_is_opaque():
    m, n = 8, 64

    @T.prim_func
    def main(A: T.Tensor((m * n,), "float32"), C: T.Tensor((m * n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((m * n,), "float32")
            c_ub = T.alloc_shared((m * n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(m, n):
                    t = i * n + j
                    c_ub[t] = a_ub[t] + 1.0
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=OPAQUE" in report
    assert "tirx.Bind" in report


def test_3d_lifted():
    b0, b1, b2 = 2, 8, 16

    @T.prim_func
    def main(
        A: T.Tensor((b0, b1, b2), "float32"),
        C: T.Tensor((b0, b1, b2), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((b0, b1, b2), "float32")
            c_ub = T.alloc_shared((b0, b1, b2), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j, k in T.Parallel(b0, b1, b2):
                    c_ub[i, j, k] = a_ub[i, j, k] + 1.0
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=LIFTED" in report
    assert "T.grid(2, 8, 16)" in report
    assert "dep={0,1,2}" in report
    assert "verify: OK" in report


def test_4d_recursive_lift():
    b0, b1, b2, b3 = 2, 2, 4, 8

    @T.prim_func
    def main(
        A: T.Tensor((b0, b1, b2, b3), "float32"),
        C: T.Tensor((b0, b1, b2, b3), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((b0, b1, b2, b3), "float32")
            c_ub = T.alloc_shared((b0, b1, b2, b3), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j, k, l in T.Parallel(b0, b1, b2, b3):
                    c_ub[i, j, k, l] = a_ub[i, j, k, l] + 1.0
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=LIFTED" in report
    assert "T.grid(2, 2, 4, 8)" in report
    assert "dep={0,1,2,3}" in report
    assert "verify: OK" in report


def test_no_vf_reports_empty():
    assert "no SIMD_VF" in _report(no_vf_kernel())


def test_region_matches_its_original_parallel_loop():
    """The reconstruction oracle accepts every lifted region."""

    def build(n):
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

    for n in (16, 64):
        pairs = _lifted_regions(build(n))
        assert len(pairs) == 1
        region, for_node = pairs[0]
        assert ir.check_region_against(region, for_node) == ""


def test_region_mismatch_reports_a_path():
    """A region that does not describe the loop is rejected with a location."""

    def build(n):
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

    region, _ = _lifted_regions(build(64))[0]
    _, other_for = _lifted_regions(build(16))[0]
    error = str(ir.check_region_against(region, other_for))
    assert error
    assert "body[0]" in error
    assert "differs" in error


def test_region_verify_switch_is_observable_and_side_effect_free():
    """The oracle switch shows up in the report and does not change P1 output.

    The failure path itself is not reachable from valid input (it means a
    compiler bug), so the switch is observed through the report's `verify:`
    line; the annotations must be identical either way.
    """
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

    def lift(verify):
        with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ASCEND_SIMD_REGION_VERIFY: verify}):
            report = _report(main)
            module = at.AscendLiftVSimd()(tvm.IRModule({"main": main}))
        return report, module

    report_on, module_on = lift(True)
    report_off, module_off = lift(False)
    assert "verify: OK" in report_on
    assert "verify: skipped" in report_off
    # Skipping the check is side-effect free: identical annotations either way.
    assert tvm.ir.structural_equal(module_on["main"], module_off["main"])


def test_combined_dump_report_has_all_sections():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), C: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((16,), "float32")
            c_ub = T.alloc_shared((16,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(16):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    report = str(at.AscendDumpVSimdReport(main))
    assert "=== memory legality ===" in report
    assert "=== capture / scope ===" in report
    assert "=== region lift ===" in report

    mod = tvm.IRModule({"main": main})
    out = at.AscendDumpVSimd()(mod)
    assert tvm.ir.structural_equal(out["main"], main)


if __name__ == "__main__":
    tilelang.testing.main()
