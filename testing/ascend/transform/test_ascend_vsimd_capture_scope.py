"""Structural tests for the read-only P1 capture/scope analysis (PR-B)."""

import tilelang
import tilelang.ascend.language as T
import tvm
from tilelang.ascend import transform as at
from tilelang.layout import make_ascend_compact_nz_layout
from testing.ascend.transform._vsimd_report_helpers import no_vf_kernel


def _report(func):
    return str(at.AscendAnalyzeSimdVFCaptureReport(func))


def _info(func):
    return at.AscendAnalyzeSimdVFCaptureInfo(func)


def _info_injected(func):
    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.InjectAssumes()(mod)
    return at.AscendAnalyzeSimdVFCaptureInfo(mod["main"])


def test_simple_closed():
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
    assert "status=CLOSED" in report
    assert "captures:" in report
    assert "a_ub[alloc]" in report
    assert "c_ub[alloc]" in report

    infos = _info(main)
    assert len(infos) == 1
    assert infos[0]["status"] == "CLOSED"
    assert "a_ub[alloc]" in infos[0]["captures"]
    assert "c_ub[alloc]" in infos[0]["captures"]


def test_versioned_outer_var():
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
    assert "status=CLOSED" in report
    assert "outer_vars:" in report
    assert "w[outer_loop]" in report
    assert "w[outer_loop]" in _info(main)[0]["outer_vars"]


def test_view_local_buffer():
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
    assert "status=CLOSED" in report
    assert "local_views:" in report
    assert "v -> base" in report
    assert "v -> base" in _info(main)[0]["local_views"]


def test_layout_capture():
    rows, cols = 32, 128

    @T.prim_func
    def main(S: T.Tensor((rows, cols), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((rows, cols), "float32")
            p_nz = T.alloc_shared((rows + 1, cols), "float32")
            T.annotate_layout({p_nz: make_ascend_compact_nz_layout(p_nz)})
            T.copy(S, s_ub)
            with T.SimdVF():
                for i, j in T.Parallel(rows, cols):
                    p_nz[i, j] = s_ub[i, j] + 1.0

    report = _report(main)
    assert "status=CLOSED" in report
    assert "layouts:" in report
    assert "p_nz" in report
    assert "p_nz" in _info(main)[0]["layouts"]


def test_tail_scope_closed():
    n = 1000
    bn = 256

    @T.prim_func
    def main(A: T.Tensor((n + bn,), "float32")):
        with T.Kernel(T.ceildiv(n, bn)) as bx:
            a_ub = T.alloc_shared((n + bn,), "float32")
            T.copy(A, a_ub)
            t = T.min(n - bx * bn, bn)
            with T.SimdVF():
                for i in T.Parallel(t):
                    a_ub[i] = a_ub[i + bn]

    report = _report(main)
    assert "status=CLOSED" in report
    assert "t[bind]" in report
    info = _info(main)[0]
    assert "t[bind]" in info["outer_vars"]
    assert any("t = " in b for b in info["bindings"])


def test_dynamic_shape_capture():
    e = T.dynamic("E")

    @T.prim_func
    def main(A: T.Tensor((e,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((e,), "float32")
            c_ub = T.alloc_shared((e,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(e):
                    c_ub[i] = a_ub[i]

    report = _report(main)
    assert "status=CLOSED" in report
    assert "shape_syms:" in report
    assert "E" in report


def test_manual_vf_opaque():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                value = T.simd.vld(a_ub[0])
                T.simd.vsts(c_ub[0], value, mask)

    report = _report(main)
    assert "status=OPAQUE" in report


def test_unbound_var_open():
    n = 64
    idx = T.dynamic("IDX")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[idx] = a_ub[i]

    report = _report(main)
    assert "status=OPEN" in report
    assert "unbound var IDX" in report
    info = _info(main)[0]
    assert info["status"] == "OPEN"
    assert any("IDX" in e for e in info["errors"])


def test_assume_capture_info():
    e = T.dynamic("E")

    @T.prim_func
    def main(A: T.Tensor((e,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((e,), "float32")
            c_ub = T.alloc_shared((e,), "float32")
            T.copy(A, a_ub)
            T.assume(e == 1)
            with T.SimdVF():
                for i in T.Parallel(e):
                    c_ub[0] = a_ub[i]

    infos = _info_injected(main)
    assert len(infos) == 1
    assert infos[0]["status"] == "CLOSED"
    assert any("E == 1" in a for a in infos[0]["assumptions"])


def test_alloc_var_inside_opaque():
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
    assert _info(main)[0]["status"] == "OPAQUE"


def test_alloc_var_outside_opaque():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            v = T.alloc_var("float32", init=0.0)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + v
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=OPAQUE" in report
    assert "local.var" in report
    assert _info(main)[0]["status"] == "OPAQUE"


def test_alloc_var_unused_inside_opaque():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                _v = T.alloc_var("float32", init=0.0)
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i]
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=OPAQUE" in report
    assert "local.var" in report
    assert _info(main)[0]["status"] == "OPAQUE"


def test_opaque_evaluate_opaque():
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
    assert _info(main)[0]["status"] == "OPAQUE"


def test_no_vf_reports_empty():
    assert "no SIMD_VF" in _report(no_vf_kernel())


if __name__ == "__main__":
    tilelang.testing.main()
