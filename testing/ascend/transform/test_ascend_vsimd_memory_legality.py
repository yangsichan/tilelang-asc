"""Structural tests for the read-only P1 memory-legality analysis (PR-A).

These tests do not lower or execute a kernel; they build the SIMD_VF DSL
directly and inspect the analyzer report.  The analyzer is intentionally
read-only, so it can be merged before the vsimd lifter.
"""

import tilelang
import tilelang.ascend.language as T
import tvm
from tilelang.ascend import transform as at
from tilelang.layout import make_ascend_compact_nz_layout
from testing.ascend.transform._vsimd_report_helpers import no_vf_kernel


def _report(func):
    return str(at.AscendAnalyzeSimdVFMemoryReport(func))


def _report_injected(func):
    """Report on the P1-form IR after InjectAssumes (tl.assume AttrStmt)."""
    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.InjectAssumes()(mod)
    return str(at.AscendAnalyzeSimdVFMemoryReport(mod["main"]))


def test_simple_legal():
    n = 256

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
    assert "status=LEGAL" in report
    assert "class=affine" in report
    assert "injective=yes" in report
    assert "accesses=2/2" in report
    assert "queries=0" in report


def test_broadcast_legal():
    m, n = 8, 64

    @T.prim_func
    def main(
        A: T.Tensor((m, n), "float32"),
        B: T.Tensor((m,), "float32"),
        C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((m, n), "float32")
            b_ub = T.alloc_shared((m,), "float32")
            c_ub = T.alloc_shared((m, n), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                for i, j in T.Parallel(m, n):
                    c_ub[i, j] = a_ub[i, j] + b_ub[i]
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=LEGAL" in report


def test_transpose_legal():
    m, n = 64, 64

    @T.prim_func
    def main(A: T.Tensor((m, n), "float32"), C: T.Tensor((n, m), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((m, n), "float32")
            c_ub = T.alloc_shared((n, m), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i, j in T.Parallel(m, n):
                    c_ub[j, i] = a_ub[i, j]
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=LEGAL" in report


def test_overlapping_copy_illegal():
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
    assert "status=ILLEGAL" in report
    assert "cross-iteration" in report
    assert "queries=1" in report


def test_view_alias_legal():
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            base = T.alloc_shared((n,), "float32")
            T.copy(A, base)
            with T.SimdVF():
                for i in T.Parallel(n // 2):
                    view = T.view(base, shape=(n // 2, 2), dtype="float32")
                    view[i, 0] = base[2 * i] + 1.0
            T.copy(base, C)

    report = _report(main)
    assert "status=LEGAL" in report
    assert "local_views: view -> base" in report


def test_versioned_buffer_legal():
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
    assert "status=LEGAL" in report


def test_direct_nz_store_layout_changing():
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
    assert "status=LEGAL" in report
    assert "layout=changing" in report


def test_manual_simd_opaque():
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
    assert "manual SIMD" in report


def test_write_race_illegal():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[0] = a_ub[i]

    report = _report(main)
    assert "status=ILLEGAL" in report
    assert "not injective" in report


def test_data_dependent_store_unproven():
    n = 64

    @T.prim_func
    def main(
        A: T.Tensor((n,), "float32"),
        IDX: T.Tensor((n,), "int32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            idx_ub = T.alloc_shared((n,), "int32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            T.copy(IDX, idx_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[idx_ub[i]] = a_ub[i]

    report = _report(main)
    assert "status=UNPROVEN" in report
    assert "data-dependent" in report
    assert "scatter" in report


def test_duplicate_loads_are_deduplicated():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + a_ub[i]
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=LEGAL" in report
    assert "accesses=3/2" in report
    assert "queries=0" in report


def test_div_mod_fast_paths():
    n = 32

    @T.prim_func
    def quotient_bad(A: T.Tensor((n,), "float32"), C: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((8,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i // 16] = a_ub[i]
            T.copy(c_ub, C)

    report = _report(quotient_bad)
    assert "status=ILLEGAL" in report
    assert "i // C" in report
    assert "queries=0" in report

    @T.prim_func
    def remainder_ok(A: T.Tensor((8,), "float32"), C: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((8,), "float32")
            c_ub = T.alloc_shared((8,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(8):
                    c_ub[i % 16] = a_ub[i]
            T.copy(c_ub, C)

    assert "status=LEGAL" in _report(remainder_ok)

    @T.prim_func
    def remainder_bad(A: T.Tensor((n,), "float32"), C: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((16,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i % 16] = a_ub[i]
            T.copy(c_ub, C)

    report = _report(remainder_bad)
    assert "status=ILLEGAL" in report
    assert "i % C" in report

    @T.prim_func
    def quotient_remainder(
        A: T.Tensor((n,), "float32"),
        C: T.Tensor((2, 16), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((2, 16), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i // 16, i % 16] = a_ub[i]
            T.copy(c_ub, C)

    assert "status=LEGAL" in _report(quotient_remainder)


def test_symbolic_extent_without_assume_illegal():
    e = T.dynamic("E")

    @T.prim_func
    def main(A: T.Tensor((e,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((e,), "float32")
            c_ub = T.alloc_shared((e,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(e):
                    c_ub[0] = a_ub[i]

    report = _report_injected(main)
    assert "status=ILLEGAL" in report
    assert "symbolic extent" in report
    assert "counterexample" in report


def test_symbolic_extent_with_assume_legal():
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

    report = _report_injected(main)
    assert "status=LEGAL" in report


def test_assume_upper_bound_legal():
    e = T.dynamic("E")

    @T.prim_func
    def main(A: T.Tensor((e,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((e,), "float32")
            c_ub = T.alloc_shared((4,), "float32")
            T.copy(A, a_ub)
            T.assume(e <= 4)
            with T.SimdVF():
                for i in T.Parallel(e):
                    c_ub[i % 4] = a_ub[i]

    report = _report_injected(main)
    assert "status=LEGAL" in report


def test_inconsistent_assumptions_unproven():
    e = T.dynamic("E")

    @T.prim_func
    def main(A: T.Tensor((e,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((e,), "float32")
            c_ub = T.alloc_shared((e,), "float32")
            T.copy(A, a_ub)
            T.assume(e == 1)
            T.assume(e == 2)
            with T.SimdVF():
                for i in T.Parallel(e):
                    c_ub[0] = a_ub[i]

    report = _report_injected(main)
    assert "status=UNPROVEN" in report
    assert "inconsistent" in report


def test_tail_min_legal():
    n = 1000
    bn = 256

    @T.prim_func
    def main(A: T.Tensor((n + bn,), "float32")):
        with T.Kernel(T.ceildiv(n, bn)) as bx:
            a_ub = T.alloc_shared((n + bn,), "float32")
            T.copy(A, a_ub)
            tail = T.min(n - bx * bn, bn)
            with T.SimdVF():
                for i in T.Parallel(tail):
                    a_ub[i] = a_ub[i + bn]

    report = _report_injected(main)
    assert "status=LEGAL" in report
    assert "queries=1" in report


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


def test_alloc_var_outside_unsupported():
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
    assert "status=UNSUPPORTED" in report
    assert "local.var" in report


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


def test_opaque_evaluate_unsupported():
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
    assert "status=UNSUPPORTED" in report
    assert "opaque Evaluate" in report


def test_no_vf_reports_empty():
    assert "no SIMD_VF" in _report(no_vf_kernel())


def test_debug_pass_is_read_only():
    n = 32

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

    mod = tvm.IRModule({"main": main})
    out = at.AscendDumpVSimd()(mod)
    assert tvm.ir.structural_equal(out["main"], main)


# ---------------------------------------------------------------------------
# branch guards
# ---------------------------------------------------------------------------
#
# Two arms of one ``if`` are mutually exclusive *for a lane*, but different
# lanes can take different arms, so the pair is a real cross-iteration hazard
# when the address maps overlap. The prover therefore adds both guard paths to
# its violation query: lane-invariant conditions (or provably disjoint address
# sets) become safe, intersecting addresses stay ILLEGAL.


def test_two_arms_with_overlapping_addresses_stay_illegal():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            c_ub = T.alloc_shared((128,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(64):
                    if a_ub[i] > 0.0:
                        c_ub[2 * i] = 1.0
                    else:
                        c_ub[i] = 2.0
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=ILLEGAL" in report
    assert "cross-iteration RAW/WAR/WAW" in report


def test_two_arms_with_disjoint_addresses_are_legal():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            c_ub = T.alloc_shared((128,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(64):
                    if a_ub[i] > 0.0:
                        c_ub[2 * i] = 1.0
                    else:
                        c_ub[2 * i + 1] = 2.0
            T.copy(c_ub, C)

    assert "status=LEGAL" in _report(main)


def test_lane_range_arms_are_legal():
    """``i < 64`` and ``i >= 64`` write disjoint lanes of the same buffer."""
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if i < 64:
                        c_ub[i] = 1.0
                    else:
                        c_ub[i] = 2.0
            T.copy(c_ub, C)

    assert "status=LEGAL" in _report(main)


def test_lane_invariant_guard_makes_one_arm_dead():
    """A condition that reads one scalar takes one arm for *every* lane."""
    n = 64

    @T.prim_func
    def main(
        A: T.Tensor((n,), "float32"),
        F: T.Tensor((1,), "int32"),
        C: T.Tensor((2 * n,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            f_ub = T.alloc_shared((1,), "int32")
            c_ub = T.alloc_shared((2 * n,), "float32")
            T.copy(A, a_ub)
            T.copy(F, f_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if f_ub[0] > 0:
                        c_ub[2 * i] = 1.0
                    else:
                        c_ub[i] = 2.0
            T.copy(c_ub, C)

    report = _report(main)
    assert "status=LEGAL" in report


if __name__ == "__main__":
    tilelang.testing.main()
