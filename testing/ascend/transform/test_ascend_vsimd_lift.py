"""Phase 0 F3: the P1 pass attaches a region to every top-level T.Parallel loop."""

import tilelang
import tilelang.ascend.language as T
import tvm
from tvm import tirx
from tilelang.ascend import transform as at
from tilelang.ascend import vsimd_ir as ir

REGION_KEY = "tl.vsimd.region"


def _lift(func):
    mod = tvm.IRModule({"main": func})
    return at.AscendLiftVSimd()(mod)


def _regions(mod):
    """Return the region annotations in body order."""
    regions = []

    def visit(node):
        if isinstance(node, tirx.For) and REGION_KEY in dict(node.annotations):
            regions.append(node.annotations[REGION_KEY])

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return regions


def _vf_bodies(mod):
    bodies = []

    def visit(node):
        if isinstance(node, tirx.SBlock) and node.name_hint == "SIMD_VF":
            bodies.append(node.body)

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return bodies


def _strip_for_annotations(stmt):
    """Drop For annotations so a lifted body can be compared structurally."""

    def visit(node):
        if isinstance(node, tirx.For) and dict(node.annotations):
            return tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                {},
            )
        return None

    return tirx.stmt_functor.ir_transform(stmt, None, visit)


def test_region_attached_to_parallel_and_body_unchanged():
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

    out = _lift(main)
    before = _vf_bodies(tvm.IRModule({"main": main}))[0]
    after = _vf_bodies(out)[0]
    assert tvm.ir.structural_equal(_strip_for_annotations(before), _strip_for_annotations(after))

    regions = _regions(out)
    assert len(regions) == 1
    region = regions[0]
    assert region.status == "LIFTED"
    assert region.reason_code == ""
    assert int(region.vf_index) == 0
    assert int(region.region_index) == 0
    assert [str(axis.extent) for axis in region.axes] == ["64"]
    assert [op.kind for op in region.ops] == ["store"]
    assert [capture.buffer.name for capture in region.captures] == [
        "c_ub",
        "a_ub",
    ]


def test_independent_parallels_lift_independently():
    @T.prim_func
    def main(
        A: T.Tensor((96,), "float32"),
        D: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((96,), "float32")
            d_ub = T.alloc_shared((64,), "float32")
            c_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            T.copy(D, d_ub)
            with T.SimdVF():
                for i in T.Parallel(64):
                    a_ub[i] = a_ub[i + 32]  # cross-iteration RAW: illegal
                for i in T.Parallel(64):
                    c_ub[i] = d_ub[i] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 2
    assert regions[0].status == "OPAQUE"
    assert regions[0].reason_code == "ILLEGAL"
    assert regions[1].status == "LIFTED"


def test_shared_buffer_dependency_does_not_block_the_consumer():
    """A buffer dependency must not force the consumer down.

    Shared/global buffers always have the scalar fallback, so P1 keeps the
    regions independent and leaves the decision to P2.
    """

    @T.prim_func
    def main(
        A: T.Tensor((96,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((96,), "float32")
            c_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(64):
                    a_ub[i] = a_ub[i + 32]
                for i in T.Parallel(64):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert [region.status for region in regions] == ["OPAQUE", "LIFTED"]


def test_serial_wrapped_parallel_is_lifted():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            c_ub = T.alloc_shared((64,), "float32")
            tmp = T.alloc_shared((32,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for w in T.serial(2):
                    for i in T.Parallel(32):
                        tmp[i] = 1.0 + w
                for i in T.Parallel(64):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    mod = _lift(main)
    regions = _regions(mod)
    assert len(regions) == 2
    assert [region.status for region in regions] == ["LIFTED", "LIFTED"]
    assert [int(axis.extent) for axis in regions[0].axes] == [32]
    assert [int(axis.extent) for axis in regions[1].axes] == [64]
    # The enclosing serial loop stays in place: it becomes the scalar shell
    # around the emitted vector code and keeps binding ``w``.
    serial = []

    def visit(node):
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.SERIAL:
            serial.append(node)

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert any(node.loop_var.name == "w" for node in serial)


def test_serial_wrapper_does_not_hide_the_nest():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for w in T.serial(2):
                    for i in T.Parallel(64):
                        a_ub[i] = 1.0 + w
            T.copy(a_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"
    assert [int(axis.extent) for axis in regions[0].axes] == [64]


def test_no_parallel_no_annotation():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n, 64), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, 64), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                T.reduce_sum(a_ub, c_ub, dim=1, clear=True)
            T.copy(c_ub, C)

    out = _lift(main)
    assert _regions(out) == []


def test_assume_wrapper_is_transparent():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                T.attr(n > 0, "tl.assume", "test assume")
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"


def test_dynamic_shell_bound_is_lifted():
    """A lane-invariant runtime shell bound is liftable (P2 keeps it as a loop)."""
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n, 64), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, 64), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    for j in T.serial(64):
                        c_ub[i] = a_ub[i, j]
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"


def test_lane_dependent_shell_bound_is_opaque():
    """A shell bound that reads the lane index is not lane invariant -> OPAQUE."""
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n, n), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, n), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    for j in T.serial(i + 1):
                        c_ub[i] = a_ub[i, j]
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    assert regions[0].status == "OPAQUE"
    assert "lane index" in regions[0].reason


def test_multi_shell_body_keeps_one_region():
    """Sibling serial shells keep their tree and one original parallel domain."""
    n, m = 64, 32

    @T.prim_func
    def main(A: T.Tensor((n, m), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, m), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    for j in T.serial(m):
                        a_ub[i, j] = a_ub[i, j] * 0.5
                    for j in T.serial(m):
                        c_ub[i] = T.max(c_ub[i], a_ub[i, j])
            T.copy(c_ub, C)

    mod = _lift(main)
    regions = _regions(mod)
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"
    assert [op.kind for op in regions[0].ops] == ["serial", "serial"]
    assert [int(axis.extent) for axis in regions[0].axes] == [n]
    tvm.ir.assert_structural_equal(_strip_for_annotations(mod["main"].body), main.body)


def test_region_annotation_survives_simplify():
    n = 32

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    lifted = _lift(main)
    simplified = tilelang.transform.Simplify()(lifted)
    regions = _regions(simplified)
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"


def test_if_inside_parallel_is_lifted():
    """A bundled ``if`` is modelled as a guarded op and stays vectorizable."""
    n = 128

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    if a_ub[i] > 0.0:
                        c_ub[i] = a_ub[i] + 1.0
                    else:
                        c_ub[i] = 0.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    region = regions[0]
    assert region.status == "LIFTED"
    assert region.reason_code == ""
    assert [op.kind for op in region.ops] == ["if"]
    # The branch condition is the op guard; both arms are modelled.
    assert "a_ub" in str(region.ops[0].guard)
    assert [op.kind for op in region.ops[0].ops] == ["store"]
    assert [op.kind for op in region.ops[0].else_ops] == ["store"]


def test_tileop_inside_parallel_is_opaque_per_nest():
    """A tile-op inside ``T.Parallel`` is not modelled (front-end forbids it).

    The semantic check rejects such a program before P1 runs, so P1 has no
    tile-op grammar: the nest is reported OPAQUE with a diagnostic and the
    sibling Parallel in the same VF still lifts on its own.
    """
    br, bc = 64, 128

    @T.prim_func
    def main(A: T.Tensor((br, bc), "float32"), M: T.Tensor((br,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((br, bc), "float32")
            m_ub = T.alloc_shared((br,), "float32")
            T.copy(A, s_ub)
            with T.SimdVF():
                for i in T.Parallel(br):
                    T.reduce_max(s_ub[i, :], m_ub[i : i + 1], dim=1, clear=False)
                for i in T.Parallel(br):
                    m_ub[i] = s_ub[i, 0] + 1.0
            T.copy(m_ub, M)

    regions = _regions(_lift(main))
    assert [region.status for region in regions] == ["OPAQUE", "LIFTED"]
    assert regions[0].reason_code == "UNSUPPORTED"
    assert "tl.tileop.reduce" in regions[0].reason


def test_unsupported_tileop_rejects_only_its_nest():
    """An unmodelled tile-op is a per-nest fallback, not a VF-level poison."""
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for _i in T.Parallel(n):
                    T.fill(c_ub, 0.0)
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert [region.status for region in regions] == ["OPAQUE", "LIFTED"]
    assert regions[0].reason_code == "UNSUPPORTED"
    assert "tl.tileop.fill" in regions[0].reason
    assert regions[1].reason_code == ""


def test_copy_tileop_does_not_poison_the_sibling_nest():
    """A copy inside ``T.Parallel`` is a per-nest fallback, not a VF poison."""
    m, n = 8, 32

    @T.prim_func
    def main(A: T.Tensor((n, m), "float32"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, m), "float32")
            c_ub = T.alloc_shared((m, n), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(m):
                    T.copy(a_ub[:, i], c_ub[i, :], transpose=True)
                for i in T.Parallel(m):
                    c_ub[i, 0] = c_ub[i, 0] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert [region.status for region in regions] == ["OPAQUE", "LIFTED"]
    assert regions[0].reason_code == "UNSUPPORTED"
    assert "tl.tileop.ascend_copy" in regions[0].reason


def test_sibling_parallels_are_unsupported_not_internal():
    """A Parallel body that is a sequence of loops is not one rectangular nest.

    The nest model collapses only a body that is exactly one Parallel loop; a
    body that mixes sibling loops with other statements must be reported as
    unsupported input, not as a P1 internal error. This pins the target-FA shape
    and the unit-extent variant that used to reach the reconstruction oracle.
    """
    br, bc = 64, 128

    @T.prim_func
    def fa_shape(A: T.Tensor((br, bc), "float32"), M: T.Tensor((br,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((br, bc), "float32")
            m_ub = T.alloc_shared((br,), "float32")
            T.copy(A, s_ub)
            with T.SimdVF():
                for i in T.Parallel(br):
                    for j in T.Parallel(bc):
                        s_ub[i, j] = s_ub[i, j] * 0.5
                    T.reduce_max(s_ub[i, 0:bc], m_ub[i : i + 1], dim=1, clear=False)
            T.copy(m_ub, M)

    @T.prim_func
    def unit_siblings(A: T.Tensor((4,), "float32"), C: T.Tensor((4,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((4,), "float32")
            c_ub = T.alloc_shared((4,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(4):
                    for _j in T.Parallel(1):
                        a_ub[i] = a_ub[i] * 2.0
                    for _k in T.Parallel(1):
                        c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    for func in (fa_shape, unit_siblings):
        regions = _regions(_lift(func))
        assert len(regions) == 1
        region = regions[0]
        assert region.status == "OPAQUE"
        assert region.reason_code == "UNSUPPORTED"
        assert "not a single rectangular nest" in region.reason


def test_unused_parallel_axis_is_rejected(capfd):
    """An unused parallel axis that duplicates stores is ILLEGAL by dependence."""
    m, n = 8, 64

    @T.prim_func
    def main(S: T.Tensor((m, n), "float32"), D: T.Tensor((m,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((m, n), "float32")
            d_ub = T.alloc_shared((m,), "float32")
            T.copy(S, s_ub)
            T.fill(d_ub, 0.0)
            with T.SimdVF():
                for i, _j in T.Parallel(m, n):
                    for k in T.serial(n):
                        d_ub[i] = d_ub[i] + s_ub[i, k]
            T.copy(d_ub, D)

    report = str(at.AscendAnalyzeVSimdRegionReport(main))
    assert "not injective across parallel instances" in report
    assert "write address does not reference parallel axis '_j'" in report
    assert "[vsimd][P1]" not in capfd.readouterr().err
    region = _regions(_lift(main))[0]
    assert region.status == "OPAQUE"
    assert region.reason_code == "ILLEGAL"
    assert "W d_ub[i]" in region.reason
    assert "write address does not reference parallel axis '_j'" in region.reason
    assert "T.serial" in region.reason
    assert "remove it after checking the intended semantics" in region.reason
    warning = capfd.readouterr().err
    assert warning.count("[vsimd][P1][ILLEGAL]") == 1
    assert "VF#0 region#0" in warning
    assert region.reason in warning
    assert "no successful fallback is implied" in warning


def test_unit_extent_parallel_axis_is_legal(capfd):
    """Extent one is semantically legal; hardware suitability belongs to P2/P3."""
    n = 100

    @T.prim_func
    def main(A: T.Tensor((1, n), "float32"), C: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((1, n), "float32")
            c_ub = T.alloc_shared((1,), "float32")
            T.copy(A, a_ub)
            T.fill(c_ub, 0.0)
            with T.SimdVF():
                for i in T.Parallel(1):
                    for j in T.serial(n):
                        c_ub[i] = c_ub[i] + a_ub[i, j]
            T.copy(c_ub, C)

    assert _regions(_lift(main))[0].status == "LIFTED"
    assert "[vsimd][P1]" not in capfd.readouterr().err


def test_reduction_over_a_dead_parallel_axis_is_rejected():
    """A sibling store using j does not legalize a reduction that races over j."""
    m, n = 8, 64

    @T.prim_func
    def main(
        A: T.Tensor((m, n), "float32"),
        S: T.Tensor((m, n), "float32"),
        C: T.Tensor((m, n), "float32"),
        D: T.Tensor((m,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((m, n), "float32")
            s_ub = T.alloc_shared((m, n), "float32")
            c_ub = T.alloc_shared((m, n), "float32")
            d_ub = T.alloc_shared((m,), "float32")
            T.copy(A, a_ub)
            T.copy(S, s_ub)
            T.fill(d_ub, 0.0)
            with T.SimdVF():
                for i, j in T.Parallel(m, n):
                    c_ub[i, j] = a_ub[i, j] * 2.0
                    for k in T.serial(n):
                        d_ub[i] = d_ub[i] + s_ub[i, k]
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    region = _regions(_lift(main))[0]
    assert region.status == "OPAQUE"
    assert region.reason_code == "ILLEGAL"
    assert "W d_ub[i]" in region.reason
    assert "write address does not reference parallel axis 'j'" in region.reason
    assert "T.serial" in region.reason


def test_parallel_reduction_without_contract_is_illegal():
    """RMW syntax cannot permit multiple parallel instances to write d[i]."""
    m, n = 8, 64

    @T.prim_func
    def main(S: T.Tensor((m, n), "float32"), D: T.Tensor((m,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((m, n), "float32")
            d_ub = T.alloc_shared((m,), "float32")
            T.copy(S, s_ub)
            T.fill(d_ub, 0.0)
            with T.SimdVF():
                for i, j in T.Parallel(m, n):
                    d_ub[i] = d_ub[i] + s_ub[i, j]
            T.copy(d_ub, D)

    region = _regions(_lift(main))[0]
    assert region.status == "OPAQUE"
    assert region.reason_code == "ILLEGAL"
    assert "write address does not reference parallel axis 'j'" in region.reason
    assert "using the axis only in the stored value does not resolve a write conflict" in region.reason


def test_manual_vf_region_keeps_footprint(capfd):
    """A manual VF rejects its regions, but they still record what they touch."""

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

    regions = _regions(_lift(main))
    assert len(regions) == 1
    region = regions[0]
    assert region.status == "OPAQUE"
    # Manual is a VF-level verdict, so the region reports UNSUPPORTED and the
    # VF-level message stays in the reason text.
    assert region.reason_code == "UNSUPPORTED"
    assert "manual" in region.reason
    assert "VF-level rejection" in region.reason
    assert "[vsimd][P1]" not in capfd.readouterr().err

    captures = {capture.buffer.name: capture for capture in region.captures}
    assert {"a_ub", "c_ub"} <= set(captures)
    assert all(capture.kind == "unknown" for capture in region.captures)
    op = region.ops[0]
    assert op.kind == "opaque"
    assert "c_ub" in [buffer.name for buffer in op.writes]
    assert "a_ub" in [buffer.name for buffer in op.reads]


def test_rejection_diagnostics_are_per_region_and_keep_primary_reason(capfd):
    @T.prim_func
    def main(IDX: T.Tensor((8,), "int32"), C: T.Tensor((8,), "int32")):
        with T.Kernel(1):
            indices = T.alloc_shared((8,), "int32")
            scatter = T.alloc_shared((8,), "int32")
            shared_acc = T.alloc_shared((1,), "int32")
            c = T.alloc_shared((8,), "int32")
            frag = T.alloc_fragment((8,), "int32")
            T.copy(IDX, indices)
            with T.SimdVF():
                for i in T.Parallel(8):
                    scatter[indices[i]] = i
                    shared_acc[0] = i
                for j in T.Parallel(8):
                    scatter[indices[j]] = j
                for k in T.Parallel(8):
                    frag[k] = k
                for t in T.Parallel(8):
                    c[t] = t
            T.copy(c, C)

    # Read-only analyses report their findings without printing pass warnings.
    at.AscendAnalyzeSimdVFMemoryReport(main)
    at.AscendAnalyzeSimdVFCaptureReport(main)
    report = str(at.AscendAnalyzeVSimdRegionReport(main))
    assert "[vsimd]" not in capfd.readouterr().err

    mod = _lift(main)
    regions = _regions(mod)
    assert [r.reason_code for r in regions] == ["ILLEGAL", "UNPROVEN", "UNSUPPORTED", ""]
    assert "W shared_acc[0]" in regions[0].reason
    assert "data-dependent" not in regions[0].reason  # later proven conflict takes precedence
    assert "data-dependent store map" in regions[1].reason
    assert "fragment" in regions[2].reason
    fragment_line = next(line for line in report.splitlines() if "VF#0 nest 2:" in line)
    assert "status=OPAQUE" in fragment_line
    assert fragment_line.count(regions[2].reason) == 1
    warning = capfd.readouterr().err
    assert warning.count("[vsimd][P1]") == 3
    for index, region in enumerate(regions[:3]):
        assert f"VF#0 region#{index}" in warning
        assert region.reason in warning
    assert "region#3" not in warning
    assert "to legacy" not in warning
    tvm.ir.assert_structural_equal(_strip_for_annotations(mod["main"].body), main.body)


def test_unbound_guard_reports_capture_error(capfd):
    missing_guard = T.dynamic("missing_guard")

    @T.prim_func
    def main(C: T.Tensor((8,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((8,), "int32")
            with T.SimdVF():
                for i in T.Parallel(8):
                    if missing_guard > 0:
                        c[i] = missing_guard
                        c[i] = c[i] + missing_guard
                    else:
                        c[i] = missing_guard + 1
            T.copy(c, C)

    capture = at.AscendAnalyzeSimdVFCaptureInfo(main)[0]
    assert list(capture["errors"]) == ["unbound var missing_guard"]
    report = str(at.AscendAnalyzeVSimdRegionReport(main))
    assert report.count("unbound var missing_guard") == 1
    region = _regions(_lift(main))[0]
    assert region.reason_code == "UNSUPPORTED"
    assert region.reason.count("unbound var missing_guard") == 1
    warning = capfd.readouterr().err
    assert warning.count("[vsimd][P1][UNSUPPORTED]") == 1
    assert region.reason in warning
    assert warning.count("unbound var missing_guard") == 1


def _diagnostic_vf(body, output):
    """Construct invalid scope deliberately, bypassing front-end declarations."""
    block = tirx.SBlock([], [], [], "SIMD_VF", body)
    return tirx.PrimFunc([output.data], tirx.SBlockRealize([], True, block), buffer_map={output.data: output})


def test_unbound_diagnostics_deduplicate_by_identity(capfd):
    i = tirx.Var("i", "int32")
    first = tirx.Var("missing", "int32")
    second = tirx.Var("missing", "int32")
    output = tirx.decl_buffer((8,), "int32", name="output")
    store = tirx.BufferStore(output, first + second, [i])
    func = _diagnostic_vf(tirx.For(i, 0, 8, tirx.ForKind.PARALLEL, tirx.SeqStmt([store, store])), output)

    # Repeated uses are deduplicated; distinct same-named variables are not.
    capture = at.AscendAnalyzeSimdVFCaptureInfo(func)[0]
    assert list(capture["errors"]) == ["unbound var missing", "unbound var missing"]
    region = _regions(_lift(func))[0]
    assert region.reason.count("unbound var missing") == 2
    assert capfd.readouterr().err.count("unbound var missing") == 2


def test_repeated_unbound_buffer_is_reported_once(capfd):
    i = tirx.Var("i", "int32")
    output = tirx.decl_buffer((8,), "int32", name="output")
    missing = tirx.decl_buffer((8,), "int32", name="missing_buffer")
    store = tirx.BufferStore(output, tirx.BufferLoad(missing, [i]), [i])
    func = _diagnostic_vf(tirx.For(i, 0, 8, tirx.ForKind.PARALLEL, tirx.SeqStmt([store, store])), output)

    capture = at.AscendAnalyzeSimdVFCaptureInfo(func)[0]
    assert list(capture["errors"]).count("unbound buffer missing_buffer") == 1
    report = str(at.AscendAnalyzeVSimdRegionReport(func))
    assert report.count("unbound buffer missing_buffer") == 1
    region = _regions(_lift(func))[0]
    assert region.reason.count("unbound buffer missing_buffer") == 1
    assert capfd.readouterr().err.count("unbound buffer missing_buffer") == 1


def test_distinct_capture_and_memory_failures_are_both_retained(capfd):
    missing = T.dynamic("missing_value")

    @T.prim_func
    def main(C: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((1,), "int32")
            with T.SimdVF():
                for i in T.Parallel(8):
                    c[0] = missing + i
            T.copy(c, C)

    region = _regions(_lift(main))[0]
    assert region.reason_code == "ILLEGAL"
    assert region.reason.count("unbound var missing_value") == 1
    assert region.reason.count("memory legality is ILLEGAL") == 1
    assert "W c[0]" in region.reason
    assert region.reason in capfd.readouterr().err


def test_modulo_conflict_does_not_suggest_removing_an_unused_axis(capfd):
    @T.prim_func
    def main(C: T.Tensor((2,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((2,), "int32")
            with T.SimdVF():
                for i in T.Parallel(8):
                    c[i % 2] = i
            T.copy(c, C)

    region = _regions(_lift(main))[0]
    assert region.reason_code == "ILLEGAL"
    assert "revise the output indices/guards" in region.reason
    assert "redundant" not in region.reason
    assert "write address does not reference" not in region.reason
    assert region.reason in capfd.readouterr().err


def test_view_and_base_share_one_capture():
    """A footprint seeing a T.view and its base must not duplicate the capture.

    Capture identity is the data var, and the verifier rejects two captures for
    one data var; here the opaque region is produced from the syntactic
    footprint, which sees both buffers.
    """

    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            base = T.alloc_shared((n,), "float32")
            T.copy(A, base)
            with T.SimdVF():
                v = T.alloc_var("float32", init=0.0)  # forces the manual path
                for i in T.Parallel(n // 2):
                    view = T.view(base, shape=(n // 2, 2), dtype="float32")
                    view[i, 0] = base[2 * i] + v
            T.copy(base, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    region = regions[0]
    assert region.status == "OPAQUE"
    data_vars = [str(capture.buffer.data) for capture in region.captures]
    assert len(data_vars) == len(set(data_vars))
    assert ir.verify_region(region) == ""


def test_p1_can_be_disabled_by_config():
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

    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ASCEND_SIMD_LIFT: False}):
        out = _lift(main)
    assert _regions(out) == []


def test_default_pipeline_vectorizes_the_parallel_nest():
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

    with tvm.target.Target("ascend"):
        out = tilelang.lower(main)
    # The default pipeline runs P1 -> P2 -> P3, so the nest becomes vector code.
    assert "main_kernel_simd_vf_0" in out.kernel_source
    assert "simd_inst::" in out.kernel_source
    assert "tl.vsimd.region" not in out.kernel_source
    assert "tl.vsimd.plan" not in out.kernel_source


def _rewrite_loop_bounds(func, kind, *, min_value=None, step=None):
    """Rewrite the first loop of ``kind`` into a hand-written shifted/strided form.

    ``T.Parallel`` / ``T.serial`` always build a unit-step loop from the frontend,
    so a non-zero ``min`` or a non-unit ``step`` can only come from hand-written
    TIR; this helper produces exactly that input for the P1 shape gate.
    """

    def visit(node):
        if isinstance(node, tirx.For) and node.kind == kind:
            return tirx.For(
                node.loop_var,
                node.min if min_value is None else min_value,
                node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                node.annotations,
                node.step if step is None else step,
            )
        return None

    return func.with_body(tirx.stmt_functor.ir_transform(func.body, None, visit))


def _parallel_kernel(n=64):
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


def test_parallel_with_non_zero_min_is_opaque():
    """The region models ``[0, extent)``: a shifted Parallel must not be lifted."""
    out = _lift(_rewrite_loop_bounds(_parallel_kernel(), tirx.ForKind.PARALLEL, min_value=tirx.IntImm("int32", 3)))
    regions = _regions(out)
    assert len(regions) == 1
    assert regions[0].status == "OPAQUE"
    assert regions[0].reason_code == "UNSUPPORTED"
    assert "must start at 0" in regions[0].reason


def test_parallel_with_non_unit_step_is_opaque():
    """``step != 1`` visits a different index set, so the nest falls back."""
    out = _lift(_rewrite_loop_bounds(_parallel_kernel(), tirx.ForKind.PARALLEL, step=tirx.IntImm("int32", 2)))
    regions = _regions(out)
    assert len(regions) == 1
    assert regions[0].status == "OPAQUE"
    assert regions[0].reason_code == "UNSUPPORTED"
    assert "must have step 1" in regions[0].reason


def test_serial_shell_with_non_unit_step_is_opaque(capfd):
    """The planner replicates a shell as ``min + iter``; a stride breaks that."""
    n, steps = 16, 4

    @T.prim_func
    def main(A: T.Tensor((n, steps), "float32"), C: T.Tensor((n, steps), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, steps), "float32")
            c_ub = T.alloc_shared((n, steps), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    for k in T.serial(steps):
                        c_ub[i, k] = a_ub[i, k] + 1.0
            T.copy(c_ub, C)

    assert _regions(_lift(main))[0].status == "LIFTED"

    unsupported = _rewrite_loop_bounds(main, tirx.ForKind.SERIAL, step=tirx.IntImm("int32", 2))
    cause = "T.serial shell inside T.Parallel must have step 1"
    report = str(at.AscendAnalyzeVSimdRegionReport(unsupported))
    assert report.count(cause) == 1
    out = _lift(unsupported)
    regions = _regions(out)
    assert len(regions) == 1
    assert regions[0].status == "OPAQUE"
    assert regions[0].reason_code == "UNSUPPORTED"
    assert regions[0].reason == cause
    warning = capfd.readouterr().err
    assert warning.count(cause) == 1
    assert warning.count("[vsimd][P1][UNSUPPORTED]") == 1


if __name__ == "__main__":
    tilelang.testing.main()
