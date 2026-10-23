"""InsertNd2Nz recognizes physical layouts before scheduling."""

import operator

import numpy as np
import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_nz_layout
from tvm import tirx
from testing.ascend._ir import calls, nodes


def _module(*, rows=16, cols=128, padding=1, strides=None, staged=False, reshape=False, vf=None):
    src = tirx.decl_buffer((2, rows, cols) if staged else (rows, cols), "bfloat16", name="src", scope="shared.dyn", strides=strides)
    owner = tirx.decl_buffer((rows // 2, cols * 2), "bfloat16", name="storage", scope="shared.dyn", data=src.data) if reshape else src
    dst = tirx.decl_buffer((2, rows + padding, cols) if staged else (rows + padding, cols), "float32", name="dst", scope="shared.dyn")
    i = tirx.Var("stage", "int32")

    def region(buffer):
        ranges = [tvm.ir.Range.from_min_extent(i, 1)] if staged else []
        return tirx.BufferRegion(buffer, ranges + [tvm.ir.Range(0, rows), tvm.ir.Range(0, cols)])

    body = tirx.Evaluate(T.copy(region(src), region(dst)))
    if vf:
        body = tirx.SBlockRealize([], True, tirx.SBlock([], [], [], vf, body))
    if staged:
        body = tirx.For(i, 0, 2, tirx.ForKind.SERIAL, body)
    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=[owner, dst],
        annotations={"layout_map": {dst: make_ascend_compact_nz_layout(dst)}},
    )
    return tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, root))})


@pytest.mark.parametrize("case", ["plain", "staged", "reshape", "simd", "simt"])
def test_scatter_regions_and_vf_boundary(case):
    before = _module(staged=case == "staged", reshape=case == "reshape", vf={"simd": "SIMD_VF", "simt": "SIMT_VF"}.get(case))
    after = transform.InsertNd2Nz()(before)
    if case == "simt":
        tvm.ir.assert_structural_equal(after, before)
        return
    (scatter,) = calls(after, "tl.ascend_nd2nz_scatter")
    assert [int(x) for x in scatter.args[2:4]] == [16, 128]
    assert [str(x.value) for x in scatter.args[5:]] == ["bfloat16_t", "float"]
    assert not calls(after, "tl.tileop.ascend_copy")
    # The destination footprint includes the bank-conflict padding row.
    analyzer = tvm.arith.Analyzer()
    assert int(analyzer.simplify(scatter.args[0].args[1])) == 16 * 128
    assert int(analyzer.simplify(scatter.args[1].args[1])) == ((128 // 8 - 1) * 17 + 16) * 8
    if case == "staged":
        src_index = scatter.args[0].args[0].indices[0]
        dst_index = scatter.args[1].args[0].indices[0]
        assert src_index.same_as(dst_index)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"padding": 0}, None),
        ({"strides": (256, 1)}, "compact trailing source matrix"),
        ({"staged": True, "strides": (2047, 128, 1)}, "32-byte-aligned source address"),
        ({"staged": True, "strides": (2048, 128, 1)}, None),
    ],
)
def test_scatter_storage_requirements(kwargs, error):
    before = _module(**kwargs)
    if error:
        with pytest.raises(ValueError, match=error):
            transform.InsertNd2Nz()(before)
    else:
        assert len(calls(transform.InsertNd2Nz()(before), "tl.ascend_nd2nz_scatter")) == 1


@pytest.mark.parametrize("padding", [0, 1, 7])
@pytest.mark.parametrize("parts", [1, 2, 4, 8])
def test_partial_nz_copy_uses_complete_storage_pitches(padding, parts):
    from tilelang.layout import make_ascend_nz_layout
    from testing.ascend._ir import copy, kernel, seq

    nz = tirx.decl_buffer((2, 64 + padding, 128), "bfloat16", name="nz", scope="shared.dyn")
    dst = tirx.decl_buffer((2, 64, 128), "bfloat16", name="dst", scope="shared.l1")
    rows = 64 // parts
    before = kernel(
        seq(
            *[
                copy(nz, dst, src_indices=[1, r, 16], dst_indices=[1, r, 32], src_shape=[1, rows, 64], dst_shape=[1, rows, 64])
                for r in range(0, 64, rows)
            ]
        ),
        buffers=[nz, dst],
        annotations={"layout_map": {nz: make_ascend_compact_nz_layout(nz), dst: make_ascend_nz_layout(dst)}},
    )
    after = transform.InsertNd2Nz()(before)
    copies = calls(after, "tl.ascend_nd2nz_post_copy")
    assert len(copies) == parts
    for i, call in enumerate(copies):
        assert [int(x) for x in call.args[2:6]] == [rows, 64, 64 + padding, 64]
        assert [int(x) for x in call.args[1].args[0].indices] == [1, i * rows, 16]
        assert [int(x) for x in call.args[0].args[0].indices] == [1, i * rows, 32]
        assert int(tvm.arith.Analyzer().simplify(call.args[1].args[1])) == (3 * (64 + padding) + rows) * 16


@pytest.mark.parametrize("padding", [0, 1, 7])
def test_partial_scatter_uses_destination_storage_pitch(padding):
    from testing.ascend._ir import copy, kernel

    src = tirx.decl_buffer((16, 128), "bfloat16", scope="shared.dyn")
    dst = tirx.decl_buffer((64 + padding, 256), "bfloat16", scope="shared.dyn")
    before = kernel(
        copy(src, dst, dst_indices=[32, 16], dst_shape=[16, 128]),
        buffers=[src, dst],
        annotations={"layout_map": {dst: make_ascend_compact_nz_layout(dst)}},
    )
    after = transform.InsertNd2Nz()(before)
    (scatter,) = calls(after, "tl.ascend_nd2nz_scatter")
    assert [int(x) for x in scatter.args[2:5]] == [16, 128, 64 + padding]
    assert [int(x) for x in scatter.args[1].args[0].indices] == [32, 16]


@pytest.mark.parametrize(
    "case,error",
    [
        ("out-of-bounds", "provably in bounds"),
        ("column-start", "C0-aligned"),
        ("column-width", "C0-aligned"),
        ("dtype", "dtype conversion"),
        ("destination-layout", "MajorK destination"),
    ],
)
def test_invalid_partial_nz_regions(case, error):
    from tilelang.layout import make_ascend_nz_layout, make_ascend_major_mn_layout
    from testing.ascend._ir import copy, kernel

    src = tirx.decl_buffer((64, 128), "bfloat16", scope="shared.dyn")
    dst = tirx.decl_buffer((64, 128), "float16" if case == "dtype" else "bfloat16", scope="shared.l1")
    rows = 65 if case == "out-of-bounds" else 32
    cols = 63 if case == "column-width" else 64
    before = kernel(
        copy(src, dst, src_indices=[0, 1 if case == "column-start" else 0], src_shape=[rows, cols], dst_shape=[rows, cols]),
        buffers=[src, dst],
        annotations={
            "layout_map": {
                src: make_ascend_compact_nz_layout(src),
                dst: (make_ascend_major_mn_layout if case == "destination-layout" else make_ascend_nz_layout)(dst),
            }
        },
    )
    with pytest.raises(ValueError, match=error):
        transform.InsertNd2Nz()(before)


@pytest.mark.parametrize("parts", [1, 2, 4])
def test_explicit_physical_view_pitch_does_not_depend_on_partition(parts):
    from testing.ascend._ir import kernel, seq

    # The BF16 view retains the 8192-byte FP32 owner's capacity and 64-block
    # C0-group stride without allocating additional storage.
    owner = tirx.decl_buffer((32, 64), "float32", name="owner", scope="shared.dyn")
    view = tirx.decl_buffer((4096,), "bfloat16", data=owner.data, scope="shared.dyn")
    dst = tirx.decl_buffer((4096,), "bfloat16", scope="shared.l1")
    rows = 32 // parts
    before = kernel(
        seq(
            *[
                tirx.Evaluate(
                    T.ascend_nd2nz_post_copy(dst[(32 + r) * 16], view[r * 16], rows, 64, src_pitch=64, dst_pitch=64, dtype="bfloat16")
                )
                for r in range(0, 32, rows)
            ]
        ),
        buffers=[owner, dst],
    )
    after = transform.InsertNd2Nz()(before)
    copies = calls(after, "tl.ascend_nd2nz_post_copy")
    assert len(copies) == parts
    assert all([int(x) for x in call.args[2:6]] == [rows, 64, 64, 64] for call in copies)
    assert [int(x) for x in after["main"].body.block.alloc_buffers[0].shape] == [32, 64]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"src_pitch": 7}, "pitches >= rows"),
        ({"src_pitch": 65544}, "instruction field range"),
        ({"cols": 4096 * 16}, "instruction field range"),
        ({"offset": 1}, "32-byte-aligned"),
        ({"offset": 4000}, "physical span exceeds"),
    ],
)
def test_explicit_post_copy_rejects_invalid_geometry(kwargs, error):
    from testing.ascend._ir import kernel

    src = tirx.decl_buffer((4096,), "bfloat16", scope="shared.dyn")
    dst = tirx.decl_buffer((4096,), "bfloat16", scope="shared.l1")
    args = {"src_pitch": 64, "dst_pitch": 64, "cols": 64, "offset": 0, **kwargs}
    call = T.ascend_nd2nz_post_copy(dst[0], src[args.pop("offset")], 8, dtype="bfloat16", **args)
    with pytest.raises(ValueError, match=error):
        transform.InsertNd2Nz()(kernel(tirx.Evaluate(call), buffers=[src, dst]))


@pytest.mark.parametrize("rows,cols", [(0, 64), (8, 0)])
def test_empty_post_copy_does_not_compute_a_negative_span(rows, cols):
    from testing.ascend._ir import kernel

    src = tirx.decl_buffer((4096,), "bfloat16", scope="shared.dyn")
    dst = tirx.decl_buffer((4096,), "bfloat16", scope="shared.l1")
    before = kernel(
        tirx.Evaluate(T.ascend_nd2nz_post_copy(dst[0], src[0], rows, cols, src_pitch=64, dst_pitch=64, dtype="bfloat16")),
        buffers=[src, dst],
    )
    assert not calls(transform.InsertNd2Nz()(before), "tl.ascend_nd2nz_post_copy")


@pytest.mark.parametrize("source_rows,rows,pitch", [(1, 1, 1), (8, 8, 8), (17, 8, 32)])
def test_canonical_nz_source_pitch_includes_fractal_rounding(source_rows, rows, pitch):
    from tilelang.layout import make_ascend_nz_layout
    from testing.ascend._ir import copy, kernel

    src = tirx.decl_buffer((source_rows, 64), "bfloat16", scope="shared.dyn")
    dst = tirx.decl_buffer((32, 64), "bfloat16", scope="shared.l1")
    before = kernel(
        copy(src, dst, src_shape=[rows, 64], dst_shape=[rows, 64]),
        buffers=[src, dst],
        annotations={"layout_map": {src: make_ascend_nz_layout(src), dst: make_ascend_nz_layout(dst)}},
    )
    (call,) = calls(transform.InsertNd2Nz()(before), "tl.ascend_nd2nz_post_copy")
    assert [int(x) for x in call.args[2:6]] == [rows, 64, pitch, 32]


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("explicit_strides", [False, True])
def test_explicit_post_copy_uses_dense_physical_layout_strides(staged, explicit_strides):
    from tilelang.layout import make_ascend_nz_layout
    from testing.ascend._ir import kernel

    src_shape = (2, 65, 128) if staged else (65, 128)
    dst_shape = (2, 64, 16) if staged else (64, 16)
    src_strides = ((65 * 128, 128, 1) if staged else (128, 1)) if explicit_strides else None
    dst_strides = ((64 * 16, 16, 1) if staged else (16, 1)) if explicit_strides else None
    src = tirx.decl_buffer(src_shape, "bfloat16", scope="shared.dyn", strides=src_strides)
    dst = tirx.decl_buffer(dst_shape, "bfloat16", scope="shared.l1", strides=dst_strides)
    prefix = (1,) if staged else ()
    before = kernel(
        tirx.Evaluate(
            T.ascend_nd2nz_post_copy(dst[prefix + (0, 0)], src[prefix + (0, 16)], 64, 16, src_pitch=65, dst_pitch=64, dtype="bfloat16")
        ),
        buffers=[src, dst],
        annotations={"layout_map": {src: make_ascend_compact_nz_layout(src), dst: make_ascend_nz_layout(dst)}},
    )
    assert len(calls(transform.InsertNd2Nz()(before), "tl.ascend_nd2nz_post_copy")) == 1


@pytest.mark.parametrize("elem_offset,index,valid", [(16, 0, True), (1024, 0, True), (16, -16, False), (16, 16, False), (-16, 0, False)])
def test_explicit_post_copy_checks_capacity_relative_to_view(elem_offset, index, valid):
    from testing.ascend._ir import kernel

    owner = tirx.decl_buffer((2048,), "bfloat16", scope="shared.dyn")
    view = tirx.decl_buffer((1024,), "bfloat16", scope="shared.dyn", data=owner.data, elem_offset=elem_offset)
    dst = tirx.decl_buffer((1024,), "bfloat16", scope="shared.l1")
    before = kernel(
        tirx.Evaluate(T.ascend_nd2nz_post_copy(dst[0], view[index], 64, 16, src_pitch=64, dst_pitch=64, dtype="bfloat16")),
        buffers=[owner, dst],
    )
    if valid:
        assert len(calls(transform.InsertNd2Nz()(before), "tl.ascend_nd2nz_post_copy")) == 1
    else:
        with pytest.raises(ValueError, match="physical span exceeds"):
            transform.InsertNd2Nz()(before)


def _fp4_transpose_module(rows=32, cols=128, partial=False, mode=None):
    src = tirx.decl_buffer(
        (cols, rows),
        "float4_e2m1fn",
        name="src",
        strides=(rows * 2, 1) if mode == "strided" else None,
        elem_offset=1 if mode == "odd-offset" else 2 if mode == "offset" else 0,
    )
    dst = tirx.decl_buffer((rows, cols), "float4_e2m1fn", name="dst", scope="shared.l1", elem_offset=2 if mode == "dst-offset" else 0)
    source = tirx.BufferRegion(src, [tvm.ir.Range(0, cols // 2), tvm.ir.Range(0, rows)]) if partial else src
    destination = tirx.BufferRegion(dst, [tvm.ir.Range(0, rows), tvm.ir.Range(0, cols // 2)]) if partial else dst
    options = {"l2_cache_ctrl": "NOTALLOC_KEEP"} if mode == "cache" else {}
    body = tirx.Evaluate(T.copy(source, destination, transpose=True, **options))
    if mode in ("CUBE", "VECTOR", "SIMD_VF"):
        body = tirx.SBlock([], [], [], mode, body)
    if mode in ("task", "per_core_task"):
        body = tirx.AttrStmt(tirx.IntImm("int32", 0), "tl.ascend_" + mode, 1, body)
    if mode == "mixed":
        axis = tirx.IterVar(tvm.ir.Range(0, 2), tirx.Var("sid", "int32"), tirx.IterVar.ThreadIndex, "cthread")
        body = tirx.AttrStmt(axis, "thread_extent", 2, body)
    root = tirx.SBlock([], [], [], "root", body, alloc_buffers=[dst], annotations={"layout_map": {dst: make_ascend_nz_layout(dst)}})
    func = tirx.PrimFunc([src.data], tirx.SBlockRealize([], True, root), buffer_map={src.data: src})
    if mode == "pto":
        func = func.with_attr("target", tvm.target.Target({"kind": "ascend", "keys": ["pto", "ascend"]}))
    return tvm.IRModule({"main": func})


@pytest.mark.parametrize("rows,cols,partial", [(32, 128, False), (64, 256, False), (24, 128, False), (32, 96, False), (32, 128, True)])
def test_fp4_gm_to_l1_element_transpose(rows, cols, partial):
    before = _fp4_transpose_module(rows, cols, partial)
    if partial or rows % 16 or cols % 64:
        with pytest.raises(ValueError, match="FP4 GM->L1 element transpose requires"):
            transform.InsertNd2Nz()(before)
        return
    after = transform.InsertNd2Nz()(before)
    assert not calls(after, "tl.tileop.ascend_copy")
    (transfer,) = calls(after, "tl.ascend_copy_ubuf_to_cbuf")
    assert int(tvm.arith.Analyzer().simplify(transfer.args[4])) == rows * cols // 64
    assert int(transfer.args[1].args[1]) == rows * cols // 2


@pytest.mark.parametrize("mode", ["CUBE", "VECTOR", "SIMD_VF", "mixed", "manual", "cache", "task", "per_core_task", "pto"])
def test_fp4_transpose_execution_requirements(mode):
    before = _fp4_transpose_module(mode=mode)
    config = {tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False} if mode == "manual" else {}
    error = "PTO" if mode == "pto" else "l2_cache_ctrl" if mode == "cache" else "automatic core scheduling"
    with (
        tvm.transform.PassContext(config=config),
        pytest.raises(ValueError, match=error),
    ):
        transform.InsertNd2Nz()(before)


@pytest.mark.parametrize("mode", ["strided", "odd-offset", "offset", "dst-offset"])
def test_fp4_transpose_rejects_unsupported_source_storage(mode):
    with pytest.raises(ValueError, match="FP4 GM->L1 element transpose requires"):
        transform.InsertNd2Nz()(_fp4_transpose_module(mode=mode))


def test_fp4_transpose_rejects_dynamic_shapes():
    with pytest.raises(ValueError, match="FP4 transpose rows must be static"):
        transform.InsertNd2Nz()(_fp4_transpose_module(rows=tirx.Var("rows", "int32")))


@pytest.mark.parametrize("rows,cols", [(16, 64), (32, 128), (64, 256)])
def test_fp4_transpose_preserves_packed_codes(rows, cols):
    after = transform.InsertNd2Nz()(_fp4_transpose_module(rows, cols))
    (store,) = nodes(after, tirx.BufferStore)
    (loop,) = nodes(after, tirx.For)
    thread = next(
        node.node.var
        for node in nodes(after, tirx.AttrStmt)
        if isinstance(node.node, tirx.IterVar) and node.node.thread_tag == "threadIdx.x"
    )
    codes = np.random.default_rng(0).integers(0, 16, size=(cols, rows), dtype=np.uint8)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).reshape(-1)
    transposed = codes.T
    expected = (transposed[:, ::2] | (transposed[:, 1::2] << 4)).reshape(rows, cols // 64, 32).transpose(1, 0, 2).reshape(-1)
    analyzer = tvm.arith.Analyzer()
    operations = {
        "tirx.shift_right": operator.rshift,
        "tirx.shift_left": operator.lshift,
        "tirx.bitwise_and": operator.and_,
        "tirx.bitwise_or": operator.or_,
    }
    for index, expected_byte in enumerate(expected):
        value = tirx.stmt_functor.substitute(store.value, {loop.loop_var: index // 128, thread: index % 128})

        def load_byte(node):
            if isinstance(node, tirx.BufferLoad):
                address = int(analyzer.simplify(node.indices[0]))
                return tirx.const(int(packed[address]), "uint8")
            if isinstance(node, tirx.Call) and node.op.name in operations:
                arguments = [int(analyzer.simplify(argument)) for argument in node.args]
                return tirx.const(operations[node.op.name](*arguments), node.dtype)

        evaluated = tirx.stmt_functor.ir_transform(tirx.Evaluate(value), None, load_byte)
        assert int(analyzer.simplify(evaluated.value)) == int(expected_byte)
