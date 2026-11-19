"""Schema, verifier and reference-expander tests for the typed vsimd region IR.

These tests build regions directly through the `tilelang.ascend.vsimd_ir`
constructors, so every op kind is covered even before the P1 lift pass emits it.
"""

import pytest
import tilelang
from tvm import tirx
from tilelang.ascend import vsimd_ir as ir
from tilelang.layout import Layout


def _decl(name, shape, dtype="float32", scope="shared", data=None):
    return tirx.decl_buffer(shape, dtype, name=name, scope=scope, data=data)


def _captures(*buffers):
    return [ir.make_capture(buffer, "alloc", None) for buffer in buffers]


def _int32(value):
    return tirx.IntImm("int32", value)


def _store_region():
    i = tirx.Var("i", "int32")
    a = _decl("a", (64,))
    c = _decl("c", (64,))
    write = ir.make_access(c, [i], tirx.BufferLoad(a, [i]) + 1.0, True, [0], None)
    read = ir.make_access(a, [i], None, False, [0], None)
    region = ir.region(
        [ir.make_axis(i, 0, 64)],
        [ir.make_op_store([write, read])],
        captures=_captures(a, c),
    )
    return region


def test_store_region_verifies_and_dumps():
    region = _store_region()
    assert ir.verify_region(region) == ""
    text = str(ir.dump_region(region))
    assert "region VF#0 region#0 status=LIFTED" in text
    assert "axes: i in [0, 64)" in text
    assert "captures: a[alloc,SHARED] c[alloc,SHARED]" in text
    assert "store" in text
    assert "W c[i] = a[i] + T.float32(1.0) dep={0} layout=none" in text
    assert "R a[i] dep={0} layout=none" in text


def test_store_region_expands_to_serial_loop():
    stmt, error = ir.expand_region(_store_region())
    assert error == ""
    text = str(stmt)
    assert "for i in range(64)" in text
    assert "c[i] = a[i] + T.float32(1.0)" in text


def test_nested_parallel_op_is_checked_and_expanded():
    i = tirx.Var("i", "int32")
    j = tirx.Var("j", "int32")
    a = _decl("a", (64, 32))
    c = _decl("c", (64, 32))
    inner = ir.make_op_store(
        [
            ir.make_access(c, [i, j], tirx.BufferLoad(a, [i, j]), True, [0, 1], None),
            ir.make_access(a, [i, j], None, False, [0, 1], None),
        ]
    )
    region = ir.region(
        [ir.make_axis(i, 0, 64)],
        [ir.make_op_parallel([ir.make_axis(j, 0, 32)], [inner])],
        captures=_captures(a, c),
    )
    assert ir.verify_region(region) == ""
    text = str(ir.dump_region(region))
    assert "parallel j in [0, 32)" in text
    assert "W c[i, j] = a[i, j] dep={0,1} layout=none" in text

    stmt, error = ir.expand_region(region)
    assert error == ""
    assert "T.grid(64, 32)" in str(stmt)


def test_serial_op_is_checked_and_expanded():
    """A shell counter is not a region axis but is reconstructed as a loop."""
    i = tirx.Var("i", "int32")
    k = tirx.Var("k", "int32")
    a = _decl("a", (64, 4))
    c = _decl("c", (64,))
    store = ir.make_op_store(
        [
            ir.make_access(c, [i], tirx.BufferLoad(a, [i, k]), True, [0], None),
            ir.make_access(a, [i, k], None, False, [0], None),
        ]
    )
    region = ir.region(
        [ir.make_axis(i, 0, 64)],
        [ir.make_op_serial(k, 0, 4, [store])],
        captures=_captures(a, c),
    )
    assert ir.verify_region(region) == ""
    text = str(ir.dump_region(region))
    assert "serial k in [0, 4)" in text
    assert "W c[i] = a[i, k] dep={0} layout=none" in text

    stmt, error = ir.expand_region(region)
    assert error == ""
    assert "T.grid(64, 4)" in str(stmt)


def test_opaque_op_requires_reason():
    i = tirx.Var("i", "int32")
    c = _decl("c", (64,))
    op = ir.make_op_opaque("unknown side effect", [c], [c])
    region = ir.region(
        [ir.make_axis(i, 0, 64)],
        [op],
        captures=_captures(c),
        status="OPAQUE",
        reason="opaque op",
    )
    assert ir.verify_region(region) == ""
    assert "opaque reason=unknown side effect" in str(ir.dump_region(region))


def test_view_resolves_to_captured_base():
    i = tirx.Var("i", "int32")
    base = _decl("base", (64,))
    view = tirx.decl_buffer((32, 2), "float32", name="v", data=base.data)
    store = ir.make_op_store(
        [
            ir.make_access(
                view,
                [i, _int32(0)],
                tirx.BufferLoad(base, [i * 2]) + 1.0,
                True,
                [0],
                None,
            )
        ]
    )
    region = ir.region(
        [ir.make_axis(i, 0, 64)],
        [store],
        captures=[ir.make_capture(base, "alloc", None)],
        local_views=[ir.make_view(view, base)],
    )
    assert ir.verify_region(region) == ""
    assert "v[i, 0]" in str(ir.dump_region(region))


def test_dump_finds_layout_through_a_view():
    i = tirx.Var("i", "int32")
    base = _decl("base", (64,))
    view = tirx.decl_buffer((32, 2), "float32", name="v", data=base.data)
    layout = Layout([64], lambda x: [x // 32, x % 32])
    store = ir.make_op_store(
        [
            ir.make_access(
                view,
                [i, _int32(0)],
                tirx.BufferLoad(base, [i * 2]) + 1.0,
                True,
                [0],
                None,
            )
        ]
    )
    region = ir.region(
        [ir.make_axis(i, 0, 64)],
        [store],
        captures=[ir.make_capture(base, "alloc", layout)],
        local_views=[ir.make_view(view, base)],
    )
    assert ir.verify_region(region) == ""
    # The layout lives on the base capture; the access is on the view buffer.
    assert "layout=annotated" in str(ir.dump_region(region))


def test_uncaptured_buffer_is_rejected():
    i = tirx.Var("i", "int32")
    a = _decl("a", (64,))
    c = _decl("c", (64,))
    write = ir.make_access(c, [i], tirx.BufferLoad(a, [i]), True, [0], None)
    region = ir.region(
        [ir.make_axis(i, 0, 64)],
        [ir.make_op_store([write])],
        captures=[ir.make_capture(c, "alloc", None)],
    )
    assert "uncaptured buffer" in ir.verify_region(region)


def test_store_with_two_writes_is_rejected():
    i = tirx.Var("i", "int32")
    c = _decl("c", (64,))
    first = ir.make_access(c, [i], tirx.const(0.0, "float32"), True, [0], None)
    second = ir.make_access(c, [i], tirx.const(1.0, "float32"), True, [0], None)
    region = ir.region(
        [ir.make_axis(i, 0, 64)],
        [ir.make_op_store([first, second])],
        captures=_captures(c),
    )
    assert "exactly one write" in ir.verify_region(region)


def test_store_without_access_is_rejected():
    i = tirx.Var("i", "int32")
    region = ir.region([ir.make_axis(i, 0, 64)], [ir.make_op_store([])])
    assert "store op has no access" in ir.verify_region(region)


def test_dep_axis_out_of_range_is_rejected():
    i = tirx.Var("i", "int32")
    c = _decl("c", (64,))
    write = ir.make_access(c, [i], tirx.const(0.0, "float32"), True, [1], None)
    region = ir.region(
        [ir.make_axis(i, 0, 64)],
        [ir.make_op_store([write])],
        captures=_captures(c),
    )
    assert "dep_axis out of range" in ir.verify_region(region)


def test_parallel_without_body_is_rejected():
    i = tirx.Var("i", "int32")
    j = tirx.Var("j", "int32")
    region = ir.region(
        [ir.make_axis(i, 0, 64)],
        [ir.make_op_parallel([ir.make_axis(j, 0, 32)], [])],
    )
    assert "parallel op has no body" in ir.verify_region(region)


def test_empty_region_is_rejected():
    i = tirx.Var("i", "int32")
    region = ir.region([ir.make_axis(i, 0, 64)], [])
    assert "region has no op" in ir.verify_region(region)
    # Every region carries at least the op that stands for its body; the P1
    # pass always emits one, including for OPAQUE regions.
    opaque = ir.region([ir.make_axis(i, 0, 64)], [], status="OPAQUE", reason="nothing modelled")
    assert "region has no op" in ir.verify_region(opaque)


def test_opaque_region_without_reason_is_rejected():
    i = tirx.Var("i", "int32")
    c = _decl("c", (64,))
    op = ir.make_op_opaque("side effect", [c], [c])
    region = ir.region([ir.make_axis(i, 0, 64)], [op], captures=_captures(c), status="OPAQUE")
    assert "OPAQUE region has no reason" in ir.verify_region(region)


def test_lifted_region_with_opaque_op_is_rejected():
    i = tirx.Var("i", "int32")
    c = _decl("c", (64,))
    op = ir.make_op_opaque("side effect", [c], [c])
    region = ir.region([ir.make_axis(i, 0, 64)], [op], captures=_captures(c))
    assert "LIFTED region contains an OPAQUE op" in ir.verify_region(region)


def test_opaque_op_cannot_be_expanded():
    i = tirx.Var("i", "int32")
    c = _decl("c", (64,))
    op = ir.make_op_opaque("side effect", [c], [c])
    region = ir.region(
        [ir.make_axis(i, 0, 64)],
        [op],
        captures=_captures(c),
        status="OPAQUE",
        reason="side effect",
    )
    assert ir.verify_region(region) == ""
    stmt, error = ir.expand_region(region)
    # An opaque op has no modelled effect; the reference expander must say so
    # rather than dropping it.
    assert stmt is None
    assert "opaque op" in error


def test_duplicate_capture_is_rejected():
    i = tirx.Var("i", "int32")
    c = _decl("c", (64,))
    write = ir.make_access(c, [i], tirx.const(0.0, "float32"), True, [0], None)
    region = ir.region(
        [ir.make_axis(i, 0, 64)],
        [ir.make_op_store([write])],
        captures=_captures(c) + _captures(c),
    )
    assert "duplicate capture" in ir.verify_region(region)


def test_binding_without_value_is_rejected():
    i = tirx.Var("i", "int32")
    c = _decl("c", (64,))
    write = ir.make_access(c, [i], tirx.const(0.0, "float32"), True, [0], None)
    t = tirx.Var("t", "int32")
    region = ir.region(
        [ir.make_axis(i, 0, 64)],
        [ir.make_op_store([write])],
        captures=_captures(c),
        bindings=[ir.make_var_binding(t, "bind", None)],
    )
    assert "binding has no value" in ir.verify_region(region)


def test_dump_and_expand_validate_their_input():
    i = tirx.Var("i", "int32")
    invalid = ir.region([ir.make_axis(i, 0, 64)], [])
    with pytest.raises(Exception, match="invalid vsimd region"):
        ir.dump_region(invalid)
    with pytest.raises(Exception, match="invalid vsimd region"):
        ir.expand_region(invalid)


if __name__ == "__main__":
    tilelang.testing.main()
