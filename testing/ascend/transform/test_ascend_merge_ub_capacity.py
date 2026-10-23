"""On-chip capacity check in MergeUBAllocations.

The pass must reject a plan whose footprint exceeds the physical capacity of the
targeted scope, and must stay silent when no capacity is known (no target
attribute, or a non-Ascend target).
"""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

UB_CAPACITY = 248 * 1024  # no SIMT_VF in these bodies
SIMT_VF_UB_CAPACITY = UB_CAPACITY - 32 * 1024  # thread context reservation

# (scope, per-buffer element count, dtype, capacity) — every entry overflows.
OVERFLOW_SCOPES = [
    ("shared.dyn", 32768, "int32", UB_CAPACITY),  # 2 x 128 KB
    ("shared.l1", 76800, "int32", 512 * 1024),  # 2 x 300 KB
    ("shared.l0c", 51200, "int32", 256 * 1024),  # 2 x 200 KB
    ("shared.l0a", 12288, "int32", 64 * 1024),  # 2 x 48 KB
    ("shared.l0b", 12288, "int32", 64 * 1024),  # 2 x 48 KB
]


def module(buffers, body=None, contract=None, target="ascend", wrap=None):
    """Module whose buffers are allocated at the top level.

    Pass ``target=None`` for a target-less module (the pass must degrade to no
    checking) and ``contract`` to attach ``tl.buffer_alias_map``.
    """
    decls = [tirx.decl_buffer(shape, dtype, name=name, scope=scope) for name, shape, dtype, scope in buffers]
    body = (
        body
        if body is not None
        else tirx.SeqStmt([*[tirx.AllocBuffer(buffer) for buffer in decls], *[tirx.BufferStore(buffer, 0, [0]) for buffer in decls]])
    )
    if wrap is not None:
        body = wrap(body)
    if contract is not None:
        by_name = {buffer.name: buffer for buffer in decls}
        table = {by_name[src].data: [by_name[dst].data for dst in dsts] for src, dsts in contract.items()}
        body = tirx.AttrStmt(table, "tl.buffer_alias_map", 1, body)
    func = tirx.PrimFunc([], body)
    if target is not None:
        func = func.with_attr("target", tvm.target.Target(target))
    return tvm.IRModule({"main": func})


def run(mod, **kwargs):
    return transform.MergeUBAllocations(**kwargs)(mod)


def test_overflow_message_is_actionable():
    mod = module([("ub0", (32768,), "int32", "shared.dyn"), ("ub1", (32768,), "int32", "shared.dyn")])
    with pytest.raises(tvm.error.InternalError) as error:
        run(mod)
    message = str(error.value)
    assert "on-chip memory overflow" in message
    assert '"shared.dyn"' in message
    assert "Unified Buffer (UB)" in message
    assert str(2 * 128 * 1024) in message and str(UB_CAPACITY) in message
    # unlimit is explicitly called out as no help.
    assert "T.annotate_unlimit_memory" in message


def test_reuse_decides_the_verdict():
    """The check reads the plan, not the sum: reuse makes the same shapes fit."""
    buffers = [("a", (32768,), "int32", "shared.dyn"), ("b", (32768,), "int32", "shared.dyn")]
    shared = module(buffers, contract={"a": ["b"], "b": ["a"]})
    run(shared)  # one 128 KB slot: fits
    with pytest.raises(tvm.error.InternalError, match="on-chip memory overflow"):
        run(shared, disable_reuse=True)


@pytest.mark.parametrize("scope,elements,dtype,capacity", OVERFLOW_SCOPES)
def test_every_scope_is_checked(scope, elements, dtype, capacity):
    mod = module([("x", (elements,), dtype, scope), ("y", (elements,), dtype, scope)])
    with pytest.raises(tvm.error.InternalError) as error:
        run(mod)
    message = str(error.value)
    assert f'"{scope}"' in message
    assert str(capacity) in message


def test_simt_vf_reserves_ub():
    """SIMT_VF parks its 32 KB thread context in the UB, lowering the usable limit."""
    buffers = [("ub0", (61440,), "int32", "shared.dyn")]  # 240 KB: fits 248 KB, not 216 KB
    run(module(buffers))  # no SIMT_VF
    simt_vf = module(buffers, wrap=lambda body: tirx.SBlock([], [], [], "SIMT_VF", body))
    with pytest.raises(tvm.error.InternalError) as error:
        run(simt_vf)
    assert str(SIMT_VF_UB_CAPACITY) in str(error.value)


def test_constant_floor_of_a_mixed_scope_is_fatal():
    """A symbolic allocation must not hide constants that alone exceed capacity.

    The plan total is `n*4 + padding + 262144` and the analyzer cannot bound it
    (a bare extent variable may not be assumed non-negative), so the check also
    judges the constant-sized allocations it can never fall below.
    """
    n = tirx.Var("n", "int32")
    decls = [tirx.decl_buffer((32768,), "int32", name=name, scope="shared.dyn") for name in ("ub0", "ub1")]
    decls.append(tirx.decl_buffer((n,), "int32", name="dyn", scope="shared.dyn"))
    body = tirx.SeqStmt([*[tirx.AllocBuffer(buffer) for buffer in decls], *[tirx.BufferStore(buffer, 0, [0]) for buffer in decls]])
    with pytest.raises(tvm.error.InternalError) as error:
        run(module([], body=body))
    message = str(error.value)
    assert "is provably at least" in message
    assert str(2 * 128 * 1024) in message and str(UB_CAPACITY) in message


# --- end to end through the pipeline (source only, no NPU required) ---------


def _two_ub_copy_kernel(tile_elements, unlimit):
    import tilelang
    import tilelang.ascend.language as T

    @T.prim_func
    def kernel(a: T.Tensor((tile_elements,), "float32"), c: T.Tensor((tile_elements,), "float32")):
        with T.Kernel(1):
            if unlimit:
                T.annotate_unlimit_memory("shared")
            ub0 = T.alloc_shared((tile_elements,), "float32")
            ub1 = T.alloc_shared((tile_elements,), "float32")
            T.copy(a, ub0)
            T.copy(ub0, c)
            T.copy(a, ub1)
            T.copy(ub1, c)

    return tilelang.lower(kernel, target="ascend")


def test_lowering_rejects_ub_overflow():
    """2 x 128 KB of UB: rejected by the planner instead of silently overflowing."""
    with pytest.raises(tvm.error.InternalError, match="on-chip memory overflow"):
        _two_ub_copy_kernel(32768, unlimit=False)


def test_lowering_fits_when_ub_is_small_enough():
    assert _two_ub_copy_kernel(16384, unlimit=False).kernel_source


def test_unlimit_does_not_waive_the_physical_capacity():
    """unlimit only lifts the scheduling budget; the physical check still applies."""
    with pytest.raises(tvm.error.InternalError, match="on-chip memory overflow"):
        _two_ub_copy_kernel(32768, unlimit=True)
