"""Typed vsimd region objects produced by the P1 pass.

The P1 pass will attach one ``tl.vsimd.Region`` object to each outer
``T.Parallel`` loop (Phase 0 step F3). The objects are plain FFI types
registered with reflection, so every field is readable from Python; the thin
classes below bind the C++ type keys to stable Python names.

The ``make_*`` functions are the programmatic constructors for the same IR.
They are what the reference-expander/oracle tests use to cover the supported op
kinds, and what a future handwritten ``T.vsimd`` producer emits; the C++ lift
pass builds the identical objects in-process. Tile-ops are not modelled: the
front-end semantic check rejects them inside ``T.Parallel`` before P1 runs.
"""

from __future__ import annotations

import tvm_ffi
from tvm.ir import Node

tvm_ffi.init_ffi_api("tl.vsimd", __name__)


@tvm_ffi.register_object("tl.vsimd.Axis")
class VSimdAxis(Node):
    """One ``T.Parallel`` loop level: ``var`` in ``[min, min + extent)``."""


@tvm_ffi.register_object("tl.vsimd.Access")
class VSimdAccess(Node):
    """One logical load/store with its index map, ``dep_axes`` and ``guard``."""


@tvm_ffi.register_object("tl.vsimd.Op")
class VSimdOp(Node):
    """One operation inside a region (store/if/parallel/serial/opaque)."""


@tvm_ffi.register_object("tl.vsimd.Capture")
class VSimdCapture(Node):
    """A captured buffer with its capture kind, storage class and layout."""


@tvm_ffi.register_object("tl.vsimd.VarBinding")
class VSimdVarBinding(Node):
    """A captured non-domain variable, or a scalar definition."""


@tvm_ffi.register_object("tl.vsimd.View")
class VSimdView(Node):
    """A local ``T.view`` alias: ``view`` mapped onto ``base``."""


@tvm_ffi.register_object("tl.vsimd.Region")
class VSimdRegion(Node):
    """A lifted ``T.Parallel`` region: ``axes`` plus the ordered ``ops``."""


# ``make_region`` has no defaults on the C++ side; fill them here once so callers
# describe only what matters.
_make_region = make_region  # noqa: F821  (bound by init_ffi_api)


def region(
    axes,
    ops,
    *,
    vf_index=0,
    region_index=0,
    captures=(),
    outer_vars=(),
    shape_syms=(),
    local_views=(),
    assumptions=(),
    bindings=(),
    status="LIFTED",
    reason_code="",
    reason="",
):
    """Build a region from ``axes`` + ``ops`` with the usual empty context.

    An OPAQUE region defaults to ``reason_code="UNSUPPORTED"`` because the
    verifier requires a known reason code for every non-lifted region.
    """
    if status == "OPAQUE" and not reason_code:
        reason_code = "UNSUPPORTED"
    return _make_region(
        vf_index,
        region_index,
        list(axes),
        list(ops),
        list(captures),
        list(outer_vars),
        list(shape_syms),
        list(local_views),
        list(assumptions),
        list(bindings),
        status,
        reason_code,
        reason,
    )


__all__ = [
    "VSimdAxis",
    "VSimdAccess",
    "VSimdOp",
    "VSimdCapture",
    "VSimdVarBinding",
    "VSimdView",
    "VSimdRegion",
    "region",
]
