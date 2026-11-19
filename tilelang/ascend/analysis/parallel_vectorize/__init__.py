"""C++-only facade for the parallel_vectorize P2/P3 vectorizer.

Every symbol here is backed by the compiled C++ implementation
(``src/ascend/transform/parallel_vectorize``) through the TVM FFI; nothing in this
package plans, emits, or renders in Python.

Data flow (all C++):

============================================ ===============================
``build_plan(stmt)``                         P2: ``T.Parallel`` nest -> ``VectorProgram``
``emit_plan(plan)``                          P3: plan -> ``T.simd.*`` TIR
``render_plan(plan)``                        plan -> readable text (diff oracle)
``finite_difference`` / ``classify_delta``   the mathematical core
``Axis`` / ``AccessInfo`` / ``StoreSpec`` /  FFI plan objects (fields match the
``ValueNode`` / ``StorePlan`` /              C++ ``def_ro`` registration)
``ReduceInfo`` / ``VectorPlan`` /
``VectorProgram``
============================================ ===============================

The pipeline passes live in ``tilelang.ascend.transform``
(``AscendLiftVSimd`` -> ``AscendPlanVSimd`` -> ``AscendEmitVSimd``).
"""

from __future__ import annotations

from ._ffi import (
    AccessClass,
    AccessInfo,
    Axis,
    PlanNode,
    ReduceInfo,
    StorePlan,
    StoreSpec,
    Unsupported,
    ValueNode,
    VectorPlan,
    VectorProgram,
    build_plan,
    classify_delta,
    emit_plan,
    finite_difference,
    render_plan,
    verify_no_scalar_data_ops,
)

__all__ = [
    "Axis",
    "AccessClass",
    "AccessInfo",
    "PlanNode",
    "StoreSpec",
    "ValueNode",
    "StorePlan",
    "ReduceInfo",
    "VectorPlan",
    "VectorProgram",
    "Unsupported",
    "finite_difference",
    "classify_delta",
    "build_plan",
    "emit_plan",
    "render_plan",
    "verify_no_scalar_data_ops",
]
