/*!
 * \file plan.h
 * \brief Planning layer: nest splitting, axis ranking, reductions, build_plan.
 *
 * C++ counterpart of ``parallel_vectorize/plan.py``. ``BuildPlan`` always
 * returns a :class:`VectorProgram` with 1..N segments; the plan holds no
 * instruction names at all (instruction selection is the emit layer's job).
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_PLAN_H_
#define TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_PLAN_H_

#include <optional>
#include <utility>
#include <vector>

#include "ascend/layout/ascend_layouts.h"
#include "model.h"

namespace tvm {
namespace tl {
namespace parallel_vectorize {

/*!
 * \brief Context P1 hands to the planner alongside the nest statement.
 *
 * The region is the P1 contract; rebuilding the plan input from it reproduces
 * the statement but not its annotations, so the remaining fields travel here:
 *
 *  - ``layouts`` decides the store form: a buffer without an explicit layout is
 *    ND (the historical contiguous store), a buffer explicitly annotated with
 *    an Ascend NZ layout is scattered as 32B blocks straight from the register;
 *    dropping this list would silently write NZ buffers in ND order.
 *  - ``assumptions`` / ``bindings`` / ``shape_syms`` are the arithmetic context
 *    reserved for the region-contract analyzer mode (unused in legacy parity).
 */
struct PlanOptions {
  std::vector<std::pair<tirx::Buffer, Layout>> layouts;
  std::vector<PrimExpr> assumptions;
  std::vector<std::pair<tirx::Var, PrimExpr>> bindings;
  std::vector<tirx::Var> shape_syms;
};

/*!
 * \brief ``T.Parallel`` nest -> :class:`VectorProgram` (1..N segments).
 *
 * ``lanes`` defaults to the vector register width derived from the store
 * element dtype (``2048 / elem_bits``), so callers normally leave it empty.
 * ``options`` carries the P1 region context (layouts, ...) and may be empty.
 */
TVM_DLL VectorProgram BuildPlan(const tirx::Stmt &stmt,
                                std::optional<int64_t> lanes = std::nullopt,
                                const PlanOptions &options = PlanOptions());

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_PLAN_H_
