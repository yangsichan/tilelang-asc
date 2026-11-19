/*!
 * \file vsimd_group_passes.h
 * \brief P2/P3 passes over the P1 per-Parallel ``tl.vsimd.region`` annotations.
 *
 * The chain vectorizes ``T.Parallel`` loops inside a ``SIMD_VF``:
 *
 *   P1  For.annotations["tl.vsimd.region"] = tl.vsimd.Region
 *   P2  For.annotations["tl.vsimd.plan"]   =
 * tl.parallel_vectorize.VSimdRegionPlan P3  replaces every loop carrying a
 * plan; strips both annotations.
 *
 * Only Parallel loops take part. Tile-ops outside a Parallel are left to the
 * regular tile-op lowering; tile-ops inside a Parallel are planned as part of
 * that loop's program.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_VSIMD_GROUP_PASSES_H_
#define TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_VSIMD_GROUP_PASSES_H_

#include <tvm/tirx/function.h>

namespace tvm {
namespace tl {
namespace parallel_vectorize {

/*! \brief P2: plan every annotated Parallel; body is unchanged. */
TVM_DLL tirx::PrimFunc PlanVSimd(tirx::PrimFunc func);

/*! \brief P3: emit planned Parallel loops and strip the P1/P2 annotations. */
TVM_DLL tirx::PrimFunc EmitVSimd(tirx::PrimFunc func);

/*! \brief P2 + P3 (combination entry point). */
TVM_DLL tirx::PrimFunc VectorizeVSimd(tirx::PrimFunc func);

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_VSIMD_GROUP_PASSES_H_
