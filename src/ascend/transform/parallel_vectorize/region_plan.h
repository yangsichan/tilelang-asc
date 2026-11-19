/*!
 * \file region_plan.h
 * \brief P2→P3 plan carrier attached to the T.Parallel it plans.
 *
 * P1 annotates every top-level ``T.Parallel`` with a ``tl.vsimd.Region``. P2
 * plans exactly those loops and writes the resulting plan back to the **same
 * loop node**:
 *
 *     For.annotations["tl.vsimd.region"] = tl.vsimd.Region    // P1
 *     For.annotations["tl.vsimd.plan"]   =
 * tl.parallel_vectorize.VSimdRegionPlan  // P2
 *
 * Keeping the plan on the loop means P3 does not need spans, flattened body
 * indices or fingerprints: the annotation is the contract, and P3 replaces the
 * loop it is attached to. P2 only plans LIFTED regions; a LIFTED region it
 * cannot plan is a hard error, because leaving scalar code inside
 * ``T.SimdVF`` is rejected by ccec. OPAQUE regions carry no plan and are not
 * part of this chain.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_REGION_PLAN_H_
#define TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_REGION_PLAN_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/node_functor.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/tirx/expr.h>

#include <cstdint>

#include "model.h"

namespace tvm {
namespace tl {
namespace parallel_vectorize {

/*! \brief Contract version of the ``tl.vsimd.plan`` payload. */
inline constexpr int64_t kRegionPlanVersion = 1;

/*!
 * \brief The plan of one ``T.Parallel`` loop.
 *
 * ``program`` is the vector program P3 feeds to ``EmitPlan``. ``vf_index`` and
 * ``region_index`` are copied from the P1 region for diagnostics; consumers
 * locate the plan by the node that carries it, never by these indices.
 */
class VSimdRegionPlanNode : public ffi::Object {
public:
  int64_t version{kRegionPlanVersion};
  int64_t vf_index{-1};
  int64_t region_index{-1};
  VectorProgram program;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.parallel_vectorize.VSimdRegionPlan",
                                    VSimdRegionPlanNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VSimdRegionPlanNode>()
        .def_ro("version", &VSimdRegionPlanNode::version)
        .def_ro("vf_index", &VSimdRegionPlanNode::vf_index)
        .def_ro("region_index", &VSimdRegionPlanNode::region_index)
        .def_ro("program", &VSimdRegionPlanNode::program);
  }
};

class VSimdRegionPlan : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(VSimdRegionPlan, ffi::ObjectRef,
                                             VSimdRegionPlanNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VSimdRegionPlanNode);
};

/*! \brief Build the plan attached to one planned Parallel loop. */
TVM_DLL VSimdRegionPlan MakeVSimdRegionPlan(int64_t vf_index,
                                            int64_t region_index,
                                            VectorProgram program);

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_REGION_PLAN_H_
