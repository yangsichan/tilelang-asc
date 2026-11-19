/*!
 * \file vsimd_group_passes.cc
 * \brief P2/P3 over the P1 per-Parallel ``tl.vsimd.region`` annotations.
 *
 * Scope: the passes only touch ``T.Parallel`` loops. P1 annotates every
 * top-level Parallel inside a ``SIMD_VF``; P2 plans exactly those loops and
 * writes the plan back to the same ``For`` node; P3 replaces the loops that
 * carry a plan and strips the P1/P2 annotations. Tile-ops outside a Parallel
 * (top-level in the VF) are deliberately left alone: they belong to the
 * regular tile-op lowering, not to this vectorization chain.
 */
#include "vsimd_group_passes.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../vsimd_ir.h"
#include "emit.h"
#include "layout/layout.h"
#include "model.h"
#include "plan.h"
#include "primitives.h"
#include "region_plan.h"

namespace tvm {
namespace tl {
namespace parallel_vectorize {

using namespace tirx;
using namespace tirx::transform;

using ffi::String;

namespace {

constexpr const char *kRegionKey = "tl.vsimd.region";
constexpr const char *kPlanKey = "tl.vsimd.plan";

/*! \brief Layouts recorded by the regions of one VF, keyed by data var. */
std::vector<std::pair<Buffer, Layout>> CollectVfLayouts(const Stmt &body) {
  std::vector<std::pair<Buffer, Layout>> layouts;
  std::unordered_set<const ffi::Object *> seen;

  class Collector final : public StmtExprVisitor {
  public:
    Collector(std::vector<std::pair<Buffer, Layout>> *layouts,
              std::unordered_set<const ffi::Object *> *seen)
        : layouts_(layouts), seen_(seen) {}

    void VisitStmt_(const ForNode *op) final {
      auto it = op->annotations.find(kRegionKey);
      if (it != op->annotations.end()) {
        if (auto region = (*it).second.as<vsimd::VSimdRegion>();
            region.has_value() && region.value().defined()) {
          for (const vsimd::VSimdCapture &capture : region.value()->captures) {
            if (!capture->layout.has_value() ||
                !capture->layout.value().defined())
              continue;
            if (seen_->insert(capture->buffer->data.get()).second)
              layouts_->emplace_back(capture->buffer, capture->layout.value());
          }
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }

  private:
    std::vector<std::pair<Buffer, Layout>> *layouts_;
    std::unordered_set<const ffi::Object *> *seen_;
  } collector(&layouts, &seen);

  collector(body);
  return layouts;
}

/*! \brief Planner context: the VF's captured layouts. */
PlanOptions MakeOptions(const std::vector<std::pair<Buffer, Layout>> &layouts) {
  PlanOptions options;
  options.layouts = layouts;
  return options;
}

/*! \brief Planner context: the VF layouts plus the region's own context. */
PlanOptions
MakeRegionOptions(const vsimd::VSimdRegion &region,
                  const std::vector<std::pair<Buffer, Layout>> &vf_layouts) {
  PlanOptions options = MakeOptions(vf_layouts);
  for (const PrimExpr &condition : region->assumptions)
    options.assumptions.push_back(condition);
  for (const vsimd::VSimdVarBinding &binding : region->bindings) {
    if (binding->value.has_value() && binding->value.value().defined())
      options.bindings.emplace_back(binding->var, binding->value.value());
  }
  for (const Var &var : region->shape_syms)
    options.shape_syms.push_back(var);
  return options;
}

/*!
 * \brief Fail when a region P1 lifted cannot be vectorized by P2 or P3.
 *
 * P1 LIFTED regions are the only units this chain owns: P1 has already proved
 * the nest legal, so leaving it as a scalar ``T.Parallel`` inside
 * ``__simd_vf__`` would lower to scalar instructions, which ccec rejects far
 * from the actual cause. Report the region and the planner/emitter reason at
 * the stage that owns the failure instead of deferring to codegen.
 */
void FailUnvectorizedRegion(const char *stage, int64_t vf_index,
                            int64_t region_index, const Unsupported &error) {
  std::ostringstream message;
  message << "[vsimd] " << stage << " cannot vectorize VF#" << vf_index
          << " region#" << region_index
          << " that P1 lifted: " << error.message();
  for (const std::string &diagnostic : error.diagnostics())
    message << "\n  " << diagnostic;
  message << "\nLeaving scalar code inside T.SimdVF is rejected by ccec; fix "
             "the nest shape so P2/P3 can vectorize it.";
  LOG(FATAL) << message.str();
}

/*! \brief Short P1 summary for a reason code (mirrors the P1 warning text). */
const char *P1RejectionSummary(const ffi::String &reason_code) {
  if (reason_code == vsimd::reason_code::kIllegal)
    return "parallel independence violation";
  if (reason_code == vsimd::reason_code::kUnproven)
    return "parallel independence could not be proven";
  if (reason_code == vsimd::reason_code::kInternal)
    return "internal compiler consistency check failed";
  return "automatic lifting is unsupported";
}

/*!
 * \brief Fail on a region P1 did not lift (status != LIFTED).
 *
 * Leaving an OPAQUE nest in a ``T.SimdVF`` would lower to scalar instructions,
 * which ccec rejects far from the cause. P2 is the fail-fast gate: it prints
 * P1's reason code / reason and repeats the warning P1 logged, then aborts.
 * ``TL_ASCEND_SIMD_LIFT=False`` selects the legacy Parallel lowering instead.
 */
[[noreturn]] void FailOpaqueRegion(int64_t vf_index, int64_t region_index,
                                   const vsimd::VSimdRegion &region) {
  std::ostringstream message;
  message << "[vsimd] P2 cannot lower VF#" << vf_index << " region#"
          << region_index << ": P1 status=" << region->status
          << " (reason_code=" << region->reason_code << "): " << region->reason;
  message << "\nP1 reported: [vsimd][P1][" << region->reason_code << "] VF#"
          << vf_index << " region#" << region_index << ": "
          << P1RejectionSummary(region->reason_code) << "; " << region->reason
          << "; original IR retained; no successful fallback is implied";
  message << "\nLeaving scalar code inside T.SimdVF is rejected by ccec; fix "
             "the nest so P1 can lift it, or set TL_ASCEND_SIMD_LIFT=False to "
             "use the legacy Parallel lowering.";
  LOG(FATAL) << message.str();
}

/*!
 * \brief Fail on a ``T.Parallel`` P1 never visited (no region annotation).
 *
 * P1 only annotates nests reachable through ``SeqStmt`` / transparent
 * attributes / serial shells. A Parallel nested under a branch or a ``while``
 * is not part of that model; P1 logs a warning and P2 must fail instead of
 * leaving it scalar inside the VF.
 */
[[noreturn]] void FailUnliftedParallel(int64_t vf_index) {
  std::ostringstream message;
  message << "[vsimd] P2 cannot lower a T.Parallel in VF#" << vf_index
          << " that P1 did not lift: the loop has no P1 region (it is nested "
             "under a branch or a while loop)";
  message << "\nP1 reported: [vsimd][P1][UNSUPPORTED] VF#" << vf_index
          << ": T.Parallel loop(s) have no P1 region (nested under a branch "
             "or a while loop); original IR retained; no successful fallback "
             "is implied";
  message << "\nLeaving scalar code inside T.SimdVF is rejected by ccec; "
             "hoist the loop to the top level of the VF, or set "
             "TL_ASCEND_SIMD_LIFT=False to use the legacy Parallel lowering.";
  LOG(FATAL) << message.str();
}

/*!
 * \brief Plan every annotated top-level Parallel of one VF.
 *
 * Every Parallel in the VF must carry a LIFTED P1 region: an OPAQUE region or
 * an unannotated Parallel aborts with P1's reason instead of remaining scalar
 * inside the VF. Nothing else in the VF body is inspected, so top-level
 * tile-ops are not consumed by this chain.
 */
class VfRegionPlanner final : public StmtMutator {
public:
  VfRegionPlanner(const std::vector<std::pair<Buffer, Layout>> &layouts,
                  int64_t vf_index)
      : layouts_(layouts), vf_index_(vf_index) {}

  Stmt VisitStmt_(const ForNode *op) final {
    auto it = op->annotations.find(kRegionKey);
    if (it == op->annotations.end()) {
      if (op->kind == ForKind::kParallel)
        FailUnliftedParallel(vf_index_);
      return StmtMutator::VisitStmt_(op);
    }

    For loop = ffi::GetRef<For>(op);
    auto region = (*it).second.as<vsimd::VSimdRegion>();
    if (!region.has_value() || !region.value().defined()) {
      LOG(FATAL) << "[vsimd] P2 found a malformed tl.vsimd.region annotation "
                    "on VF#"
                 << vf_index_
                 << "; leaving the loop scalar inside T.SimdVF is rejected by "
                    "ccec";
    }
    const vsimd::VSimdRegion &typed = region.value();
    if (typed->status != String(vsimd::region_status::kLifted)) {
      FailOpaqueRegion(vf_index_, typed->region_index, typed);
    }
    try {
      VectorProgram program =
          BuildPlan(loop, std::nullopt, MakeRegionOptions(typed, layouts_));
      VSimdRegionPlan plan = MakeVSimdRegionPlan(
          typed->vf_index, typed->region_index, std::move(program));
      loop.CopyOnWrite()->annotations.Set(kPlanKey, ffi::Any(plan));
      return loop;
    } catch (const Unsupported &e) {
      FailUnvectorizedRegion("P2", typed->vf_index, typed->region_index, e);
    }
    return loop; // unreachable: FailUnvectorizedRegion aborts
  }

private:
  const std::vector<std::pair<Buffer, Layout>> &layouts_;
  int64_t vf_index_{0};
};

/*! \brief Walk the function and plan the regions of every SIMD_VF. */
class RegionPlanner final : public StmtMutator {
public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF")
      return StmtMutator::VisitStmt_(op);

    SBlock block = ffi::GetRef<SBlock>(op);
    const std::vector<std::pair<Buffer, Layout>> layouts =
        CollectVfLayouts(block->body);
    VfRegionPlanner planner(layouts, vf_index_++);
    Stmt body = planner(block->body);
    if (!body.same_as(block->body))
      block.CopyOnWrite()->body = body;
    return block;
  }

private:
  int64_t vf_index_{0};
};

/*! \brief Drop the P1/P2 annotations of a loop, keeping everything else. */
Stmt StripLoopAnnotations(For loop) {
  ForNode *writer = loop.CopyOnWrite();
  writer->annotations.erase(kRegionKey);
  writer->annotations.erase(kPlanKey);
  return loop;
}

/*!
 * \brief Emit every planned Parallel and clean the P1/P2 annotations.
 *
 * The plan lives on the loop it describes, so replacement needs no span or
 * kind check: a plan annotation is by construction attached to the loop it was
 * built from. A failed emission is a hard error: P1 already lifted the region,
 * so leaving the scalar loop inside ``__simd_vf__`` would be rejected by ccec
 * later. Every annotation is stripped on the success path so P1/P2 metadata
 * never reaches codegen.
 */
class RegionEmitter final : public StmtMutator {
public:
  Stmt VisitStmt_(const ForNode *op) final {
    auto it = op->annotations.find(kPlanKey);
    if (it != op->annotations.end()) {
      const auto *plan_node = (*it).second.as<VSimdRegionPlanNode>();
      if (plan_node != nullptr && plan_node->version == kRegionPlanVersion) {
        VSimdRegionPlan plan = ffi::GetRef<VSimdRegionPlan>(plan_node);
        try {
          Stmt emitted = EmitPlan(ffi::ObjectRef(plan->program));
          if (serial_shells_ > 0) {
            // The replaced Parallel sits inside a serial loop that stays in
            // place (a shell around the nest). Its iterations run in memory
            // order and may read back what the previous one stored, so the
            // replacement ends with a VST->VLD barrier.
            emitted = tirx::SeqStmt({emitted, VstVldBarrier()});
          }
          return emitted;
        } catch (const Unsupported &e) {
          FailUnvectorizedRegion("P3", plan->vf_index, plan->region_index, e);
        }
      } else if (plan_node == nullptr) {
        LOG(FATAL) << "[vsimd] P3 found a malformed tl.vsimd.plan payload; "
                      "leaving the loop scalar inside T.SimdVF is rejected by "
                      "ccec";
      } else {
        LOG(FATAL) << "[vsimd] P3 found an incompatible tl.vsimd.plan on VF#"
                   << plan_node->vf_index << " region#"
                   << plan_node->region_index << " (version "
                   << plan_node->version << ", expected " << kRegionPlanVersion
                   << "); leaving the loop scalar inside T.SimdVF is rejected "
                      "by ccec";
      }
    }
    const bool serial = op->kind == ForKind::kSerial;
    if (serial)
      ++serial_shells_;
    Stmt visited = StmtMutator::VisitStmt_(op);
    if (serial)
      --serial_shells_;
    const auto *loop = visited.as<ForNode>();
    if (loop == nullptr)
      return visited;
    if (loop->annotations.count(kRegionKey) == 0 &&
        loop->annotations.count(kPlanKey) == 0)
      return visited;
    return StripLoopAnnotations(ffi::GetRef<For>(loop));
  }

private:
  /*! \brief Serial loops currently enclosing the planned Parallel. */
  int serial_shells_{0};
};

tvm::transform::Pass MakeGroupPass(const char *name,
                                   tirx::PrimFunc (*fn)(tirx::PrimFunc)) {
  auto pass_func = [fn](tirx::PrimFunc func, const IRModule &, PassContext) {
    return fn(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, name, {});
}

} // namespace

tirx::PrimFunc PlanVSimd(tirx::PrimFunc func) {
  RegionPlanner planner;
  tirx::PrimFuncNode *writer = func.CopyOnWrite();
  writer->body = planner(std::move(writer->body));
  return func;
}

tirx::PrimFunc EmitVSimd(tirx::PrimFunc func) {
  RegionEmitter emitter;
  tirx::PrimFuncNode *writer = func.CopyOnWrite();
  writer->body = emitter(std::move(writer->body));
  return func;
}

tirx::PrimFunc VectorizeVSimd(tirx::PrimFunc func) {
  func = PlanVSimd(std::move(func));
  return EmitVSimd(std::move(func));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendPlanVSimd", []() {
    return MakeGroupPass("tl.AscendPlanVSimd", PlanVSimd);
  });
  refl::GlobalDef().def("tl.transform.AscendEmitVSimd", []() {
    return MakeGroupPass("tl.AscendEmitVSimd", EmitVSimd);
  });
  refl::GlobalDef().def("tl.transform.AscendVectorizeVSimd", []() {
    return MakeGroupPass("tl.AscendVectorizeVSimd", VectorizeVSimd);
  });
}

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm
