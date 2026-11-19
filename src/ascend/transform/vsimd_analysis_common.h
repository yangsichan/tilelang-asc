/*!
 * \file vsimd_analysis_common.h
 * \brief Shared read-only IR helpers for the P1 vsimd analyses (PR-A/PR-B).
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_VSIMD_ANALYSIS_COMMON_H_
#define TVM_TL_ASCEND_TRANSFORM_VSIMD_ANALYSIS_COMMON_H_

#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace vsimd {

using namespace tirx;
using ffi::Array;

using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

inline void AddUndefinedVars(const PrimExpr &expr, VarSet *vars) {
  if (!expr.defined())
    return;
  for (const Var &v : tirx::UndefinedVars(expr))
    vars->insert(v);
}

/*! \brief Collect VF-local tl.assume conditions for PR-A/PR-B. */
class VFAssumeCollector : public StmtExprVisitor {
public:
  std::vector<PrimExpr> assumptions;
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::tilelang_assume ||
        op->attr_key == "tl.assume") {
      assumptions.push_back(Downcast<PrimExpr>(op->node));
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

/*! \brief One T.Parallel loop level: loop var and its iteration range. */
struct LoopInfo {
  Var var;
  PrimExpr min;
  PrimExpr extent;
};

/*!
 * \brief One branch literal of the guard path around an access.
 *
 * ``condition`` is the ``if`` condition as written in the source and
 * ``polarity`` says whether the access sits in the then arm (true) or the else
 * arm (false). Mutual exclusion is decided by the dependence prover, not by
 * comparing literals: opposite polarities of the same condition are a
 * contradiction only when the condition is invariant across both executions;
 * different lanes or serial times can take opposite arms.
 */
struct GuardLiteral {
  PrimExpr condition;
  bool polarity{true};
};

/*!
 * \brief One buffer access inside a parallel nest.
 *
 * `value` is only meaningful for writes; it is held so a reference expander
 * can reproduce the stored expression without re-inspecting the IR.
 */
struct AccessInfo {
  Buffer buffer;
  Array<PrimExpr> indices;
  PrimExpr value;
  bool is_write{false};
  // Ordered enclosing serial domains, internal to this parallel region.
  // Each side of a dependence query gets independent copies of these counters.
  std::vector<For> serial_loops;
  /*!
   * \brief Branch literals guarding this access (outermost first).
   *
   * The legality prover instantiates each path with that access's own
   * parallel and serial coordinates. Opposite arms are mutually exclusive
   * only if their conditions agree across both executions.
   */
  std::vector<GuardLiteral> guards;
};

/*! \brief A single top-level T.Parallel nest collected from a SIMD_VF. */
struct ParallelNest {
  /*! \brief Outermost Parallel loop of the nest.
   *
   * Identity of the analysis unit: the P1 pass attaches the region to this
   * node, so consumers match nests by pointer instead of by ordinal.
   */
  const ForNode *root{nullptr};
  std::vector<LoopInfo> loops;      // parallel domain, outer to inner
  std::vector<AccessInfo> accesses; // loads/stores in the nest
  std::vector<Buffer> local_views;  // DeclBuffer/AllocBuffer in the VF body
  /*!
   * \brief Counters of accepted ``T.serial`` shells inside the nest.
   *
   * They are bound by the nest itself (not captured from outside), so the
   * capture analysis must not report them as unbound variables.
   */
  std::vector<tirx::Var> shell_vars;
  /*!
   * \brief Serial loops enclosing the nest inside the same SIMD_VF.
   *
   * They stay in place around the emitted vector code, so their counters are
   * bound (not captured) and may appear in indices / values as lane-uniform
   * scalars. Recorded so the capture closure does not report them as unbound.
   */
  std::vector<LoopInfo> outer_shells;
  /*!
   * \brief Whether this nest alone cannot be modelled.
   *
   * The verdict is per nest: one unsupported nest must not disable its
   * siblings, which stay independent planning units.
   */
  bool unsupported{false};
  std::string reason; // why this nest is unsupported
};

inline bool IsManualCall(const CallNode *call) {
  const auto *op = call->op.as<OpNode>();
  if (op == nullptr)
    return false;
  std::string name = op->name;
  return name.rfind("tl.simd.", 0) == 0 ||
         name.rfind("tl.ascend_nd2nz", 0) == 0;
}

class ManualDetector : public StmtExprVisitor {
public:
  bool manual{false};

  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      if (buffer.scope() == "local.var")
        manual = true;
    }
    for (const auto &match : op->match_buffers) {
      if (match->buffer.scope() == "local.var")
        manual = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (IsManualCall(op))
      manual = true;
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    if (op->buffer.scope() == "local.var")
      manual = true;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    if (op->buffer.scope() == "local.var")
      manual = true;
    StmtExprVisitor::VisitStmt_(op);
  }
};

class NestCollector : public StmtExprVisitor {
public:
  std::vector<ParallelNest> nests;

  // Positive statement grammar for a liftable T.Parallel body. Unknown
  // statements are rejected instead of being silently ignored; this prevents
  // effects or declarations from disappearing during region expansion.
  // Bundled control flow (``if``) is part of the grammar; every other
  // control-flow/effect statement is rejected. Tile-ops inside ``T.Parallel``
  // never reach this analysis: the front-end semantic check
  // (``nested_loop_checker.py``) rejects them before P1 runs.
  void VisitStmt(const Stmt &stmt) final {
    if (depth_ == 0) {
      StmtExprVisitor::VisitStmt(stmt);
      return;
    }
    if (stmt.as<BufferStoreNode>() || stmt.as<SeqStmtNode>() ||
        stmt.as<DeclBufferNode>() || stmt.as<EvaluateNode>() ||
        stmt.as<IfThenElseNode>()) {
      StmtExprVisitor::VisitStmt(stmt);
      return;
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      // Loop statements go to their own handler: it accepts the Parallel chain
      // and a constant-bound serial shell, and records the precise reason for
      // everything else.
      if (loop->kind == ForKind::kParallel || loop->kind == ForKind::kSerial) {
        StmtExprVisitor::VisitStmt(stmt);
        return;
      }
    }
    MarkUnsupported("unsupported statement inside T.Parallel: " +
                    stmt->GetTypeKey());
  }

  /*! \brief Record the first reason the current nest cannot be modelled. */
  void MarkUnsupported(const std::string &reason) {
    if (depth_ == 0 || cur_unsupported_)
      return;
    cur_unsupported_ = true;
    cur_reason_ = reason;
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      bool outer = depth_ == 0;
      if (outer) {
        root_ = op;
        chain_open_ = true;
        loops_.clear();
        accesses_.clear();
        local_views_.clear();
        shell_vars_.clear();
        cur_unsupported_ = false;
        cur_reason_.clear();
      } else if (!chain_open_) {
        // A Parallel loop that is not the sole body of its parent is a sibling,
        // not a deeper level of the same rectangular nest: the merged iteration
        // space would mix statements that the source runs sequentially.
        MarkUnsupported(
            "T.Parallel body is not a single rectangular nest (sibling "
            "parallel/serial/if statement); P1 models one rectangular nest "
            "per region");
        return;
      }
      loops_.push_back(LoopInfo{op->loop_var, op->min, op->extent});
      ++depth_;
      // A region models every Parallel axis as the half-open interval
      // [0, extent): a non-zero `min` or a non-unit `step` enumerates a
      // different index set, so such a nest must not be lifted as if it were
      // the unit shape. Checking it here (rather than at plan time) keeps the
      // failure per nest and reports the real cause; ``plan.cc`` re-checks the
      // lower bound as a defence-in-depth measure for non-P1 entry points.
      if (!IsZeroLowerBound(op->min))
        MarkUnsupported("T.Parallel loop must start at 0 to be vectorized");
      else if (!IsUnitStep(op->step))
        MarkUnsupported("T.Parallel loop must have step 1 to be vectorized");
      // Only a body that is exactly one Parallel loop continues the chain.
      if (!IsSingleParallelLoop(op->body))
        chain_open_ = false;
      StmtExprVisitor::VisitStmt_(op);
      --depth_;
      if (outer) {
        ParallelNest nest;
        nest.root = root_;
        nest.loops = loops_;
        nest.accesses = accesses_;
        nest.local_views = local_views_;
        nest.shell_vars = shell_vars_;
        nest.outer_shells = enclosing_serial_;
        nest.unsupported = cur_unsupported_;
        nest.reason = cur_reason_;
        nests.push_back(std::move(nest));
        root_ = nullptr;
        chain_open_ = false;
        loops_.clear();
        accesses_.clear();
        local_views_.clear();
        shell_vars_.clear();
        cur_unsupported_ = false;
        cur_reason_.clear();
      }
      return;
    } else {
      if (depth_ == 0) {
        // A serial loop *enclosing* a nest: it is not part of the iteration
        // domain, but it stays in place around the emitted vector code, so its
        // counter is a lane-uniform scalar. Nests inside it are still their own
        // analysis units (one region per Parallel).
        if (op->kind == ForKind::kSerial) {
          enclosing_serial_.push_back(
              LoopInfo{op->loop_var, op->min, op->extent});
          StmtExprVisitor::VisitStmt_(op);
          enclosing_serial_.pop_back();
          return;
        }
        StmtExprVisitor::VisitStmt_(op);
        return;
      }
      // A ``T.serial`` shell is part of the nest: P1 models it as a nested op.
      // P2 either absorbs it as a lane-uniform axis, reduces along it, or keeps
      // it as a real scalar loop around the vectorized body, so its trip count
      // may be a runtime value as long as it is *lane invariant* (a lane
      // dependent bound would make the shell a different program per lane).
      if (op->kind == ForKind::kSerial) {
        if (!IsUnitStep(op->step)) {
          // Absorption / reduction replica the strict sequence min..min+extent.
          MarkUnsupported("T.serial shell inside T.Parallel must have step 1");
        } else if (!IsZeroLowerBound(op->min)) {
          MarkUnsupported("T.serial shell inside T.Parallel must start at 0 to "
                          "be modelled");
        } else if (!IsLaneInvariantBound(op->extent)) {
          MarkUnsupported(
              "T.serial shell bound depends on the lane index; a lane-uniform "
              "trip count is required");
        } else {
          // Nest-scoped, not statement-scoped: the list is consumed when the
          // nest is published and cleared when the next nest starts.
          shell_vars_.push_back(op->loop_var);
          serial_loops_.push_back(ffi::GetRef<For>(op));
          StmtExprVisitor::VisitStmt_(op);
          serial_loops_.pop_back();
          return;
        }
      } else {
        MarkUnsupported("nested non-serial loop inside T.Parallel");
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  /*!
   * \brief Whether a shell bound is the same for every lane.
   *
   * The bound may use outer scalars and enclosing shell counters (all uniform),
   * but not a Parallel axis of this nest, and no value read from memory: a
   * buffer read would have to be re-evaluated per lane / per shell iteration.
   */
  bool IsLaneInvariantBound(const PrimExpr &expr) const {
    if (!expr.defined())
      return true;
    for (const Var &var : tirx::UndefinedVars(expr)) {
      for (const LoopInfo &loop : loops_) {
        if (loop.var.same_as(var))
          return false;
      }
    }
    class MemoryRead : public StmtExprVisitor {
    public:
      bool found{false};
      void VisitExpr_(const BufferLoadNode *) final { found = true; }
      void VisitExpr_(const ProducerLoadNode *) final { found = true; }
    } read;
    read(expr);
    return !read.found;
  }

  /*! \brief A shell bound must be knowable without the surrounding lanes. */
  static bool IsZeroLowerBound(const PrimExpr &expr) {
    const auto *imm = expr.as<IntImmNode>();
    return imm != nullptr && imm->value == 0;
  }

  /*! \brief A unit step enumerates the same set as ``range(min, min+extent)``.
   */
  static bool IsUnitStep(const ffi::Optional<PrimExpr> &step) {
    if (!step.has_value())
      return true; // unspecified means one
    const auto *imm = step.value().as<IntImmNode>();
    return imm != nullptr && imm->value == 1;
  }

  // ``if`` is admissible inside a T.Parallel: P1 models it as a guarded op and
  // P2 predicates (or lazily selects) the stores. Each arm's accesses record
  // the branch literal, so the legality prover can add both guards to its
  // violation queries (opposite arms collide only across lanes).
  void VisitStmt_(const IfThenElseNode *op) final {
    if (depth_ == 0) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    // The condition itself is evaluated on every lane, so its loads belong to
    // the enclosing guard path, not to either arm.
    VisitExpr(op->condition);
    guard_path_.push_back(GuardLiteral{op->condition, true});
    VisitStmt(op->then_case);
    guard_path_.pop_back();
    if (op->else_case.defined() && !IsEmptyElse(*op->else_case)) {
      guard_path_.push_back(GuardLiteral{op->condition, false});
      VisitStmt(*op->else_case);
      guard_path_.pop_back();
    }
  }

  /*! \brief ``else`` that carries no work: absent or ``Evaluate(0)``. */
  static bool IsEmptyElse(const Stmt &stmt) {
    const auto *evaluate = stmt.as<EvaluateNode>();
    if (evaluate == nullptr)
      return false;
    const auto *imm = evaluate->value.as<IntImmNode>();
    return imm != nullptr && imm->value == 0;
  }

  void VisitStmt_(const WhileNode *op) final {
    if (depth_ > 0)
      MarkUnsupported("while inside T.Parallel");
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (depth_ > 0) {
      if (op->buffer.scope() == "local.var") {
        MarkUnsupported("local.var access inside T.Parallel");
        return;
      }
      AccessInfo info;
      info.buffer = op->buffer;
      info.indices = op->indices;
      info.value = op->value;
      info.is_write = true;
      info.serial_loops = serial_loops_;
      info.guards = guard_path_;
      accesses_.push_back(std::move(info));
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (depth_ > 0) {
      if (op->buffer.scope() == "local.var") {
        MarkUnsupported("local.var access inside T.Parallel");
        return;
      }
      AccessInfo info;
      info.buffer = op->buffer;
      info.indices = op->indices;
      info.is_write = false;
      info.serial_loops = serial_loops_;
      info.guards = guard_path_;
      accesses_.push_back(std::move(info));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    local_views_.push_back(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    local_views_.push_back(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    // A bare Evaluate is an effectful/opaque statement. The region capture
    // only models values, BufferStores, branches and serial shells, so lifting
    // anything else would silently drop the effect. Reject it here instead of
    // letting it pass A/B/C. Tile-ops never reach P1 (the front-end semantic
    // check forbids them inside T.Parallel), so no tile-op is expandable here.
    if (depth_ > 0) {
      const auto *call = op->value.as<CallNode>();
      const auto *callee = call == nullptr ? nullptr : call->op.as<OpNode>();
      const std::string name = callee == nullptr ? op->value->GetTypeKey()
                                                 : std::string(callee->name);
      MarkUnsupported("opaque Evaluate inside T.Parallel: " + name);
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (depth_ > 0 && !cur_unsupported_) {
      const auto *callee = op->op.as<OpNode>();
      if (callee != nullptr) {
        std::string name = callee->name;
        if (name.rfind("tl.tileop.", 0) == 0) {
          // The front-end rejects every tile-op inside T.Parallel; keep a
          // per-nest diagnostic for direct analysis callers.
          MarkUnsupported("tile-op inside T.Parallel is forbidden: " + name);
          return;
        }
        if (name.rfind("tl.ascend_", 0) == 0) {
          MarkUnsupported("TileOp/copy inside T.Parallel: " + name);
          return;
        }
        if (name == "tirx.call_extern" || name == "tirx.call_pure_extern") {
          MarkUnsupported("opaque call_extern inside T.Parallel");
          return;
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

private:
  /*! \brief True while the current nest may still be deepened by one loop. */
  static bool IsSingleParallelLoop(const Stmt &body) {
    if (const auto *loop = body.as<ForNode>())
      return loop->kind == ForKind::kParallel;
    if (const auto *seq = body.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        if (const auto *only = seq->seq[0].as<ForNode>())
          return only->kind == ForKind::kParallel;
      }
    }
    return false;
  }

  int depth_{0};
  bool chain_open_{false};
  const ForNode *root_{nullptr};
  bool cur_unsupported_{false};
  std::string cur_reason_;
  std::vector<LoopInfo> loops_;
  std::vector<AccessInfo> accesses_;
  std::vector<Buffer> local_views_;
  std::vector<tirx::Var> shell_vars_;
  /*! \brief Serial loops enclosing the nest currently being collected. */
  std::vector<LoopInfo> enclosing_serial_;
  /*! \brief Enclosing serial domains of the access being recorded. */
  std::vector<For> serial_loops_;
  /*! \brief Branch literals of the statement currently being visited. */
  std::vector<GuardLiteral> guard_path_;
};

class AssumptionFilter : public StmtExprVisitor {
public:
  bool ok{true};
  void VisitExpr_(const BufferLoadNode *) final { ok = false; }
  void VisitExpr_(const ProducerLoadNode *) final { ok = false; }
  void VisitExpr_(const CallNode *) final { ok = false; }
};

inline bool IsSupportedAssumption(const PrimExpr &cond) {
  if (!cond.defined() || !cond.dtype().is_bool())
    return false;
  if (tirx::SideEffect(cond) > tirx::CallEffectKind::kPure)
    return false;
  AssumptionFilter filter;
  filter(cond);
  return filter.ok;
}

} // namespace vsimd
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VSIMD_ANALYSIS_COMMON_H_
