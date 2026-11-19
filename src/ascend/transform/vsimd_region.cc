/*!
 * \file vsimd_region.cc
 * \brief Build a VSimdRegion from an analyzed Parallel nest, plus the reference
 *        scalar expander used as a structural self-check.
 */
#include "vsimd_region.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "op/builtin.h"
#include "vsimd_memory_legality.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace vsimd;
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::Optional;
using ffi::String;

namespace {

bool ExprUsesVar(const PrimExpr &expr, const Var &var) {
  if (!expr.defined())
    return false;
  for (const Var &used : tirx::UndefinedVars(expr)) {
    if (used.same_as(var))
      return true;
  }
  return false;
}

std::vector<int> ComputeDepAxes(const std::vector<PrimExpr> &indices,
                                const std::vector<LoopInfo> &domain) {
  std::vector<int> dep;
  for (size_t axis = 0; axis < domain.size(); ++axis) {
    bool used = false;
    for (const PrimExpr &index : indices) {
      if (ExprUsesVar(index, domain[axis].var)) {
        used = true;
        break;
      }
    }
    if (used)
      dep.push_back(static_cast<int>(axis));
  }
  return dep;
}

/*! \brief One region access with its dependency axes. */
VSimdAccess MakeRegionAccess(const Buffer &buffer,
                             const Array<PrimExpr> &indices,
                             const Optional<PrimExpr> &value, bool is_write,
                             const std::vector<LoopInfo> &domain) {
  std::vector<PrimExpr> flat(indices.begin(), indices.end());
  Array<Integer> dep_axes;
  for (int axis : ComputeDepAxes(flat, domain))
    dep_axes.push_back(Integer(axis));
  return VSimdAccess(buffer, indices, value, is_write, dep_axes, TrueGuard());
}

/*! \brief Loads of one expression, in source order. */
class LoadCollector : public StmtExprVisitor {
public:
  std::vector<AccessInfo> loads;

  void VisitExpr_(const BufferLoadNode *op) final {
    AccessInfo info;
    info.buffer = op->buffer;
    info.indices = op->indices;
    info.is_write = false;
    loads.push_back(std::move(info));
    StmtExprVisitor::VisitExpr_(op);
  }
};

void CollectExprLoads(const PrimExpr &expr, std::vector<AccessInfo> *out) {
  if (!expr.defined())
    return;
  LoadCollector collector;
  collector(expr);
  for (AccessInfo &load : collector.loads)
    out->push_back(std::move(load));
}

/*! \brief Evaluate(0): the DSL's empty-statement marker. */
bool IsNoOpStmt(const Stmt &stmt) {
  const auto *evaluate = stmt.as<EvaluateNode>();
  if (evaluate == nullptr)
    return false;
  const auto *imm = evaluate->value.as<IntImmNode>();
  return imm != nullptr && imm->value == 0;
}

/*! \brief Innermost body of the collapsed Parallel chain described by info. */
const Stmt &InnermostBody(const VSimdCaptureInfo &info) {
  const ForNode *cur = info.root_for;
  for (size_t level = 1; level < info.domain.size(); ++level) {
    const ForNode *next = cur->body.as<ForNode>();
    if (next == nullptr) {
      if (const auto *seq = cur->body.as<SeqStmtNode>()) {
        if (seq->seq.size() == 1)
          next = seq->seq[0].as<ForNode>();
      }
    }
    ICHECK(next != nullptr)
        << "collapsed Parallel axis chain does not match the loop body";
    cur = next;
  }
  return cur->body;
}

/*!
 * \brief Build the region ops from the actual statement structure.
 *
 * The region is a faithful description of the Parallel body: one store op per
 * BufferStore (with the loads it uses), one ``if`` op per branch, and one
 * structured op per planner-expandable tile-op call. Building from the
 * statements — instead of from the flat access list — is what lets the
 * structural oracle compare the reconstruction with the original body
 * statement by statement.
 */
bool BuildOpsFromStmt(const Stmt &stmt, const std::vector<LoopInfo> &domain,
                      Array<VSimdOp> *ops, std::string *reason);

void BuildStoreOp(const BufferStoreNode *store,
                  const std::vector<LoopInfo> &domain, VSimdOp *out) {
  VSimdOp op(op_kind::kStore);
  op->accesses.push_back(MakeRegionAccess(store->buffer, store->indices,
                                          Optional<PrimExpr>(store->value),
                                          /*is_write=*/true, domain));
  std::vector<AccessInfo> loads;
  CollectExprLoads(store->value, &loads);
  for (const PrimExpr &index : store->indices)
    CollectExprLoads(index, &loads);
  for (const AccessInfo &load : loads)
    op->accesses.push_back(MakeRegionAccess(load.buffer, load.indices,
                                            Optional<PrimExpr>(),
                                            /*is_write=*/false, domain));
  *out = std::move(op);
}

bool BuildOpsFromStmt(const Stmt &stmt, const std::vector<LoopInfo> &domain,
                      Array<VSimdOp> *ops, std::string *reason) {
  if (!stmt.defined())
    return true;
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &child : seq->seq) {
      if (!BuildOpsFromStmt(child, domain, ops, reason))
        return false;
    }
    return true;
  }
  if (const auto *store = stmt.as<BufferStoreNode>()) {
    VSimdOp op(op_kind::kStore);
    BuildStoreOp(store, domain, &op);
    ops->push_back(op);
    return true;
  }
  if (const auto *branch = stmt.as<IfThenElseNode>()) {
    VSimdOp op(op_kind::kIf);
    op->guard = branch->condition;
    Array<VSimdOp> then_ops;
    if (!BuildOpsFromStmt(branch->then_case, domain, &then_ops, reason))
      return false;
    Array<VSimdOp> else_ops;
    if (branch->else_case.defined() && !IsNoOpStmt(*branch->else_case)) {
      if (!BuildOpsFromStmt(*branch->else_case, domain, &else_ops, reason))
        return false;
    }
    if (then_ops.empty() && else_ops.empty()) {
      *reason = "if inside T.Parallel has no modelled branch body";
      return false;
    }
    for (const VSimdOp &child : then_ops)
      op->ops.push_back(ffi::ObjectRef(child));
    for (const VSimdOp &child : else_ops)
      op->else_ops.push_back(ffi::ObjectRef(child));
    ops->push_back(op);
    return true;
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    // A scalar shell inside the Parallel: the counter is *not* a region axis
    // (P2 may absorb it as a lane-uniform axis), so it is modelled as a nested
    // op whose children are ordinary ops over the enclosing axes.
    if (loop->kind != ForKind::kSerial) {
      *reason = "nested non-serial loop inside T.Parallel is not modelled";
      return false;
    }
    VSimdOp op(op_kind::kSerial);
    op->axes.push_back(VSimdAxis(loop->loop_var, loop->min, loop->extent));
    Array<VSimdOp> children;
    if (!BuildOpsFromStmt(loop->body, domain, &children, reason))
      return false;
    if (children.empty()) {
      *reason = "serial shell inside T.Parallel has no modelled body";
      return false;
    }
    for (const VSimdOp &child : children)
      op->ops.push_back(ffi::ObjectRef(child));
    ops->push_back(op);
    return true;
  }
  if (stmt.as<EvaluateNode>()) {
    // An empty-statement marker carries no effect; every other Evaluate is an
    // effect the region cannot describe. Tile-ops never reach P1 (the
    // front-end semantic check rejects them inside T.Parallel).
    if (IsNoOpStmt(stmt))
      return true;
    *reason = "opaque Evaluate inside T.Parallel";
    return false;
  }
  // ``T.view`` declarations are captured as local views, not as ops.
  if (stmt.as<DeclBufferNode>())
    return true;
  *reason = "unsupported statement inside T.Parallel: " + stmt->GetTypeKey();
  return false;
}

/*! \brief True when an op list writes at least one buffer. */
bool WritesSomething(const Array<VSimdOp> &ops) {
  for (const VSimdOp &op : ops) {
    if (op->kind == op_kind::kStore) {
      for (const VSimdAccess &access : op->accesses) {
        if (access->is_write)
          return true;
      }
    } else if (op->kind == op_kind::kParallel || op->kind == op_kind::kSerial) {
      if (WritesSomething(ChildOps(op)))
        return true;
    } else if (op->kind == op_kind::kIf) {
      if (WritesSomething(ChildOps(op)) || WritesSomething(ElseOps(op)))
        return true;
    }
  }
  return false;
}

Optional<Layout>
FindLayout(const std::vector<std::pair<Buffer, Layout>> &layouts,
           const Buffer &buffer) {
  for (const auto &entry : layouts) {
    if (entry.first->data.same_as(buffer->data))
      return entry.second;
  }
  return Optional<Layout>();
}

Stmt ExpandOp(const VSimdOp &op, std::string *unsupported) {
  const VSimdOpNode *node = op.get();
  if (node->kind == op_kind::kStore) {
    std::vector<Stmt> stores;
    for (const VSimdAccess &access : node->accesses) {
      if (!access->is_write)
        continue;
      stores.push_back(
          BufferStore(access->buffer, access->value.value(), access->indices));
    }
    if (stores.empty())
      return Evaluate(0);
    return stores.size() == 1 ? stores[0] : SeqStmt(stores);
  }
  if (node->kind == op_kind::kParallel) {
    std::vector<Stmt> body;
    for (const ffi::ObjectRef &child : node->ops) {
      Stmt expanded = ExpandOp(AsOp(child), unsupported);
      if (!expanded.defined())
        return Stmt();
      body.push_back(expanded);
    }
    Stmt stmt = body.size() == 1 ? body[0] : SeqStmt(body);
    for (auto it = node->axes.rbegin(); it != node->axes.rend(); ++it) {
      stmt = For((*it)->var, (*it)->min, (*it)->extent, ForKind::kSerial, stmt);
    }
    return stmt;
  }
  if (node->kind == op_kind::kSerial) {
    std::vector<Stmt> body;
    for (const ffi::ObjectRef &child : node->ops) {
      Stmt expanded = ExpandOp(AsOp(child), unsupported);
      if (!expanded.defined())
        return Stmt();
      body.push_back(expanded);
    }
    const VSimdAxisNode *axis = node->axes[0].get();
    Stmt stmt = body.size() == 1 ? body[0] : SeqStmt(body);
    return For(axis->var, axis->min, axis->extent, ForKind::kSerial, stmt);
  }
  if (node->kind == op_kind::kIf) {
    std::vector<Stmt> then_stmts;
    for (const ffi::ObjectRef &child : node->ops) {
      Stmt expanded = ExpandOp(AsOp(child), unsupported);
      if (!expanded.defined())
        return Stmt();
      then_stmts.push_back(expanded);
    }
    std::vector<Stmt> else_stmts;
    for (const ffi::ObjectRef &child : node->else_ops) {
      Stmt expanded = ExpandOp(AsOp(child), unsupported);
      if (!expanded.defined())
        return Stmt();
      else_stmts.push_back(expanded);
    }
    Stmt then_case =
        then_stmts.empty()
            ? Evaluate(0)
            : (then_stmts.size() == 1 ? then_stmts[0] : SeqStmt(then_stmts));
    Optional<Stmt> else_case;
    if (!else_stmts.empty()) {
      else_case = else_stmts.size() == 1 ? else_stmts[0] : SeqStmt(else_stmts);
    }
    return IfThenElse(node->guard, then_case, else_case);
  }
  // An opaque op has no modelled effect, so it cannot be reconstructed at all.
  if (node->kind == op_kind::kOpaque) {
    if (unsupported != nullptr) {
      *unsupported = "reference expander cannot represent an opaque op (" +
                     std::string(node->reason) + ")";
    }
    return Stmt();
  }
  // No builder should produce an op the expander cannot reconstruct; refuse
  // loudly rather than dropping it (silently losing an op would make the
  // structural oracle unsound).
  if (unsupported != nullptr) {
    *unsupported = "reference expander cannot represent op kind " +
                   std::string(node->kind);
  }
  return Stmt();
}

void CollectExpectedAxes(const Array<VSimdOp> &ops,
                         std::vector<const VSimdAxisNode *> *axes) {
  for (const VSimdOp &op : ops) {
    if (op->kind == op_kind::kParallel) {
      for (const VSimdAxis &axis : op->axes)
        axes->push_back(axis.get());
      CollectExpectedAxes(ChildOps(op), axes);
    } else if (op->kind == op_kind::kIf) {
      CollectExpectedAxes(ChildOps(op), axes);
      CollectExpectedAxes(ElseOps(op), axes);
    } else if (op->kind == op_kind::kSerial) {
      // A shell counter is not one of ``region->axes`` (those are the Parallel
      // levels), but the reconstruction emits a loop for it at this position,
      // so the expanded-loop walk must include it here to stay aligned.
      for (const VSimdAxis &axis : op->axes)
        axes->push_back(axis.get());
      CollectExpectedAxes(ChildOps(op), axes);
    }
  }
}

int CountWrites(const Array<VSimdOp> &ops) {
  int writes = 0;
  for (const VSimdOp &op : ops) {
    if (op->kind == op_kind::kStore) {
      for (const VSimdAccess &access : op->accesses)
        writes += access->is_write ? 1 : 0;
    } else if (op->kind == op_kind::kParallel || op->kind == op_kind::kSerial) {
      writes += CountWrites(ChildOps(op));
    } else if (op->kind == op_kind::kIf) {
      writes += CountWrites(ChildOps(op));
      writes += CountWrites(ElseOps(op));
    }
  }
  return writes;
}

/*! \brief Write accesses in the order the expanded body visits them. */
void CollectExpectedStores(const Array<VSimdOp> &ops,
                           std::vector<const VSimdAccessNode *> *stores) {
  for (const VSimdOp &op : ops) {
    if (op->kind == op_kind::kStore) {
      for (const VSimdAccess &access : op->accesses) {
        if (access->is_write)
          stores->push_back(access.get());
      }
    } else if (op->kind == op_kind::kParallel || op->kind == op_kind::kSerial) {
      CollectExpectedStores(ChildOps(op), stores);
    } else if (op->kind == op_kind::kIf) {
      CollectExpectedStores(ChildOps(op), stores);
      CollectExpectedStores(ElseOps(op), stores);
    }
  }
}

/* ------------------------------------------------------------------------- */
/* Reconstruction oracle                                                     */
/* ------------------------------------------------------------------------- */

Stmt NormalizeStmt(const Stmt &stmt, std::vector<Buffer> *decls);

/*! \brief Normalize a statement sequence: flatten, drop no-ops, keep order. */
std::vector<Stmt> NormalizeSeq(const Stmt &stmt, std::vector<Buffer> *decls) {
  std::vector<Stmt> out;
  if (!stmt.defined())
    return out;
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &child : seq->seq) {
      std::vector<Stmt> normalized = NormalizeSeq(child, decls);
      out.insert(out.end(), normalized.begin(), normalized.end());
    }
    return out;
  }
  if (const auto *decl = stmt.as<DeclBufferNode>()) {
    decls->push_back(decl->buffer);
    return out;
  }
  if (IsNoOpStmt(stmt))
    return out;
  Stmt normalized = NormalizeStmt(stmt, decls);
  if (normalized.defined())
    out.push_back(normalized);
  return out;
}

Stmt SeqOrMarker(const std::vector<Stmt> &stmts) {
  if (stmts.empty())
    return Evaluate(0); // an empty body still needs a comparable statement
  if (stmts.size() == 1)
    return stmts[0];
  return SeqStmt(Array<Stmt>(stmts.begin(), stmts.end()));
}

/*!
 * \brief Make both sides comparable.
 *
 * Parallel and Serial are the same control flow for the oracle, annotations are
 * not part of the region's meaning, and T.view declarations are compared
 * separately because the reconstruction hoists them out of the loops.
 */
Stmt NormalizeStmt(const Stmt &stmt, std::vector<Buffer> *decls) {
  if (const auto *loop = stmt.as<ForNode>()) {
    auto node = ffi::make_object<ForNode>(*loop);
    node->kind = ForKind::kSerial;
    node->annotations = Map<String, Any>();
    node->body = SeqOrMarker(NormalizeSeq(loop->body, decls));
    return ffi::GetRef<For>(node.get());
  }
  if (stmt.as<SeqStmtNode>())
    return SeqOrMarker(NormalizeSeq(stmt, decls));
  if (const auto *branch = stmt.as<IfThenElseNode>()) {
    auto node = ffi::make_object<IfThenElseNode>(*branch);
    node->then_case = SeqOrMarker(NormalizeSeq(branch->then_case, decls));
    std::vector<Stmt> else_stmts;
    if (branch->else_case.defined())
      else_stmts = NormalizeSeq(*branch->else_case, decls);
    node->else_case = else_stmts.empty()
                          ? Optional<Stmt>()
                          : Optional<Stmt>(SeqOrMarker(else_stmts));
    return ffi::GetRef<IfThenElse>(node.get());
  }
  if (const auto *decl = stmt.as<DeclBufferNode>()) {
    decls->push_back(decl->buffer);
    return Stmt();
  }
  return stmt;
}

std::string DescribeStmt(const Stmt &stmt) {
  return stmt.defined() ? stmt->GetTypeKey() : std::string("<none>");
}

bool CompareExpr(const PrimExpr &expected, const PrimExpr &actual,
                 const std::string &path, std::string *reason) {
  if (ffi::StructuralEqual()(expected, actual))
    return true;
  std::ostringstream os;
  os << path << ": expression differs (expected `" << expected << "`, got `"
     << actual << "`)";
  *reason = os.str();
  return false;
}

bool CompareStmts(const Stmt &expected, const Stmt &actual,
                  const std::string &path, std::string *reason) {
  if (const auto *expected_loop = expected.as<ForNode>()) {
    const auto *actual_loop = actual.as<ForNode>();
    if (actual_loop == nullptr) {
      *reason = path + ": expected a loop, got " + DescribeStmt(actual);
      return false;
    }
    if (!expected_loop->loop_var.same_as(actual_loop->loop_var)) {
      *reason = path + ": loop variable differs";
      return false;
    }
    if (!ffi::StructuralEqual()(expected_loop->min, actual_loop->min) ||
        !ffi::StructuralEqual()(expected_loop->extent, actual_loop->extent)) {
      std::ostringstream os;
      os << path << ": loop range differs (expected [" << expected_loop->min
         << ", " << expected_loop->min + expected_loop->extent << "), got ["
         << actual_loop->min << ", " << actual_loop->min + actual_loop->extent
         << "))";
      *reason = os.str();
      return false;
    }
    return CompareStmts(expected_loop->body, actual_loop->body, path + ".body",
                        reason);
  }
  if (const auto *expected_store = expected.as<BufferStoreNode>()) {
    const auto *actual_store = actual.as<BufferStoreNode>();
    if (actual_store == nullptr) {
      *reason = path + ": expected a store, got " + DescribeStmt(actual);
      return false;
    }
    if (!expected_store->buffer.same_as(actual_store->buffer)) {
      *reason = path + ": store target differs";
      return false;
    }
    if (!ffi::StructuralEqual()(expected_store->indices,
                                actual_store->indices)) {
      *reason = path + ": store indices differ";
      return false;
    }
    return CompareExpr(expected_store->value, actual_store->value,
                       path + ".value", reason);
  }
  if (const auto *expected_eval = expected.as<EvaluateNode>()) {
    const auto *actual_eval = actual.as<EvaluateNode>();
    if (actual_eval == nullptr) {
      *reason = path + ": expected an evaluate, got " + DescribeStmt(actual);
      return false;
    }
    return CompareExpr(expected_eval->value, actual_eval->value,
                       path + ".value", reason);
  }
  if (const auto *expected_if = expected.as<IfThenElseNode>()) {
    const auto *actual_if = actual.as<IfThenElseNode>();
    if (actual_if == nullptr) {
      *reason = path + ": expected a branch, got " + DescribeStmt(actual);
      return false;
    }
    if (!CompareExpr(expected_if->condition, actual_if->condition,
                     path + ".condition", reason))
      return false;
    if (!CompareStmts(expected_if->then_case, actual_if->then_case,
                      path + ".then", reason))
      return false;
    if (!expected_if->else_case.defined() && !actual_if->else_case.defined())
      return true;
    if (!expected_if->else_case.defined() || !actual_if->else_case.defined()) {
      *reason =
          path + ": one branch carries an else arm and the other does not";
      return false;
    }
    return CompareStmts(*expected_if->else_case, *actual_if->else_case,
                        path + ".else", reason);
  }
  if (const auto *expected_seq = expected.as<SeqStmtNode>()) {
    const auto *actual_seq = actual.as<SeqStmtNode>();
    if (actual_seq == nullptr) {
      *reason =
          path + ": expected a statement sequence, got " + DescribeStmt(actual);
      return false;
    }
    if (expected_seq->seq.size() != actual_seq->seq.size()) {
      std::ostringstream os;
      os << path << ": statement count differs (expected "
         << expected_seq->seq.size() << ", got " << actual_seq->seq.size()
         << ")";
      *reason = os.str();
      return false;
    }
    for (size_t i = 0; i < expected_seq->seq.size(); ++i) {
      if (!CompareStmts(expected_seq->seq[i], actual_seq->seq[i],
                        path + "[" + std::to_string(i) + "]", reason))
        return false;
    }
    return true;
  }
  if (ffi::StructuralEqual()(expected, actual))
    return true;
  *reason = path + ": statement differs (expected " + DescribeStmt(expected) +
            ", got " + DescribeStmt(actual) + ")";
  return false;
}

} // namespace

VSimdRegionMatch CompareRegionWithOriginal(const VSimdRegion &region,
                                           const Stmt &original) {
  VSimdRegionMatch result;
  std::string unsupported;
  Stmt expanded = ExpandVSimdRegionToLoops(region, &unsupported);
  if (!expanded.defined()) {
    result.ok = false;
    result.reason = unsupported.empty()
                        ? "reference expander produced no statement"
                        : unsupported;
    return result;
  }

  std::vector<Buffer> original_decls;
  std::vector<Buffer> expanded_decls;
  std::vector<Stmt> original_stmts = NormalizeSeq(original, &original_decls);
  std::vector<Stmt> expanded_stmts = NormalizeSeq(expanded, &expanded_decls);
  // DeclBuffer statements are extracted only so they do not disturb the
  // statement streams. They are not compared: a T.view declaration may live
  // outside the Parallel loop (the DSL hoists it), and the reconstruction
  // re-declares exactly `region->local_views` by construction, which
  // VerifyExpandedRegion checks separately.
  (void)original_decls;
  (void)expanded_decls;
  if (original_stmts.size() != expanded_stmts.size()) {
    std::ostringstream os;
    os << "body statement count differs (original " << original_stmts.size()
       << ", reconstructed " << expanded_stmts.size() << ")";
    result.ok = false;
    result.reason = os.str();
    return result;
  }
  for (size_t i = 0; i < original_stmts.size(); ++i) {
    if (!CompareStmts(original_stmts[i], expanded_stmts[i],
                      "body[" + std::to_string(i) + "]", &result.reason)) {
      result.ok = false;
      return result;
    }
  }
  return result;
}

/*!
 * \brief Whether a Parallel subtree allocates or touches a ``local.fragment``.
 *
 * ``T.alloc_fragment`` reaches P1 as an ``alloc_buffers`` entry of a block, and
 * a nest that reads or writes the tile shows up as a ``local.fragment`` buffer
 * access. Fragment vectorization is not supported on this branch, so P1 reports
 * every nest that touches one as OPAQUE and retains its original body;
 * nests that do not touch a fragment are unaffected and stay independent
 * planning units.
 */
std::string FragmentRejectionReason(const For &loop) {
  class Collector final : public StmtExprVisitor {
  public:
    bool found{false};

    void VisitStmt_(const SBlockNode *op) final {
      for (const Buffer &buffer : op->alloc_buffers) {
        if (buffer.scope() == "local.fragment") {
          found = true;
          return;
        }
      }
      if (!found)
        StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const BufferLoadNode *op) final {
      if (op->buffer.scope() == "local.fragment") {
        found = true;
        return;
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      if (op->buffer.scope() == "local.fragment") {
        found = true;
        return;
      }
      StmtExprVisitor::VisitStmt_(op);
    }
  } collector;
  collector(loop);
  return collector.found
             ? "nest touches a local.fragment buffer (T.alloc_fragment); "
               "fragment vectorization is not supported"
             : "";
}

VSimdRegionLift BuildRegion(const VSimdCaptureInfo &info,
                            const std::string &memory_status, bool verify_body,
                            const std::string &memory_reason) {
  VSimdRegionLift result;
  if (info.root_for != nullptr) {
    result.reason = FragmentRejectionReason(ffi::GetRef<For>(info.root_for));
    if (!result.reason.empty())
      return result;
  }
  if (info.status != "CLOSED") {
    // Keep the per-nest diagnostic: an unsupported nest knows exactly which
    // statement it could not model, and the reason is the user-facing part.
    result.reason =
        info.message.empty() ? "capture/scope is not CLOSED" : info.message;
    for (const std::string &error : info.errors)
      result.reason += "; " + error;
    // Unsupported syntax is collected once and shared by capture and memory.
    // Retain genuinely different findings without repeating the shared cause.
    if (!memory_reason.empty() && memory_reason != info.message &&
        std::find(info.errors.begin(), info.errors.end(), memory_reason) ==
            info.errors.end())
      result.reason +=
          "; memory legality is " + memory_status + ": " + memory_reason;
    return result;
  }
  if (memory_status != "LEGAL") {
    result.reason = "memory legality is " + memory_status;
    if (!memory_reason.empty())
      result.reason += ": " + memory_reason;
    return result;
  }
  if (info.domain.empty()) {
    result.reason = "region requires at least one T.Parallel axis";
    return result;
  }

  for (const AccessInfo &acc : info.accesses) {
    if (acc.buffer.scope() == "local.var") {
      result.reason = "local.var access is not liftable";
      return result;
    }
    if (acc.buffer->shape.size() != acc.indices.size()) {
      result.reason = "access rank does not match buffer rank";
      return result;
    }
  }

  VSimdRegion region;
  region->vf_index = info.vf_index;
  region->region_index = info.nest_index;
  for (const LoopInfo &loop : info.domain)
    region->axes.push_back(VSimdAxis(loop.var, loop.min, loop.extent));
  ICHECK(info.root_for != nullptr)
      << "a lifted region must come from an analysed Parallel nest";
  Array<VSimdOp> ops;
  std::string build_error;
  if (!BuildOpsFromStmt(InnermostBody(info), info.domain, &ops, &build_error)) {
    result.reason = build_error;
    return result;
  }
  // A nest that writes nothing cannot be vectorized: the store map is what the
  // planner turns into a vector register write. A structured op writes its
  // destination region, so a tile-op-only nest counts as productive.
  if (!WritesSomething(ops)) {
    result.reason = "nest has no BufferStore";
    return result;
  }
  region->ops = std::move(ops);
  for (const auto &capture : info.captures) {
    region->captures.push_back(VSimdCapture(
        capture.first, String(capture.second), StorageKindOf(capture.first),
        FindLayout(info.layouts, capture.first)));
  }
  for (const auto &var : info.outer_vars)
    region->outer_vars.push_back(
        VSimdVarBinding(var.first, String(var.second), Optional<PrimExpr>()));
  for (const Var &var : info.shape_syms)
    region->shape_syms.push_back(var);
  for (const auto &view : info.local_views)
    region->local_views.push_back(VSimdView(view.first, view.second));
  for (const PrimExpr &cond : info.assumptions)
    region->assumptions.push_back(cond);
  for (const auto &binding : info.bindings)
    region->bindings.push_back(VSimdVarBinding(
        binding.first, String("bind"), Optional<PrimExpr>(binding.second)));
  region->status = String(region_status::kLifted);

  std::string verify_error;
  if (!VerifyRegion(region, &verify_error)) {
    result.internal_error = true;
    result.reason = "internal region verification failed: " + verify_error;
    return result;
  }

  std::string unsupported;
  Stmt expanded = ExpandVSimdRegionToLoops(region, &unsupported);
  if (!unsupported.empty() || !expanded.defined()) {
    result.internal_error = true;
    result.reason = "internal expander failure: " + unsupported;
    return result;
  }
  VSimdRegionVerify verify = VerifyExpandedRegion(region, expanded);
  if (!verify.ok) {
    result.internal_error = true;
    result.reason = "internal expander verification failed: " + verify.reason;
    return result;
  }

  // Structural oracle: the reconstruction must reproduce the original Parallel
  // subtree statement by statement. A mismatch means the region lost, reordered
  // or altered something, so P3 could not be trusted to replace the loop.
  if (verify_body && info.root_for != nullptr) {
    VSimdRegionMatch match =
        CompareRegionWithOriginal(region, ffi::GetRef<Stmt>(info.root_for));
    if (!match.ok) {
      result.internal_error = true;
      result.reason =
          "region does not match the original Parallel body: " + match.reason;
      return result;
    }
  }

  result.lifted = true;
  result.region = std::move(region);
  return result;
}

Stmt ExpandVSimdRegionToLoops(const VSimdRegion &region,
                              std::string *unsupported) {
  std::vector<Stmt> body;
  for (const VSimdOp &op : region->ops) {
    Stmt expanded = ExpandOp(op, unsupported);
    if (!expanded.defined())
      return Stmt();
    body.push_back(expanded);
  }
  Stmt result = body.size() == 1 ? body[0] : SeqStmt(body);
  // Reference expansion is intentionally serial scalar loops; it is a
  // structural oracle, not the P3 physical lowering.
  for (auto it = region->axes.rbegin(); it != region->axes.rend(); ++it) {
    result =
        For((*it)->var, (*it)->min, (*it)->extent, ForKind::kSerial, result);
  }
  // T.view aliases are DeclBuffer definitions, not allocations. They must be
  // declared before the reconstructed loop or the expanded IR references an
  // undeclared view buffer.
  for (auto it = region->local_views.rbegin(); it != region->local_views.rend();
       ++it) {
    result = SeqStmt({DeclBuffer((*it)->view), result});
  }
  return result;
}

VSimdRegionVerify VerifyExpandedRegion(const VSimdRegion &region,
                                       const Stmt &expanded) {
  using BufferSet =
      std::unordered_set<Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
  BufferSet visible;
  for (const VSimdCapture &capture : region->captures)
    visible.insert(capture->buffer);
  for (const VSimdView &view : region->local_views)
    visible.insert(view->view);

  class VerifyVisitor : public StmtExprVisitor {
  public:
    BufferSet declared;
    std::vector<const BufferStoreNode *> stores;
    std::vector<const ForNode *> loops;
    std::vector<Buffer> loads;
    std::string error;

    void VisitStmt_(const DeclBufferNode *op) final {
      declared.insert(op->buffer);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const ForNode *op) final {
      loops.push_back(op);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      stores.push_back(op);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const BufferLoadNode *op) final {
      loads.push_back(op->buffer);
      StmtExprVisitor::VisitExpr_(op);
    }
  } verify;

  verify(expanded);

  VSimdRegionVerify result;
  for (const VSimdView &view : region->local_views) {
    if (!verify.declared.count(view->view)) {
      result.ok = false;
      result.reason =
          "local view is not declared: " + std::string(view->view->name);
      return result;
    }
  }

  auto visible_buffer = [&](const Buffer &buffer) {
    return visible.count(buffer) || verify.declared.count(buffer);
  };
  for (const Buffer &buffer : verify.loads) {
    if (!visible_buffer(buffer)) {
      result.ok = false;
      result.reason =
          "load from undeclared buffer: " + std::string(buffer->name);
      return result;
    }
  }

  std::vector<const VSimdAxisNode *> expected;
  for (const VSimdAxis &axis : region->axes)
    expected.push_back(axis.get());
  CollectExpectedAxes(region->ops, &expected);
  if (verify.loops.size() != expected.size()) {
    result.ok = false;
    result.reason = "expanded loop count does not match region axes";
    return result;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    const ForNode *loop = verify.loops[i];
    if (!loop->loop_var.same_as(expected[i]->var) ||
        !ffi::StructuralEqual()(loop->min, expected[i]->min) ||
        !ffi::StructuralEqual()(loop->extent, expected[i]->extent)) {
      result.ok = false;
      result.reason = "expanded domain does not match region axes";
      return result;
    }
  }

  const int writes = CountWrites(region->ops);
  if (static_cast<int>(verify.stores.size()) != writes) {
    result.ok = false;
    result.reason = "expanded store count does not match region stores";
    return result;
  }
  std::vector<const VSimdAccessNode *> expected_stores;
  CollectExpectedStores(region->ops, &expected_stores);
  for (size_t i = 0; i < expected_stores.size(); ++i) {
    const BufferStoreNode *store = verify.stores[i];
    const VSimdAccessNode *acc = expected_stores[i];
    if (!store->buffer.same_as(acc->buffer) ||
        !ffi::StructuralEqual()(store->indices, acc->indices) ||
        !ffi::StructuralEqual()(store->value, acc->value.value())) {
      result.ok = false;
      result.reason = "expanded store does not match region access";
      return result;
    }
  }
  return result;
}

bool RegionVerifyEnabled() {
  return tvm::transform::PassContext::Current()
      ->GetConfig<Bool>("tl.ascend_simd_region_verify", Bool(true))
      .value();
}

String
RenderVSimdRegionReport(const std::vector<VSimdCaptureInfo> &infos,
                        const std::vector<VSimdMemoryRegionStatus> &memory,
                        bool verify_body) {
  std::ostringstream out;
  for (const VSimdCaptureInfo &info : infos) {
    if (info.nest_index < 0) {
      out << "VF#" << info.vf_index << " status=OPAQUE";
      if (!info.message.empty())
        out << " reason=" << info.message;
      out << "\n";
      continue;
    }
    // Gate each nest on its own legality status: an illegal sibling nest must
    // not disable a legal one (the P1 pass already works per nest).
    std::string memory_status = "MISSING";
    std::string memory_reason;
    for (const VSimdMemoryRegionStatus &status : memory) {
      if (status.vf_index == info.vf_index &&
          status.nest_index == info.nest_index) {
        memory_status = status.status;
        memory_reason = status.reason;
        break;
      }
    }
    VSimdRegionLift lift =
        BuildRegion(info, memory_status, verify_body, memory_reason);
    out << "VF#" << info.vf_index << " nest " << info.nest_index << ": ";
    if (!lift.lifted) {
      if (lift.internal_error) {
        out << "status=INTERNAL_ERROR reason=" << lift.reason << "\n";
      } else {
        out << "status=OPAQUE reason=" << lift.reason << "\n";
      }
      continue;
    }
    Stmt expanded_stmt = ExpandVSimdRegionToLoops(lift.region);
    if (!expanded_stmt.defined()) {
      out << "status=INTERNAL_ERROR reason=reference expander cannot represent "
             "this region\n";
      continue;
    }
    out << "status=LIFTED\n";
    // Serial loops *around* the region stay in place and keep binding their
    // counters for the emitted vector code; surface them so a reader can tell
    // this nest apart from a top-level one.
    if (!info.outer_shells.empty()) {
      out << "  outer shells:";
      for (const LoopInfo &loop : info.outer_shells) {
        out << " " << loop.var->name_hint << " in [" << loop.min << ", "
            << (loop.min + loop.extent) << ")";
      }
      out << "\n";
    }
    out << DumpRegion(lift.region);
    std::ostringstream expanded;
    expanded << expanded_stmt;
    out << "  expanded:\n    " << expanded.str() << "\n";
    // Reflect the switch instead of always claiming a check that may not have
    // run: the report is the only place the difference is observable.
    out << "  verify: " << (verify_body ? "OK" : "skipped") << "\n";
  }
  if (out.str().empty())
    return String("no SIMD_VF blocks\n");
  return String(out.str());
}

String AnalyzeVSimdRegionReport(const PrimFunc &func) {
  return RenderVSimdRegionReport(AnalyzeVSimdCaptureInfo(func),
                                 AnalyzeVSimdMemoryRegions(func),
                                 RegionVerifyEnabled());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.vsimd.expand_region",
           [](VSimdRegion region) {
             // The expander reads per-kind fields unconditionally, so validate
             // the region at the entry point instead of dereferencing a
             // malformed one.
             std::string error;
             if (!VerifyRegion(region, &error))
               TVM_FFI_THROW(ValueError) << "invalid vsimd region: " << error;
             std::string unsupported;
             Stmt expanded = ExpandVSimdRegionToLoops(region, &unsupported);
             Array<ffi::Any> out;
             out.push_back(expanded.defined() ? ffi::Any(expanded)
                                              : ffi::Any());
             out.push_back(ffi::Any(String(unsupported)));
             return out;
           })
      .def("tl.vsimd.check_region_against", [](VSimdRegion region,
                                               Stmt original) {
        std::string error;
        if (!VerifyRegion(region, &error))
          TVM_FFI_THROW(ValueError) << "invalid vsimd region: " << error;
        VSimdRegionMatch match = CompareRegionWithOriginal(region, original);
        return String(match.ok ? "" : match.reason);
      });
}

} // namespace tl
} // namespace tvm
