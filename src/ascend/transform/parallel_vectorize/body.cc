/*!
 * \file body.cc
 * \brief Body normalization implementation.
 */
#include "body.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "access.h"
#include "primitives.h"

namespace tvm {
namespace tl {
namespace parallel_vectorize {

namespace {

/*! \brief True when the statement writes an address with no loop variable. */
bool WritesScalarAddress(const tirx::Stmt &stmt) {
  bool found = false;
  tirx::PostOrderVisit(stmt, [&](const ffi::ObjectRef &node) {
    const auto *store = node.as<tirx::BufferStoreNode>();
    if (store == nullptr || found)
      return;
    bool varies = false;
    for (const PrimExpr &index : store->indices) {
      if (!tirx::UndefinedVars(index).empty())
        varies = true;
    }
    found = !varies;
  });
  return found;
}

/*!
 * \brief Largest constant-bound serial shell P2 unrolls iteration by iteration.
 *
 * A shell the planner cannot absorb as an axis is replicated instead of being
 * modelled as a plan-level loop; the bound keeps the emitted program (and the
 * plan) small.
 */
constexpr int64_t kShellUnrollLimit = 8;

/*! \brief Literal zero only: statement normalization has no analyzer. */
bool IsZeroLiteral(const PrimExpr &expr) {
  if (const auto *int_imm = expr.as<IntImmNode>())
    return int_imm->value == 0;
  return false;
}

/*! \brief ``else`` that carries no work: absent, or ``Evaluate(0)``. */
bool IsTruthyElse(const ffi::Optional<tirx::Stmt> &else_case) {
  if (!else_case.has_value())
    return true;
  const tirx::Stmt &stmt = *else_case;
  const auto *evaluate = stmt.as<tirx::EvaluateNode>();
  return evaluate != nullptr && IsZeroLiteral(evaluate->value);
}

} // namespace

bool IsScalarShell(const tirx::ForNode *loop) {
  if (loop == nullptr || loop->kind != tirx::ForKind::kSerial)
    return false;
  // The shell is replayed with unit stride (``min + iter``); P1 guarantees
  // ``step == 1``, direct planner callers need the guard here.
  if (loop->step.has_value()) {
    const auto *step = loop->step.value().as<IntImmNode>();
    if (step == nullptr || step->value != 1)
      return false;
  }
  std::optional<int64_t> min = ConstInt(loop->min, nullptr);
  if (!min.has_value() || *min != 0)
    return false;
  if (WritesScalarAddress(loop->body))
    return false;
  std::optional<int64_t> extent = ConstInt(loop->extent, nullptr);
  if (extent.has_value() && *extent >= 0 && *extent <= kShellUnrollLimit)
    return false; // unrolled iteration by iteration, not a shell
  return true;
}

StoreSpec MakeStoreSpec(tirx::Buffer buffer, ffi::Array<PrimExpr> indices,
                        PrimExpr value, ffi::Optional<PrimExpr> mask) {
  auto node = ffi::make_object<StoreSpecNode>();
  node->buffer = std::move(buffer);
  node->indices = std::move(indices);
  node->value = std::move(value);
  node->mask = std::move(mask);
  return StoreSpec(std::move(node));
}

AliasMap CollectAliases(const tirx::Stmt &stmt) {
  AliasMap aliases;
  tirx::PostOrderVisit(stmt, [&](const ffi::ObjectRef &node) {
    const auto *bind = node.as<tirx::BindNode>();
    if (bind == nullptr)
      return;
    // Buffer declarations / type constructors are Calls, not value aliases.
    if (bind->value.as<tirx::CallNode>() != nullptr)
      return;
    aliases.values[bind->var.get()] = bind->value;
  });
  return aliases;
}

PrimExpr ApplyAliases(const PrimExpr &expr, const AliasMap &aliases) {
  if (aliases.empty())
    return expr;
  PrimExpr current = expr;
  for (int pass = 0; pass < 8; ++pass) {
    PrimExpr replaced = tirx::Substitute(
        current, [&](const tirx::Var &var) -> ffi::Optional<PrimExpr> {
          auto it = aliases.values.find(var.get());
          if (it == aliases.values.end())
            return ffi::Optional<PrimExpr>();
          return ffi::Optional<PrimExpr>(it->second);
        });
    if (replaced.same_as(current))
      break;
    current = replaced;
  }
  return current;
}

namespace {

/*! \brief Replace one loop counter by a constant (used to unroll a shell). */
class ShellIteration : public tirx::StmtExprMutator {
public:
  ShellIteration(tirx::Var var, PrimExpr value)
      : var_(std::move(var)), value_(std::move(value)) {}

  PrimExpr VisitExpr_(const tirx::VarNode *op) final {
    if (var_.same_as(ffi::GetRef<tirx::Var>(op)))
      return value_;
    return ffi::GetRef<PrimExpr>(op);
  }

private:
  tirx::Var var_;
  PrimExpr value_;
};

/*! \brief ``extra & mask``: an absent operand is the identity. */
ffi::Optional<PrimExpr> ConjoinMask(const ffi::Optional<PrimExpr> &mask,
                                    const PrimExpr &extra) {
  if (!mask.has_value())
    return ffi::Optional<PrimExpr>(extra);
  return ffi::Optional<PrimExpr>(logical_and(*mask, extra));
}

/*! \brief Append ``stores`` with ``mask`` conjoined into each store's own mask.
 */
void AppendPredicated(std::vector<StoreSpec> *stores,
                      const ffi::Optional<PrimExpr> &mask,
                      std::vector<StoreSpec> *out) {
  for (const StoreSpec &original : *stores) {
    StoreSpec spec = original;
    spec.CopyOnWrite()->mask =
        spec->mask.has_value() && mask.has_value()
            ? ffi::Optional<PrimExpr>(logical_and(*mask, *spec->mask))
            : (spec->mask.has_value() ? spec->mask : mask);
    out->push_back(spec);
  }
}

} // namespace

void NormalizeBodyInto(const tirx::Stmt &stmt, const AliasMap &aliases,
                       std::vector<StoreSpec> *out,
                       ffi::Optional<PrimExpr> mask,
                       std::vector<ShellLoop> *shells) {
  if (const auto *seq = stmt.as<tirx::SeqStmtNode>()) {
    // Sibling scalar shells cannot be modelled as one shell chain: the shell
    // segmentation must have cut them into one segment each, so reaching here
    // means the shape is not distributable (e.g. inside a branch). Reject
    // instead of silently nesting the siblings.
    int shell_like = 0;
    for (const tirx::Stmt &child : seq->seq)
      shell_like += IsScalarShell(child.as<tirx::ForNode>()) ? 1 : 0;
    if (shell_like >= 2) {
      throw Unsupported(
          "a T.Parallel body with multiple sibling scalar shells cannot be "
          "planned as one segment",
          {}, "body.sibling_shells");
    }
    for (const tirx::Stmt &child : seq->seq) {
      NormalizeBodyInto(child, aliases, out, mask, shells);
    }
    return;
  }

  if (const auto *store = stmt.as<tirx::BufferStoreNode>()) {
    PrimExpr value = ApplyAliases(store->value, aliases);
    out->push_back(MakeStoreSpec(store->buffer, store->indices, value, mask));
    return;
  }

  if (const auto *evaluate = stmt.as<tirx::EvaluateNode>()) {
    if (IsZeroLiteral(evaluate->value))
      return;
    if (const auto *call = evaluate->value.as<tirx::CallNode>()) {
      if (IsAtomicName(CallTail(ffi::GetRef<tirx::Call>(call)))) {
        throw Unsupported("atomic op (T.atomic_add) is not supported yet: no "
                          "vector+scalar atomic recipe");
      }
      std::string name = "Evaluate";
      if (const auto *op = call->op.as<OpNode>())
        name = op->name;
      throw Unsupported(
          "opaque Evaluate inside a planned nest is not supported: " + name);
    }
    throw Unsupported(
        "opaque Evaluate inside a planned nest is not supported: " +
        TypeKeyTail(evaluate->value));
  }

  if (const auto *branch = stmt.as<tirx::IfThenElseNode>()) {
    PrimExpr condition = ApplyAliases(branch->condition, aliases);
    std::vector<StoreSpec> then_stores;
    NormalizeBodyInto(branch->then_case, aliases, &then_stores, std::nullopt,
                      shells);
    std::vector<StoreSpec> else_stores;
    if (!IsTruthyElse(branch->else_case)) {
      NormalizeBodyInto(*branch->else_case, aliases, &else_stores, std::nullopt,
                        shells);
    }

    if (else_stores.empty()) {
      // Single-arm if -> predication: every write carries (cond & mask).
      ffi::Optional<PrimExpr> combined(
          mask.has_value() ? logical_and(*mask, condition) : condition);
      for (const StoreSpec &spec_const : then_stores) {
        StoreSpec spec = spec_const;
        ffi::Optional<PrimExpr> spec_mask = spec->mask;
        spec.CopyOnWrite()->mask =
            spec_mask.has_value()
                ? ffi::Optional<PrimExpr>(logical_and(*combined, *spec_mask))
                : combined;
        out->push_back(spec);
      }
      return;
    }

    if (then_stores.size() == 1 && else_stores.size() == 1) {
      const StoreSpec &then_spec = then_stores[0];
      const StoreSpec &else_spec = else_stores[0];
      // Merging needs one unpredicated store per arm: with a nested else-if one
      // arm already carries its own predicate, and there is no single select
      // condition. Such an arm falls through to per-arm predication below.
      if (SameBuffer(then_spec->buffer, else_spec->buffer) &&
          SameIndices(then_spec->indices, else_spec->indices) &&
          !then_spec->mask.has_value() && !else_spec->mask.has_value()) {
        // Statement-level if/else arms are *lazy*: use ``if_then_else`` so the
        // arm loads stay guarded (a plain Select would load eagerly, D16).
        PrimExpr merged =
            if_then_else(condition, then_spec->value, else_spec->value);
        out->push_back(
            MakeStoreSpec(then_spec->buffer, then_spec->indices, merged, mask));
        return;
      }
    }

    // Divergent arms: each destination is written by one arm only, so instead
    // of merging values each arm keeps its own store and carries the arm
    // condition as a predicate. A destination written by both arms is still
    // correct: the two predicates are complementary and the stores are
    // emitted in source order.
    AppendPredicated(&then_stores, ConjoinMask(mask, condition), out);
    AppendPredicated(&else_stores, ConjoinMask(mask, logical_not(condition)),
                     out);
    return;
  }

  if (const auto *loop = stmt.as<tirx::ForNode>()) {
    // A scalar shell the planner could not absorb as an axis (its writes do not
    // vary along the counter, so it is not a reduction either) is emitted
    // iteration by iteration: the loop is unrolled at plan time, which keeps
    // the shell semantics (last write wins, side effects in order) without a
    // new plan-level loop construct.
    std::optional<int64_t> min = ConstInt(loop->min, nullptr);
    std::optional<int64_t> extent = ConstInt(loop->extent, nullptr);
    if (loop->kind == tirx::ForKind::kSerial && min.has_value() &&
        extent.has_value() && *extent >= 0 && *extent <= kShellUnrollLimit &&
        !WritesScalarAddress(loop->body)) {
      for (int64_t iter = 0; iter < *extent; ++iter) {
        PrimExpr value = IntImm(loop->loop_var->dtype, *min + iter);
        tirx::Stmt replaced = ShellIteration(loop->loop_var, value)(loop->body);
        NormalizeBodyInto(replaced, aliases, out, mask, shells);
      }
      return;
    }
    // Otherwise keep the shell as a real scalar loop around the vectorized
    // body (the shape PR-2 uses): the counter stays a free lane-uniform value
    // in the store maps, and P3 wraps the chunk program in it. A write to a
    // scalar address is not lane-parallel work (it would be replicated once per
    // lane), so such a shell stays a rejection.
    if (shells != nullptr && IsScalarShell(loop)) {
      shells->push_back(ShellLoop{loop->loop_var, loop->min, loop->extent});
      NormalizeBodyInto(loop->body, aliases, out, mask, shells);
      return;
    }
    throw Unsupported(
        "nested " + ForKindName(loop->kind) +
            " loop inside T.Parallel is not affine-vectorizable "
            "(design note §6): only directly-nested, dependence-free "
            "serial loops are absorbed as axes, "
            "only a constant-bound serial shell up to " +
            std::to_string(kShellUnrollLimit) +
            " iterations is unrolled, and only a "
            "lane-invariant shell without scalar-address writes "
            "is kept as a scalar loop",
        {}, "body.nested_loop");
  }

  if (stmt.as<tirx::BindNode>() != nullptr) {
    return;
  }

  throw Unsupported("unsupported statement in T.Parallel body: " +
                        TypeKeyTail(stmt),
                    {}, "body.unsupported_statement");
}

std::vector<StoreSpec> NormalizeBody(const tirx::Stmt &stmt,
                                     const AliasMap &aliases,
                                     std::vector<ShellLoop> *shells) {
  std::vector<StoreSpec> out;
  NormalizeBodyInto(stmt, aliases, &out, std::nullopt, shells);
  return out;
}

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm
