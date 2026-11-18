/*!
 * \file plan.cc
 * \brief Planning layer implementation (nest split / axis ranking /
 * reductions).
 */
#include "plan.h"

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "primitives.h"

namespace tvm {
namespace tl {
namespace parallel_vectorize {

namespace {

using tirx::ForKind;
using tirx::Stmt;

/*! \brief Python repr of a list of strings: ``['a', 'b']``. */
std::string PyStrList(const std::vector<std::string> &items) {
  std::string out = "[";
  for (size_t i = 0; i < items.size(); ++i) {
    out += (i == 0 ? "'" : ", '") + items[i] + "'";
  }
  return out + "]";
}

/*! \brief Python repr of a list of ints: ``[0, 5]``. */
std::string PyIntList(const std::vector<int64_t> &items) {
  std::string out = "[";
  for (size_t i = 0; i < items.size(); ++i) {
    out += (i == 0 ? "" : ", ") + std::to_string(items[i]);
  }
  return out + "]";
}

template <typename T> ffi::Array<T> ToArray(const std::vector<T> &items) {
  ffi::Array<T> out;
  for (const T &item : items)
    out.push_back(item);
  return out;
}

std::vector<std::string> ToStrings(const ffi::Array<ffi::String> &items) {
  std::vector<std::string> out;
  out.reserve(items.size());
  for (const ffi::String &item : items)
    out.push_back(item.operator std::string());
  return out;
}

ffi::Array<ffi::String> ToStringArray(const std::vector<std::string> &items) {
  ffi::Array<ffi::String> out;
  for (const std::string &item : items)
    out.push_back(item);
  return out;
}

std::string JoinNames(const std::vector<Axis> &axes) {
  std::string out;
  for (size_t i = 0; i < axes.size(); ++i) {
    out += (i == 0 ? "" : ", ") + axes[i]->name.operator std::string();
  }
  return out;
}

/*! \brief ``_seq`` over statement vectors. */
Stmt SeqStmts(const std::vector<Stmt> &stmts) {
  ffi::Array<Stmt> array;
  for (const Stmt &stmt : stmts)
    array.push_back(stmt);
  return SeqOf(array);
}

/*! \brief Substitute a variable mapping into an expression. */
PrimExpr
SubstituteVars(const PrimExpr &expr,
               const std::map<const tirx::VarNode *, PrimExpr> &mapping) {
  return tirx::Substitute(expr,
                          [&](const tirx::Var &var) -> ffi::Optional<PrimExpr> {
                            auto it = mapping.find(var.get());
                            if (it == mapping.end())
                              return ffi::Optional<PrimExpr>();
                            return ffi::Optional<PrimExpr>(it->second);
                          });
}

int AxisWeight(AccessClass kind) {
  switch (kind) {
  case AccessClass::kContiguous:
    return 8;
  case AccessClass::kPiecewise:
    return 4;
  case AccessClass::kStrided:
    return 3;
  case AccessClass::kPermute:
    return 3;
  case AccessClass::kIndirect:
    return 2;
  case AccessClass::kUniform:
    return 1;
  case AccessClass::kTrueRandom:
    return 0;
  }
  return 0;
}

/*! \brief ``_SegmentSpecs``: normalized analysis input of one segment. */
struct SegmentSpecs {
  std::vector<Axis> axes;
  std::optional<Axis> serial_axis;
  std::vector<StoreSpec> pre_specs;
  std::vector<AccessInfo> pre_infos;
  std::vector<StoreSpec> main_specs;
  std::vector<AccessInfo> main_infos;
  std::vector<StoreSpec> post_specs;
  std::vector<AccessInfo> post_infos;
  /*! \brief Scalar shells of ``main`` (see :struct:`ShellLoop`), outermost
   * first. */
  std::vector<ShellLoop> shells;
  int64_t lanes{0};
  /*! \brief Load classification memo, keyed by the statement's load nodes.
   *
   * Filled once per chunk; ``BuildValue`` reuses it instead of running
   * ``ClassifyAccess`` again for every candidate axis. */
  LoadInfoCache load_info_cache;
  /*! \brief Guard inherited from an enclosing ``if`` (not yet applied to the
   * specs). ``BuildPlanForAxis`` builds its value node once per chunk so every
   * store shares one predicate binding and the condition is evaluated before
   * the stores, matching the source ``if``. */
  ffi::Optional<PrimExpr> guard;

  std::vector<AccessInfo> AllInfos() const {
    std::vector<AccessInfo> infos;
    infos.insert(infos.end(), pre_infos.begin(), pre_infos.end());
    infos.insert(infos.end(), main_infos.begin(), main_infos.end());
    infos.insert(infos.end(), post_infos.begin(), post_infos.end());
    return infos;
  }
  std::vector<StoreSpec> AllSpecs() const {
    std::vector<StoreSpec> specs;
    specs.insert(specs.end(), pre_specs.begin(), pre_specs.end());
    specs.insert(specs.end(), main_specs.begin(), main_specs.end());
    specs.insert(specs.end(), post_specs.begin(), post_specs.end());
    return specs;
  }
  VectorPlan plan_for(const Axis &vector_axis, arith::Analyzer *analyzer,
                      const PlanOptions &options) const;
};

/* ------------------------------------------------------------------------- */
/* forward declarations                                                       */
/* ------------------------------------------------------------------------- */

VectorPlan BuildPlanForAxis(
    const std::vector<Axis> &axes, std::optional<Axis> serial_axis,
    const std::vector<StoreSpec> &pre_specs,
    const std::vector<AccessInfo> &pre_infos,
    const std::vector<StoreSpec> &main_specs,
    const std::vector<AccessInfo> &main_infos,
    const std::vector<StoreSpec> &post_specs,
    const std::vector<AccessInfo> &post_infos, const Axis &vector_axis,
    int64_t lanes, arith::Analyzer *analyzer, const PlanOptions &options,
    const LoadInfoCache *load_infos, const ffi::Optional<PrimExpr> &guard);

VectorPlan SegmentSpecs::plan_for(const Axis &vector_axis,
                                  arith::Analyzer *analyzer,
                                  const PlanOptions &options) const {
  return BuildPlanForAxis(axes, serial_axis, pre_specs, pre_infos, main_specs,
                          main_infos, post_specs, post_infos, vector_axis,
                          lanes, analyzer, options, &load_info_cache, guard);
}

/* ------------------------------------------------------------------------- */
/* iteration domain                                                          */
/* ------------------------------------------------------------------------- */

void RequireZeroMin(const tirx::For &loop, arith::Analyzer *analyzer) {
  std::optional<int64_t> min = ConstInt(loop->min, analyzer);
  if (!min.has_value() || *min != 0) {
    throw Unsupported(ForKindName(loop->kind) + " loop '" +
                      loop->loop_var->name_hint.operator std::string() +
                      "' must start at 0 to be vectorized (min=" +
                      PrintExpr(loop->min) + ")");
  }
}

std::pair<std::vector<Axis>, Stmt>
AbsorbParallelChain(const Stmt &stmt, arith::Analyzer *analyzer) {
  std::vector<Axis> axes;
  Stmt current = stmt;
  while (const auto *loop = current.as<tirx::ForNode>()) {
    if (loop->kind != ForKind::kParallel)
      break;
    RequireZeroMin(ffi::GetRef<tirx::For>(loop), analyzer);
    axes.push_back(MakeAxis(loop->loop_var, loop->extent,
                            static_cast<int64_t>(axes.size()),
                            ForKind::kParallel));
    current = loop->body;
  }
  return {axes, current};
}

bool ValueVariesAlong(const PrimExpr &expr, const Axis &axis,
                      arith::Analyzer *analyzer);

/*!
 * \brief Whether a directly-nested ``T.serial`` loop can be absorbed (design
 * note §6).
 *
 * Absorbing makes the loop iteration a candidate vector axis, so loop-carried
 * dependences must be excluded. The criterion is conservative on purpose:
 * reject rather than miscompile.
 */
std::pair<bool, std::string> SerialIsClean(const tirx::For &loop,
                                           arith::Analyzer *analyzer) {
  tirx::Var var = loop->loop_var;
  std::vector<const tirx::BufferStoreNode *> stores;
  std::vector<const tirx::BufferLoadNode *> loads;
  tirx::PostOrderVisit(loop->body, [&](const ffi::ObjectRef &node) {
    if (const auto *store = node.as<tirx::BufferStoreNode>()) {
      stores.push_back(store);
    } else if (const auto *load = node.as<tirx::BufferLoadNode>()) {
      loads.push_back(load);
    }
  });

  Axis axis = MakeAxis(var, loop->extent, 0, ForKind::kSerial);
  std::vector<std::pair<const tirx::BufferStoreNode *, PrimExpr>> offsets;
  for (const tirx::BufferStoreNode *store : stores) {
    offsets.emplace_back(
        store, LinearElementOffset(store->buffer, store->indices, analyzer));
  }
  for (const auto &entry : offsets) {
    std::optional<int64_t> delta =
        ConstInt(FiniteDifference(entry.second, var, analyzer), analyzer);
    if (delta.has_value() && *delta == 0) {
      if (!ExprHasSameAddressLoad(entry.first->value, entry.first->buffer,
                                  entry.first->indices)) {
        return {false, "write to " +
                           entry.first->buffer->name.operator std::string() +
                           " is invariant along '" +
                           var->name_hint.operator std::string() +
                           "' but not a reduction"};
      }
      if (!ValueVariesAlong(entry.first->value, axis, analyzer)) {
        return {false, "reduction into " +
                           entry.first->buffer->name.operator std::string() +
                           " does not vary along '" +
                           var->name_hint.operator std::string() + "'"};
      }
    }
  }

  for (size_t i = 0; i < offsets.size(); ++i) {
    for (size_t j = i + 1; j < offsets.size(); ++j) {
      if (SameBuffer(offsets[i].first->buffer, offsets[j].first->buffer)) {
        tirx::ExprDeepEqual equal;
        if (!equal(offsets[i].second, offsets[j].second)) {
          return {false,
                  "multiple writes to " +
                      offsets[i].first->buffer->name.operator std::string() +
                      " at different addresses along '" +
                      var->name_hint.operator std::string() +
                      "' may alias across iterations (last-write-wins is not "
                      "preserved by vectorization, R6)"};
        }
      }
    }
  }

  for (const tirx::BufferLoadNode *load : loads) {
    bool touches_written = false;
    bool same_address = false;
    for (const auto &entry : offsets) {
      if (!SameBuffer(load->buffer, entry.first->buffer))
        continue;
      touches_written = true;
      if (SameIndices(load->indices, entry.first->indices))
        same_address = true;
    }
    if (touches_written && !same_address) {
      return {false, "cross-iteration read of " +
                         load->buffer->name.operator std::string() +
                         " along '" + var->name_hint.operator std::string() +
                         "'"};
    }
  }
  return {true, ""};
}

/*!
 * \brief One chunk source: axes + (pre, main, post) statements.
 *
 * The unit handed to :func:`PlanSegment`; the tree planner builds it from the
 * context scope plus the loop chain it can absorb.
 */
struct SegmentSource {
  std::vector<Axis> axes;
  std::optional<Axis> serial_axis;
  std::vector<Stmt> pre;
  Stmt main;
  std::vector<Stmt> post;
  /*! \brief Aliases visible to this segment (``CollectAliases`` of its pieces).
   */
  AliasMap aliases;
  /*! \brief Whether the whole segment may collapse into one full-lane axis. */
  bool allow_flatten{true};
};

/*!
 * \brief Whether a statement keeps two sibling scalar shells at one level.
 *
 * Such a body cannot be one chunk (a plan carries a single shell chain). The
 * planner keeps its structure (one ``Loop``/chunk per chain) instead of nesting
 * the shells; the recursion into ``if`` arms is deliberate, because a sibling
 * pair inside a branch must become a structural ``If`` or a hard error.
 */
bool HasSiblingShell(const Stmt &stmt) {
  if (const auto *seq = stmt.as<tirx::SeqStmtNode>()) {
    int shells = 0;
    for (const Stmt &child : seq->seq)
      shells += IsScalarShell(child.as<tirx::ForNode>()) ? 1 : 0;
    if (shells >= 2)
      return true;
    for (const Stmt &child : seq->seq) {
      if (HasSiblingShell(child))
        return true;
    }
    return false;
  }
  if (const auto *loop = stmt.as<tirx::ForNode>())
    return HasSiblingShell(loop->body);
  if (const auto *branch = stmt.as<tirx::IfThenElseNode>()) {
    if (HasSiblingShell(branch->then_case))
      return true;
    return branch->else_case.defined() && HasSiblingShell(*branch->else_case);
  }
  return false;
}

/* ------------------------------------------------------------------------- */
/* scoring / diagnostics                                                     */
/* ------------------------------------------------------------------------- */

int64_t ActiveLanes(const Axis &axis, arith::Analyzer *analyzer,
                    int64_t lanes) {
  std::optional<int64_t> extent = ConstInt(axis->extent, analyzer);
  if (!extent.has_value())
    return lanes;
  return std::max<int64_t>(std::min<int64_t>(*extent, lanes), 1);
}

int64_t AxisWeightSum(const std::vector<AccessInfo> &infos, const Axis &axis) {
  int64_t total = 0;
  for (const AccessInfo &info : infos)
    total += AxisWeight(AxisClassOf(info, axis).kind);
  return total;
}

int64_t AxisScore(const std::vector<AccessInfo> &store_infos,
                  const std::vector<AccessInfo> &load_infos, const Axis &axis,
                  arith::Analyzer *analyzer, int64_t lanes) {
  int64_t weight = 100 * AxisWeightSum(store_infos, axis) +
                   10 * AxisWeightSum(load_infos, axis);
  return weight * ActiveLanes(axis, analyzer, lanes) / lanes + axis->index;
}

std::vector<Axis> RankVectorAxes(const std::vector<Axis> &axes,
                                 const std::vector<AccessInfo> &store_infos,
                                 const std::vector<AccessInfo> &load_infos,
                                 arith::Analyzer *analyzer, int64_t lanes) {
  std::vector<std::pair<int64_t, Axis>> ranked;
  for (const Axis &axis : axes) {
    bool rejected = false;
    for (const AccessInfo &info : store_infos) {
      if (AxisClassOf(info, axis).kind == AccessClass::kTrueRandom) {
        rejected = true;
        break;
      }
    }
    if (rejected)
      continue;
    ranked.emplace_back(
        AxisScore(store_infos, load_infos, axis, analyzer, lanes), axis);
  }
  // Python's ``list.sort(reverse=True)`` is stable: equal scores keep nest
  // order.
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const std::pair<int64_t, Axis> &lhs,
                      const std::pair<int64_t, Axis> &rhs) {
                     return lhs.first > rhs.first;
                   });
  std::vector<Axis> out;
  out.reserve(ranked.size());
  for (const auto &entry : ranked)
    out.push_back(entry.second);
  return out;
}

std::vector<std::string>
AxisDiagnostics(const std::vector<Axis> &axes,
                const std::vector<AccessInfo> &store_infos,
                const std::vector<AccessInfo> &load_infos,
                arith::Analyzer *analyzer, int64_t lanes) {
  std::vector<std::string> lines;
  for (const Axis &axis : axes) {
    std::string label = "axis '" + axis->name.operator std::string() + "'[" +
                        std::to_string(axis->index) + "]" +
                        (axis->is_parallel ? "" : "(serial)");
    std::vector<std::string> store_classes;
    bool rejected = false;
    for (const AccessInfo &info : store_infos) {
      AccessClass kind = AxisClassOf(info, axis).kind;
      store_classes.push_back(AccessClassName(kind));
      if (kind == AccessClass::kTrueRandom)
        rejected = true;
    }
    if (rejected) {
      std::set<std::string> reasons;
      for (const AccessInfo &info : store_infos) {
        if (!info->reason.empty())
          reasons.insert(info->reason.operator std::string());
      }
      std::string detail;
      if (!reasons.empty()) {
        detail = " (";
        bool first = true;
        for (const std::string &reason : reasons) {
          detail += (first ? "" : "; ") + reason;
          first = false;
        }
        detail += ")";
      }
      lines.push_back(label + ": store=" + PyStrList(store_classes) +
                      " -> rejected (true-random store)" + detail);
      continue;
    }
    std::vector<std::string> load_classes;
    for (const AccessInfo &info : load_infos) {
      load_classes.push_back(AccessClassName(AxisClassOf(info, axis).kind));
    }
    int64_t active = ActiveLanes(axis, analyzer, lanes);
    lines.push_back(label + ": store=" + PyStrList(store_classes) + " load=" +
                    PyStrList(load_classes) + " -> candidate (score=" +
                    std::to_string(AxisScore(store_infos, load_infos, axis,
                                             analyzer, lanes)) +
                    ", active=" + std::to_string(active) + "/" +
                    std::to_string(lanes) + ")");
  }
  return lines;
}

/* ------------------------------------------------------------------------- */
/* body normalization helpers                                                */
/* ------------------------------------------------------------------------- */

std::vector<PrimExpr> CollectLoadExprs(const std::vector<StoreSpec> &specs) {
  std::vector<PrimExpr> found;
  for (const StoreSpec &spec : specs) {
    tirx::PostOrderVisit(spec->value, [&](const ffi::ObjectRef &node) {
      if (node.as<tirx::BufferLoadNode>() != nullptr)
        found.push_back(Downcast<PrimExpr>(node));
    });
    if (spec->mask.has_value()) {
      tirx::PostOrderVisit(*spec->mask, [&](const ffi::ObjectRef &node) {
        if (node.as<tirx::BufferLoadNode>() != nullptr)
          found.push_back(Downcast<PrimExpr>(node));
      });
    }
  }
  return found;
}

std::vector<AccessInfo> LoadInfos(const std::vector<Axis> &axes,
                                  const std::vector<StoreSpec> &specs,
                                  arith::Analyzer *analyzer,
                                  LoadInfoCache *cache) {
  std::vector<AccessInfo> infos;
  for (const PrimExpr &expr : CollectLoadExprs(specs)) {
    const auto *load = expr.as<tirx::BufferLoadNode>();
    AccessInfo info = ClassifyAccess(load->buffer, load->indices, axes,
                                     analyzer, /*is_store=*/false);
    (*cache)[load] = info;
    infos.push_back(info);
  }
  return infos;
}

bool ValueVariesAlong(const PrimExpr &expr, const Axis &axis,
                      arith::Analyzer *analyzer) {
  bool varies = false;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
    const auto *load = node.as<tirx::BufferLoadNode>();
    if (load == nullptr)
      return;
    AccessInfo info =
        ClassifyAccess(load->buffer, load->indices, {axis}, analyzer,
                       /*is_store=*/false);
    if (AxisClassOf(info, axis).kind != AccessClass::kUniform)
      varies = true;
  });
  if (varies)
    return true;
  for (const tirx::Var &var : FreeVars(expr)) {
    if (axis->var.same_as(var))
      return true;
  }
  return false;
}

bool IsReduceAxis(const Axis &axis, const std::vector<StoreSpec> &specs,
                  const std::vector<AccessInfo> &store_infos,
                  arith::Analyzer *analyzer) {
  for (const AccessInfo &info : store_infos) {
    if (AxisClassOf(info, axis).kind != AccessClass::kUniform)
      return false;
  }
  for (const StoreSpec &spec : specs) {
    if (ValueVariesAlong(spec->value, axis, analyzer))
      return true;
  }
  return false;
}

std::string ReduceOpOf(const std::vector<StoreSpec> &specs) {
  for (const StoreSpec &spec : specs) {
    std::string op;
    if (ReduceOpFor(TypeKeyTail(spec->value), &op))
      return op;
  }
  std::set<std::string> names;
  for (const StoreSpec &spec : specs)
    names.insert(TypeKeyTail(spec->value));
  std::string joined;
  if (names.empty()) {
    joined = "<empty>";
  } else {
    bool first = true;
    for (const std::string &name : names) {
      joined += (first ? "" : ", ") + name;
      first = false;
    }
  }
  throw Unsupported(
      "unsupported reduction operator(s) " + joined +
      " (design note §2 Case II): only Add/Max/Min reductions are supported");
}

bool ExprReadsBuffers(const PrimExpr &expr,
                      const std::vector<tirx::Buffer> &buffers) {
  bool found = false;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
    const auto *load = node.as<tirx::BufferLoadNode>();
    if (load == nullptr)
      return;
    for (const tirx::Buffer &buffer : buffers) {
      if (SameBuffer(load->buffer, buffer))
        found = true;
    }
  });
  return found;
}

/* ------------------------------------------------------------------------- */
/* flatten channel                                                           */
/* ------------------------------------------------------------------------- */

struct FlattenResult {
  Axis flat_axis;
  std::map<const tirx::VarNode *, PrimExpr> mapping;
  int64_t total{0};
};

std::optional<FlattenResult> FlattenAxes(const std::vector<Axis> &axes,
                                         arith::Analyzer *analyzer) {
  if (axes.size() < 2)
    return std::nullopt;
  // An extent-1 axis carries no lane information: mapping it to 0 keeps the
  // synthesized axis free of ``range(1)`` padding, so e.g. a sliced tile-op
  // dimension (``b[i, :]`` keeps a leading 1) does not block flattening.
  std::vector<Axis> chain;
  std::vector<int64_t> extents;
  std::map<const tirx::VarNode *, PrimExpr> trivial;
  for (const Axis &axis : axes) {
    std::optional<int64_t> extent = ConstInt(axis->extent, analyzer);
    if (!extent.has_value() || *extent < 1)
      return std::nullopt;
    if (*extent == 1) {
      trivial[axis->var.get()] = IntImm32(0);
      continue;
    }
    if ((*extent & (*extent - 1)) != 0)
      return std::nullopt;
    chain.push_back(axis);
    extents.push_back(*extent);
  }
  if (chain.size() < 2)
    return std::nullopt;
  int64_t total = 1;
  for (int64_t extent : extents)
    total *= extent;

  std::string name = "k_";
  for (size_t i = 0; i < chain.size(); ++i) {
    name += (i == 0 ? "" : "_") + chain[i]->name.operator std::string();
  }
  tirx::Var var(name, DataType::Int(32));

  FlattenResult result;
  result.flat_axis = MakeAxis(var, IntImm32(total), 0, ForKind::kParallel);
  result.total = total;
  result.mapping = std::move(trivial);
  int64_t suffix = 1;
  int64_t count = static_cast<int64_t>(chain.size());
  for (int64_t position = 0; position < count; ++position) {
    // Right-to-left decomposition; the outermost axis is NOT taken modulo,
    // since
    // ``(k // suffix) % E`` would collapse to ``k % (prod E)`` and lose
    // Delta==1.
    const Axis &axis = chain[static_cast<size_t>(count - 1 - position)];
    int64_t extent = extents[static_cast<size_t>(count - 1 - position)];
    PrimExpr mapping;
    if (position == 0) {
      mapping = floormod(var, IntImm32(extent));
    } else if (position == count - 1) {
      mapping = floordiv(var, IntImm32(suffix));
    } else {
      mapping = floormod(floordiv(var, IntImm32(suffix)), IntImm32(extent));
    }
    result.mapping[axis->var.get()] = mapping;
    suffix *= extent;
  }
  return result;
}

SegmentSpecs
PrepareSegment(const std::vector<Axis> &axes, std::optional<Axis> serial_axis,
               const std::vector<Stmt> &pre_stmts, const Stmt &main_stmt,
               const std::vector<Stmt> &post_stmts, const AliasMap &aliases,
               arith::Analyzer *analyzer, std::optional<int64_t> lanes,
               const ffi::Optional<PrimExpr> &predicate = std::nullopt);

int64_t SelectSegmentLanes(const SegmentSpecs &specs,
                           std::optional<int64_t> override_lanes);

std::pair<std::optional<SegmentSpecs>, std::string>
TryFlatten(const std::vector<Axis> &axes, const std::vector<Stmt> &pre_stmts,
           const std::vector<Stmt> &post_stmts, int64_t lanes,
           arith::Analyzer *analyzer, const SegmentSpecs &base_specs) {
  if (!pre_stmts.empty() || !post_stmts.empty())
    return {std::nullopt, ""};
  if (axes.size() < 2)
    return {std::nullopt, ""};
  // The innermost *real* axis decides whether flattening is needed; a trailing
  // extent-1 axis carries no lanes.
  std::optional<int64_t> innermost;
  int64_t real_axes = 0;
  for (auto it = axes.rbegin(); it != axes.rend(); ++it) {
    std::optional<int64_t> extent = ConstInt((*it)->extent, analyzer);
    if (!extent.has_value())
      return {std::nullopt, ""};
    if (*extent == 1)
      continue;
    if (!innermost.has_value())
      innermost = extent;
    ++real_axes;
  }
  if (real_axes < 2)
    return {std::nullopt, ""};
  if (!innermost.has_value() || *innermost >= lanes)
    return {std::nullopt, ""};
  std::optional<FlattenResult> flat = FlattenAxes(axes, analyzer);
  if (!flat.has_value()) {
    return {std::nullopt, "flatten rejected: extent product of [" +
                              JoinNames(axes) +
                              "] is not a power-of-two chain"};
  }
  Axis flat_axis = flat->flat_axis;
  int64_t total = flat->total;
  // Reuse the normalization already done for the original axes: flattening is a
  // pure variable substitution on the specs, so the body is not normalized
  // twice.
  SegmentSpecs specs;
  specs.axes = {flat_axis};
  specs.guard = base_specs.guard;
  for (const StoreSpec &original : base_specs.main_specs) {
    ffi::Array<PrimExpr> indices;
    for (const PrimExpr &index : original->indices)
      indices.push_back(SubstituteVars(index, flat->mapping));
    PrimExpr value = SubstituteVars(original->value, flat->mapping);
    ffi::Optional<PrimExpr> mask;
    if (original->mask.has_value())
      mask = ffi::Optional<PrimExpr>(
          SubstituteVars(*original->mask, flat->mapping));
    specs.main_specs.push_back(
        MakeStoreSpec(original->buffer, indices, value, mask));
  }
  try {
    specs.lanes = SelectSegmentLanes(specs, lanes);
    for (const StoreSpec &spec : specs.main_specs) {
      specs.main_infos.push_back(ClassifyAccess(spec->buffer, spec->indices,
                                                specs.axes, analyzer,
                                                /*is_store=*/true));
    }
  } catch (const Unsupported &error) {
    return {std::nullopt, "flatten rejected: " + error.message()};
  }
  for (const AccessInfo &info : specs.main_infos) {
    if (AxisClassOf(info, flat_axis).kind != AccessClass::kContiguous) {
      return {std::nullopt, "flatten rejected: not every store is contiguous "
                            "along the synthesized axis '" +
                                flat_axis->name.operator std::string() +
                                "' (extent " + std::to_string(total) + ")"};
    }
  }
  // Read-side gate: flattening only removes contiguity, never a gather.
  std::vector<AccessInfo> load_infos = LoadInfos(
      {flat_axis}, specs.AllSpecs(), analyzer, &specs.load_info_cache);
  for (const AccessInfo &info : load_infos) {
    AccessClass kind = AxisClassOf(info, flat_axis).kind;
    if (kind != AccessClass::kContiguous && kind != AccessClass::kUniform) {
      return {std::nullopt,
              "flatten rejected: a load along the synthesized axis '" +
                  flat_axis->name.operator std::string() +
                  "' still needs an index vector (gather) — flattening would "
                  "not remove it"};
    }
  }
  std::vector<tirx::Buffer> targets;
  for (const StoreSpec &spec : specs.main_specs)
    targets.push_back(spec->buffer);
  for (size_t i = 0; i < targets.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (SameBuffer(targets[i], targets[j])) {
        return {std::nullopt,
                "flatten rejected: two statements write the same buffer '" +
                    targets[i]->name.operator std::string() +
                    "' along the synthesized axis '" +
                    flat_axis->name.operator std::string() +
                    "' (their addresses may overlap across k) — extent " +
                    std::to_string(total)};
      }
    }
  }
  for (const StoreSpec &spec : specs.main_specs) {
    const std::vector<PrimExpr> exprs = {
        spec->value, spec->mask.has_value() ? *spec->mask : PrimExpr()};
    for (const PrimExpr &expr : exprs) {
      if (expr.defined() && ExprReadsBuffers(expr, targets)) {
        return {std::nullopt,
                "flatten rejected: a written buffer is also read inside the "
                "nest (no write-conflict-free flattening) — extent " +
                    std::to_string(total)};
      }
    }
  }
  return {specs, ""};
}

/* ------------------------------------------------------------------------- */
/* segment preparation / reduction helpers                                    */
/* ------------------------------------------------------------------------- */

void RequireSupportedElement(const tirx::Buffer &buffer) {
  std::string elem = ElemDtype(buffer->dtype);
  try {
    MaskBits(elem);
  } catch (const Unsupported &error) {
    // Python converts the ValueError into Unsupported at build time (D11).
    throw Unsupported("store element dtype '" + elem + "' of buffer '" +
                      buffer->name.operator std::string() +
                      "' is not supported: " + error.message());
  }
}

std::string PyBufferDtypeName(const tirx::Buffer &buffer) {
  return ffi::DLDataTypeToString(buffer->dtype).operator std::string();
}

/*! \brief Whether ``buffer`` holds an 8-bit float element type. */
bool IsFp8Buffer(const tirx::Buffer &buffer) {
  std::string elem = ElemDtype(PyBufferDtypeName(buffer));
  return elem == "float8_e4m3" || elem == "float8_e4m3fn" ||
         elem == "float8_e5m2";
}

/*! \brief Natural vector-register width of a store's element type (2048/bits).
 */
int64_t StoreRegisterLanes(const StoreSpec &spec) {
  return VRegLanes(ElemDtype(PyBufferDtypeName(spec->buffer)));
}

/*! \brief Element bit width shared by every load inside ``expr`` (0 =
 * unknown/mixed). */
int CommonLoadBits(const PrimExpr &expr) {
  int bits = 0;
  bool mixed = false;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
    const auto *load = node.as<tirx::BufferLoadNode>();
    if (load == nullptr)
      return;
    int width = load->buffer->dtype.bits();
    if (bits == 0) {
      bits = width;
    } else if (bits != width) {
      mixed = true;
    }
  });
  return mixed ? 0 : bits;
}

/*!
 * \brief Register lanes of a genuine narrowing conversion, if any.
 *
 * A cast only narrows the *register* when the converted expression is itself a
 * full register of its own element type: every load inside it must share that
 * element width (``cast(bf16, f32_load)``). A cast that merely closes a
 * widening round trip (``f16 load -> f32 arithmetic -> f16 store``) keeps the
 * store's own register and must not shrink the chunk.
 */
std::optional<int64_t> NarrowingSourceLanes(const PrimExpr &value,
                                            const tirx::Buffer &buffer) {
  const auto *cast = value.as<tirx::CastNode>();
  if (cast == nullptr)
    return std::nullopt;
  if (cast->value.dtype().bits() <= buffer->dtype.bits())
    return std::nullopt;
  if (CommonLoadBits(cast->value) != cast->value.dtype().bits())
    return std::nullopt;
  // fp8 destinations are only reachable through a float32 intermediate: the
  // verified cast kernel converts 16-bit floats with ``UNPK_B16`` + f16->f32
  // (``PART_EVEN``) + f32->fp8 (``PART_P0``) + ``PK4_B32``, so one chunk holds
  // ``VRegLanes(float32)`` elements regardless of the 16-bit source width.
  if (IsFp8Buffer(buffer) && cast->value.dtype().bits() == 16) {
    return VRegLanes("float32");
  }
  return VRegLanes(ElemDtype(ffi::DLDataTypeToString(cast->value.dtype())));
}

/*!
 * \brief Lanes of the vector a store hands to its write port.
 *
 * A narrowing conversion fills only ``VRegLanes(src)`` lanes of the wider
 * destination register; every other recipe produces a full destination
 * register.
 */
int64_t ProducedLanes(const PrimExpr &value, const tirx::Buffer &buffer) {
  std::optional<int64_t> narrowed = NarrowingSourceLanes(value, buffer);
  if (narrowed.has_value())
    return *narrowed;
  return VRegLanes(ElemDtype(PyBufferDtypeName(buffer)));
}

/*!
 * \brief Pick the chunk width of one segment (mixed element widths, Stage 1).
 *
 * The chunk covers ``lanes`` logical elements; a store whose element type has
 * ``2048 / bits`` register lanes writes exactly ``lanes`` of them, so every
 * store width must be ``lanes``, ``2 * lanes`` or ``4 * lanes``. Stage 1 keeps
 * the model deliberately narrow:
 *
 *  - ``lanes`` is the smallest width the stores actually produce, so one chunk
 *    is never wider than the data a store hands to its write port;
 *  - a store wider than the chunk must be a narrowing conversion whose source
 *    register is exactly ``lanes`` wide (it is emitted in the packed form);
 *  - mixed widths with sibling (``pre``/``post``) statements or a statement
 *    predicate are rejected here; the extent and access-shape checks live in
 *    ``BuildPlanForAxis`` once the vector axis is known.
 */
int64_t SelectSegmentLanes(const SegmentSpecs &specs,
                           std::optional<int64_t> override_lanes) {
  int64_t min_effective = 0;
  for (const StoreSpec &spec : specs.main_specs) {
    int64_t effective = std::min(StoreRegisterLanes(spec),
                                 ProducedLanes(spec->value, spec->buffer));
    min_effective =
        min_effective == 0 ? effective : std::min(min_effective, effective);
  }
  if (min_effective == 0)
    return override_lanes.value_or(0);
  int64_t lanes = override_lanes.value_or(min_effective);
  if (lanes <= 0)
    return min_effective;

  bool mixed = false;
  for (const StoreSpec &spec : specs.main_specs) {
    int64_t register_lanes = StoreRegisterLanes(spec);
    std::string name = spec->buffer->name.operator std::string();
    if (register_lanes % lanes != 0 || register_lanes / lanes > 4) {
      throw Unsupported(
          "store '" + name + "' has " + std::to_string(register_lanes) +
          " register lanes, which the segment chunk width " +
          std::to_string(lanes) + " cannot cover (split the nest)");
    }
    if (register_lanes / lanes > 1)
      mixed = true;
  }
  if (!mixed)
    return lanes;

  if (!specs.pre_specs.empty() || !specs.post_specs.empty()) {
    throw Unsupported(
        "mixed element widths in one segment are only supported without "
        "sibling statements "
        "around the loop; split the nest or stage an explicit cast");
  }
  for (const StoreSpec &spec : specs.main_specs) {
    int64_t register_lanes = StoreRegisterLanes(spec);
    int64_t ratio = register_lanes / lanes;
    std::string name = spec->buffer->name.operator std::string();
    if (spec->mask.has_value() || specs.guard.has_value()) {
      throw Unsupported("mixed element widths in one segment are not supported "
                        "with a statement predicate on "
                        "store '" +
                            name + "'",
                        {}, "body.statement_predicate");
    }
    const auto *cast = spec->value.as<tirx::CastNode>();
    if (ratio == 1) {
      if (cast != nullptr &&
          cast->value.dtype().bits() < spec->buffer->dtype.bits()) {
        throw Unsupported("store '" + name +
                          "' widens its value inside a mixed-width segment; "
                          "widening stores are "
                          "not supported yet");
      }
      continue;
    }
    std::optional<int64_t> narrowed =
        NarrowingSourceLanes(spec->value, spec->buffer);
    if (!narrowed.has_value()) {
      throw Unsupported("store '" + name + "' needs " + std::to_string(ratio) +
                        "x the segment chunk width " + std::to_string(lanes) +
                        " but its value is not a narrowing conversion; split "
                        "the nest or stage "
                        "an explicit cast");
    }
    if (*narrowed != lanes) {
      throw Unsupported("store '" + name + "' narrows from a " +
                        std::to_string(*narrowed) +
                        "-lane register but the segment chunk width is " +
                        std::to_string(lanes) + " (split the nest)");
    }
  }
  return lanes;
}

SegmentSpecs
PrepareSegment(const std::vector<Axis> &axes, std::optional<Axis> serial_axis,
               const std::vector<Stmt> &pre_stmts, const Stmt &main_stmt,
               const std::vector<Stmt> &post_stmts, const AliasMap &aliases,
               arith::Analyzer *analyzer, std::optional<int64_t> lanes,
               const ffi::Optional<PrimExpr> &predicate) {
  SegmentSpecs specs;
  specs.axes = axes;
  specs.serial_axis = serial_axis;
  if (!pre_stmts.empty())
    specs.pre_specs = NormalizeBody(SeqStmts(pre_stmts), aliases);
  specs.main_specs = NormalizeBody(main_stmt, aliases, &specs.shells);
  if (!post_stmts.empty()) {
    std::vector<ShellLoop> post_shells;
    specs.post_specs =
        NormalizeBody(SeqStmts(post_stmts), aliases, &post_shells);
    if (!post_shells.empty()) {
      throw Unsupported("a statement after a scalar shell cannot itself be a "
                        "shell; split the nest");
    }
  }
  if (specs.main_specs.empty()) {
    throw Unsupported("T.Parallel body has no store to vectorize");
  }
  // The enclosing guard is kept on the segment, not folded into the store
  // masks: lane selection must see it (a mixed-width segment with a predicate
  // is rejected), and ``BuildPlanForAxis`` builds it once per chunk so every
  // store shares the same predicate binding.
  specs.guard = predicate;
  std::vector<StoreSpec> all = specs.AllSpecs();
  for (const StoreSpec &spec : all)
    RequireSupportedElement(spec->buffer);
  specs.lanes = SelectSegmentLanes(specs, lanes);

  auto classify = [&](const std::vector<StoreSpec> &group) {
    std::vector<AccessInfo> infos;
    for (const StoreSpec &spec : group) {
      infos.push_back(ClassifyAccess(spec->buffer, spec->indices, axes,
                                     analyzer,
                                     /*is_store=*/true));
    }
    return infos;
  };
  specs.pre_infos = classify(specs.pre_specs);
  specs.main_infos = classify(specs.main_specs);
  specs.post_infos = classify(specs.post_specs);
  return specs;
}

bool IsSameConst(const PrimExpr &expr, const PrimExpr &value) {
  const auto *lhs_int = expr.as<IntImmNode>();
  const auto *rhs_int = value.as<IntImmNode>();
  if (lhs_int != nullptr && rhs_int != nullptr) {
    return static_cast<double>(lhs_int->value) ==
           static_cast<double>(rhs_int->value);
  }
  const auto *lhs_float = expr.as<FloatImmNode>();
  const auto *rhs_float = value.as<FloatImmNode>();
  if (lhs_float != nullptr && rhs_float != nullptr) {
    return lhs_float->value == rhs_float->value;
  }
  return false;
}

void FoldIdentityInit(VectorPlan &plan) {
  if (plan->pre_stores.empty())
    return;
  std::vector<StorePlan> kept;
  for (const StorePlan &pre : plan->pre_stores) {
    bool folded = false;
    if (pre->kind == AccessClass::kUniform && !pre->mask.has_value()) {
      for (size_t store_index = 0; store_index < plan->stores.size();
           ++store_index) {
        const StorePlan &store = plan->stores[store_index];
        ReduceInfo reduce = ReduceOf(store);
        if (!reduce.defined() || !reduce->accumulate)
          continue;
        if (!(SameBuffer(pre->spec->buffer, store->spec->buffer) &&
              SameIndices(pre->spec->indices, store->spec->indices))) {
          continue;
        }
        std::string elem = ElemDtype(PyBufferDtypeName(store->spec->buffer));
        PrimExpr identity =
            ReduceIdentity(reduce->recipe.operator std::string(), elem);
        if (IsSameConst(pre->spec->value, identity)) {
          // Defect L1: `reduce` is a by-value ObjectRef; mutating its COW copy
          // alone loses the change. Write the updated reduce back to the store
          // (the store still has exclusive ownership here, so COW is a no-op
          // copy of the node we are about to replace).
          ReduceInfo updated_reduce = reduce;
          updated_reduce.CopyOnWrite()->accumulate = false;
          StorePlan updated_store = store;
          updated_store.CopyOnWrite()->reduce =
              ffi::Optional<ffi::ObjectRef>(updated_reduce);
          plan.CopyOnWrite()->stores.Set(static_cast<int64_t>(store_index),
                                         updated_store);
          folded = true;
          break;
        }
      }
    }
    if (!folded)
      kept.push_back(pre);
  }
  plan.CopyOnWrite()->pre_stores = ffi::Array<StorePlan>(kept);
}

void DropDeadPreStores(VectorPlan &plan, arith::Analyzer *analyzer) {
  if (plan->pre_stores.empty())
    return;
  std::vector<tirx::Var> serial_vars;
  for (const Axis &axis : plan->axes) {
    if (!axis->is_parallel)
      serial_vars.push_back(axis->var);
  }
  if (serial_vars.empty())
    return;
  // A zero-trip serial loop never runs the body: pre writes stay live.
  for (const Axis &axis : plan->axes) {
    if (axis->is_parallel)
      continue;
    std::optional<int64_t> extent = ConstInt(axis->extent, analyzer);
    if (extent.has_value() && *extent == 0)
      return;
  }
  std::vector<StorePlan> kept;
  for (const StorePlan &pre : plan->pre_stores) {
    bool dead = false;
    for (const StorePlan &store : plan->stores) {
      if (store->mask.has_value() ||
          !SameBuffer(pre->spec->buffer, store->spec->buffer)) {
        continue;
      }
      // A read-modify-write depends on the value the pre write left behind.
      if (ReadsBuffer(store->value, store->spec->buffer))
        continue;
      PrimExpr offset = store->access->offset;
      std::map<const tirx::VarNode *, PrimExpr> mapping;
      for (const tirx::Var &var : serial_vars)
        mapping[var.get()] = IntImm32(0);
      offset = SubstituteVars(offset, mapping);
      PrimExpr diff = analyzer->Simplify(offset - pre->access->offset);
      std::optional<int64_t> constant = ConstInt(diff, analyzer);
      if (constant.has_value() && *constant == 0) {
        dead = true;
        break;
      }
    }
    if (!dead)
      kept.push_back(pre);
  }
  plan.CopyOnWrite()->pre_stores = ffi::Array<StorePlan>(kept);
}

/*!
 * \brief Whether ``pre`` can be folded into ``store``'s accumulator initial
 * value.
 *
 * Folding drops the pre store's memory write, so the value must be exactly the
 * accumulator's initial value (uniform, unpredicated, same address) and no
 * sibling pre store may read the buffer afterwards (it would see a stale
 * value).
 */
bool PreFoldableInto(const StorePlan &pre, const StorePlan &store,
                     const std::vector<StorePlan> &pre_stores) {
  if (pre->mask.has_value() || !pre->value->is_uniform)
    return false;
  if (!SameBuffer(pre->spec->buffer, store->spec->buffer))
    return false;
  if (!SameIndices(pre->spec->indices, store->spec->indices))
    return false;
  for (const StorePlan &other : pre_stores) {
    if (other.same_as(pre))
      continue;
    if (ReadsBuffer(other->value, pre->spec->buffer))
      return false;
  }
  return true;
}

/*!
 * \brief Record the two non-default store handlings in the plan.
 *
 * Both rules used to be re-derived by the emitter, and the identity-init rule
 * was derived twice with different strength (the hoisting pass used the weak
 * form, the emission block the strong one). Marking them here makes the plan
 * authoritative and keeps both emitter paths consistent.
 */
void MarkStoreRoles(VectorPlan &plan) {
  // PRE_FOLDED: the pre store writes the accumulator's initial value.
  std::set<const StorePlanNode *> consumed;
  for (int64_t index = 0; index < static_cast<int64_t>(plan->stores.size());
       ++index) {
    const StorePlan &store = plan->stores[index];
    if (!store->promote.has_value())
      continue;
    for (int64_t pre_index = 0;
         pre_index < static_cast<int64_t>(plan->pre_stores.size());
         ++pre_index) {
      const StorePlan &pre = plan->pre_stores[pre_index];
      if (consumed.count(pre.get()) != 0)
        continue;
      if (!PreFoldableInto(pre, store,
                           std::vector<StorePlan>(plan->pre_stores.begin(),
                                                  plan->pre_stores.end()))) {
        continue;
      }
      consumed.insert(pre.get());
      StorePlan updated = pre;
      updated.CopyOnWrite()->role =
          ffi::String(StoreRoleName(StoreRole::kPreFolded));
      plan.CopyOnWrite()->pre_stores.Set(pre_index, updated);
      break;
    }
  }

  // POST_SUPERSEDED: a post store rewrites the same address and consumes the
  // accumulator value, so the write-back belongs to the post (register
  // forwarding, one ``vsts`` instead of two).
  for (int64_t index = 0; index < static_cast<int64_t>(plan->stores.size());
       ++index) {
    const StorePlan &store = plan->stores[index];
    if (!store->promote.has_value())
      continue;
    bool superseded = false;
    for (const StorePlan &post : plan->post_stores) {
      if (post->mask.has_value())
        continue;
      if (!SameBuffer(post->spec->buffer, store->spec->buffer))
        continue;
      if (!SameIndices(post->spec->indices, store->spec->indices))
        continue;
      if (!SameAddressLoad(post->value, post->spec->buffer, post->spec->indices)
               .defined()) {
        continue;
      }
      superseded = true;
      break;
    }
    if (!superseded)
      continue;
    StorePlan updated = store;
    updated.CopyOnWrite()->role =
        ffi::String(StoreRoleName(StoreRole::kPostSuperseded));
    plan.CopyOnWrite()->stores.Set(index, updated);
  }
}

/*!
 * \brief Shrink the segment to the chunks whose stores can actually fire.
 *
 * ``GuardFreeBounds`` already proves "this predicate holds over a chunk range"
 * for the mask-free fast path. Asking it about ``not(mask)`` yields the
 * complementary fact: the leading / trailing chunks where every store's
 * predicate is *false*, i.e. the whole chunk performs no write. Those chunks
 * used to be emitted (a compare, a load and a masked store that writes
 * nothing).
 *
 * Only whole-chunk elision is considered, and only when every store is masked
 * and no pre/post statement shares the chunk body: a single unmasked store, or
 * a sibling statement, keeps the chunk live.
 */
void MarkLiveChunkRange(VectorPlan &plan, const ValueContext &ctx) {
  if (!plan->chunks.has_value() || *plan->chunks <= 1)
    return;
  if (plan->stores.empty() || !plan->pre_stores.empty() ||
      !plan->post_stores.empty())
    return;
  if (ReduceOf(plan).defined())
    return;
  const int64_t chunks = *plan->chunks;
  int64_t live_lo = 0;
  int64_t live_hi = chunks;
  for (const StorePlan &store : plan->stores) {
    if (!store->mask.has_value())
      return; // an unconditional write keeps every chunk
    const auto dead = GuardFreeBounds(logical_not(*store->spec->mask), ctx);
    live_lo = std::max(live_lo, dead.first);
    live_hi = std::min(live_hi, dead.second.value_or(chunks));
  }
  if (live_lo <= 0 && live_hi >= chunks)
    return; // nothing to elide
  if (live_lo >= live_hi)
    return; // degenerate: the whole nest is dead
  plan.CopyOnWrite()->chunk_lo = live_lo;
  plan.CopyOnWrite()->chunk_hi = live_hi;
}

std::pair<ValueNode, int64_t> StripReduce(const ValueNode &value,
                                          const std::string &op,
                                          const StoreSpec &site) {
  std::string scalar_recipe;
  ReduceScalarRecipeFor(op, &scalar_recipe);
  std::set<ValueRecipe> spine = {ValueRecipeFromName(op),
                                 ValueRecipeFromName(scalar_recipe)};
  const tirx::Buffer &buffer = site->buffer;
  int64_t accumulated = 0;
  bool repeated = false;

  std::function<ValueNode(const ValueNode &)> strip =
      [&](const ValueNode &node) -> ValueNode {
    if (node->id == ValueRecipe::kUniformLoad && node->access.has_value() &&
        SameBuffer((*node->access)->buffer, buffer) &&
        SameIndices((*node->access)->indices, site->indices)) {
      ++accumulated;
      if (accumulated > 1)
        repeated = true;
      return ValueNode();
    }
    if (spine.count(node->id) != 0 && node->args.size() == 2) {
      std::vector<ValueNode> kept;
      for (const ffi::ObjectRef &arg : node->args) {
        ValueNode child = strip(Downcast<ValueNode>(arg));
        if (child.defined())
          kept.push_back(child);
      }
      if (kept.empty())
        return ValueNode();
      if (kept.size() == 1)
        return kept[0];
      ValueSpec spec;
      spec.recipe = node->recipe;
      spec.dtype = node->dtype;
      spec.args = kept;
      spec.detail = node->detail;
      return MakeValueNode(std::move(spec));
    }
    return node;
  };

  ValueNode varying = strip(value);
  if (repeated) {
    throw Unsupported(
        "the old value of '" + buffer->name.operator std::string() +
        "' appears more than once inside the reduction expression "
        "(design note §2 Case II): only a single same-address old-value "
        "operand can "
        "be folded into the "
        "reduction's initial value");
  }
  if (varying.defined() && ReadsBuffer(varying, buffer)) {
    throw Unsupported(
        "the reduction body reads the destination buffer '" +
        buffer->name.operator std::string() +
        "' outside the accumulate position "
        "(design note §2 Case II): the old value must be a direct "
        "operand of the reduction operator");
  }
  return {varying, accumulated};
}

ReduceInfo PlanReduce(const StorePlan &store, const Axis &vector_axis) {
  const tirx::Buffer &buffer = store->spec->buffer;
  if (store->mask.has_value()) {
    // A conditional reduction has no vector form yet: the cross-lane mask only
    // carries the tail predicate, so the user guard would be silently dropped.
    throw Unsupported(
        "conditional reduction into '" + buffer->name.operator std::string() +
            "' is not supported yet: the cross-lane reduction mask only "
            "carries the tail predicate, so the guard would be silently "
            "dropped "
            "(design note §2 Case II / R3)",
        {}, "reduce.conditional");
  }
  if (!ReadsBuffer(store->value, buffer)) {
    throw Unsupported(
        "store to '" + buffer->name.operator std::string() +
        "' is lane-uniform along axis '" +
        vector_axis->name.operator std::string() +
        "' but its value does not read '" +
        buffer->name.operator std::string() +
        "': only the accumulate form 'dst[...] = dst[...] <op> "
        "x' (op ∈ Add/Max/Min) is a "
        "reduction (design note §2 Case II); a lane-varying value "
        "written to a lane-uniform address is a "
        "last-write-wins overwrite and has no vector form");
  }
  std::string op = ReduceOpOf({store->spec});
  auto stripped = StripReduce(store->value, op, store->spec);
  if (stripped.second == 0 || !stripped.first.defined()) {
    throw Unsupported(
        "the old value of '" + buffer->name.operator std::string() +
        "' must be a direct operand of the '" + op +
        "' reduction "
        "(design note §2 Case II); it cannot be scaled or nested inside "
        "another operation");
  }
  std::string recipe;
  ReduceRecipeFor(op, &recipe);
  return MakeReduceInfo(vector_axis, op, recipe, stripped.first,
                        /*accumulate=*/true);
}

/*! \brief Segment-level reduction classification (D13 single criterion). */
ReduceInfo ClassifyReduce(VectorPlan &plan, arith::Analyzer *analyzer) {
  std::vector<StorePlan> stores;
  for (const StorePlan &store : plan->stores) {
    // A private scalar has one instance per parallel iteration: a lane-uniform
    // write along a parallel axis is a per-lane accumulator, not a reduction.
    if (plan->vector_axis->is_parallel && IsPrivateScope(store->spec->buffer))
      continue;
    stores.push_back(store);
  }
  if (stores.empty())
    return ReduceInfo();

  std::vector<StoreSpec> specs;
  std::vector<AccessInfo> infos;
  for (const StorePlan &store : stores) {
    specs.push_back(store->spec);
    infos.push_back(store->access);
  }
  if (!IsReduceAxis(plan->vector_axis, specs, infos, analyzer))
    return ReduceInfo();

  ReduceInfo primary;
  for (size_t i = 0; i < stores.size(); ++i) {
    if (!ValueVariesAlong(stores[i]->spec->value, plan->vector_axis,
                          analyzer)) {
      continue; // lane-uniform value: a single broadcast write, no reduction
    }
    ReduceInfo reduce = PlanReduce(stores[i], plan->vector_axis);
    for (size_t j = 0; j < plan->stores.size(); ++j) {
      if (plan->stores[j].get() == stores[i].get()) {
        StorePlan copy = plan->stores[j];
        copy.CopyOnWrite()->reduce = ffi::Optional<ffi::ObjectRef>(reduce);
        plan.CopyOnWrite()->stores.Set(static_cast<int64_t>(j), copy);
        break;
      }
    }
    if (!primary.defined())
      primary = reduce;
  }
  return primary;
}

/* ------------------------------------------------------------------------- */
/* stencil / guard cuts / per-axis plan                                      */
/* ------------------------------------------------------------------------- */

std::optional<std::string> SpecCrossAddressRead(
    const StoreSpec &spec,
    const std::vector<std::pair<tirx::Buffer, ffi::Array<PrimExpr>>> &sites,
    const Axis &vector_axis, arith::Analyzer *analyzer) {
  std::vector<std::string> bad;
  tirx::PostOrderVisit(spec->value, [&](const ffi::ObjectRef &node) {
    const auto *load = node.as<tirx::BufferLoadNode>();
    if (load == nullptr)
      return;
    bool touches = false;
    bool same_address = false;
    for (const auto &site : sites) {
      if (!SameBuffer(load->buffer, site.first))
        continue;
      touches = true;
      if (SameIndices(load->indices, site.second))
        same_address = true;
    }
    if (!touches || same_address)
      return;
    AccessInfo info = ClassifyAccess(load->buffer, load->indices, {vector_axis},
                                     analyzer, false);
    if (AxisClassOf(info, vector_axis).kind != AccessClass::kUniform) {
      bad.push_back(load->buffer->name.operator std::string());
    }
  });
  if (bad.empty())
    return std::nullopt;
  return bad.front();
}

std::optional<std::string> CrossAddressRead(const std::vector<StoreSpec> &specs,
                                            const Axis &vector_axis,
                                            arith::Analyzer *analyzer) {
  std::vector<std::pair<tirx::Buffer, ffi::Array<PrimExpr>>> sites;
  for (const StoreSpec &spec : specs)
    sites.emplace_back(spec->buffer, spec->indices);
  for (const StoreSpec &spec : specs) {
    std::optional<std::string> name =
        SpecCrossAddressRead(spec, sites, vector_axis, analyzer);
    if (name.has_value()) {
      return "read of '" + *name +
             "' at a different address than any write in the same nest is a "
             "stencil along "
             "vector axis '" +
             vector_axis->name.operator std::string() +
             "': vectorizing would read a different element than the scalar "
             "order";
    }
  }
  return std::nullopt;
}

std::vector<int64_t> GuardCuts(const VectorPlan &plan) {
  if (!plan->chunks.has_value() || *plan->chunks <= 1)
    return {};
  int64_t chunks = *plan->chunks;
  std::set<int64_t> cuts;
  std::vector<ValueNode> stack;
  std::vector<StorePlan> stores(plan->stores.begin(), plan->stores.end());
  if (plan->vector_axis->is_parallel) {
    stores.insert(stores.end(), plan->pre_stores.begin(),
                  plan->pre_stores.end());
    stores.insert(stores.end(), plan->post_stores.begin(),
                  plan->post_stores.end());
  }
  for (const StorePlan &store : stores) {
    if (store->mask.has_value()) {
      cuts.insert(store->mask_free_until);
      if (store->mask_free_from.has_value())
        cuts.insert(*store->mask_free_from);
    }
    stack.push_back(store->value);
    if (store->mask.has_value())
      stack.push_back(*store->mask);
    if (store->index.has_value())
      stack.push_back(*store->index);
  }
  while (!stack.empty()) {
    ValueNode node = stack.back();
    stack.pop_back();
    ValueNode predicate = PredicateOf(node);
    if (predicate.defined()) {
      cuts.insert(node->predicate_free_until);
      if (node->predicate_free_from.has_value())
        cuts.insert(*node->predicate_free_from);
    }
    for (const ffi::ObjectRef &arg : node->args)
      stack.push_back(Downcast<ValueNode>(arg));
  }
  std::vector<int64_t> interior;
  for (int64_t cut : cuts) {
    if (cut > 0 && cut < chunks)
      interior.push_back(cut);
  }
  if (static_cast<int64_t>(interior.size()) > kGuardCutLimit)
    return {};
  return interior;
}

/*! \brief Whether the chunk is narrower than at least one store's register. */
bool HasPartialLaneStore(const VectorPlan &plan) {
  for (const StorePlan &store : plan->stores) {
    if (store->store_lane_mode.operator std::string() != "full")
      return true;
  }
  return false;
}

/*! \brief Collect the register widths of every real load in a value DAG. */
void CollectLoadLanes(const ValueNode &node, std::set<int64_t> *lanes) {
  if (!node.defined())
    return;
  if (node->id == ValueRecipe::kVLoad ||
      node->id == ValueRecipe::kVLoadStrided ||
      node->id == ValueRecipe::kGather) {
    lanes->insert(VRegLanes(ElemDtype(node->dtype.operator std::string())));
  }
  for (const ffi::ObjectRef &arg : node->args) {
    CollectLoadLanes(Downcast<ValueNode>(arg), lanes);
  }
  if (node->predicate.has_value()) {
    CollectLoadLanes(Downcast<ValueNode>(*node->predicate), lanes);
  }
}

/*!
 * \brief Stage 1 guards for a segment whose chunk is narrower than some store.
 *
 * A partial-lane segment writes ``lanes`` elements per chunk out of a wider
 * store register. It is only proved for straight-line, unpredicated,
 * contiguous element-wise stores whose values load exactly ``lanes`` wide and
 * whose extent divides evenly (no tail); everything else is rejected (the nest
 * stays a serial loop).
 */
void ValidatePartialLaneSegment(const VectorPlan &plan,
                                const std::optional<int64_t> &extent,
                                int64_t lanes) {
  if (!extent.has_value()) {
    throw Unsupported("mixed element widths in one segment require a static "
                      "vector-axis extent that is a "
                      "multiple of the chunk width " +
                      std::to_string(lanes) + " (got a symbolic extent)");
  }
  if (*extent % lanes != 0) {
    throw Unsupported(
        "mixed element widths in one segment require the vector-axis extent (" +
        std::to_string(*extent) + ") to be a multiple of the chunk width " +
        std::to_string(lanes) + " (tails are not supported yet)");
  }
  for (const StorePlan &store : plan->stores) {
    const std::string name = store->spec->buffer->name.operator std::string();
    const std::string kind = AccessClassName(store->kind);
    if (store->kind != AccessClass::kContiguous &&
        store->kind != AccessClass::kUniform) {
      throw Unsupported("mixed element widths in one segment only support "
                        "contiguous or uniform stores ('" +
                        name + "' is " + kind + ")");
    }
    std::set<int64_t> load_lanes;
    CollectLoadLanes(store->value, &load_lanes);
    if (store->mask.has_value())
      CollectLoadLanes(*store->mask, &load_lanes);
    if (store->index.has_value())
      CollectLoadLanes(*store->index, &load_lanes);
    // A narrowing conversion from a 16-bit source (the fp8 recipe) reads a
    // full 16-bit register and spreads it with ``UNPK_B16``, so its load is
    // wider than the float32-wide chunk. Every other load must match the chunk.
    int64_t source_lanes = 0;
    if (const auto *cast = store->spec->value.as<tirx::CastNode>()) {
      source_lanes =
          VRegLanes(ElemDtype(ffi::DLDataTypeToString(cast->value.dtype())));
    }
    for (int64_t width : load_lanes) {
      if (width != lanes && !(width > lanes && width == source_lanes)) {
        throw Unsupported(
            "mixed element widths in one segment require every load to be " +
            std::to_string(lanes) + "-lane wide (store '" + name +
            "' loads a " + std::to_string(width) + "-lane register)");
      }
    }
  }
}

/*! \brief Whether any store of the segment targets an explicitly annotated NZ
 * buffer. */
bool HasNzStore(const VectorPlan &plan) {
  auto has_nz = [](const ffi::Array<StorePlan> &stores) {
    for (const StorePlan &store : stores) {
      if (store->layout_kind.operator std::string() == "nz")
        return true;
    }
    return false;
  };
  // A pre/post sibling is a store of this segment too: the NZ contract (whole
  // chunks, no siblings) must see it, otherwise a packed pre-store built here
  // can only be refused by the emitter.
  return has_nz(plan->pre_stores) || has_nz(plan->stores) ||
         has_nz(plan->post_stores);
}

/*! \brief Whether one value node is an 8-bit load of the given recipe. */
bool IsByteLoad(const ValueNode &node, ValueRecipe recipe) {
  if (!node.defined() || node->id != recipe)
    return false;
  int bits = 0;
  return ElemBits(ElemDtype(node->dtype.operator std::string()), &bits) &&
         bits == 8;
}

template <typename Pred> bool DAGSome(const ValueNode &node, Pred &&pred) {
  if (!node.defined())
    return false;
  if (pred(node))
    return true;
  for (const ffi::ObjectRef &arg : node->args) {
    if (DAGSome(Downcast<ValueNode>(arg), pred))
      return true;
  }
  return false;
}

/*!
 * \brief The load of a pure value chain (a load or same-width reinterprets).
 *
 * A masked 8-bit load is gathered into 16-bit containers; only a chain that
 * merely reinterprets the register can be written back with the packed
 * ``PK_B16`` store. Arithmetic would read the widened container as a 256-lane
 * byte register, so it must not reach the gather.
 */
const ValueNodeNode *PureByteLoadSource(const ValueNode &node) {
  const ValueNodeNode *current = node.get();
  while (current != nullptr && current->id == ValueRecipe::kReinterpret &&
         current->args.size() == 1) {
    current = Downcast<ValueNode>(current->args[0]).get();
  }
  if (current == nullptr || current->id != ValueRecipe::kVLoad)
    return nullptr;
  int bits = 0;
  if (!ElemBits(ElemDtype(current->dtype.operator std::string()), &bits) ||
      bits != 8) {
    return nullptr;
  }
  return current;
}

/*!
 * \brief P2/P3 agreement for masked 8-bit sources.
 *
 * ``vgather2`` widens a byte source into 16-bit lanes, and its destination has
 * only half a register of lanes. Therefore:
 *
 *  - a contiguous 8-bit load can only be gathered on a static tail of at most
 *    half a register, and its value must be a pure load/reinterpret chain that
 *    the packed ``PK_B16`` store narrows back (the ``NORM_B8`` form would write
 *    interleaved garbage);
 *  - an 8-bit strided/indirect load has no full-register gather form at all;
 *  - a predicated 8-bit load cannot be gathered when the chunk is a whole byte
 *    register (up to 256 active lanes, the gather has 128).
 *
 * Rejecting here keeps the "build succeeds => emit succeeds" contract instead
 * of leaving a plan that P3 can only refuse (or, worse, miscompile).
 */
void ValidateByteGatherSegment(const VectorPlan &plan,
                               const std::optional<int64_t> &extent) {
  const bool may_mask =
      !extent.has_value() || (plan->tail.has_value() && *plan->tail != 0);
  auto check = [&](const StoreSpec &spec, const StorePlan &store) {
    if (!store->value.defined())
      return;
    const std::string name = spec->buffer->name.operator std::string();
    const bool strided = DAGSome(store->value, [](const ValueNode &node) {
      return IsByteLoad(node, ValueRecipe::kVLoadStrided) ||
             IsByteLoad(node, ValueRecipe::kGather);
    });
    if (strided) {
      throw Unsupported(
          "an 8-bit strided/indirect load has no full-register gather form "
          "(vgather2 widens a byte source to half a register); store '" +
              name + "' cannot be vectorized",
          {}, "body.byte_gather");
    }
    const bool contiguous = DAGSome(store->value, [](const ValueNode &node) {
      return IsByteLoad(node, ValueRecipe::kVLoad);
    });
    if (!contiguous)
      return;
    const bool byte_store =
        VRegLanes(ElemDtype(PyBufferDtypeName(spec->buffer))) == 256;
    if (!byte_store)
      return; // an 8 -> 16 conversion gathers one lane per byte
    if (!may_mask)
      return; // full chunks stay plain byte loads and stores
    if (store->mask.has_value()) {
      throw Unsupported(
          "a predicated 8-bit store cannot be gathered: the widened gather has "
          "half a register of lanes, so up to 256 active lanes have no form "
          "(store '" +
              name + "')",
          {}, "body.byte_gather");
    }
    if (PureByteLoadSource(store->value) == nullptr) {
      throw Unsupported(
          "an 8-bit value that is not a plain load/reinterpret needs a "
          "256-lane masked gather, which the ISA does not have (store '" +
              name + "')",
          {}, "body.byte_gather");
    }
    if (!extent.has_value()) {
      throw Unsupported(
          "an 8-bit tail needs a static tail of at most half a register "
          "(store '" +
              name + "')",
          {}, "body.byte_gather");
    }
    if (*plan->tail > plan->lanes / 2) {
      throw Unsupported("an 8-bit tail of " + std::to_string(*plan->tail) +
                            " lanes exceeds the half-register gather limit " +
                            std::to_string(plan->lanes / 2) + " (store '" +
                            name + "')",
                        {}, "body.byte_gather");
    }
  };
  for (const StorePlan &store : plan->pre_stores)
    check(store->spec, store);
  for (const StorePlan &store : plan->stores)
    check(store->spec, store);
  for (const StorePlan &store : plan->post_stores)
    check(store->spec, store);
}

/*!
 * \brief P2/P3 agreement for widening loads.
 *
 * A widening conversion reads the source spread into wider containers, which
 * only exists for a 2-way 8/16-bit source (``UNPK_B8`` / ``UNPK_B16``) and a
 * 4-way 8-bit source (``UNPK4_B8``). A 32-bit source (``i32/f32 -> i64``) has a
 * two-way unpack but no 64-bit ``NORM`` store on the ISA, so the conversion has
 * no usable consumer and the plan must be rejected here instead of at emit.
 */
void ValidateWideningLoads(const VectorPlan &plan) {
  auto check = [&](const StoreSpec &spec, const StorePlan &store) {
    DAGSome(store->value, [&](const ValueNode &node) {
      if (node->id != ValueRecipe::kVLoad &&
          node->id != ValueRecipe::kVLoadStrided &&
          node->id != ValueRecipe::kGather) {
        return false;
      }
      if (plan->lanes <= 0)
        return false;
      const std::string elem = ElemDtype(node->dtype.operator std::string());
      int bits = 0;
      if (!ElemBits(elem, &bits))
        return false;
      const int reg_lanes = VRegLanes(elem);
      if (reg_lanes <= plan->lanes)
        return false;
      const int64_t factor = reg_lanes / plan->lanes;
      const bool contiguous = node->id == ValueRecipe::kVLoad;
      const bool ok =
          contiguous && ((factor == 2 && (bits == 8 || bits == 16)) ||
                         (factor == 4 && bits == 8));
      if (!ok) {
        throw Unsupported("a widening load of '" + elem +
                              "' has no verified unpack layout "
                              "(store '" +
                              spec->buffer->name.operator std::string() + "')",
                          {}, "body.widening_load");
      }
      return false;
    });
  };
  for (const StorePlan &store : plan->pre_stores)
    check(store->spec, store);
  for (const StorePlan &store : plan->stores)
    check(store->spec, store);
  for (const StorePlan &store : plan->post_stores)
    check(store->spec, store);
}

/*!
 * \brief Reject a contiguous stream whose chunk base is not 32B aligned.
 *
 * The block register transfers (``vlds_*`` / ``vsts_*``: ``NORM``, ``UNPK``,
 * ``PK``) decode their address as a 32B block plus an element offset, and the
 * emitter folds the whole element offset into the base pointer. A stream whose
 * chunk base is not a multiple of ``32B / elem_size`` elements is therefore
 * mis-decoded on every chunk after the first (device report: ``rows, k = 8,
 * 100`` returns 507035 because the f32 row stride is 400B, not a multiple of
 * 32B; ``8, 104`` is 416B and passes). The ISA can express the split through
 * the offset operand, but the emitter does not pass it yet, so the segment is
 * rejected here instead of being miscompiled.
 *
 * Chunk starts are ``chunk * lanes`` and ``lanes`` is a multiple of the block
 * size, so checking the access with the vector axis at 0 covers every chunk.
 * Element-addressed transfers are exempt: ``gather`` / ``scatter`` take
 * per-lane indices, and the single-element ``ONEPT`` store tolerates a
 * misaligned base (measured with ``rows, k = 8, 104``).
 */
void ValidateStreamAlignment(const VectorPlan &plan, const Axis &vector_axis,
                             arith::Analyzer *analyzer) {
  auto chunk_base = [&](const AccessInfo &access) {
    ffi::Array<PrimExpr> indices;
    for (const PrimExpr &index : access->indices) {
      indices.push_back(analyzer->Simplify(tirx::Substitute(
          index, [&](const tirx::Var &var) -> ffi::Optional<PrimExpr> {
            if (var.same_as(vector_axis->var)) {
              return ffi::Optional<PrimExpr>(IntImm32(0));
            }
            return ffi::Optional<PrimExpr>();
          })));
    }
    return LinearElementOffset(access->buffer, indices, analyzer);
  };
  auto check = [&](const AccessInfo &access) {
    const std::string name = access->buffer->name.operator std::string();
    const std::string elem = ElemDtype(PyBufferDtypeName(access->buffer));
    int bits = 0;
    if (!ElemBits(elem, &bits) || bits == 0)
      return;
    const int64_t per_block = (32 * 8) / bits;
    if (plan->lanes > 0 && plan->lanes % per_block != 0) {
      throw Unsupported("the " + std::to_string(plan->lanes) +
                            "-lane chunk of '" + name +
                            "' is not a whole number of 32B blocks (" + elem +
                            " x " + std::to_string(per_block) +
                            "); the vector load/store base cannot be aligned",
                        {}, "plan.alignment");
    }
    PrimExpr flat = chunk_base(access);
    PrimExpr residual =
        analyzer->Simplify(floormod(flat, IntImm(flat.dtype(), per_block)));
    if (analyzer->CanProveEqual(residual, IntImm(flat.dtype(), 0)))
      return;
    throw Unsupported("the vector stream of '" + name +
                          "' does not start on a 32B boundary (" + elem +
                          " x " + std::to_string(per_block) +
                          " = 32B, residual " + PrintExpr(residual) +
                          "): a block load/store needs a 32B-aligned base, so "
                          "pad the row stride to a "
                          "multiple of " +
                          std::to_string(per_block) +
                          " elements or split the nest",
                      {}, "plan.alignment");
  };
  // A load emitted through ``vgather2`` (statement predicate or the masked tail
  // chunk) is element-addressed, so it needs no alignment. When the whole plan
  // is one masked tail chunk there is no block load at all.
  const bool block_loads = !plan->chunks.has_value() ||
                           !plan->tail.has_value() || *plan->chunks > 1 ||
                           *plan->tail == 0;
  auto check_store = [&](const StorePlan &store) {
    if (!store->value.defined())
      return;
    // NZ scatters write 32B blocks through ``vsstb``'s own block stride; their
    // physical origin is the layout's block base and is not covered here.
    if (store->layout_kind.operator std::string() == "nz")
      return;
    if (store->kind == AccessClass::kContiguous)
      check(store->access);
    if (!block_loads)
      return;
    DAGSome(store->value, [&](const ValueNode &node) {
      if (node->id != ValueRecipe::kVLoad || !node->access.has_value())
        return false;
      if (node->predicate.has_value() && node->predicate.value().defined())
        return false;
      check(node->access.value());
      return false;
    });
  };
  for (const StorePlan &store : plan->pre_stores)
    check_store(store);
  for (const StorePlan &store : plan->stores)
    check_store(store);
  for (const StorePlan &store : plan->post_stores)
    check_store(store);
}

/*!
 * \brief Stage 2b: validate the paired form of a half-register NZ store.
 *
 * ``vsstb`` has no packed variant, so a ``pk_half`` NZ store cannot be
 * scattered as planned: the value register only holds ``lanes`` valid elements
 * out of its
 * ``2 * lanes`` lanes. It is emitted in the *paired* form instead — two
 * consecutive chunks are de-interleaved into one full register (``vdintlv`` +
 * two ``vcvt`` parts + ``vor``) and written with a single ``vsstb``, the same
 * recipe as the hand-written FA softmax. That form only exists when the segment
 * is a clean element-wise body over whole chunk pairs and the packed store owns
 * its destination buffer, so everything else stays ``Unsupported``.
 */
void ValidatePackedNzSegment(const VectorPlan &plan,
                             const std::vector<Axis> &axes,
                             const Axis &vector_axis,
                             const std::optional<int64_t> &extent,
                             int64_t lanes) {
  std::optional<size_t> packed;
  for (size_t i = 0; i < plan->stores.size(); ++i) {
    const StorePlan &store = plan->stores[i];
    if (store->layout_kind.operator std::string() != "nz")
      continue;
    if (store->store_lane_mode.operator std::string() != "pk_half")
      continue;
    if (packed.has_value()) {
      throw Unsupported("at most one half-register NZ store per segment is "
                        "supported (split the nest)");
    }
    packed = i;
  }
  if (!packed.has_value())
    return;

  const StorePlan &target = plan->stores[*packed];
  const std::string name = target->spec->buffer->name.operator std::string();
  // Defect L3: the paired emitter requires the vector axis to be the innermost
  // axis; without this gate P2 succeeds and P3 rejects the same plan.
  for (const Axis &axis : axes) {
    if (axis->index > vector_axis->index) {
      throw Unsupported("the packed NZ store to '" + name +
                        "' needs the vector axis to be the innermost axis "
                        "(split the nest)");
    }
  }
  if (*packed != plan->stores.size() - 1) {
    throw Unsupported("the packed NZ store to '" + name +
                      "' must be the last store of the segment (the merged "
                      "register only exists "
                      "after both chunks)");
  }
  for (const StorePlan &store : plan->stores) {
    if (store.same_as(target))
      continue;
    if (store->spec->buffer->data.same_as(target->spec->buffer->data)) {
      throw Unsupported(
          "the packed NZ store to '" + name +
          "' must own its buffer (another store of the segment writes it)");
    }
  }
  if (target->value->recipe.operator std::string() != "vcvt" ||
      target->value->args.size() != 1) {
    throw Unsupported("the packed NZ store to '" + name +
                      "' needs a plain narrowing cast as its value");
  }
  if (!extent.has_value() || *extent % (2 * lanes) != 0) {
    throw Unsupported(
        "the packed NZ store to '" + name +
        "' needs the vector axis to cover whole chunk pairs (extent=" +
        (extent.has_value() ? std::to_string(*extent) : std::string("?")) +
        ", pair=" + std::to_string(2 * lanes) + ")");
  }
}

/*! \brief Find the explicit layout of a store buffer (same buffer or shared
 * data var). */
const Layout *LookupStoreLayout(const PlanOptions &options,
                                const tirx::Buffer &buffer) {
  for (const auto &entry : options.layouts) {
    if (entry.first.same_as(buffer) ||
        entry.first->data.same_as(buffer->data)) {
      return &entry.second;
    }
  }
  return nullptr;
}

/*! \brief Row-major linearized physical offset of a logical index under
 * ``layout``. */
PrimExpr PhysicalOffset(const Layout &layout,
                        const ffi::Array<PrimExpr> &logical) {
  ffi::Array<PrimExpr> out = layout->Forward(logical);
  ffi::Array<PrimExpr> shape = layout->OutputShape();
  PrimExpr offset = IntImm(DataType::Int(64), 0);
  for (size_t i = 0; i < out.size(); ++i) {
    offset = offset * tirx::Cast(DataType::Int(64), shape[i]) +
             tirx::Cast(DataType::Int(64), out[i]);
  }
  return offset;
}

/*!
 * \brief Classify one store's destination layout (Stage 2).
 *
 * Default is ND: a buffer without an explicit layout annotation keeps the plain
 * contiguous store. A buffer explicitly annotated with an Ascend NZ layout is
 * scattered as 32B blocks straight from the register (``vsstb``); the block and
 * element strides are derived from the layout forward map instead of being
 * hard-coded, and any shape the scatter cannot express stays ``Unsupported`` so
 * the nest is left as a serial loop instead of writing a wrong layout.
 *
 * Returns the classification instead of mutating ``store``: a by-value
 * ``StorePlan`` parameter would bump the node refcount and turn the caller's
 * ``CopyOnWrite`` into a private copy (the write would be lost).
 */
struct StoreLayoutPlan {
  bool nz{false};
  int64_t block_stride{0}; // 32B blocks between two neighbouring C0 groups
};

StoreLayoutPlan ClassifyStoreLayout(const StorePlan &store,
                                    const PlanOptions &options, int64_t lanes,
                                    arith::Analyzer *analyzer) {
  StoreLayoutPlan result;
  const tirx::Buffer &buffer = store->spec->buffer;
  const Layout *layout = LookupStoreLayout(options, buffer);
  if (layout == nullptr || !layout->defined() ||
      layout->as<FragmentNode>() != nullptr) {
    return result; // no explicit (physical) layout: default ND
  }
  const std::string name = buffer->name.operator std::string();
  if (!IsAscendNZLayout(*layout, buffer)) {
    throw Unsupported(
        "store to '" + name +
        "' has an explicit non-NZ layout; only ND (default) and Ascend NZ "
        "destination layouts are supported");
  }
  const std::string lane_mode = store->store_lane_mode.operator std::string();
  if (lane_mode == "pk_quarter") {
    throw Unsupported(
        "NZ store to '" + name +
        "' would need a quarter-register source; only the half-register packed "
        "form is implemented");
  }
  if (store->kind != AccessClass::kContiguous) {
    throw Unsupported(
        "NZ store to '" + name +
        "' requires a contiguous access along the vector axis (got " +
        AccessClassName(store->kind) + ")");
  }
  if (store->mask.has_value()) {
    throw Unsupported("NZ store to '" + name +
                          "' does not support a statement predicate yet",
                      {}, "body.statement_predicate");
  }
  const int c0 = AscendC0(buffer->dtype.bits());
  std::optional<int64_t> cols = ConstInt(buffer->shape.back(), analyzer);
  if (c0 <= 0 || lanes % c0 != 0) {
    throw Unsupported("NZ store to '" + name +
                      "' needs a chunk width multiple of the 32B block "
                      "(" +
                      std::to_string(c0) + " elements), got " +
                      std::to_string(lanes));
  }
  if (!cols.has_value() || *cols % lanes != 0 || lanes > *cols) {
    throw Unsupported(
        "NZ store to '" + name +
            "' needs a static column extent covering a whole number of chunks "
            "(cols=" +
            (cols.has_value() ? std::to_string(*cols) : std::string("?")) +
            ", chunk=" + std::to_string(lanes) + ")",
        {}, "plan.partial_chunks");
  }
  // A half-register source is written in the paired form (Stage 2b): two chunks
  // are merged into one full register, so the columns must cover whole pairs.
  if (lane_mode == "pk_half" && *cols % (2 * lanes) != 0) {
    throw Unsupported("NZ store to '" + name +
                      "' narrows to a half register, so it needs whole " +
                      "chunk pairs (cols=" + std::to_string(*cols) +
                      ", pair=" + std::to_string(2 * lanes) + ")");
  }
  // Block stride from the layout forward map: the distance between the 32B
  // block holding (row, col) and the one holding (row, col + C0).
  ffi::Array<PrimExpr> indices = store->spec->indices;
  ffi::Array<PrimExpr> next;
  for (size_t i = 0; i < indices.size(); ++i) {
    next.push_back(i + 1 == indices.size() ? indices[i] + IntImm32(c0)
                                           : indices[i]);
  }
  PrimExpr step = analyzer->Simplify(PhysicalOffset(*layout, next) -
                                     PhysicalOffset(*layout, indices));
  std::optional<int64_t> block = ConstInt(step, analyzer);
  if (!block.has_value() || *block <= 0 || *block % c0 != 0) {
    throw Unsupported("NZ store to '" + name +
                      "' has a non-constant 32B block stride under its layout");
  }
  result.nz = true;
  result.block_stride = *block / c0;
  return result;
}

VectorPlan BuildPlanForAxis(
    const std::vector<Axis> &axes, std::optional<Axis> serial_axis,
    const std::vector<StoreSpec> &pre_specs,
    const std::vector<AccessInfo> &pre_infos,
    const std::vector<StoreSpec> &main_specs,
    const std::vector<AccessInfo> &main_infos,
    const std::vector<StoreSpec> &post_specs,
    const std::vector<AccessInfo> &post_infos, const Axis &vector_axis,
    int64_t lanes, arith::Analyzer *analyzer, const PlanOptions &options,
    const LoadInfoCache *load_infos, const ffi::Optional<PrimExpr> &guard) {
  bool along_serial =
      serial_axis.has_value() && vector_axis->var.same_as((*serial_axis)->var);
  if (!pre_specs.empty() || !post_specs.empty()) {
    int64_t serial_count = 0;
    for (const Axis &axis : axes)
      serial_count += axis->is_parallel ? 0 : 1;
    if (!along_serial && !(vector_axis->is_parallel && serial_count == 1)) {
      throw Unsupported("a T.Parallel body with sibling statements around a "
                        "serial loop can only be vectorized "
                        "along that serial axis or a single outer parallel "
                        "axis (design note §6, step 2)");
    }
  }

  std::optional<int64_t> extent = ConstInt(vector_axis->extent, analyzer);
  // Defect L2: a zero-extent plan has no chunks, and the emitter indexes an
  // empty statement sequence. Reject it here instead of producing a plan that
  // crashes (or silently truncates) in P3.
  if (extent.has_value() && *extent == 0) {
    throw Unsupported("zero-extent parallel axis '" +
                      vector_axis->name.operator std::string() + "'");
  }
  std::optional<int64_t> chunks;
  std::optional<int64_t> tail;
  if (extent.has_value()) {
    chunks = (*extent + lanes - 1) / lanes;
    tail = *extent % lanes;
  }

  // A local.fragment buffer is a V-core register tile, not an addressable
  // memory operand: planning it would emit an un-compilable vld/vsts pair, so
  // the nest is rejected instead (P1 already keeps fragment nests out of the
  // vectorization path; this guard protects direct planner callers).
  auto reject_fragment = [](const StoreSpec &spec) {
    if (StorageClassOf(spec->buffer) != StorageClass::kFragment)
      return;
    throw Unsupported("store to '" + spec->buffer->name.operator std::string() +
                      "' targets a local.fragment register tile; fragment "
                      "vectorization is not supported");
  };
  for (const StoreSpec &spec : pre_specs)
    reject_fragment(spec);
  for (const StoreSpec &spec : main_specs)
    reject_fragment(spec);
  for (const StoreSpec &spec : post_specs)
    reject_fragment(spec);

  ValueContext ctx;
  ctx.axes = axes;
  ctx.vector_axis = vector_axis;
  ctx.analyzer = analyzer;
  ctx.lanes = lanes;
  ctx.load_infos = load_infos;

  // Build the enclosing guard once: every store of this chunk reuses the same
  // predicate value node, so the condition is evaluated before the stores (as
  // the source ``if`` does) and P3 binds one mask compare for all of them.
  ffi::Optional<ValueNode> guard_node;
  int64_t guard_free_until = 0;
  std::optional<int64_t> guard_free_from;
  if (guard.has_value()) {
    guard_node = ffi::Optional<ValueNode>(BuildValue(*guard, ctx));
    auto bounds = GuardFreeBounds(*guard, ctx);
    guard_free_until = bounds.first;
    guard_free_from = bounds.second;
  }

  // Build each distinct condition expression once per chunk. ``NormalizeBody``
  // hands the *same* expression node to every store of one source ``if``, so
  // all of them share one predicate value node and P3 emits a single compare.
  struct MaskBuild {
    ffi::Optional<ValueNode> node;
    int64_t free_until{0};
    std::optional<int64_t> free_from;
  };
  std::map<const PrimExprNode *, MaskBuild> mask_builds;

  VectorPlan plan =
      MakeVectorPlan(ToArray(axes), vector_axis, lanes, chunks, tail);
  // The plan is uniquely owned here, so take the writable node once instead of
  // re-checking the refcount at every field write (P6).
  VectorPlanNode *writer = plan.CopyOnWrite();

  auto plan_store = [&](const StoreSpec &spec,
                        const AccessInfo &info) -> StorePlan {
    AxisClass classified = AxisClassOf(info, vector_axis);
    ffi::Optional<ValueNode> own_mask;
    int64_t own_free_until = 0;
    std::optional<int64_t> own_free_from;
    if (spec->mask.has_value()) {
      auto build_it = mask_builds.find(spec->mask.get());
      if (build_it != mask_builds.end()) {
        own_mask = build_it->second.node;
        own_free_until = build_it->second.free_until;
        own_free_from = build_it->second.free_from;
      } else {
        own_mask = ffi::Optional<ValueNode>(BuildValue(*spec->mask, ctx));
        auto bounds = GuardFreeBounds(*spec->mask, ctx);
        own_free_until = bounds.first;
        own_free_from = bounds.second;
        mask_builds.emplace(spec->mask.get(),
                            MaskBuild{own_mask, own_free_until, own_free_from});
      }
    }
    // One shared guard value node per chunk: the compare is bound before the
    // stores and reused, so a later store cannot observe a condition changed by
    // an earlier store of the same ``if``.
    ffi::Optional<ValueNode> mask = own_mask;
    int64_t mask_free_until = own_free_until;
    std::optional<int64_t> mask_free_from = own_free_from;
    if (guard_node.has_value()) {
      mask =
          own_mask.has_value()
              ? ffi::Optional<ValueNode>(AndPredicate(*guard_node, *own_mask))
              : guard_node;
      mask_free_until = own_mask.has_value()
                            ? CombineFreeUntil(guard_free_until, own_free_until)
                            : guard_free_until;
      mask_free_from = own_mask.has_value()
                           ? CombineFreeFrom(guard_free_from, own_free_from)
                           : guard_free_from;
    }
    ValueContext value_ctx = ctx;
    if (mask.has_value()) {
      value_ctx = WithGuard(ctx, *mask, mask_free_until, mask_free_from);
    }
    ValueNode value = BuildValue(spec->value, value_ctx);
    bool needs_index =
        classified.kind == AccessClass::kPermute ||
        classified.kind == AccessClass::kPiecewise ||
        classified.kind == AccessClass::kIndirect ||
        (classified.kind == AccessClass::kStrided &&
         classified.stride.has_value() && *classified.stride < 0);
    ffi::Optional<ValueNode> index;
    if (needs_index) {
      index = ffi::Optional<ValueNode>(IndexValue(info, ctx));
    }
    StorePlan store =
        MakeStorePlan(spec, info, value, mask, classified.kind, index);
    store.CopyOnWrite()->mask_free_until = mask_free_until;
    if (mask_free_from.has_value()) {
      store.CopyOnWrite()->mask_free_from =
          ffi::Optional<int64_t>(*mask_free_from);
    }
    // Mixed element widths (Stage 1): record how many of this store's register
    // lanes one chunk actually writes. ``full`` keeps the historical form.
    int64_t register_lanes =
        VRegLanes(ElemDtype(PyBufferDtypeName(spec->buffer)));
    int64_t ratio = lanes > 0 ? register_lanes / lanes : 1;
    store.CopyOnWrite()->store_lanes = register_lanes;
    if (ratio == 2) {
      store.CopyOnWrite()->store_lane_mode = ffi::String("pk_half");
      store.CopyOnWrite()->lane_mode_id = StoreLaneMode::kPkHalf;
    } else if (ratio == 4) {
      store.CopyOnWrite()->store_lane_mode = ffi::String("pk_quarter");
      store.CopyOnWrite()->lane_mode_id = StoreLaneMode::kPkQuarter;
    } else {
      store.CopyOnWrite()->store_lane_mode = ffi::String("full");
      store.CopyOnWrite()->lane_mode_id = StoreLaneMode::kFull;
    }
    if (ratio > 1) {
      store.CopyOnWrite()->store_pack_bits =
          ffi::Optional<int64_t>(2048 / lanes);
    }
    // Destination layout (Stage 2): the classification is applied here, where
    // the plan node is still uniquely owned, so ``CopyOnWrite`` stays in place.
    StoreLayoutPlan layout_plan =
        ClassifyStoreLayout(store, options, lanes, analyzer);
    if (layout_plan.nz) {
      store.CopyOnWrite()->layout_kind = ffi::String("nz");
      store.CopyOnWrite()->layout_id = StoreLayoutKind::kNz;
      store.CopyOnWrite()->nz_block_stride =
          ffi::Optional<int64_t>(layout_plan.block_stride);
    }
    return store;
  };

  // Early axis screen: these rejections only need the classified specs, so a
  // doomed axis does not build every value DAG first (the store plans below are
  // the expensive part of a failed axis attempt).
  if (vector_axis->is_parallel) {
    std::vector<StoreSpec> all_specs;
    all_specs.insert(all_specs.end(), pre_specs.begin(), pre_specs.end());
    all_specs.insert(all_specs.end(), main_specs.begin(), main_specs.end());
    all_specs.insert(all_specs.end(), post_specs.begin(), post_specs.end());
    std::optional<std::string> stencil =
        CrossAddressRead(all_specs, vector_axis, analyzer);
    if (stencil.has_value())
      throw Unsupported(*stencil);
  }
  // A lane-uniform address with a varying value has no vector form unless it is
  // a reduction or a promotable private accumulator. The latter needs the store
  // plans, so only the provably impossible cases are rejected here. The
  // (all-uniform, some value varies) pair is exactly ``IsReduceAxis``, computed
  // from the same per-store checks instead of running the pass twice.
  bool all_addresses_uniform = true;
  std::vector<size_t> uniform_varying;
  for (size_t i = 0; i < main_specs.size(); ++i) {
    if (AxisClassOf(main_infos[i], vector_axis).kind != AccessClass::kUniform) {
      all_addresses_uniform = false;
      continue;
    }
    if (ValueVariesAlong(main_specs[i]->value, vector_axis, analyzer))
      uniform_varying.push_back(i);
  }
  if (!(all_addresses_uniform && !uniform_varying.empty())) {
    for (size_t i : uniform_varying) {
      if (vector_axis->is_parallel) {
        if (IsPrivateScope(main_specs[i]->buffer))
          continue; // may still be promoted by AccumulatorCandidate
        throw Unsupported(
            "store to '" + main_specs[i]->buffer->name.operator std::string() +
            "' is lane-uniform along vector axis '" +
            vector_axis->name.operator std::string() +
            "' but its value varies with it: a lane-varying value written to a "
            "lane-uniform address has no vector form (design note §2 Case II)");
      }
      throw Unsupported(
          "store to '" + main_specs[i]->buffer->name.operator std::string() +
          "' is lane-uniform along the absorbed serial axis '" +
          vector_axis->name.operator std::string() +
          "' but its value varies with it: a partial reduction mixed with "
          "element-wise stores has no vector form (design note §2 Case II)");
    }
  }

  ffi::Array<StorePlan> stores;
  for (size_t i = 0; i < main_specs.size(); ++i) {
    stores.push_back(plan_store(main_specs[i], main_infos[i]));
  }
  ffi::Array<StorePlan> pre_stores;
  for (size_t i = 0; i < pre_specs.size(); ++i) {
    pre_stores.push_back(plan_store(pre_specs[i], pre_infos[i]));
  }
  ffi::Array<StorePlan> post_stores;
  for (size_t i = 0; i < post_specs.size(); ++i) {
    post_stores.push_back(plan_store(post_specs[i], post_infos[i]));
  }
  writer->stores = stores;
  writer->pre_stores = pre_stores;
  writer->post_stores = post_stores;

  // Mixed element widths (Stage 1): a chunk narrower than some store register
  // is only proved for the straight-line element-wise subset.
  const bool partial_lanes = HasPartialLaneStore(plan);
  if (partial_lanes)
    ValidatePartialLaneSegment(plan, extent, lanes);

  // 8-bit sources (Stage 1): a masked byte load can only be gathered in the
  // half-register form; reject here so P3 never has to refuse a built plan.
  ValidateByteGatherSegment(plan, extent);
  ValidateWideningLoads(plan);
  ValidateStreamAlignment(plan, vector_axis, analyzer);

  // Destination layouts (Stage 2): an NZ scatter has no tail or sibling form.
  const bool nz_stores = HasNzStore(plan);
  if (nz_stores) {
    if (!extent.has_value() || *extent % lanes != 0) {
      throw Unsupported("NZ stores require the vector-axis extent to be a "
                        "multiple of the chunk width " +
                        std::to_string(lanes) +
                        " (tails are not supported yet)");
    }
    if (!pre_specs.empty() || !post_specs.empty()) {
      throw Unsupported("NZ stores are only supported without sibling "
                        "statements around the loop");
    }
  }
  // Stage 2b: a half-register NZ store is widened to a full register by pairing
  // two chunks; that shape has its own segment-level contract.
  ValidatePackedNzSegment(plan, axes, vector_axis, extent, lanes);

  // Along a serial axis, pre/post are siblings of the chunk loop and must be
  // lane-uniform; along a parallel axis they live inside the chunk body.
  if (along_serial) {
    const std::vector<std::pair<std::string, ffi::Array<StorePlan>>> groups = {
        {"before", plan->pre_stores}, {"after", plan->post_stores}};
    for (const auto &group : groups) {
      for (const StorePlan &store : group.second) {
        if (store->kind != AccessClass::kUniform) {
          throw Unsupported(
              "a statement " + group.first +
              " the serial loop must be uniform along vector axis '" +
              vector_axis->name.operator std::string() + "' (got " +
              AccessClassName(store->kind) + ") (design note §6, step 2)");
        }
      }
    }
  }

  if (vector_axis->is_parallel) {
    // Accumulator promotion is a plan-time decision: emit only consumes it.
    for (int64_t i = 0; i < static_cast<int64_t>(plan->stores.size()); ++i) {
      StorePlan store = plan->stores[i];
      ValueNode promote = AccumulatorCandidate(store, plan, analyzer);
      if (promote.defined()) {
        store.CopyOnWrite()->promote = ffi::Optional<ValueNode>(promote);
        writer->stores.Set(i, store);
      }
    }
    for (const StorePlan &store : plan->stores) {
      bool promoted = store->promote.has_value();
      if (store->kind == AccessClass::kUniform && !store->value->is_uniform &&
          IsPrivateScope(store->spec->buffer) && !promoted) {
        throw Unsupported(
            "store to private scalar '" +
            store->spec->buffer->name.operator std::string() +
            "' varies along vector axis '" +
            vector_axis->name.operator std::string() +
            "': vectorizing the outer axis needs a promotable accumulator "
            "(a per-iteration private scalar cannot be shared by all lanes) "
            "(design note §2 Case II)");
      }
    }
  }

  ReduceInfo reduce = ClassifyReduce(plan, analyzer);
  if (reduce.defined() && partial_lanes) {
    throw Unsupported("a reduction cannot share a segment with a wider-lane "
                      "store (mixed element widths would "
                      "fold the unused lanes into the accumulator); split the "
                      "nest or stage an explicit cast");
  }
  if (reduce.defined() && nz_stores) {
    throw Unsupported("a reduction store cannot be scattered into an NZ "
                      "destination yet; split the nest");
  }
  if (reduce.defined())
    writer->reduce = ffi::Optional<ffi::ObjectRef>(reduce);
  if (reduce.defined()) {
    // The reduction path hands pre/post to the outer-store emitter, which needs
    // lane-uniform statements (R7: close the build-ok/emit-Unsupported gap).
    const std::vector<std::pair<std::string, ffi::Array<StorePlan>>> groups = {
        {"before", plan->pre_stores}, {"after", plan->post_stores}};
    for (const auto &group : groups) {
      for (const StorePlan &store : group.second) {
        if (store->kind != AccessClass::kUniform) {
          throw Unsupported("a statement " + group.first +
                            " the serial loop must be uniform along the "
                            "reduction vector axis '" +
                            vector_axis->name.operator std::string() +
                            "' (got " + AccessClassName(store->kind) +
                            ") (design note §6, step 2)");
        }
      }
    }
    FoldIdentityInit(plan);
  }
  DropDeadPreStores(plan, analyzer);
  MarkStoreRoles(plan);
  MarkLiveChunkRange(plan, ctx);

  // A lane-varying predicate on a store to a lane-uniform address has no
  // vector form: several lanes would race on one scalar, which is a cross-lane
  // reduction the mask cannot express (e.g. ``if cond_lane: cnt[0] += 1``).
  // Reject at the leaf, before any scalar instruction can be emitted.
  for (const StorePlan &store : plan->stores) {
    if (store->kind != AccessClass::kUniform || !store->mask.has_value())
      continue;
    if ((*store->mask)->is_uniform)
      continue;
    throw Unsupported(
        "store to '" + store->spec->buffer->name.operator std::string() +
            "' has a lane-varying predicate but a lane-uniform address: the "
            "per-lane condition has no vector form (design note §2 Case II)",
        {}, "plan.lane_varying_uniform_store");
  }

  // Defect A0: a partial reduction on an absorbed serial axis has no vector
  // form.
  if (!reduce.defined()) {
    for (const StorePlan &store : plan->stores) {
      if (store->kind != AccessClass::kUniform || store->value->is_uniform) {
        continue;
      }
      if (vector_axis->is_parallel && store->promote.has_value())
        continue;
      if (vector_axis->is_parallel) {
        throw Unsupported(
            "store to '" + store->spec->buffer->name.operator std::string() +
            "' is lane-uniform along vector axis '" +
            vector_axis->name.operator std::string() +
            "' but its value varies with it: a lane-varying value written to a "
            "lane-uniform address has no vector form (design note §2 Case II)");
      }
      throw Unsupported(
          "store to '" + store->spec->buffer->name.operator std::string() +
          "' is lane-uniform along the absorbed serial axis '" +
          vector_axis->name.operator std::string() +
          "' but its value varies with it: a partial reduction mixed with "
          "element-wise stores has no vector form (design note §2 Case II)");
    }
  }

  std::vector<int64_t> cuts = GuardCuts(plan);
  ffi::Array<int64_t> guard_boundaries;
  for (int64_t cut : cuts)
    guard_boundaries.push_back(cut);
  writer->guard_boundaries = guard_boundaries;
  return plan;
}

void NoteGuardSplit(VectorPlan &plan) {
  if (plan->guard_boundaries.empty())
    return;
  std::vector<int64_t> cuts(plan->guard_boundaries.begin(),
                            plan->guard_boundaries.end());
  std::vector<std::string> diagnostics = ToStrings(plan->diagnostics);
  diagnostics.push_back("guard-free chunk cuts at " + PyIntList(cuts) +
                        ": masked gathers revert to plain vld");
  plan.CopyOnWrite()->diagnostics = ToStringArray(diagnostics);
}

/*! \brief Combine an enclosing lane-uniform predicate with a new guard. */
PrimExpr ConjoinPredicate(const ffi::Optional<PrimExpr> &outer,
                          const PrimExpr &guard, bool negate,
                          arith::Analyzer *analyzer) {
  PrimExpr current = negate ? logical_not(guard) : guard;
  if (outer.has_value())
    current = logical_and(*outer, current);
  return analyzer->Simplify(current);
}

/*! \brief Plan one segment: flatten channel first, then ranked axis fallback.
 */
std::pair<VectorPlan, bool>
PlanSegment(const std::vector<Axis> &axes, std::optional<Axis> serial_axis,
            const std::vector<Stmt> &pre_stmts, const Stmt &main_stmt,
            const std::vector<Stmt> &post_stmts, const AliasMap &aliases,
            arith::Analyzer *analyzer, std::optional<int64_t> lanes,
            bool single, const std::vector<std::string> &base_notes,
            const PlanOptions &options, bool allow_flatten = true,
            const ffi::Optional<PrimExpr> &predicate = std::nullopt) {
  std::string flatten_word = single ? "nest" : "segment";
  SegmentSpecs specs =
      PrepareSegment(axes, serial_axis, pre_stmts, main_stmt, post_stmts,
                     aliases, analyzer, lanes, predicate);
  // A chunk is a straight-line region. A scalar loop left by normalization
  // (a "shell") would be wrapped around every store of the chunk, replaying
  // stores that were source-order siblings of the loop. Materialize it as a
  // plan-level Loop instead; the recursive planner does that when this
  // speculative chunk plan fails.
  if (!specs.shells.empty()) {
    throw Unsupported(
        "a scalar loop inside a chunk would be replayed by the chunk loop; "
        "materialize it as a plan-level loop instead",
        {}, "body.shell_reorder");
  }
  std::vector<AccessInfo> all_infos = specs.AllInfos();
  std::vector<AccessInfo> load_infos =
      LoadInfos(axes, specs.AllSpecs(), analyzer, &specs.load_info_cache);
  std::vector<std::string> diagnostics = base_notes;
  std::vector<std::string> axis_notes =
      AxisDiagnostics(axes, all_infos, load_infos, analyzer, specs.lanes);
  diagnostics.insert(diagnostics.end(), axis_notes.begin(), axis_notes.end());

  auto flattened = allow_flatten
                       ? TryFlatten(axes, pre_stmts, post_stmts, specs.lanes,
                                    analyzer, specs)
                       : std::pair<std::optional<SegmentSpecs>, std::string>{
                             std::nullopt, std::string()};
  if (!flattened.second.empty())
    diagnostics.push_back(flattened.second);
  if (flattened.first.has_value()) {
    SegmentSpecs flat_specs = *flattened.first;
    Axis flat_axis = flat_specs.axes[0];
    bool ok = true;
    VectorPlan plan;
    try {
      plan = flat_specs.plan_for(flat_axis, analyzer, options);
    } catch (const Unsupported &error) {
      diagnostics.push_back("flattened axis '" +
                            flat_axis->name.operator std::string() +
                            "': " + error.message());
      ok = false;
    }
    if (ok) {
      std::vector<std::string> notes;
      notes.push_back("flattened " + flatten_word + " [" + JoinNames(axes) +
                      "] -> '" + flat_axis->name.operator std::string() +
                      "' (extent " + PrintExpr(flat_axis->extent) +
                      ", full-lane chunks)");
      notes.insert(notes.end(), diagnostics.begin(), diagnostics.end());
      plan.CopyOnWrite()->diagnostics = ToStringArray(notes);
      NoteGuardSplit(plan);
      return {plan, true};
    }
  }

  std::vector<std::string> errors;
  // Specificity survives the per-axis fallback: the first classified reason
  // (ranked axes are tried in order) becomes the rejection code.
  std::string first_classified_code;
  std::vector<Axis> ranked =
      RankVectorAxes(axes, all_infos, load_infos, analyzer, specs.lanes);
  if (ranked.empty()) {
    throw Unsupported(
        "no vectorizable axis: every store is true-random (its index cannot be "
        "built in the index-vector channel) on all axes",
        diagnostics, "plan.no_vector_axis");
  }
  for (const Axis &axis : ranked) {
    VectorPlan plan;
    try {
      plan = specs.plan_for(axis, analyzer, options);
    } catch (const Unsupported &error) {
      // Per-axis failure reasons survive even on success (improvement 7).
      errors.push_back("axis '" + axis->name.operator std::string() +
                       "': " + error.message());
      if (first_classified_code.empty() &&
          error.code() != Unsupported::kDefaultCode) {
        first_classified_code = error.code();
      }
      diagnostics.push_back("axis '" + axis->name.operator std::string() +
                            "': " + error.message() + " -> fallback");
      for (const std::string &note : error.diagnostics())
        diagnostics.push_back("  " + note);
      continue;
    }
    if (single) {
      diagnostics.push_back(
          "selected vector axis '" + axis->name.operator std::string() +
          "' (score=" +
          std::to_string(
              AxisScore(all_infos, load_infos, axis, analyzer, specs.lanes)) +
          ")");
    } else {
      diagnostics.push_back(
          "segment vector axis '" + axis->name.operator std::string() +
          "' (active=" +
          std::to_string(ActiveLanes(axis, analyzer, specs.lanes)) + "/" +
          std::to_string(specs.lanes) + ", design note §6, step 3)");
    }
    plan.CopyOnWrite()->diagnostics = ToStringArray(diagnostics);
    NoteGuardSplit(plan);
    return {plan, false};
  }
  std::string joined;
  for (size_t i = 0; i < errors.size(); ++i)
    joined += (i == 0 ? "" : "; ") + errors[i];
  std::string code = first_classified_code.empty() ? "plan.no_vector_axis"
                                                   : first_classified_code;
  throw Unsupported("no vectorizable axis: " + joined, diagnostics, code);
}

/* ------------------------------------------------------------------------- */
/* recursive tree planner (design A, stages A2/A3)                           */
/* ------------------------------------------------------------------------- */

/*!
 * \brief Planner context: visible axes and where we are in the source tree.
 *
 * ``scope`` holds the *non-materialized* axes only. A materialized axis (a
 * ``kLoop`` node) is replayed by the emitter and must not be visible to the
 * chunk planner, which keeps every chunk's axis table exactly the set it may
 * use for wrapping / vectorization (R4 without a hoisted-not-selectable rule).
 */
struct PlanContext {
  std::vector<Axis> scope;
  AliasMap aliases;
  std::optional<int64_t> lanes;
  const PlanOptions *options{nullptr};
  arith::Analyzer *analyzer{nullptr};
  std::string path;
  /*! \brief Guard inherited from an enclosing ``if`` (uniform or lane-varying).
   *
   * The VF contract only allows vector work plus masks, so every branch is
   * lowered by predicating the leaf stores of its arms instead of emitting
   * scalar control flow. The arm structure itself is preserved by the tree. */
  ffi::Optional<PrimExpr> predicate;
};

/*! \brief Append a node to ``children`` unless it is undefined/empty. */
void PushPlanNode(std::vector<PlanNode> *children, PlanNode node) {
  if (node.defined())
    children->push_back(std::move(node));
}

/*! \brief True for statements normalization drops (Bind / Evaluate(0)). */
bool IsDroppableStmt(const Stmt &stmt) {
  if (stmt.as<tirx::BindNode>() != nullptr)
    return true;
  const auto *evaluate = stmt.as<tirx::EvaluateNode>();
  if (evaluate == nullptr)
    return false;
  if (const auto *imm = evaluate->value.as<tirx::IntImmNode>())
    return imm->value == 0;
  return false;
}

[[noreturn]] void RethrowWithPath(const Unsupported &error,
                                  const std::string &path) {
  std::vector<std::string> diagnostics = error.diagnostics();
  diagnostics.push_back("at " + path);
  throw Unsupported(error.message() + " [path: " + path + "]", diagnostics,
                    error.code());
}

/*! \brief Throw a hard ``tree.*`` structural error annotated with its path. */
[[noreturn]] void ThrowTreeError(const std::string &code,
                                 const std::string &message,
                                 const std::string &path) {
  throw Unsupported(message + " [path: " + path + "]", {"at " + path}, code);
}

/*!
 * \brief Whether a statement keeps two or more sibling ``For`` children.
 *
 * Such a body is not one shell chain: each loop is an independent segment (or
 * a materialized loop), so planning it as one chunk would either unroll the
 * loops iteration by iteration or silently chain them (the old hole).
 */
bool HasMultipleSiblingLoops(const Stmt &stmt) {
  bool found = false;
  tirx::PostOrderVisit(stmt, [&](const ffi::ObjectRef &node) {
    if (found)
      return;
    const auto *seq = node.as<tirx::SeqStmtNode>();
    if (seq == nullptr)
      return;
    int loops = 0;
    for (const Stmt &child : seq->seq)
      loops += child.as<tirx::ForNode>() != nullptr ? 1 : 0;
    found = loops >= 2;
  });
  return found;
}

/*! \brief Build a segment source whose base axes are the context scope. */
SegmentSource SourceFromScope(const PlanContext &ctx, const Stmt &stmt,
                              bool allow_flatten) {
  SegmentSource source;
  source.axes = ctx.scope;
  Stmt current = stmt;
  std::optional<Axis> serial_axis;

  while (const auto *loop_ptr = current.as<tirx::ForNode>()) {
    tirx::For loop = ffi::GetRef<tirx::For>(loop_ptr);
    ForKind kind = loop->kind;
    RequireZeroMin(loop, ctx.analyzer);
    if (kind == ForKind::kSerial) {
      auto clean = SerialIsClean(loop, ctx.analyzer);
      if (!clean.first)
        break;
    } else if (kind != ForKind::kParallel) {
      break;
    }
    Axis axis = MakeAxis(loop->loop_var, loop->extent,
                         static_cast<int64_t>(source.axes.size()), kind);
    source.axes.push_back(axis);
    if (kind == ForKind::kSerial)
      serial_axis = axis;
    current = loop->body;
  }

  if (const auto *seq = current.as<tirx::SeqStmtNode>()) {
    std::vector<size_t> loop_positions;
    for (size_t i = 0; i < seq->seq.size(); ++i) {
      if (seq->seq[i].as<tirx::ForNode>() != nullptr)
        loop_positions.push_back(i);
    }
    // Exactly one clean serial loop splits the body into pre/serial/post; the
    // pre/post stores stay in this chunk (a reduction's identity initializer
    // folds into the accumulator instead of becoming a shared-memory write).
    if (loop_positions.size() == 1) {
      size_t pos = loop_positions[0];
      tirx::For loop = Downcast<tirx::For>(seq->seq[pos]);
      std::optional<int64_t> min = ConstInt(loop->min, ctx.analyzer);
      auto clean =
          loop->kind == ForKind::kSerial && min.has_value() && *min == 0
              ? SerialIsClean(loop, ctx.analyzer)
              : std::pair<bool, std::string>{false, std::string()};
      if (clean.first) {
        serial_axis = MakeAxis(loop->loop_var, loop->extent,
                               static_cast<int64_t>(source.axes.size()),
                               ForKind::kSerial);
        source.axes.push_back(*serial_axis);
        for (size_t i = 0; i < pos; ++i)
          source.pre.push_back(seq->seq[i]);
        for (size_t i = pos + 1; i < seq->seq.size(); ++i)
          source.post.push_back(seq->seq[i]);
        current = loop->body;
      }
    }
  }

  bool has_parallel = false;
  for (const Axis &axis : source.axes)
    has_parallel = has_parallel || axis->is_parallel;
  if (!has_parallel)
    throw Unsupported("no T.Parallel axis found");

  std::vector<Stmt> pieces = source.pre;
  pieces.push_back(current);
  pieces.insert(pieces.end(), source.post.begin(), source.post.end());
  source.aliases =
      ctx.aliases.empty() ? CollectAliases(SeqStmts(pieces)) : ctx.aliases;
  source.serial_axis = serial_axis;
  source.main = current;
  source.allow_flatten = allow_flatten;
  return source;
}

/*! \brief Plan one leaf statement group into a chunk (hard failure). */
PlanNode PlanLeaf(const Stmt &stmt, const PlanContext &ctx, bool single) {
  try {
    SegmentSource source = SourceFromScope(ctx, stmt, /*allow_flatten=*/true);
    std::pair<VectorPlan, bool> planned = PlanSegment(
        source.axes, source.serial_axis, source.pre, source.main, source.post,
        source.aliases, ctx.analyzer, ctx.lanes, single, {}, *ctx.options,
        source.allow_flatten, ctx.predicate);
    PlanNode chunk = MakePlanChunk(planned.first, planned.second);
    return chunk;
  } catch (const Unsupported &error) {
    RethrowWithPath(error, ctx.path);
  }
  return PlanNode();
}

/*! \brief Try one statement / chain as a single chunk; nullopt when rejected.
 */
std::optional<PlanNode> TryPlanChunk(const Stmt &stmt, const PlanContext &ctx,
                                     bool single) {
  try {
    SegmentSource source = SourceFromScope(ctx, stmt, /*allow_flatten=*/true);
    std::pair<VectorPlan, bool> planned = PlanSegment(
        source.axes, source.serial_axis, source.pre, source.main, source.post,
        source.aliases, ctx.analyzer, ctx.lanes, single, {}, *ctx.options,
        source.allow_flatten, ctx.predicate);
    return MakePlanChunk(planned.first, planned.second);
  } catch (const Unsupported &) {
    return std::nullopt;
  }
}

/*!
 * \brief Whether a materialized loop iteration must close with VST->VLD.
 *
 * A Loop is materialized for a source loop that could not be absorbed. Its
 * body iteration is replayed in source order, so a barrier is needed exactly
 * when a buffer written in the body may be read in the same or a later
 * iteration (RAW / WAR across iterations). Pure-write bodies and disjoint
 * read/write sets do not need one.
 */
void CollectExprBufferKeys(const PrimExpr &expr,
                           std::set<const tirx::VarNode *> *read) {
  if (!expr.defined())
    return;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
    if (const auto *load = node.as<tirx::BufferLoadNode>())
      read->insert(load->buffer->data.get());
  });
}

void CollectPlanAccess(const PlanNode &node,
                       std::set<const tirx::VarNode *> *read,
                       std::set<const tirx::VarNode *> *written) {
  if (!node.defined())
    return;
  if (node->kind == PlanKind::kSeq) {
    for (const ffi::ObjectRef &child : node->children)
      CollectPlanAccess(Downcast<PlanNode>(child), read, written);
    return;
  }
  if (node->kind == PlanKind::kLoop) {
    if (node->body.defined())
      CollectPlanAccess(Downcast<PlanNode>(node->body), read, written);
    return;
  }
  const VectorPlan &plan = node->chunk;
  const std::vector<ffi::Array<StorePlan>> groups = {
      plan->pre_stores, plan->stores, plan->post_stores};
  for (const ffi::Array<StorePlan> &group : groups) {
    for (const StorePlan &store : group) {
      written->insert(store->spec->buffer->data.get());
      for (const PrimExpr &index : store->spec->indices)
        CollectExprBufferKeys(index, read);
      CollectExprBufferKeys(store->spec->value, read);
      if (store->spec->mask.has_value())
        CollectExprBufferKeys(*store->spec->mask, read);
    }
  }
}

bool LoopNeedsBarrier(const PlanNode &body) {
  std::set<const tirx::VarNode *> read;
  std::set<const tirx::VarNode *> written;
  CollectPlanAccess(body, &read, &written);
  for (const tirx::VarNode *buffer : written) {
    if (read.count(buffer) != 0)
      return true;
  }
  return false;
}

/*! \brief Whether a condition consumes a parallel axis as a lane value. */
bool ConditionTouchesLaneAxis(const PrimExpr &condition,
                              const PlanContext &ctx) {
  if (!condition.defined())
    return false;
  ffi::Array<tirx::Var> vars = tirx::UndefinedVars(condition);
  for (const tirx::Var &var : vars) {
    for (const Axis &axis : ctx.scope) {
      if (axis->is_parallel && axis->var.same_as(var))
        return true;
    }
  }
  return false;
}

PlanNode PlanBody(const Stmt &stmt, const PlanContext &ctx, bool single);

bool ConditionTouchesLaneAxis(const PrimExpr &condition,
                              const PlanContext &ctx);

/*! \brief A constant bool expression, or nullopt when it is not constant. */
std::optional<bool> ConstBoolExpr(const PrimExpr &expr) {
  if (!expr.defined())
    return std::nullopt;
  if (const auto *imm = expr.as<tirx::IntImmNode>())
    return imm->value != 0;
  return std::nullopt;
}

/*! \brief Exactly one store and no loop/alloc/attr statement.
 *
 * This is the only ``if`` shape that may be planned as one leaf chunk without
 * losing structure: with no loop there is no shell chain to replay, so the
 * chunk planner may still apply its same-address ``vsel`` / predicated-store
 * decisions.
 */
bool IsSingleStoreArm(const Stmt &stmt) {
  if (!stmt.defined())
    return false;
  int stores = 0;
  bool structural = false;
  tirx::PostOrderVisit(stmt, [&](const ffi::ObjectRef &node) {
    if (structural)
      return;
    if (node.as<tirx::ForNode>() != nullptr ||
        node.as<tirx::WhileNode>() != nullptr ||
        node.as<tirx::AllocBufferNode>() != nullptr ||
        node.as<tirx::AttrStmtNode>() != nullptr)
      structural = true;
    if (node.as<tirx::BufferStoreNode>() != nullptr)
      ++stores;
  });
  return !structural && stores == 1;
}

/*! \brief Plan an ``if`` by predicating its arms (no scalar branch). */
PlanNode PlanIf(const tirx::IfThenElseNode *branch, const PlanContext &ctx,
                bool single) {
  Stmt stmt = ffi::GetRef<tirx::Stmt>(branch);
  // Narrow leaf fast path: one store per arm, no loop. The leaf planner still
  // merges same-address arms into a lazy if_then_else (vsel) and predicates
  // divergent arms, while no shell can be collected or replayed.
  if (IsSingleStoreArm(branch->then_case) &&
      (!branch->else_case.defined() || IsSingleStoreArm(*branch->else_case))) {
    if (auto chunk = TryPlanChunk(stmt, ctx, single))
      return *chunk;
  }

  PrimExpr condition = ApplyAliases(branch->condition, ctx.aliases);
  auto plan_arm = [&](const Stmt &arm, bool negate,
                      const std::string &suffix) -> PlanNode {
    PlanContext arm_ctx = ctx;
    PrimExpr guard =
        ConjoinPredicate(ctx.predicate, condition, negate, ctx.analyzer);
    if (std::optional<bool> constant = ConstBoolExpr(guard);
        constant.has_value()) {
      if (!*constant)
        return PlanNode(); // statically dead arm
      arm_ctx.predicate = std::nullopt;
    } else {
      arm_ctx.predicate = guard;
    }
    arm_ctx.path = ctx.path + suffix;
    try {
      return PlanBody(arm, arm_ctx, single);
    } catch (const Unsupported &error) {
      // A reduction has no mask semantics; a shape the recursive planner
      // cannot split stays a structural error.
      std::string code = error.code();
      if (code == "reduce.conditional")
        code = "tree.predicated_reduce";
      else if (code == "body.sibling_shells" || code == "body.nested_loop")
        code = "tree.mixed_chain";
      std::vector<std::string> diagnostics = error.diagnostics();
      diagnostics.push_back("at " + arm_ctx.path);
      throw Unsupported(
          "an if arm is lowered by predicating every leaf store; this arm is "
          "not predicable: " +
              error.message() + " [path: " + arm_ctx.path + "]",
          diagnostics, code);
    }
  };

  std::vector<PlanNode> children;
  PushPlanNode(&children,
               plan_arm(branch->then_case, /*negate=*/false, " > if(then)"));
  if (branch->else_case.defined() && !IsDroppableStmt(*branch->else_case)) {
    PushPlanNode(&children,
                 plan_arm(*branch->else_case, /*negate=*/true, " > if(else)"));
  }
  if (children.empty())
    return PlanNode();
  if (children.size() == 1)
    return children[0];
  return MakePlanSeq(std::move(children));
}

PlanNode PlanBody(const Stmt &stmt, const PlanContext &ctx, bool single);

/*! \brief Plan one loop: chunk chain if possible, otherwise a Loop node. */
PlanNode PlanFor(const tirx::ForNode *loop, const PlanContext &ctx,
                 bool single) {
  ForKind kind = loop->kind;
  if (kind != ForKind::kSerial && kind != ForKind::kParallel) {
    ThrowTreeError("tree.opaque",
                   "unsupported loop kind in a planned nest: " +
                       ForKindName(kind),
                   ctx.path);
  }
  RequireZeroMin(ffi::GetRef<tirx::For>(loop), ctx.analyzer);
  if (loop->step.has_value()) {
    std::optional<int64_t> step = ConstInt(loop->step.value(), ctx.analyzer);
    if (!step.has_value() || *step != 1) {
      ThrowTreeError("tree.opaque", "a planned loop needs unit stride",
                     ctx.path);
    }
  }
  // A structural body can never be one chunk chain: planning it speculatively
  // would normalize and classify the whole subtree before failing. Mirror the
  // root fast-path guard and go straight to the recursive planner.
  const bool structural =
      HasMultipleSiblingLoops(loop->body) || HasSiblingShell(loop->body);
  if (!structural) {
    if (auto chunk = TryPlanChunk(ffi::GetRef<tirx::Stmt>(loop), ctx, single))
      return *chunk;
  }

  // The chunk chain could not absorb this loop. A serial loop is materialized
  // (source structure kept); a parallel loop is not: parallel semantics allow
  // each descendant chunk to wrap it, which is what the old program-level
  // "hoisted prefix" did. Bounds may not depend on a parallel lane axis.
  ffi::Array<tirx::Var> bound_var_array = tirx::UndefinedVars(loop->min);
  std::vector<tirx::Var> bound_vars(bound_var_array.begin(),
                                    bound_var_array.end());
  ffi::Array<tirx::Var> extent_var_array = tirx::UndefinedVars(loop->extent);
  std::vector<tirx::Var> extent_vars(extent_var_array.begin(),
                                     extent_var_array.end());
  for (const Axis &axis : ctx.scope) {
    if (!axis->is_parallel)
      continue;
    for (const tirx::Var &var : bound_vars) {
      if (axis->var.same_as(var))
        throw Unsupported(
            "a loop bound depends on the parallel axis '" +
                axis->name.operator std::string() +
                "', which would become a scalar inside the lanes [path: " +
                ctx.path + "]",
            {}, "tree.scalar_bound");
    }
    for (const tirx::Var &var : extent_vars) {
      if (axis->var.same_as(var))
        throw Unsupported(
            "a loop bound depends on the parallel axis '" +
                axis->name.operator std::string() +
                "', which would become a scalar inside the lanes [path: " +
                ctx.path + "]",
            {}, "tree.scalar_bound");
    }
  }

  PlanContext inner = ctx;
  if (kind == ForKind::kParallel) {
    inner.scope.push_back(MakeAxis(loop->loop_var, loop->extent,
                                   static_cast<int64_t>(ctx.scope.size()),
                                   ForKind::kParallel));
    inner.path =
        ctx.path + " > for " + loop->loop_var->name_hint.operator std::string();
    return PlanBody(loop->body, inner, single);
  }

  inner.path =
      ctx.path + " > for " + loop->loop_var->name_hint.operator std::string();
  PlanNode body = PlanBody(loop->body, inner, single);
  return MakePlanLoop(loop->loop_var, loop->min, loop->extent,
                      static_cast<int64_t>(ForKind::kSerial), body,
                      LoopNeedsBarrier(body));
}

PlanNode PlanBody(const Stmt &stmt, const PlanContext &ctx, bool single) {
  if (!stmt.defined())
    return PlanNode();

  if (const auto *seq = stmt.as<tirx::SeqStmtNode>()) {
    std::vector<PlanNode> children;
    std::vector<Stmt> pending;
    auto flush = [&]() {
      if (pending.empty())
        return;
      Stmt group = pending.size() == 1 ? pending.front() : SeqStmts(pending);
      pending.clear();
      std::vector<Stmt> parts;
      if (const auto *group_seq = group.as<tirx::SeqStmtNode>()) {
        for (const Stmt &child : group_seq->seq)
          parts.push_back(child);
      } else {
        parts.push_back(group);
      }
      bool only_droppable = true;
      for (const Stmt &child : parts)
        only_droppable = only_droppable && IsDroppableStmt(child);
      if (only_droppable)
        return;
      PushPlanNode(&children, PlanLeaf(group, ctx, single));
    };
    for (const Stmt &child : seq->seq) {
      if (child.as<tirx::ForNode>() != nullptr ||
          child.as<tirx::IfThenElseNode>() != nullptr) {
        flush();
        PlanContext child_ctx = ctx;
        if (child.as<tirx::ForNode>() != nullptr) {
          PushPlanNode(&children,
                       PlanFor(child.as<tirx::ForNode>(), child_ctx, single));
        } else {
          PushPlanNode(&children, PlanIf(child.as<tirx::IfThenElseNode>(),
                                         child_ctx, single));
        }
        continue;
      }
      pending.push_back(child);
    }
    flush();
    if (children.size() == 1)
      return children[0];
    return MakePlanSeq(std::move(children));
  }

  if (const auto *branch = stmt.as<tirx::IfThenElseNode>())
    return PlanIf(branch, ctx, single);
  if (const auto *loop = stmt.as<tirx::ForNode>())
    return PlanFor(loop, ctx, single);
  if (stmt.as<tirx::WhileNode>() != nullptr) {
    ThrowTreeError("tree.while",
                   "while loops are not supported by the tree planner yet",
                   ctx.path);
  }
  if (stmt.as<tirx::AllocBufferNode>() != nullptr) {
    ThrowTreeError("tree.alloc",
                   "allocate inside a planned nest is not supported", ctx.path);
  }
  if (stmt.as<tirx::AttrStmtNode>() != nullptr) {
    ThrowTreeError("tree.opaque",
                   "a non-transparent attr statement is not supported",
                   ctx.path);
  }
  if (IsDroppableStmt(stmt))
    return PlanNode();
  return PlanLeaf(stmt, ctx, single);
}

/*! \brief Serial loops that wrap several chunks (the old shared shells). */
void CollectMaterializedLoops(const PlanNode &node, int64_t depth,
                              int64_t prefix_depth,
                              ffi::Array<Axis> *shared_shells) {
  if (!node.defined())
    return;
  switch (node->kind) {
  case PlanKind::kLoop: {
    if (depth >= prefix_depth && node->chunk_count >= 2) {
      shared_shells->push_back(MakeAxis(node->loop_var, node->extent, 0,
                                        static_cast<ForKind>(node->for_kind)));
    }
    if (node->body.defined())
      CollectMaterializedLoops(Downcast<PlanNode>(node->body), depth + 1,
                               prefix_depth, shared_shells);
    return;
  }
  case PlanKind::kSeq:
    for (const ffi::ObjectRef &child : node->children)
      CollectMaterializedLoops(Downcast<PlanNode>(child), depth, prefix_depth,
                               shared_shells);
    return;
  case PlanKind::kChunk:
    return;
  }
}

/*! \brief Minimum ``vector_axis`` index consumed by any chunk (flattened = 0).
 */
int64_t FirstConsumedAxis(const PlanNode &node, int64_t current) {
  if (!node.defined())
    return current;
  switch (node->kind) {
  case PlanKind::kChunk:
    return std::min<int64_t>(current, node->chunk->vector_axis->index);
  case PlanKind::kSeq:
    for (const ffi::ObjectRef &child : node->children)
      current = FirstConsumedAxis(Downcast<PlanNode>(child), current);
    return current;
  case PlanKind::kLoop:
    return node->body.defined()
               ? FirstConsumedAxis(Downcast<PlanNode>(node->body), current)
               : current;
  }
  return current;
}

/*!
 * \brief Wrap the leading parallel prefix that no chunk consumes as an axis.
 *
 * This is the old program-level hoist rule expressed in the tree: only the
 * leading run of parallel axes before the first consumed vector axis may be
 * hoisted; a flattened chunk consumes index 0, so nothing is hoisted.
 */
PlanNode HoistParallelPrefix(PlanNode root,
                             const std::vector<Axis> &parallel_axes,
                             ffi::Array<Axis> *hoisted) {
  int64_t first = FirstConsumedAxis(root, std::numeric_limits<int64_t>::max());
  std::vector<Axis> prefix;
  if (first != std::numeric_limits<int64_t>::max()) {
    for (const Axis &axis : parallel_axes) {
      if (!axis->is_parallel || axis->index >= first)
        break;
      prefix.push_back(axis);
    }
  }
  for (const Axis &axis : prefix)
    hoisted->push_back(axis);
  for (auto it = prefix.rbegin(); it != prefix.rend(); ++it) {
    root = MakePlanLoop((*it)->var, IntImm32(0), (*it)->extent, (*it)->kind,
                        root, /*barrier_at_end=*/false);
  }
  return root;
}

/*! \brief Assemble the tree plan into the program object P3 consumes. */
VectorProgram AssembleTreeProgram(const PlanNode &root,
                                  const ffi::Array<Axis> &hoisted,
                                  ffi::Array<Axis> shared_shells) {
  ffi::Array<VectorPlan> segments = TreeSegments(root);
  VectorProgram program =
      MakeVectorProgram(hoisted, std::move(shared_shells), std::move(segments));
  program.CopyOnWrite()->root = root;

  ffi::Array<ffi::String> diagnostics;
  if (program->segments.size() == 1) {
    const VectorPlan &plan = program->segments[0];
    bool flattened = false;
    std::function<void(const PlanNode &)> find_flat =
        [&](const PlanNode &node) {
          if (!node.defined() || flattened)
            return;
          if (node->kind == PlanKind::kChunk) {
            flattened = node->flattened;
            return;
          }
          if (node->kind == PlanKind::kSeq) {
            for (const ffi::ObjectRef &child : node->children)
              find_flat(Downcast<PlanNode>(child));
          } else if (node->kind == PlanKind::kLoop && node->body.defined()) {
            find_flat(Downcast<PlanNode>(node->body));
          }
        };
    find_flat(root);
    std::string detail =
        flattened ? "flattened axis '" +
                        plan->vector_axis->name.operator std::string() + "'"
                  : "vector axis '" +
                        plan->vector_axis->name.operator std::string() + "'";
    diagnostics.push_back("single-segment program: 1 segment, " + detail);
  } else {
    diagnostics.push_back(
        "multi-segment program: " + std::to_string(program->segments.size()) +
        " segments, source order preserved" +
        (hoisted.empty()
             ? ", flattened/consumed parallel axes are wrapped by each segment"
             : "") +
        (program->shared_shells.empty()
             ? ""
             : ", shared scalar shells wrap every segment"));
  }
  program.CopyOnWrite()->diagnostics = diagnostics;
  return program;
}

} // namespace

VectorProgram BuildPlan(const tirx::Stmt &stmt, std::optional<int64_t> lanes,
                        const PlanOptions &options) {
  arith::Analyzer analyzer;
  // P1 hands over the arithmetic contract of the region. Enter it before any
  // zero-min / injectivity / lane proof so a VF-local tl.assume is usable here
  // (the fields used to be write-only).
  for (const PrimExpr &assumption : options.assumptions) {
    if (assumption.defined())
      analyzer.EnterConstraint(assumption);
  }
  for (const auto &binding : options.bindings) {
    // Call-valued bindings are not value aliases (same rule as the old
    // RestoreBindings helper); binding them would confuse the analyzer.
    if (binding.first.defined() && binding.second.defined() &&
        binding.second.as<tirx::CallNode>() == nullptr) {
      analyzer.Bind(binding.first, binding.second);
    }
  }

  auto absorbed = AbsorbParallelChain(stmt, &analyzer);
  if (absorbed.first.empty())
    throw Unsupported("no T.Parallel axis found");
  const std::vector<Axis> &parallel_axes = absorbed.first;
  const Stmt &body = absorbed.second;

  PlanContext ctx;
  ctx.scope = parallel_axes;
  ctx.aliases = CollectAliases(body);
  ctx.lanes = lanes;
  ctx.options = &options;
  ctx.analyzer = &analyzer;
  ctx.path = "body";

  // Fast path: the whole body is one chunk chain (single-segment quality is
  // identical to the former ``SingleSegmentSource``).
  if (!HasMultipleSiblingLoops(body) && !HasSiblingShell(body)) {
    if (auto chunk = TryPlanChunk(body, ctx, /*single=*/true)) {
      ffi::Array<Axis> hoisted;
      ffi::Array<Axis> shared_shells;
      PlanNode root = HoistParallelPrefix(*chunk, parallel_axes, &hoisted);
      return AssembleTreeProgram(root, hoisted, std::move(shared_shells));
    }
  }

  // Structural path: plan the body as a tree, then hoist the leading parallel
  // prefix every chunk leaves unconsumed.
  PlanNode root = PlanBody(body, ctx, /*single=*/false);
  ffi::Array<Axis> hoisted;
  root = HoistParallelPrefix(root, parallel_axes, &hoisted);

  // Compatibility view: serial loops that wrap several chunks used to be
  // ``shared_shells``; report them from the tree.
  ffi::Array<Axis> shared_shells;
  CollectMaterializedLoops(
      root, /*depth=*/0,
      /*prefix_depth=*/static_cast<int64_t>(hoisted.size()), &shared_shells);
  return AssembleTreeProgram(root, hoisted, std::move(shared_shells));
}

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm
