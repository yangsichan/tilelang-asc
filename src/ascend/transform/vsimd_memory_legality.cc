/*!
 * \file vsimd_memory_legality.cc
 * \brief Read-only memory-legality analysis for SIMD_VF bodies (P1, PR-A).
 */
#include "vsimd_memory_legality.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ascend/layout/ascend_layouts.h"
#include "vsimd_analysis_common.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace vsimd;
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

namespace {

/* ------------------------------------------------------------------------- */
/* Small helpers                                                             */
/* ------------------------------------------------------------------------- */

enum class MapClass {
  kAffine = 0,
  kQuasiAffine = 1,
  kGeneral = 2,
  kDataDependent = 3,
};

enum class Legality {
  kLegal = 0,
  kUnproven = 1,
  kIllegal = 2,
  kUnsupported = 3,
  kOpaque = 4,
};

// Result of a single Z3-backed no-violation query.
enum class Relation {
  kSafe,
  kViolation,
  kUnknown,
};

static const char *StatusName(Legality s) {
  switch (s) {
  case Legality::kLegal:
    return "LEGAL";
  case Legality::kUnproven:
    return "UNPROVEN";
  case Legality::kIllegal:
    return "ILLEGAL";
  case Legality::kUnsupported:
    return "UNSUPPORTED";
  case Legality::kOpaque:
    return "OPAQUE";
  }
  return "UNKNOWN";
}

static const char *MapClassName(MapClass c) {
  switch (c) {
  case MapClass::kAffine:
    return "affine";
  case MapClass::kQuasiAffine:
    return "quasi-affine";
  case MapClass::kGeneral:
    return "general";
  case MapClass::kDataDependent:
    return "data-dependent";
  }
  return "unknown";
}

static MapClass MaxClass(MapClass a, MapClass b) {
  return static_cast<int>(a) >= static_cast<int>(b) ? a : b;
}

struct AccessInfo : vsimd::AccessInfo {
  MapClass map_class{MapClass::kAffine};
  bool layout_changing{false};
  // Injective across parallel instances; serial times within one instance may
  // intentionally revisit the same address (e.g. an ordered accumulator).
  bool injective{true};
  std::string injective_reason;
};

struct NestInfo {
  std::vector<LoopInfo> loops;
  std::vector<LoopInfo> outer_shells;
  std::vector<AccessInfo> accesses;
  std::vector<std::pair<Buffer, Buffer>> local_views;
  Legality status{Legality::kLegal};
  std::vector<std::string> messages;
  size_t total_accesses{0};
  size_t unique_accesses{0};
  size_t queries{0};
  size_t cache_hits{0};
  bool budget_exceeded{false};
};

/*!
 * \brief Identity of a guard path: one (condition object, polarity) per branch.
 *
 * Pointer identity is enough: every access in one arm shares the same
 * condition object, and two structurally equal conditions in different
 * statements are still different guards (being conservative here only means the
 * query is not shared, never that a conflict is missed).
 */
using GuardPathKey = std::vector<std::pair<const void *, bool>>;

static GuardPathKey GuardSignature(const std::vector<GuardLiteral> &guards) {
  GuardPathKey out;
  out.reserve(guards.size());
  for (const GuardLiteral &guard : guards)
    out.emplace_back(guard.condition.get(), guard.polarity);
  return out;
}

static size_t HashGuards(size_t h, const GuardPathKey &guards) {
  for (const auto &guard : guards)
    h = h * 31 + std::hash<const void *>()(guard.first) * 7 +
        (guard.second ? 1 : 0);
  return h;
}

static bool SameGuards(const GuardPathKey &lhs, const GuardPathKey &rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

// Identity + structural key for deduplicating repeated loads/stores.
struct AccessKey {
  Buffer buffer;
  Array<PrimExpr> indices;
  bool is_write{false};
  std::vector<For> serial_loops;
  // Different guards are different statements even over the same map.
  GuardPathKey guards;

  bool operator==(const AccessKey &other) const {
    if (!buffer.same_as(other.buffer) || is_write != other.is_write ||
        serial_loops.size() != other.serial_loops.size() ||
        !SameGuards(guards, other.guards) ||
        indices.size() != other.indices.size())
      return false;
    for (size_t i = 0; i < serial_loops.size(); ++i) {
      if (!serial_loops[i].same_as(other.serial_loops[i]))
        return false;
    }
    for (size_t i = 0; i < indices.size(); ++i) {
      if (!tirx::ExprDeepEqual()(indices[i], other.indices[i]))
        return false;
    }
    return true;
  }
};

struct AccessKeyHash {
  size_t operator()(const AccessKey &key) const {
    size_t h = ffi::ObjectPtrHash()(key.buffer);
    for (const PrimExpr &idx : key.indices)
      h = h * 31 + ffi::StructuralHash()(idx);
    h = h * 31 + (key.is_write ? 1 : 0);
    for (const For &loop : key.serial_loops)
      h = h * 31 + ffi::ObjectPtrHash()(loop);
    return HashGuards(h, key.guards);
  }
};

// Map-level key: independence is a property of (view, map, domains, guards),
// independent of whether the access is a read or a write.
struct MapKey {
  Buffer buffer;
  Array<PrimExpr> indices;
  std::vector<For> serial_loops;
  GuardPathKey guards;

  bool operator==(const MapKey &other) const {
    if (!buffer.same_as(other.buffer) ||
        serial_loops.size() != other.serial_loops.size() ||
        !SameGuards(guards, other.guards) ||
        indices.size() != other.indices.size())
      return false;
    for (size_t i = 0; i < serial_loops.size(); ++i) {
      if (!serial_loops[i].same_as(other.serial_loops[i]))
        return false;
    }
    for (size_t i = 0; i < indices.size(); ++i) {
      if (!tirx::ExprDeepEqual()(indices[i], other.indices[i]))
        return false;
    }
    return true;
  }
};

struct MapKeyHash {
  size_t operator()(const MapKey &key) const {
    size_t h = ffi::ObjectPtrHash()(key.buffer);
    for (const PrimExpr &idx : key.indices)
      h = h * 31 + ffi::StructuralHash()(idx);
    for (const For &loop : key.serial_loops)
      h = h * 31 + ffi::ObjectPtrHash()(loop);
    return HashGuards(h, key.guards);
  }
};

struct PairKey {
  MapKey a;
  MapKey b;

  bool operator==(const PairKey &other) const {
    return a == other.a && b == other.b;
  }
};

struct PairKeyHash {
  size_t operator()(const PairKey &key) const {
    return MapKeyHash()(key.a) * 31 + MapKeyHash()(key.b);
  }
};

struct RelationResult {
  Relation relation{Relation::kUnknown};
  std::string why;
};

constexpr size_t kMaxQueriesPerNest = 1024;

static MapKey MakeMapKey(const AccessInfo &access) {
  return MapKey{access.buffer, access.indices, access.serial_loops,
                GuardSignature(access.guards)};
}

static AccessKey MakeAccessKey(const AccessInfo &access) {
  return AccessKey{access.buffer, access.indices, access.is_write,
                   access.serial_loops, GuardSignature(access.guards)};
}

/*!
 * \brief Conjunction of a guard path, optionally renamed for the other
 *        iteration of a dependence query.
 *
 * ``subs`` instantiates both parallel and internal serial coordinates for one
 * side of the query. Only a condition invariant across both executions can
 * make opposite polarities contradict.
 */
static PrimExpr GuardConjunction(const std::vector<GuardLiteral> &guards,
                                 const Map<Var, PrimExpr> *subs) {
  PrimExpr result = Bool(true);
  for (const GuardLiteral &guard : guards) {
    PrimExpr condition =
        subs == nullptr ? guard.condition : Substitute(guard.condition, *subs);
    if (!guard.polarity)
      condition = Not(condition);
    result = And(result, condition);
  }
  return result;
}

static bool HasLoopVar(const PrimExpr &expr, const VarSet &loop_vars) {
  if (!expr.defined())
    return false;
  for (const Var &v : tirx::UndefinedVars(expr)) {
    if (loop_vars.count(v))
      return true;
  }
  return false;
}

static bool IsConstantDivisor(const PrimExpr &expr) {
  const auto *imm = expr.as<IntImmNode>();
  return imm != nullptr && imm->value != 0;
}

static MapClass ClassifyExpr(const PrimExpr &expr, const VarSet &loop_vars) {
  if (!expr.defined())
    return MapClass::kAffine;
  if (const auto *imm = expr.as<IntImmNode>())
    return MapClass::kAffine;
  if (const auto *imm = expr.as<FloatImmNode>())
    return MapClass::kAffine;
  if (expr.as<VarNode>())
    return MapClass::kAffine; // loop var or capture
  if (const auto *op = expr.as<AddNode>())
    return MaxClass(ClassifyExpr(op->a, loop_vars),
                    ClassifyExpr(op->b, loop_vars));
  if (const auto *op = expr.as<SubNode>())
    return MaxClass(ClassifyExpr(op->a, loop_vars),
                    ClassifyExpr(op->b, loop_vars));
  if (const auto *op = expr.as<MulNode>()) {
    bool a_has = HasLoopVar(op->a, loop_vars);
    bool b_has = HasLoopVar(op->b, loop_vars);
    if (a_has && b_has)
      return MapClass::kGeneral;
    return MaxClass(ClassifyExpr(op->a, loop_vars),
                    ClassifyExpr(op->b, loop_vars));
  }
  if (const auto *op = expr.as<DivNode>()) {
    if (!IsConstantDivisor(op->b))
      return MapClass::kGeneral;
    return MaxClass(MapClass::kQuasiAffine, ClassifyExpr(op->a, loop_vars));
  }
  if (const auto *op = expr.as<ModNode>()) {
    if (!IsConstantDivisor(op->b))
      return MapClass::kGeneral;
    return MaxClass(MapClass::kQuasiAffine, ClassifyExpr(op->a, loop_vars));
  }
  if (const auto *op = expr.as<FloorDivNode>()) {
    if (!IsConstantDivisor(op->b))
      return MapClass::kGeneral;
    return MaxClass(MapClass::kQuasiAffine, ClassifyExpr(op->a, loop_vars));
  }
  if (const auto *op = expr.as<FloorModNode>()) {
    if (!IsConstantDivisor(op->b))
      return MapClass::kGeneral;
    return MaxClass(MapClass::kQuasiAffine, ClassifyExpr(op->a, loop_vars));
  }
  if (const auto *op = expr.as<CastNode>())
    return ClassifyExpr(op->value, loop_vars);
  if (expr.as<BufferLoadNode>() || expr.as<ProducerLoadNode>())
    return MapClass::kDataDependent;
  if (expr.as<SelectNode>() || expr.as<MinNode>() || expr.as<MaxNode>())
    return MapClass::kGeneral;
  // Any remaining call / intrinsic is conservatively general.
  return MapClass::kGeneral;
}

static MapClass ClassifyMap(const Array<PrimExpr> &indices,
                            const VarSet &loop_vars) {
  MapClass result = MapClass::kAffine;
  for (const PrimExpr &idx : indices) {
    result = MaxClass(result, ClassifyExpr(idx, loop_vars));
  }
  return result;
}

static std::vector<PrimExpr> CompactStrides(const Buffer &buffer) {
  size_t ndim = buffer->shape.size();
  std::vector<PrimExpr> strides(ndim);
  PrimExpr stride = IntImm(DataType::Int(32), 1);
  for (size_t i = ndim; i-- > 0;) {
    strides[i] = stride;
    stride = stride * buffer->shape[i];
  }
  return strides;
}

static PrimExpr LinearizeAccess(const Buffer &buffer,
                                const Array<PrimExpr> &indices) {
  if (indices.size() != buffer->shape.size())
    return PrimExpr();
  std::vector<PrimExpr> strides;
  if (buffer->strides.empty()) {
    strides = CompactStrides(buffer);
  } else {
    strides.assign(buffer->strides.begin(), buffer->strides.end());
  }
  // Compare bit offsets, not element offsets: T.view may reinterpret the same
  // storage with a different dtype, and element units would then disagree.
  int64_t bit_width = static_cast<int64_t>(buffer->dtype.bits()) *
                      static_cast<int64_t>(buffer->dtype.lanes());
  PrimExpr scale = IntImm(DataType::Int(32), static_cast<int>(bit_width));
  PrimExpr off = buffer->elem_offset * scale;
  for (size_t i = 0; i < indices.size(); ++i) {
    off = off + indices[i] * strides[i] * scale;
  }
  return off;
}

/* ------------------------------------------------------------------------- */
/* Function-level collection                                                 */
/* ------------------------------------------------------------------------- */

class FunctionCollector : public StmtExprVisitor {
public:
  VarSet params;
  VarSet allocs;
  VarSet decls;
  VarSet non_restrict;
  VarSet layout_vars;

  void Collect(const PrimFunc &func) {
    for (const auto &kv : func->buffer_map) {
      params.insert(kv.second->data);
    }
    VisitStmt(func->body);
  }

  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &b : op->alloc_buffers)
      allocs.insert(b->data);
    CollectAnnotations_(op->annotations);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    allocs.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    decls.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  void CollectAnnotations_(const Map<String, Any> &annotations) {
    if (auto layout = annotations.Get("layout_map")) {
      if (auto map = layout.value().as<Map<Buffer, Layout>>()) {
        for (const auto &kv : map.value())
          layout_vars.insert(kv.first->data);
      } else if (auto vmap = layout.value().as<Map<Var, Layout>>()) {
        // The frontend may key the layout map by the buffer data var before
        // layout inference materializes Buffer-keyed maps.
        for (const auto &kv : vmap.value())
          layout_vars.insert(kv.first);
      }
    }
    if (auto non_restrict_any = annotations.Get("tl.non_restrict_params")) {
      if (auto vars = non_restrict_any.value().as<Array<Var>>()) {
        for (const Var &v : vars.value())
          non_restrict.insert(v);
      } else if (auto bufs = non_restrict_any.value().as<Array<Buffer>>()) {
        for (const Buffer &b : bufs.value())
          non_restrict.insert(b->data);
      }
    }
  }
};

/* ------------------------------------------------------------------------- */
/* Manual SIMD detection                                                     */
struct AliasKind {
  enum Kind { kSameStorage, kNoAlias, kMayAlias } kind;
};

/* ------------------------------------------------------------------------- */
/* Syntactic injectivity fast paths                                          */
/* ------------------------------------------------------------------------- */

static bool IsNonzeroConst(const PrimExpr &expr) {
  const auto *imm = expr.as<IntImmNode>();
  return imm != nullptr && imm->value != 0;
}

// True for var, a*var+b (a != 0), var-b, and the symmetric constant forms.
static bool IsAffineInVar(const PrimExpr &expr, const Var &var) {
  if (expr.same_as(var))
    return true;
  if (const auto *mul = expr.as<MulNode>()) {
    return (IsNonzeroConst(mul->a) && mul->b.same_as(var)) ||
           (IsNonzeroConst(mul->b) && mul->a.same_as(var));
  }
  if (const auto *add = expr.as<AddNode>()) {
    return (IsNonzeroConst(add->b) && IsAffineInVar(add->a, var)) ||
           (IsNonzeroConst(add->a) && IsAffineInVar(add->b, var));
  }
  if (const auto *sub = expr.as<SubNode>()) {
    return IsNonzeroConst(sub->b) && IsAffineInVar(sub->a, var);
  }
  return false;
}

static bool ExtractDivMod(const PrimExpr &expr, bool *is_mod, PrimExpr *base,
                          PrimExpr *divisor) {
  if (const auto *op = expr.as<FloorDivNode>()) {
    *is_mod = false;
    *base = op->a;
    *divisor = op->b;
    return true;
  }
  if (const auto *op = expr.as<DivNode>()) {
    *is_mod = false;
    *base = op->a;
    *divisor = op->b;
    return true;
  }
  if (const auto *op = expr.as<FloorModNode>()) {
    *is_mod = true;
    *base = op->a;
    *divisor = op->b;
    return true;
  }
  if (const auto *op = expr.as<ModNode>()) {
    *is_mod = true;
    *base = op->a;
    *divisor = op->b;
    return true;
  }
  return false;
}

// (i // C, i % C) uniquely determines i for any constant C > 0.
static bool HasQuotientRemainderPair(const Array<PrimExpr> &indices,
                                     const Var &var) {
  for (size_t i = 0; i < indices.size(); ++i) {
    for (size_t j = i + 1; j < indices.size(); ++j) {
      bool m1 = false;
      bool m2 = false;
      PrimExpr b1;
      PrimExpr d1;
      PrimExpr b2;
      PrimExpr d2;
      if (!ExtractDivMod(indices[i], &m1, &b1, &d1) ||
          !ExtractDivMod(indices[j], &m2, &b2, &d2))
        continue;
      if (m1 == m2)
        continue; // need exactly one quotient and one remainder
      if (!b1.same_as(var) || !b2.same_as(var))
        continue;
      const auto *c1 = d1.as<IntImmNode>();
      const auto *c2 = d2.as<IntImmNode>();
      if (c1 != nullptr && c2 != nullptr && c1->value == c2->value &&
          c1->value > 0)
        return true;
    }
  }
  return false;
}

static bool TryDivModExtent(const PrimExpr &extent, const PrimExpr &divisor,
                            bool is_mod, Relation *result, std::string *why) {
  const auto *c = divisor.as<IntImmNode>();
  const auto *e = extent.as<IntImmNode>();
  if (c == nullptr || c->value <= 0 || e == nullptr)
    return false;
  if (!is_mod) {
    // i // 1 is the identity.
    if (c->value == 1) {
      *result = Relation::kSafe;
      return true;
    }
    // i // C is constant 0 when extent <= C and repeats C times otherwise;
    // it is injective only for a single-iteration loop.
    if (e->value <= 1) {
      *result = Relation::kSafe;
      return true;
    }
    *result = Relation::kViolation;
    *why = "i // C is not injective for extent > 1";
    return true;
  }
  if (e->value <= c->value) {
    *result = Relation::kSafe;
    return true;
  }
  *result = Relation::kViolation;
  *why = "i % C is not injective when extent > C";
  return true;
}

static bool TryConstantExtent(const PrimExpr &extent, Relation *result,
                              std::string *why) {
  const auto *e = extent.as<IntImmNode>();
  if (e == nullptr)
    return false;
  if (e->value <= 1) {
    *result = Relation::kSafe;
    return true;
  }
  *result = Relation::kViolation;
  *why = "constant store map covers multiple iterations";
  return true;
}

static bool HasSymbolicExtent(const NestInfo &nest) {
  for (const LoopInfo &loop : nest.loops) {
    if (loop.extent.as<IntImmNode>() == nullptr ||
        loop.min.as<IntImmNode>() == nullptr)
      return true;
  }
  return false;
}

static std::string SanitizeModel(const std::string &model) {
  std::string s = model;
  for (char &c : s) {
    if (c == '\n' || c == '\r')
      c = ';';
  }
  if (s.size() > 240)
    s = s.substr(0, 240) + "...";
  return s;
}

class LegalityAnalyzer {
public:
  std::vector<std::string> nest_statuses;
  std::vector<std::string> nest_reasons;
  /*! \brief Outermost Parallel loop per nest, parallel to `nest_statuses`. */
  std::vector<const ForNode *> nest_roots;
  LegalityAnalyzer(const FunctionCollector &facts, std::ostringstream *out,
                   const std::vector<PrimExpr> &assumptions,
                   const std::vector<std::pair<Var, PrimExpr>> &bindings,
                   const std::vector<std::pair<Var, PrimExpr>> &thread_bounds,
                   const std::vector<LoopInfo> &outer_serial)
      : facts_(facts), out_(out), assumptions_(assumptions),
        bindings_(bindings), thread_bounds_(thread_bounds),
        outer_serial_(outer_serial) {}

  std::string AnalyzeVF(const SBlock &vf, int index) {
    ManualDetector manual;
    manual(vf);
    if (manual.manual) {
      *out_ << "VF#" << index << " status=OPAQUE\n";
      *out_ << "  message: manual SIMD / nd2nz / local.var\n";
      return "OPAQUE";
    }

    NestCollector collector;
    collector(vf->body);
    if (collector.nests.empty()) {
      *out_ << "VF#" << index << " status=OPAQUE\n";
      *out_ << "  message: no T.Parallel nest\n";
      return "OPAQUE";
    }

    Legality overall = Legality::kLegal;
    std::ostringstream body;
    int nest_index = 0;
    for (const ParallelNest &parallel_nest : collector.nests) {
      if (parallel_nest.unsupported) {
        // Unsupportable syntax is a per-nest verdict: it keeps this nest on the
        // scalar path without disabling the legality proof of its siblings.
        nest_statuses.push_back(StatusName(Legality::kUnsupported));
        nest_reasons.push_back(parallel_nest.reason);
        nest_roots.push_back(parallel_nest.root);
        overall = std::max(overall, Legality::kUnsupported);
        body << "  nest " << nest_index++
             << ": status=UNSUPPORTED message: " << parallel_nest.reason
             << "\n";
        continue;
      }
      NestInfo nest;
      nest.loops = parallel_nest.loops;
      nest.outer_shells = outer_serial_;
      nest.outer_shells.insert(nest.outer_shells.end(),
                               parallel_nest.outer_shells.begin(),
                               parallel_nest.outer_shells.end());
      for (const vsimd::AccessInfo &src : parallel_nest.accesses) {
        AccessInfo acc;
        acc.buffer = src.buffer;
        acc.indices = src.indices;
        acc.value = src.value;
        acc.is_write = src.is_write;
        acc.serial_loops = src.serial_loops;
        acc.guards = src.guards;
        nest.accesses.push_back(std::move(acc));
      }
      for (const Buffer &view : parallel_nest.local_views) {
        Buffer base;
        for (const AccessInfo &acc : nest.accesses) {
          if (acc.buffer->data.same_as(view->data) &&
              !acc.buffer.same_as(view)) {
            base = acc.buffer;
            break;
          }
        }
        nest.local_views.emplace_back(view, base);
      }
      // T.view may be declared outside the Parallel loop, so NestCollector's
      // local_views list can be empty. Recover the alias from access buffers
      // with the same data var: the written buffer is the view, the read
      // buffer is the base.
      for (const AccessInfo &acc : nest.accesses) {
        if (!acc.is_write)
          continue;
        for (const AccessInfo &other : nest.accesses) {
          if (other.is_write || !acc.buffer->data.same_as(other.buffer->data) ||
              acc.buffer.same_as(other.buffer)) {
            continue;
          }
          bool exists = false;
          for (const auto &entry : nest.local_views) {
            if (entry.first.same_as(acc.buffer) &&
                entry.second.same_as(other.buffer)) {
              exists = true;
              break;
            }
          }
          if (!exists)
            nest.local_views.emplace_back(acc.buffer, other.buffer);
          break;
        }
      }
      AnalyzeNest_(&nest);
      nest_statuses.push_back(StatusName(nest.status));
      // An ILLEGAL witness takes precedence over earlier UNPROVEN messages.
      // The full report retains all messages; annotations carry the primary
      // reason so a rejection does not produce an unbounded warning.
      nest_reasons.push_back(nest.messages.empty() ? ""
                             : nest.status == Legality::kIllegal
                                 ? nest.messages.back()
                                 : nest.messages.front());
      nest_roots.push_back(parallel_nest.root);
      overall = std::max(overall, nest.status);
      body << "  nest " << nest_index++ << ": domain=";
      for (const LoopInfo &loop : nest.loops) {
        body << std::string(loop.var->name_hint) << " in [" << loop.min << ", "
             << (loop.min + loop.extent) << ") ";
      }
      body << "accesses=" << nest.total_accesses << "/" << nest.unique_accesses
           << " queries=" << nest.queries << " cache_hits=" << nest.cache_hits;
      if (nest.budget_exceeded)
        body << " budget=exceeded";
      body << "\n";
      for (const AccessInfo &acc : nest.accesses) {
        body << "    " << (acc.is_write ? "W " : "R ")
             << std::string(acc.buffer->name) << "[";
        for (size_t i = 0; i < acc.indices.size(); ++i) {
          if (i)
            body << ", ";
          body << acc.indices[i];
        }
        body << "] class=" << MapClassName(acc.map_class)
             << " layout=" << (acc.layout_changing ? "changing" : "none")
             << " injective=" << (acc.injective ? "yes" : "no");
        if (!acc.injective_reason.empty())
          body << " (" << acc.injective_reason << ")";
        body << "\n";
      }
      if (!nest.local_views.empty()) {
        body << "    local_views:";
        for (const auto &view : nest.local_views) {
          body << " " << view.first->name;
          if (view.second.defined())
            body << " -> " << view.second->name;
        }
        body << "\n";
      }
      for (const std::string &msg : nest.messages)
        body << "    message: " << msg << "\n";
    }
    *out_ << "VF#" << index << " status=" << StatusName(overall) << "\n";
    *out_ << body.str();
    return StatusName(overall);
  }

private:
  void AnalyzeNest_(NestInfo *nest) {
    auto sync_counters = [&] {
      nest->queries = queries_;
      nest->cache_hits = cache_hits_;
      nest->budget_exceeded = budget_exceeded_;
    };

    nest->total_accesses = nest->accesses.size();
    std::vector<AccessInfo> unique;
    std::unordered_set<AccessKey, AccessKeyHash> seen;
    for (const AccessInfo &acc : nest->accesses) {
      if (seen.insert(MakeAccessKey(acc)).second)
        unique.push_back(acc);
    }
    nest->accesses = std::move(unique);
    nest->unique_accesses = nest->accesses.size();

    pair_cache_.clear();
    queries_ = 0;
    cache_hits_ = 0;
    budget_exceeded_ = false;

    VarSet loop_vars;
    for (const LoopInfo &loop : nest->loops)
      loop_vars.insert(loop.var);

    for (AccessInfo &acc : nest->accesses) {
      VarSet access_vars = loop_vars;
      for (const For &loop : acc.serial_loops)
        access_vars.insert(loop->loop_var);
      acc.map_class = ClassifyMap(acc.indices, access_vars);
      // Only a store can change the physical layout the consumer must observe.
      acc.layout_changing =
          acc.is_write && facts_.layout_vars.count(acc.buffer->data) != 0;
    }

    active_assumptions_.clear();
    VarSet relevant_vars;
    for (const LoopInfo &loop : nest->loops) {
      relevant_vars.insert(loop.var);
      AddUndefinedVars(loop.min, &relevant_vars);
      AddUndefinedVars(loop.extent, &relevant_vars);
    }
    for (const LoopInfo &loop : nest->outer_shells) {
      relevant_vars.insert(loop.var);
      AddUndefinedVars(loop.min, &relevant_vars);
      AddUndefinedVars(loop.extent, &relevant_vars);
    }
    for (const AccessInfo &acc : nest->accesses) {
      for (const For &loop : acc.serial_loops) {
        AddUndefinedVars(loop->min, &relevant_vars);
        AddUndefinedVars(loop->extent, &relevant_vars);
      }
      for (const PrimExpr &idx : acc.indices)
        AddUndefinedVars(idx, &relevant_vars);
      for (const GuardLiteral &guard : acc.guards)
        AddUndefinedVars(guard.condition, &relevant_vars);
      for (const PrimExpr &shape : acc.buffer->shape)
        AddUndefinedVars(shape, &relevant_vars);
      for (const PrimExpr &stride : acc.buffer->strides)
        AddUndefinedVars(stride, &relevant_vars);
      AddUndefinedVars(acc.buffer->elem_offset, &relevant_vars);
    }
    for (const PrimExpr &cond : assumptions_) {
      if (!IsSupportedAssumption(cond))
        continue;
      bool relevant = false;
      for (const Var &v : tirx::UndefinedVars(cond)) {
        if (relevant_vars.count(v)) {
          relevant = true;
          break;
        }
      }
      if (relevant)
        active_assumptions_.push_back(cond);
    }
    if (active_assumptions_.size() > 64)
      active_assumptions_.resize(64);

    if (!active_assumptions_.empty()) {
      arith::Analyzer check;
      EnterBindingsAndAssumptions_(&check);
      Var dummy("vsimd_consistency", DataType::Int(32));
      ++queries_;
      auto model = check.z3_prover.GetModel(Not(EQ(dummy, 0)));
      if (std::string(model).empty()) {
        nest->status = std::max(nest->status, Legality::kUnproven);
        nest->messages.push_back("inconsistent or unprovable assumptions");
        sync_counters();
        return;
      }
    }

    for (AccessInfo &acc : nest->accesses) {
      if (!acc.is_write)
        continue;
      if (acc.map_class == MapClass::kDataDependent ||
          acc.map_class == MapClass::kGeneral) {
        acc.injective = false;
        acc.injective_reason = acc.map_class == MapClass::kDataDependent
                                   ? "data-dependent store map; use an "
                                     "explicit scatter op/contract"
                                   : "unanalyzable store map";
        nest->status = std::max(nest->status, Legality::kUnproven);
        nest->messages.push_back("store " + std::string(acc.buffer->name) +
                                 ": " + acc.injective_reason);
        continue;
      }
      RelationResult result = CheckPair_(*nest, acc, acc);
      if (result.relation == Relation::kViolation) {
        acc.injective = false;
        acc.injective_reason = result.why;
        nest->status = std::max(nest->status, Legality::kIllegal);
        nest->messages.push_back(
            AccessName_(acc) + ": not injective across parallel instances: " +
            result.why + ConflictingStoreHint_(*nest, acc));
        sync_counters();
        return;
      }
      if (result.relation == Relation::kUnknown) {
        acc.injective = false;
        acc.injective_reason =
            result.why.empty() ? "not proven injective" : result.why;
        nest->status = std::max(nest->status, Legality::kUnproven);
        nest->messages.push_back("store " + std::string(acc.buffer->name) +
                                 ": " + acc.injective_reason);
        if (budget_exceeded_) {
          sync_counters();
          return;
        }
      }
    }

    for (size_t i = 0; i < nest->accesses.size(); ++i) {
      for (size_t j = i + 1; j < nest->accesses.size(); ++j) {
        const AccessInfo &a = nest->accesses[i];
        const AccessInfo &b = nest->accesses[j];
        if (!a.is_write && !b.is_write)
          continue;
        AliasKind alias = GetAlias_(a.buffer, b.buffer);
        if (alias.kind == AliasKind::kNoAlias)
          continue;
        if (a.map_class == MapClass::kDataDependent ||
            b.map_class == MapClass::kDataDependent ||
            a.map_class == MapClass::kGeneral ||
            b.map_class == MapClass::kGeneral) {
          nest->status = std::max(nest->status, Legality::kUnproven);
          nest->messages.push_back("pair " + AccessName_(a) + " / " +
                                   AccessName_(b) +
                                   ": unanalyzable access map");
          continue;
        }
        RelationResult result = CheckPair_(*nest, a, b);
        if (result.relation == Relation::kViolation) {
          nest->status = std::max(nest->status, Legality::kIllegal);
          nest->messages.push_back("pair " + AccessName_(a) + " / " +
                                   AccessName_(b) + ": " + result.why +
                                   "; review indices and guards to separate "
                                   "parallel instances, or use T.serial for "
                                   "intended sequential dependencies");
          sync_counters();
          return;
        }
        if (result.relation == Relation::kUnknown) {
          nest->status = std::max(nest->status, Legality::kUnproven);
          nest->messages.push_back(
              "pair " + AccessName_(a) + " / " + AccessName_(b) + ": " +
              (result.why.empty() ? "not proven independent" : result.why));
          if (budget_exceeded_) {
            sync_counters();
            return;
          }
        }
      }
    }
    sync_counters();
  }

  static std::string AccessName_(const AccessInfo &acc) {
    std::ostringstream os;
    os << (acc.is_write ? "W " : "R ") << std::string(acc.buffer->name) << "[";
    for (size_t i = 0; i < acc.indices.size(); ++i) {
      if (i)
        os << ",";
      os << acc.indices[i];
    }
    os << "]";
    return os.str();
  }

  // Explanation only, called after the dependence query proves a conflict.
  // An absent axis is not itself an error: guards and domains may restrict
  // ownership to a single parallel instance. No verdict depends on this hint.
  static std::string ConflictingStoreHint_(const NestInfo &nest,
                                           const AccessInfo &acc) {
    VarSet address_vars;
    AddUndefinedVars(LinearizeAccess(acc.buffer, acc.indices), &address_vars);
    std::ostringstream hint;
    bool missing_axis = false;
    for (const LoopInfo &loop : nest.loops) {
      const auto *extent = loop.extent.as<IntImmNode>();
      if (address_vars.count(loop.var) || (extent && extent->value <= 1))
        continue;
      missing_axis = true;
      hint << "; write address does not reference parallel axis '"
           << loop.var->name_hint << "'";
    }
    if (!missing_axis)
      return "; revise the output indices/guards to give parallel instances "
             "independent locations, or use T.serial for intended sequential "
             "dependencies";
    hint << "; if an axis is redundant, remove it after checking the intended "
            "semantics; if it represents sequential accumulation, use "
            "T.serial; otherwise revise the output indices/guards to give "
            "parallel instances independent locations (using the axis only "
            "in the stored value does not resolve a write conflict)";
    return hint.str();
  }

  bool TryFastMapInjectivity_(const NestInfo &nest,
                              const Array<PrimExpr> &indices,
                              RelationResult *result) {
    if (nest.loops.size() == 1) {
      const LoopInfo &loop = nest.loops[0];
      // A single component that uniquely determines the loop var makes the
      // whole tuple injective. This covers identity, a*i+b, (i, i%C), etc.
      for (const PrimExpr &idx : indices) {
        if (IsAffineInVar(idx, loop.var)) {
          result->relation = Relation::kSafe;
          return true;
        }
      }
      if (HasQuotientRemainderPair(indices, loop.var)) {
        result->relation = Relation::kSafe;
        return true;
      }
      if (indices.size() == 1) {
        bool is_mod = false;
        PrimExpr base;
        PrimExpr divisor;
        if (ExtractDivMod(indices[0], &is_mod, &base, &divisor) &&
            base.same_as(loop.var) &&
            TryDivModExtent(loop.extent, divisor, is_mod, &result->relation,
                            &result->why)) {
          return true;
        }
      }
      bool all_const = true;
      for (const PrimExpr &idx : indices) {
        if (idx.as<IntImmNode>() == nullptr) {
          all_const = false;
          break;
        }
      }
      if (all_const &&
          TryConstantExtent(loop.extent, &result->relation, &result->why)) {
        return true;
      }
    }

    // A permutation of the loop vars is injective.
    if (indices.size() == nest.loops.size()) {
      VarSet seen;
      bool ok = true;
      for (const PrimExpr &idx : indices) {
        const auto *node = idx.as<VarNode>();
        if (node == nullptr) {
          ok = false;
          break;
        }
        Var var = ffi::GetRef<Var>(node);
        bool is_loop = false;
        for (const LoopInfo &loop : nest.loops) {
          if (loop.var.same_as(var)) {
            is_loop = true;
            break;
          }
        }
        if (!is_loop || !seen.insert(var).second) {
          ok = false;
          break;
        }
      }
      if (ok && seen.size() == nest.loops.size()) {
        result->relation = Relation::kSafe;
        return true;
      }
    }
    return false;
  }

  RelationResult CheckPair_(const NestInfo &nest, const AccessInfo &a,
                            const AccessInfo &b) {
    PairKey key{MakeMapKey(a), MakeMapKey(b)};
    auto cached = pair_cache_.find(key);
    if (cached != pair_cache_.end()) {
      ++cache_hits_;
      return cached->second;
    }

    RelationResult result;
    PrimExpr off_a = LinearizeAccess(a.buffer, a.indices);
    PrimExpr off_b = LinearizeAccess(b.buffer, b.indices);
    if (!off_a.defined() || !off_b.defined()) {
      result = {Relation::kUnknown, "index/rank mismatch"};
      pair_cache_.emplace(key, result);
      return result;
    }

    // Retain the cheap proofs for the original pure-parallel case. A serial
    // counter is not a shared parameter: even i+k must use the general query.
    // Guarded domains may invalidate a syntactic counterexample, never a proof.
    if (a.serial_loops.empty() && b.serial_loops.empty() &&
        a.buffer.same_as(b.buffer) && a.buffer->strides.empty() &&
        tirx::ExprDeepEqual()(off_a, off_b) &&
        TryFastMapInjectivity_(nest, a.indices, &result) &&
        (result.relation == Relation::kSafe ||
         (a.guards.empty() && b.guards.empty() && nest.outer_shells.empty()))) {
      pair_cache_.emplace(key, result);
      return result;
    }
    result = RelationResult();

    if (budget_exceeded_ || queries_ >= kMaxQueriesPerNest) {
      budget_exceeded_ = true;
      result = {Relation::kUnknown, "query budget exceeded"};
      pair_cache_.emplace(key, result);
      return result;
    }

    arith::Analyzer analyzer;
    if (a.map_class == MapClass::kQuasiAffine ||
        b.map_class == MapClass::kQuasiAffine) {
      analyzer.z3_prover.SetRLimit(20000);
      analyzer.z3_prover.SetTimeoutMs(500);
    } else {
      analyzer.z3_prover.SetRLimit(50000);
      analyzer.z3_prover.SetTimeoutMs(2000);
    }
    EnterBindingsAndAssumptions_(&analyzer);
    // Enclosing serial loops are common context, not internal serial time.
    for (const LoopInfo &loop : nest.outer_shells) {
      analyzer.EnterConstraint(loop.var >= loop.min);
      analyzer.EnterConstraint(loop.var < loop.min + loop.extent);
    }
    Map<Var, PrimExpr> left, right;
    AddAccessDomain_(&analyzer, nest.loops, a.serial_loops, "_vsimd_a", &left);
    AddAccessDomain_(&analyzer, nest.loops, b.serial_loops, "_vsimd_b", &right);
    PrimExpr same_map = EQ(Substitute(off_a, left), Substitute(off_b, right));
    PrimExpr same_iter = Bool(true);
    for (const LoopInfo &loop : nest.loops) {
      same_iter = And(same_iter, EQ(left.at(loop.var), right.at(loop.var)));
    }
    PrimExpr violation = And(same_map, Not(same_iter));
    bool abstracted_guard = false;
    violation = And(violation, StableGuards_(nest, a, left, &abstracted_guard));
    violation =
        And(violation, StableGuards_(nest, b, right, &abstracted_guard));
    ++queries_;
    RelationResult query = ProveNoViolation_(&analyzer, violation);
    result.relation = query.relation;
    if (query.relation == Relation::kViolation && abstracted_guard) {
      // Dropping mutable predicates over-approximates execution. UNSAT is a
      // proof, but SAT is not a concrete witness for the original program.
      result = {Relation::kUnknown,
                "guard reads storage written in the region"};
    } else if (query.relation == Relation::kViolation) {
      std::string prefix = HasSymbolicExtent(nest) ? "symbolic extent; " : "";
      result.why = prefix + "cross-iteration RAW/WAR/WAW; " +
                   "counterexample: " + SanitizeModel(query.why);
    } else if (query.relation == Relation::kUnknown) {
      result.why = "Z3 returned unknown";
    }
    pair_cache_.emplace(key, result);
    return result;
  }

  // Three-way query: CanProve(true) => safe; a Z3 model of the violation =>
  // proven violation; otherwise unknown. GetModel(expr) solves !expr, so pass
  // Not(witness) to obtain a model of the violation. The dummy variable makes
  // an otherwise-empty witness model observable.
  RelationResult ProveNoViolation_(arith::Analyzer *analyzer,
                                   const PrimExpr &violation) {
    RelationResult result;
    if (analyzer->CanProve(Not(violation))) {
      result.relation = Relation::kSafe;
      return result;
    }
    Var dummy("vsimd_witness", DataType::Int(32));
    PrimExpr witness = And(violation, EQ(dummy, 0));
    auto model = analyzer->z3_prover.GetModel(Not(witness));
    std::string model_str = std::string(model);
    if (!model_str.empty()) {
      result.relation = Relation::kViolation;
      result.why = model_str;
    } else {
      result.relation = Relation::kUnknown;
    }
    return result;
  }

  void EnterBindingsAndAssumptions_(arith::Analyzer *analyzer) {
    for (const auto &binding : bindings_)
      analyzer->Bind(binding.first, binding.second);
    for (const auto &thread : thread_bounds_) {
      PrimExpr zero = IntImm(thread.second.dtype(), 0);
      analyzer->Bind(thread.first, Range::FromMinExtent(zero, thread.second));
    }
    for (const PrimExpr &cond : active_assumptions_)
      analyzer->EnterConstraint(cond);
  }

  // Instantiate the entire execution domain independently on each side.
  // Only parallel counters participate in the different-instance predicate.
  void AddAccessDomain_(arith::Analyzer *analyzer,
                        const std::vector<LoopInfo> &loops,
                        const std::vector<For> &serial_loops,
                        const std::string &suffix, Map<Var, PrimExpr> *subs) {
    for (const LoopInfo &loop : loops) {
      subs->Set(loop.var, Var(loop.var->name_hint + suffix, loop.var.dtype()));
    }
    for (const For &loop : serial_loops)
      subs->Set(loop->loop_var, Var(loop->loop_var->name_hint + suffix,
                                    loop->loop_var.dtype()));
    auto add_range = [&](const Var &var, const PrimExpr &min,
                         const PrimExpr &extent) {
      PrimExpr counter = subs->at(var);
      PrimExpr begin = Substitute(min, *subs);
      PrimExpr end = Substitute(min + extent, *subs);
      analyzer->EnterConstraint(counter >= begin);
      analyzer->EnterConstraint(counter < end);
    };
    for (const LoopInfo &loop : loops) {
      add_range(loop.var, loop.min, loop.extent);
    }
    for (const For &loop : serial_loops)
      add_range(loop->loop_var, loop->min, loop->extent);
  }

  PrimExpr StableGuards_(const NestInfo &nest, const AccessInfo &access,
                         const Map<Var, PrimExpr> &subs, bool *abstracted) {
    std::vector<GuardLiteral> stable;
    for (const GuardLiteral &guard : access.guards) {
      bool mutable_read = false;
      PostOrderVisit(guard.condition, [&](const ffi::ObjectRef &node) {
        const auto *load = node.as<BufferLoadNode>();
        if (load == nullptr)
          return;
        for (const AccessInfo &other : nest.accesses) {
          if (other.is_write &&
              GetAlias_(load->buffer, other.buffer).kind != AliasKind::kNoAlias)
            mutable_read = true;
        }
      });
      if (mutable_read)
        *abstracted = true;
      else
        stable.push_back(guard);
    }
    return GuardConjunction(stable, &subs);
  }

  AliasKind GetAlias_(const Buffer &a, const Buffer &b) {
    if (a->data.same_as(b->data))
      return {AliasKind::kSameStorage};
    bool a_param = facts_.params.count(a->data) != 0;
    bool b_param = facts_.params.count(b->data) != 0;
    bool a_alloc = facts_.allocs.count(a->data) != 0;
    bool b_alloc = facts_.allocs.count(b->data) != 0;
    bool a_decl = facts_.decls.count(a->data) != 0;
    bool b_decl = facts_.decls.count(b->data) != 0;
    // Declared views share their base allocation's data var in practice. If a
    // view introduced a fresh data var, be conservative.
    if (a_decl && b_decl)
      return {AliasKind::kMayAlias};
    if (a_alloc && b_alloc)
      return {AliasKind::kNoAlias};
    if (a_param && b_param) {
      if (facts_.non_restrict.count(a->data) != 0 ||
          facts_.non_restrict.count(b->data) != 0)
        return {AliasKind::kMayAlias};
      return {AliasKind::kNoAlias};
    }
    if ((a_param && b_alloc) || (a_alloc && b_param))
      return {AliasKind::kNoAlias};
    // tl.buffer_alias_map is a positive allocation-reuse contract, not a
    // no-alias proof. This first slice therefore does not use it to reduce
    // conservatism; unknown distinct data vars stay may-alias.
    return {AliasKind::kMayAlias};
  }

  std::unordered_map<PairKey, RelationResult, PairKeyHash> pair_cache_;
  size_t queries_{0};
  size_t cache_hits_{0};
  bool budget_exceeded_{false};
  const FunctionCollector &facts_;
  std::ostringstream *out_;
  std::vector<PrimExpr> assumptions_;
  std::vector<std::pair<Var, PrimExpr>> bindings_;
  std::vector<std::pair<Var, PrimExpr>> thread_bounds_;
  std::vector<PrimExpr> active_assumptions_;
  std::vector<LoopInfo> outer_serial_;
};

/* ------------------------------------------------------------------------- */
/* Top-level visitor                                                         */
/* ------------------------------------------------------------------------- */

class VFVisitor : public StmtExprVisitor {
public:
  std::ostringstream out;
  std::vector<VSimdMemoryRegionStatus> region_statuses;

  void Run(const PrimFunc &func) {
    FunctionCollector collector;
    collector.Collect(func);
    facts_ = &collector;
    VisitStmt(func->body);
  }

  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint == "SIMD_VF") {
      int vf = vf_index_++;
      VFAssumeCollector vf_assumes;
      vf_assumes(op->body);
      size_t saved_assumptions = assumptions_.size();
      assumptions_.insert(assumptions_.end(), vf_assumes.assumptions.begin(),
                          vf_assumes.assumptions.end());
      LegalityAnalyzer analyzer(*facts_, &out, assumptions_, bindings_,
                                thread_bounds_, outer_serial_);
      analyzer.AnalyzeVF(ffi::GetRef<SBlock>(op), vf);
      for (size_t i = 0; i < analyzer.nest_statuses.size(); ++i) {
        region_statuses.push_back(VSimdMemoryRegionStatus{
            vf, static_cast<int>(i), analyzer.nest_statuses[i],
            analyzer.nest_roots[i], analyzer.nest_reasons[i]});
      }
      assumptions_.resize(saved_assumptions);
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    size_t old_size = bindings_.size();
    for (const Stmt &stmt : op->seq) {
      if (const auto *bind = stmt.as<BindNode>()) {
        bindings_.emplace_back(bind->var, bind->value);
      } else {
        VisitStmt(stmt);
      }
    }
    bindings_.resize(old_size);
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kSerial) {
      outer_serial_.push_back(LoopInfo{op->loop_var, op->min, op->extent});
      StmtExprVisitor::VisitStmt_(op);
      outer_serial_.pop_back();
      return;
    }
    if (op->kind == ForKind::kThreadBinding) {
      thread_bounds_.emplace_back(op->loop_var, op->extent);
      StmtExprVisitor::VisitStmt_(op);
      thread_bounds_.pop_back();
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::tilelang_assume) {
      assumptions_.push_back(Downcast<PrimExpr>(op->node));
      StmtExprVisitor::VisitStmt_(op);
      assumptions_.pop_back();
      return;
    }
    if (op->attr_key == "thread_extent") {
      IterVar iv = Downcast<IterVar>(op->node);
      thread_bounds_.emplace_back(iv->var, op->value);
      StmtExprVisitor::VisitStmt_(op);
      thread_bounds_.pop_back();
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  const FunctionCollector *facts_{nullptr};
  int vf_index_{0};
  std::vector<LoopInfo> outer_serial_;
  std::vector<PrimExpr> assumptions_;
  std::vector<std::pair<Var, PrimExpr>> bindings_;
  std::vector<std::pair<Var, PrimExpr>> thread_bounds_;
};

} // namespace

VSimdMemoryAnalysis AnalyzeVSimdMemory(const PrimFunc &func) {
  VFVisitor visitor;
  visitor.Run(func);
  VSimdMemoryAnalysis analysis;
  analysis.report = visitor.out.str();
  if (analysis.report.empty())
    analysis.report = "no SIMD_VF blocks\n";
  analysis.regions = std::move(visitor.region_statuses);
  return analysis;
}

String AnalyzeVSimdMemoryLegality(const PrimFunc &func) {
  return String(AnalyzeVSimdMemory(func).report);
}

std::vector<VSimdMemoryRegionStatus>
AnalyzeVSimdMemoryRegions(const PrimFunc &func) {
  return AnalyzeVSimdMemory(func).regions;
}

} // namespace tl
} // namespace tvm
