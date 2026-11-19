/*!
 * \file access.cc
 * \brief SCoP modelling: linearization, finite differences, access classes.
 */
#include "access.h"

#include <tvm/ir/repr.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <string>
#include <unordered_map>

#include "primitives.h"

namespace tvm {
namespace tl {
namespace parallel_vectorize {

namespace {

const std::unordered_map<int, std::string> &AccessClassNames() {
  static const std::unordered_map<int, std::string> table = {
      {static_cast<int>(AccessClass::kContiguous), "contiguous"},
      {static_cast<int>(AccessClass::kUniform), "uniform"},
      {static_cast<int>(AccessClass::kStrided), "strided"},
      {static_cast<int>(AccessClass::kPermute), "permute"},
      {static_cast<int>(AccessClass::kPiecewise), "piecewise"},
      {static_cast<int>(AccessClass::kIndirect), "indirect"},
      {static_cast<int>(AccessClass::kTrueRandom), "true_random"},
  };
  return table;
}

const std::unordered_map<int, std::string> &StorageClassNames() {
  static const std::unordered_map<int, std::string> table = {
      {static_cast<int>(StorageClass::kShared), "shared"},
      {static_cast<int>(StorageClass::kFragment), "fragment"},
      {static_cast<int>(StorageClass::kPrivate), "private"},
  };
  return table;
}

bool IsZeroExpr(const PrimExpr &expr) {
  std::optional<int64_t> value = ConstInt(expr, nullptr);
  return value.has_value() && *value == 0;
}

std::vector<PrimExpr> CompactStrides(const ffi::Array<PrimExpr> &shape) {
  std::vector<PrimExpr> strides(shape.size(),
                                PrimExpr(IntImm(DataType::Int(32), 1)));
  PrimExpr acc = IntImm(DataType::Int(32), 1);
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = acc;
    acc = acc * shape[i];
  }
  return strides;
}

} // namespace

const std::string &AccessClassName(AccessClass kind) {
  return AccessClassNames().at(static_cast<int>(kind));
}

AccessClass AccessClassFromName(const std::string &name) {
  for (const auto &kv : AccessClassNames()) {
    if (kv.second == name)
      return static_cast<AccessClass>(kv.first);
  }
  TVM_FFI_ICHECK(false) << "unknown access class '" << name << "'";
  return AccessClass::kTrueRandom;
}

const std::string &StorageClassName(StorageClass kind) {
  return StorageClassNames().at(static_cast<int>(kind));
}

StorageClass StorageClassFromName(const std::string &name) {
  for (const auto &kv : StorageClassNames()) {
    if (kv.second == name)
      return static_cast<StorageClass>(kv.first);
  }
  TVM_FFI_ICHECK(false) << "unknown storage class '" << name << "'";
  return StorageClass::kShared;
}

StorageClass StorageClassOf(const tirx::Buffer &buffer) {
  std::string scope = buffer.scope().operator std::string();
  if (scope == "local.fragment")
    return StorageClass::kFragment;
  if (scope == "local" || scope == "local.var")
    return StorageClass::kPrivate;
  return StorageClass::kShared;
}

Axis MakeAxis(tirx::Var var, PrimExpr extent, int64_t index,
              tirx::ForKind kind) {
  auto node = ffi::make_object<AxisNode>();
  node->var = std::move(var);
  node->extent = std::move(extent);
  node->index = index;
  node->kind = static_cast<int64_t>(kind);
  node->name = node->var->name_hint;
  node->is_parallel = kind == tirx::ForKind::kParallel;
  return Axis(std::move(node));
}

AxisClass AxisClassOf(const AccessInfo &info, const Axis &axis) {
  auto it = info->per_axis.find(axis->index);
  if (it == info->per_axis.end())
    return AxisClass{AccessClass::kTrueRandom, std::nullopt};
  return it->second;
}

std::optional<int64_t> ConstInt(const PrimExpr &expr,
                                arith::Analyzer *analyzer) {
  if (const auto *int_imm = expr.as<IntImmNode>()) {
    return int_imm->value;
  }
  if (analyzer != nullptr) {
    PrimExpr simplified = analyzer->Simplify(expr);
    if (const auto *int_imm = simplified.as<IntImmNode>()) {
      return int_imm->value;
    }
    arith::ConstIntBound bound = analyzer->const_int_bound(simplified);
    if (bound->min_value == bound->max_value) {
      return bound->min_value;
    }
  }
  return std::nullopt;
}

std::vector<PrimExpr> BufferStrides(const tirx::Buffer &buffer) {
  std::vector<PrimExpr> strides;
  if (!buffer->strides.empty()) {
    for (const PrimExpr &stride : buffer->strides)
      strides.push_back(stride);
    return strides;
  }
  return CompactStrides(buffer->shape);
}

PrimExpr LinearElementOffset(const tirx::Buffer &buffer,
                             const ffi::Array<PrimExpr> &indices,
                             arith::Analyzer *analyzer) {
  std::vector<PrimExpr> strides = BufferStrides(buffer);
  if (strides.size() != indices.size()) {
    throw Unsupported("access rank " + std::to_string(indices.size()) +
                      " does not match buffer rank " +
                      std::to_string(strides.size()));
  }
  PrimExpr expr = buffer->elem_offset;
  for (size_t i = 0; i < indices.size(); ++i) {
    PrimExpr term = indices[i] * strides[i];
    expr = IsZeroExpr(expr) ? term : expr + term;
  }
  return analyzer->Simplify(expr);
}

PrimExpr FiniteDifference(const PrimExpr &offset, const tirx::Var &var,
                          arith::Analyzer *analyzer) {
  PrimExpr shifted = tirx::Substitute(
      offset, [&](const tirx::Var &candidate) -> ffi::Optional<PrimExpr> {
        if (candidate.same_as(var)) {
          return ffi::Optional<PrimExpr>(var + IntImm(DataType::Int(32), 1));
        }
        return ffi::Optional<PrimExpr>();
      });
  return analyzer->Simplify(shifted - offset);
}

AxisClass ClassifyDelta(const PrimExpr &delta, arith::Analyzer *analyzer) {
  std::optional<int64_t> value = ConstInt(delta, analyzer);
  if (!value.has_value())
    return AxisClass{AccessClass::kPiecewise, std::nullopt};
  if (*value == 0)
    return AxisClass{AccessClass::kUniform, static_cast<int64_t>(0)};
  if (*value == 1)
    return AxisClass{AccessClass::kContiguous, static_cast<int64_t>(1)};
  if (*value == -1)
    return AxisClass{AccessClass::kPermute, static_cast<int64_t>(-1)};
  return AxisClass{AccessClass::kStrided, *value};
}

std::optional<std::string> IndexVectorReason(const PrimExpr &expr,
                                             arith::Analyzer *analyzer) {
  std::vector<std::string> reasons;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
    std::string name = TypeKeyTail(node);
    if (node.as<tirx::BufferLoadNode>() != nullptr) {
      reasons.push_back("index contains a BufferLoad");
      return;
    }
    if (name == "FloorDiv" || name == "FloorMod") {
      const PrimExpr &divisor_expr =
          name == "FloorDiv"
              ? static_cast<const tirx::FloorDivNode *>(node.get())->b
              : static_cast<const tirx::FloorModNode *>(node.get())->b;
      std::optional<int64_t> divisor = ConstInt(divisor_expr, analyzer);
      if (!divisor.has_value()) {
        reasons.push_back(name +
                          " needs a compile-time constant divisor (got " +
                          PrintExpr(divisor_expr) + ")");
      } else if (*divisor <= 0 || (*divisor & (*divisor - 1)) != 0) {
        reasons.push_back(name + " by " + std::to_string(*divisor) +
                          " is not a power of two");
      }
      return;
    }
    if (!IsIndexValueOp(name)) {
      reasons.push_back("unsupported index operator " + name);
    }
  });
  if (reasons.empty())
    return std::nullopt;
  return reasons.front();
}

bool ContainsBufferLoad(const PrimExpr &expr) {
  bool found = false;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
    if (node.as<tirx::BufferLoadNode>() != nullptr)
      found = true;
  });
  return found;
}

std::vector<tirx::Var> FreeVars(const PrimExpr &expr) {
  std::vector<tirx::Var> variables;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
    if (const auto *var = node.as<tirx::VarNode>()) {
      variables.push_back(ffi::GetRef<tirx::Var>(var));
    }
  });
  return variables;
}

AccessInfo ClassifyAccess(const tirx::Buffer &buffer,
                          const ffi::Array<PrimExpr> &indices,
                          const std::vector<Axis> &axes,
                          arith::Analyzer *analyzer, bool is_store) {
  bool indirect = false;
  for (const PrimExpr &index : indices) {
    if (ContainsBufferLoad(index)) {
      indirect = true;
      break;
    }
  }
  PrimExpr offset = LinearElementOffset(buffer, indices, analyzer);

  auto node = ffi::make_object<AccessInfoNode>();
  node->buffer = buffer;
  node->indices = indices;
  node->offset = offset;
  node->is_store = is_store;
  node->indirect = indirect;
  node->storage = ffi::String(StorageClassName(StorageClassOf(buffer)));
  AccessInfo info(std::move(node));

  std::optional<std::string> piecewise_reason =
      indirect ? std::nullopt : IndexVectorReason(offset, analyzer);

  for (const Axis &axis : axes) {
    PrimExpr delta = FiniteDifference(offset, axis->var, analyzer);
    AxisClass classified = ClassifyDelta(delta, analyzer);
    if (indirect) {
      // An indirect subscript that does *not* vary along the axis is just a
      // lane-uniform scalar address (``c[b[0]] = 1.0``); only a varying one has
      // to go through the per-lane index-vector channel.
      if (classified.kind != AccessClass::kUniform) {
        classified = AxisClass{AccessClass::kIndirect, std::nullopt};
      }
      info->per_axis[axis->index] = classified;
      continue;
    }
    if (classified.kind == AccessClass::kPiecewise &&
        piecewise_reason.has_value()) {
      classified = AxisClass{AccessClass::kTrueRandom, std::nullopt};
      info->reason = *piecewise_reason;
    } else if (classified.kind == AccessClass::kStrided &&
               classified.stride.has_value() && *classified.stride < 0 &&
               piecewise_reason.has_value()) {
      // A negative stride must go through the index-vector channel; when that
      // is not constructible, reject instead of silently using a strided load
      // (D8).
      classified = AxisClass{AccessClass::kTrueRandom, std::nullopt};
      info->reason = *piecewise_reason;
    }
    info->per_axis[axis->index] = classified;
  }
  return info;
}

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm
