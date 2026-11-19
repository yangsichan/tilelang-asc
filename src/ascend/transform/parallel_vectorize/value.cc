/*!
 * \file value.cc
 * \brief Value DAG construction (scalar expression -> per-lane vector value).
 */
#include "value.h"

#include <tvm/ir/expr.h>
#include <tvm/ir/repr.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "primitives.h"

namespace tvm {
namespace tl {
namespace parallel_vectorize {

namespace {

std::string DtypeOf(const PrimExpr &expr) {
  return ffi::DLDataTypeToString(expr.dtype()).operator std::string();
}

/*! \brief Element dtype -> vector register dtype (``_vec_of``). */
std::string VecOf(const std::string &dtype) {
  return VecDtype(ElemDtype(dtype));
}

/*! \brief ``-expr`` exactly as Python's ``PrimExpr.__neg__`` (multiply by -1).
 */
PrimExpr NegateExpr(const PrimExpr &expr) {
  return mul(expr, tirx::make_const(expr.dtype(), -1));
}

PrimExpr AndExpr(const PrimExpr &lhs, const PrimExpr &rhs) {
  return logical_and(lhs, rhs);
}

ValueNode PredicateNode(const ffi::Optional<ffi::ObjectRef> &predicate) {
  if (!predicate.has_value())
    return ValueNode();
  return Downcast<ValueNode>(*predicate);
}

std::optional<int64_t> ElementCount(const tirx::Buffer &buffer) {
  int64_t total = 1;
  for (const PrimExpr &dim : buffer->shape) {
    std::optional<int64_t> value = ConstInt(dim, nullptr);
    if (!value.has_value())
      return std::nullopt;
    total *= *value;
  }
  return total;
}

const Axis *AxisOfVar(const std::vector<Axis> &axes, const tirx::Var &var) {
  for (const Axis &axis : axes) {
    if (axis->var.same_as(var))
      return &axis;
  }
  return nullptr;
}

bool ProveGuard(const PrimExpr &expr, const ValueContext &ctx,
                arith::Analyzer *probe, int64_t begin, int64_t end) {
  // A predicate that depends on memory cannot be decided from the iteration
  // range alone: the probe analyzer only constrains the vector axis, so the
  // final Z3 fallback (one solver query per probe, ~0.7ms) is pure overhead and
  // always fails. Keeping the guard is conservative; skipping the query is not.
  if (ContainsBufferLoad(expr))
    return false;
  // One analyzer per probe batch: ``GuardFreeBounds`` asks the same predicate
  // once per chunk, and rebuilding the sub-analyzers (plus the Z3 client) for
  // every chunk dominated the proof. Re-binding is safe because every
  // sub-analyzer accepts an override and only the axis range changes.
  // ``Range`` in this fork is ``Range(begin, end)`` (not ``(min, extent)``).
  probe->Bind(ctx.vector_axis->var, Range(IntImm32(begin), IntImm32(end)),
              /*allow_override=*/true);
  return probe->CanProve(expr);
}

/*! \brief Lane-uniform sub-expression -> scalar; validates masked loads (P0-3).
 */
ValueNode Uniform(const PrimExpr &expr) {
  ValueSpec spec;
  spec.recipe = "uniform_expr";
  spec.dtype = DtypeOf(expr);
  spec.scalar = expr;
  spec.detail = "lane-uniform";
  return MakeValueNode(std::move(spec));
}

ValueNode ValueWithAccess(const std::string &recipe, const std::string &dtype,
                          std::optional<std::string> detail,
                          const AccessInfo &access, const ValueNode &guard,
                          int64_t free_until,
                          const std::optional<int64_t> &free_from,
                          std::vector<ValueNode> args = {}) {
  ValueSpec spec;
  spec.recipe = recipe;
  spec.dtype = dtype;
  spec.args = std::move(args);
  spec.access = ffi::Optional<AccessInfo>(access);
  spec.detail = detail.value_or("");
  spec.predicate = guard;
  spec.predicate_free_until = free_until;
  spec.predicate_free_from = free_from;
  return MakeValueNode(std::move(spec));
}

ValueNode SelectNode(const std::vector<ValueNode> &args,
                     const ValueContext &ctx, const PrimExpr &expr);

ValueNode Elementwise(const std::string &recipe, const PrimExpr &expr,
                      std::vector<ValueNode> args, const ValueContext &ctx);

ValueNode Compare(const std::vector<ValueNode> &args,
                  const std::string &predicate, const ValueContext &ctx,
                  const PrimExpr &expr);

ValueNode BoolNode(const std::string &recipe, const PrimExpr &expr,
                   const std::vector<ValueNode> &args, const ValueContext &ctx);

ValueNode FloorDivMod(const std::string &node_name, const PrimExpr &numer_expr,
                      const PrimExpr &denom_expr, const ValueContext &ctx,
                      const PrimExpr &whole_expr);

ValueNode BuildCall(const tirx::Call &call, const ValueContext &ctx);

ValueNode NegateUniform(const ValueNode &node) {
  ffi::Optional<PrimExpr> scalar;
  if (node->id == ValueRecipe::kImm || node->id == ValueRecipe::kUniformExpr) {
    scalar = node->scalar;
  } else if (node->id == ValueRecipe::kUniformLoad &&
             node->access.has_value()) {
    const AccessInfo &access = *node->access;
    scalar = ffi::Optional<PrimExpr>(
        tirx::BufferLoad(access->buffer, access->indices));
  } else if (node->id == ValueRecipe::kVbrc && !node->args.empty()) {
    return NegateUniform(Downcast<ValueNode>(node->args[0]));
  } else {
    return ValueNode();
  }
  if (!scalar.has_value())
    return ValueNode();
  ValueSpec spec;
  spec.recipe = "uniform_expr";
  spec.dtype = node->dtype;
  spec.scalar = NegateExpr(*scalar);
  spec.detail = "negated uniform";
  return MakeValueNode(std::move(spec));
}

ValueNode BroadcastImpl(const ValueNode &node) {
  if (node->id == ValueRecipe::kVbrc)
    return node;
  ValueSpec spec;
  spec.recipe = "vbrc";
  spec.dtype = VecOf(node->dtype);
  spec.args = {node};
  spec.detail = "broadcast uniform";
  return MakeValueNode(std::move(spec));
}

ValueNode Elementwise(const std::string &recipe, const PrimExpr &expr,
                      std::vector<ValueNode> args, const ValueContext &ctx) {
  bool all_uniform = true;
  for (const ValueNode &arg : args)
    all_uniform = all_uniform && arg->is_uniform;
  if (all_uniform)
    return UniformGuarded(expr, ctx);

  if (args.size() == 2) {
    bool lhs_uniform = args[0]->is_uniform;
    bool rhs_uniform = args[1]->is_uniform;
    // ``vector - scalar`` -> ``vadds(vector, -scalar)``: negating a scalar is a
    // scalar cost and saves one vdup. The reverse fold is NOT valid: vadds is
    // ``vector + scalar``, which would flip the sign of ``scalar - vector``.
    if (recipe == "vsub" && rhs_uniform && !lhs_uniform) {
      ValueNode negated = NegateUniform(args[1]);
      if (negated.defined()) {
        ValueSpec spec;
        spec.recipe = "vadds";
        spec.dtype = VecOf(DtypeOf(expr));
        spec.args = {args[0], negated};
        spec.detail = "vector - scalar -> vadds(-)";
        return MakeValueNode(std::move(spec));
      }
    }
    if (lhs_uniform || rhs_uniform) {
      std::string scalar_recipe = VecScalarBinary(recipe);
      if (!scalar_recipe.empty()) {
        std::vector<ValueNode> ordered =
            lhs_uniform ? std::vector<ValueNode>{args[1], args[0]} : args;
        ValueSpec spec;
        spec.recipe = scalar_recipe;
        spec.dtype = VecOf(DtypeOf(expr));
        spec.args = std::move(ordered);
        spec.detail = "scalar " + recipe + "->" + scalar_recipe;
        return MakeValueNode(std::move(spec));
      }
      if (lhs_uniform) {
        ValueSpec spec;
        spec.recipe = recipe;
        spec.dtype = VecOf(DtypeOf(expr));
        spec.args = {BroadcastImpl(args[0]), args[1]};
        return MakeValueNode(std::move(spec));
      }
      ValueSpec spec;
      spec.recipe = recipe;
      spec.dtype = VecOf(DtypeOf(expr));
      spec.args = {args[0], BroadcastImpl(args[1])};
      return MakeValueNode(std::move(spec));
    }
  }

  ValueSpec spec;
  spec.recipe = recipe;
  spec.dtype = VecOf(DtypeOf(expr));
  spec.args = std::move(args);
  return MakeValueNode(std::move(spec));
}

ValueNode Compare(const std::vector<ValueNode> &args,
                  const std::string &predicate, const ValueContext &ctx,
                  const PrimExpr &expr) {
  bool lhs_uniform = args[0]->is_uniform;
  bool rhs_uniform = args[1]->is_uniform;
  // A boolean operand is a predicate, not a number: ``vcmp``/``vcmps`` have no
  // mask-vs-mask form, so EQ/NE are lowered with the predicate truth table
  // (``a == b`` is ``not(a xor b)``, ``a != b`` is ``a xor b``). This runs
  // before the lane-uniform logic: a uniform bool pair has no numeric
  // broadcast.
  if (ElemDtype(args[0]->dtype) == "bool" &&
      ElemDtype(args[1]->dtype) == "bool") {
    if (predicate != "eq" && predicate != "ne") {
      throw Unsupported("comparison '" + predicate +
                        "' on mask operands is not supported (only eq/ne): a "
                        "predicate has no ordering");
    }
    ValueSpec xor_spec;
    xor_spec.recipe = "pxor";
    xor_spec.dtype = MaskDtype();
    xor_spec.args = args;
    xor_spec.detail = "mask " + predicate;
    ValueNode xored = MakeValueNode(std::move(xor_spec));
    if (predicate == "ne")
      return xored;
    ValueSpec not_spec;
    not_spec.recipe = "pnot";
    not_spec.dtype = MaskDtype();
    not_spec.args = {xored};
    not_spec.detail = "mask eq";
    return MakeValueNode(std::move(not_spec));
  }
  if (lhs_uniform && rhs_uniform) {
    // A compile-time predicate folds to the scalar form, which the emitter
    // turns into a constant mask. A predicate backed by memory stays in the
    // *vector* form: materializing it as a scalar comparison would put a scalar
    // instruction inside ``__simd_vf__``, which bisheng refuses ("Unsupported
    // scalar instruction in AIV loop").
    if (!ContainsBufferLoad(expr) && FreeVars(expr).empty()) {
      return UniformGuarded(expr, ctx);
    }
    // A scalar load has no masked form under a lazy guard: keep that check.
    RequireGuardableScalar(expr, ctx);
    // Neither side is per-lane, so ``vcmps`` cannot take a plain operand as its
    // vector: broadcast the left side (``vbrc`` -> ``vdup``) and compare
    // against the scalar right side. Same instruction shape as the mixed case,
    // but the scalar read stays an operand of ``vdup`` instead of becoming its
    // own instruction.
    ValueSpec spec;
    spec.recipe = "vcmps";
    spec.dtype = MaskDtype();
    spec.args = {Broadcast(args[0], ctx), args[1]};
    spec.detail = predicate;
    return MakeValueNode(std::move(spec));
  }
  // The scalar form requires (vector, scalar); swapping the operands flips the
  // predicate, otherwise ``i < j`` with a uniform ``i`` would become ``j < i``.
  if (lhs_uniform) {
    ValueSpec spec;
    spec.recipe = "vcmps";
    spec.dtype = MaskDtype();
    spec.args = {args[1], args[0]};
    spec.detail = CompareFlip(predicate);
    return MakeValueNode(std::move(spec));
  }
  if (rhs_uniform) {
    ValueSpec spec;
    spec.recipe = "vcmps";
    spec.dtype = MaskDtype();
    spec.args = {args[0], args[1]};
    spec.detail = predicate;
    return MakeValueNode(std::move(spec));
  }
  ValueSpec spec;
  spec.recipe = "vcmp";
  spec.dtype = MaskDtype();
  spec.args = args;
  spec.detail = predicate;
  return MakeValueNode(std::move(spec));
}

ValueNode BoolNode(const std::string &recipe, const PrimExpr &expr,
                   const std::vector<ValueNode> &args,
                   const ValueContext &ctx) {
  // Only a predicate whose operands are all *scalar* (imm / uniform expr) can
  // fold into one lane-uniform expression. A uniform comparison stays a vector
  // mask (``vcmps``) with no scalar form, so ``and``/``or``/``not`` compose it
  // with ``pand``/``por``/``pnot`` instead of falling back to scalar code.
  bool all_scalar = true;
  for (const ValueNode &arg : args) {
    all_scalar = all_scalar && arg->is_uniform && arg->scalar.has_value();
  }
  if (all_scalar)
    return UniformGuarded(expr, ctx);
  ValueSpec spec;
  spec.recipe = recipe;
  spec.dtype = MaskDtype();
  spec.args = args;
  return MakeValueNode(std::move(spec));
}

/*!
 * \brief Whether a *mask* node is lane-uniform.
 *
 * ``is_uniform`` tracks "no per-lane value" (a scalar), which is why a
 * broadcast
 * (``vbrc``) and a vector mask are both flagged false. A mask built only from
 * uniform leaves is still a lane-uniform *condition* for labelling purposes: a
 * ``vcmps`` over a broadcast produces the all-ones or the empty mask.
 */
bool LaneUniformMask(const ValueNode &node) {
  if (!node.defined())
    return false;
  if (node->is_uniform)
    return true; // scalar bool / imm
  if (node->id == ValueRecipe::kVbrc)
    return true; // broadcast scalar
  switch (node->id) {
  case ValueRecipe::kVcmp:
  case ValueRecipe::kVcmps:
  case ValueRecipe::kPnot:
  case ValueRecipe::kPand:
  case ValueRecipe::kPor:
  case ValueRecipe::kPxor:
    break;
  default:
    return false;
  }
  for (const ffi::ObjectRef &arg : node->args) {
    if (!LaneUniformMask(Downcast<ValueNode>(arg)))
      return false;
  }
  return true;
}

ValueNode SelectNode(const std::vector<ValueNode> &args,
                     const ValueContext &ctx, const PrimExpr &expr) {
  ValueSpec spec;
  spec.recipe = "vsel";
  spec.dtype = VecOf(DtypeOf(expr));
  spec.args = args;
  spec.detail =
      LaneUniformMask(args[0]) ? "uniform-cond select" : "predicated select";
  return MakeValueNode(std::move(spec));
}

ValueNode FloorDivMod(const std::string &node_name, const PrimExpr &numer_expr,
                      const PrimExpr &denom_expr, const ValueContext &ctx,
                      const PrimExpr &whole_expr) {
  ValueNode numer = BuildValue(numer_expr, ctx);
  ValueNode denom = BuildValue(denom_expr, ctx);
  if (numer->is_uniform && denom->is_uniform)
    return UniformGuarded(whole_expr, ctx);

  std::optional<int64_t> divisor = ConstInt(denom_expr, ctx.analyzer);
  if (!divisor.has_value() || *divisor <= 0 || !denom->is_uniform) {
    throw Unsupported(node_name +
                      " as a value operator needs a positive compile-time "
                      "constant divisor (got " +
                      PrintExpr(denom_expr) + ")");
  }
  if ((*divisor & (*divisor - 1)) != 0) {
    throw Unsupported(
        node_name + " by non-power-of-two constant " +
        std::to_string(*divisor) +
        " is not strength-reduced yet: "
        "the SIMD layer has no integer vdiv/vmod (design note §5.4(a))");
  }

  std::string elem = ElemDtype(numer->dtype);
  if (*divisor == 1) {
    if (node_name == "FloorDiv")
      return numer;
    ValueSpec spec;
    spec.recipe = "imm";
    spec.dtype = elem;
    spec.scalar = IntImm(DataType(ffi::StringToDLDataType(elem)), 0);
    return MakeValueNode(std::move(spec));
  }
  if (node_name == "FloorDiv") {
    int shift = 0;
    for (int64_t value = *divisor; value > 1; value >>= 1)
      ++shift; // bit_length - 1
    ValueSpec spec;
    spec.recipe = "vshrs";
    spec.dtype = VecDtype(elem);
    spec.args = {numer};
    spec.scalar = IntImm32(shift);
    spec.detail = "//" + std::to_string(*divisor);
    return MakeValueNode(std::move(spec));
  }
  ValueSpec remainder_spec;
  remainder_spec.recipe = "imm";
  remainder_spec.dtype = elem;
  remainder_spec.scalar =
      IntImm(DataType(ffi::StringToDLDataType(elem)), *divisor - 1);
  ValueNode remainder = MakeValueNode(std::move(remainder_spec));

  ValueSpec mask_spec;
  mask_spec.recipe = "vbrc";
  mask_spec.dtype = VecDtype(elem);
  mask_spec.args = {remainder};
  mask_spec.detail = "broadcast uniform";
  ValueNode mask_vec = MakeValueNode(std::move(mask_spec));

  ValueSpec spec;
  spec.recipe = "vand";
  spec.dtype = VecDtype(elem);
  spec.args = {numer, mask_vec};
  spec.detail = "%" + std::to_string(*divisor);
  return MakeValueNode(std::move(spec));
}

ValueNode BuildCall(const tirx::Call &call, const ValueContext &ctx) {
  std::string tail = CallTail(call);
  if (IsAtomicName(tail)) {
    throw Unsupported(
        "atomic op inside T.Parallel is not supported (design note §5)");
  }
  if (tail == "if_then_else" || tail == "select") {
    const PrimExpr &condition_expr = call->args[0];
    ValueNode condition = BuildValue(condition_expr, ctx);
    ValueContext then_ctx = ctx;
    ValueContext else_ctx = ctx;
    if (tail == "if_then_else") {
      auto cond_bounds = GuardFreeBounds(condition_expr, ctx);
      auto neg_bounds = GuardFreeBounds(logical_not(condition_expr), ctx);
      int64_t then_until = cond_bounds.first;
      std::optional<int64_t> then_from = cond_bounds.second;
      int64_t else_until = neg_bounds.first;
      std::optional<int64_t> else_from = neg_bounds.second;
      if (ctx.guard.defined()) {
        then_until = CombineFreeUntil(ctx.guard_free_until, then_until);
        then_from = CombineFreeFrom(ctx.guard_free_from, then_from);
        else_until = CombineFreeUntil(ctx.guard_free_until, else_until);
        else_from = CombineFreeFrom(ctx.guard_free_from, else_from);
      }
      then_ctx = WithGuard(ctx, AndPredicate(ctx.guard, condition), then_until,
                           then_from);
      else_ctx =
          WithGuard(ctx, AndPredicate(ctx.guard, NegatePredicate(condition)),
                    else_until, else_from);
    }
    std::vector<ValueNode> args = {condition,
                                   BuildValue(call->args[1], then_ctx),
                                   BuildValue(call->args[2], else_ctx)};
    bool all_uniform = true;
    for (const ValueNode &arg : args)
      all_uniform = all_uniform && arg->is_uniform;
    if (all_uniform) {
      // Wholly lane-uniform: keep the original scalar expression (``?:`` stays
      // lazy through codegen).
      return UniformGuarded(call, ctx);
    }
    return SelectNode(args, ctx, call);
  }
  std::string pred;
  if (ComparePredFor(tail, &pred) && call->args.size() == 2) {
    std::vector<ValueNode> args = {BuildValue(call->args[0], ctx),
                                   BuildValue(call->args[1], ctx)};
    return Compare(args, pred, ctx, call);
  }
  std::string bool_recipe;
  if (BoolBinaryRecipeFor(tail, &bool_recipe) && call->args.size() == 2) {
    std::vector<ValueNode> args = {BuildValue(call->args[0], ctx),
                                   BuildValue(call->args[1], ctx)};
    return BoolNode(bool_recipe, call, args, ctx);
  }
  if (BoolUnaryRecipeFor(tail, &bool_recipe) && call->args.size() == 1) {
    std::vector<ValueNode> args = {BuildValue(call->args[0], ctx)};
    return BoolNode(bool_recipe, call, args, ctx);
  }
  std::string recipe;
  if (BinaryRecipeFor(tail, &recipe) && call->args.size() == 2) {
    std::vector<ValueNode> args = {BuildValue(call->args[0], ctx),
                                   BuildValue(call->args[1], ctx)};
    return Elementwise(recipe, call, std::move(args), ctx);
  }
  if (UnaryRecipeFor(tail, &recipe) && call->args.size() == 1) {
    std::vector<ValueNode> args = {BuildValue(call->args[0], ctx)};
    return Elementwise(recipe, call, std::move(args), ctx);
  }
  throw Unsupported("unsupported intrinsic call in T.Parallel body: " + tail,
                    {}, "value.unsupported_intrinsic");
}

} // namespace

namespace {

/*! \brief The single recipe table: name + lowering family per recipe. */
const std::vector<ValueRecipeInfo> &ValueRecipeInfos() {
  static const std::vector<ValueRecipeInfo> infos = {
      {ValueRecipe::kImm, "imm", ValueRecipeFamily::kImm},
      {ValueRecipe::kLaneIndex, "lane_index", ValueRecipeFamily::kLaneIndex},
      {ValueRecipe::kVLoad, "vload", ValueRecipeFamily::kVLoad},
      {ValueRecipe::kVLoadStrided, "vload_strided",
       ValueRecipeFamily::kVLoadStrided},
      {ValueRecipe::kGather, "gather", ValueRecipeFamily::kGather},
      {ValueRecipe::kUniformLoad, "uniform_load", ValueRecipeFamily::kUniform},
      {ValueRecipe::kUniformExpr, "uniform_expr", ValueRecipeFamily::kUniform},
      {ValueRecipe::kVbrc, "vbrc", ValueRecipeFamily::kVbrc},
      {ValueRecipe::kVadd, "vadd", ValueRecipeFamily::kElementwiseBinary},
      {ValueRecipe::kVsub, "vsub", ValueRecipeFamily::kElementwiseBinary},
      {ValueRecipe::kVmul, "vmul", ValueRecipeFamily::kElementwiseBinary},
      {ValueRecipe::kVdiv, "vdiv", ValueRecipeFamily::kElementwiseBinary},
      {ValueRecipe::kVmax, "vmax", ValueRecipeFamily::kElementwiseBinary},
      {ValueRecipe::kVmin, "vmin", ValueRecipeFamily::kElementwiseBinary},
      {ValueRecipe::kVand, "vand", ValueRecipeFamily::kElementwiseBinary},
      {ValueRecipe::kVor, "vor", ValueRecipeFamily::kElementwiseBinary},
      {ValueRecipe::kVxor, "vxor", ValueRecipeFamily::kElementwiseBinary},
      {ValueRecipe::kVshl, "vshl", ValueRecipeFamily::kElementwiseBinary},
      {ValueRecipe::kVshr, "vshr", ValueRecipeFamily::kElementwiseBinary},
      {ValueRecipe::kVadds, "vadds", ValueRecipeFamily::kScalarVectorBinary},
      {ValueRecipe::kVmuls, "vmuls", ValueRecipeFamily::kScalarVectorBinary},
      {ValueRecipe::kVmaxs, "vmaxs", ValueRecipeFamily::kScalarVectorBinary},
      {ValueRecipe::kVmins, "vmins", ValueRecipeFamily::kScalarVectorBinary},
      {ValueRecipe::kVshrs, "vshrs", ValueRecipeFamily::kScalarVectorField},
      {ValueRecipe::kVshls, "vshls", ValueRecipeFamily::kScalarVectorField},
      {ValueRecipe::kVneg, "vneg", ValueRecipeFamily::kElementwiseUnary},
      {ValueRecipe::kVabs, "vabs", ValueRecipeFamily::kElementwiseUnary},
      {ValueRecipe::kVexp, "vexp", ValueRecipeFamily::kElementwiseUnary},
      {ValueRecipe::kVsqrt, "vsqrt", ValueRecipeFamily::kElementwiseUnary},
      {ValueRecipe::kVln, "vln", ValueRecipeFamily::kElementwiseUnary},
      {ValueRecipe::kVcmp, "vcmp", ValueRecipeFamily::kVcmp},
      {ValueRecipe::kVcmps, "vcmps", ValueRecipeFamily::kVcmps},
      {ValueRecipe::kVsel, "vsel", ValueRecipeFamily::kVsel},
      {ValueRecipe::kVcvt, "vcvt", ValueRecipeFamily::kVcvt},
      {ValueRecipe::kReinterpret, "reinterpret",
       ValueRecipeFamily::kReinterpret},
      {ValueRecipe::kPand, "pand", ValueRecipeFamily::kPredicateBinary},
      {ValueRecipe::kPor, "por", ValueRecipeFamily::kPredicateBinary},
      {ValueRecipe::kPxor, "pxor", ValueRecipeFamily::kPredicateBinary},
      {ValueRecipe::kPnot, "pnot", ValueRecipeFamily::kPredicateNot},
  };
  return infos;
}

/*! \brief Dense id -> info table (the emitter reads this per value node). */
const std::array<ValueRecipeInfo, kValueRecipeCount> &ValueRecipeInfoTable() {
  static const auto table = [] {
    std::array<ValueRecipeInfo, kValueRecipeCount> dense{};
    for (size_t i = 0; i < kValueRecipeCount; ++i) {
      dense[i] = {ValueRecipe::kUnknown, "unknown",
                  ValueRecipeFamily::kUnknown};
    }
    for (const ValueRecipeInfo &info : ValueRecipeInfos()) {
      dense[static_cast<size_t>(info.id)] = info;
    }
    return dense;
  }();
  return table;
}

} // namespace

const ValueRecipeInfo &ValueRecipeInfoOf(ValueRecipe recipe) {
  size_t index = static_cast<size_t>(recipe);
  const auto &table = ValueRecipeInfoTable();
  if (index >= table.size())
    return table[0];
  return table[index];
}

const char *ValueRecipeName(ValueRecipe recipe) {
  return ValueRecipeInfoOf(recipe).name;
}

ValueRecipe ValueRecipeFromName(const std::string &name) {
  static const std::unordered_map<std::string, ValueRecipe> table = [] {
    std::unordered_map<std::string, ValueRecipe> map;
    for (const ValueRecipeInfo &info : ValueRecipeInfos()) {
      map.emplace(info.name, info.id);
    }
    return map;
  }();
  auto it = table.find(name);
  return it == table.end() ? ValueRecipe::kUnknown : it->second;
}

bool IsElementwiseBinary(ValueRecipe recipe) {
  return ValueRecipeInfoOf(recipe).family ==
         ValueRecipeFamily::kElementwiseBinary;
}

bool IsElementwiseUnary(ValueRecipe recipe) {
  return ValueRecipeInfoOf(recipe).family ==
         ValueRecipeFamily::kElementwiseUnary;
}

bool IsScalarVectorBinary(ValueRecipe recipe) {
  return ValueRecipeInfoOf(recipe).family ==
         ValueRecipeFamily::kScalarVectorBinary;
}

bool IsUniformValueRecipe(ValueRecipe recipe) {
  // ``imm`` is a lane-uniform scalar but keeps its own lowering family.
  ValueRecipeFamily family = ValueRecipeInfoOf(recipe).family;
  return family == ValueRecipeFamily::kUniform ||
         family == ValueRecipeFamily::kImm;
}

ValueRecipe ScalarVariantOf(ValueRecipe recipe) {
  switch (recipe) {
  case ValueRecipe::kVadd:
    return ValueRecipe::kVadds;
  case ValueRecipe::kVmul:
    return ValueRecipe::kVmuls;
  case ValueRecipe::kVmax:
    return ValueRecipe::kVmaxs;
  case ValueRecipe::kVmin:
    return ValueRecipe::kVmins;
  default:
    return ValueRecipe::kUnknown;
  }
}

ValueRecipe VectorVariantOf(ValueRecipe recipe) {
  switch (recipe) {
  case ValueRecipe::kVadds:
    return ValueRecipe::kVadd;
  case ValueRecipe::kVmuls:
    return ValueRecipe::kVmul;
  case ValueRecipe::kVmaxs:
    return ValueRecipe::kVmax;
  case ValueRecipe::kVmins:
    return ValueRecipe::kVmin;
  default:
    return ValueRecipe::kUnknown;
  }
}

ValueNode MakeValueNode(ValueSpec spec) {
  auto node = ffi::make_object<ValueNodeNode>();
  node->recipe = spec.recipe;
  node->id = ValueRecipeFromName(spec.recipe);
  node->dtype = spec.dtype;
  for (const ValueNode &arg : spec.args)
    node->args.push_back(arg);
  node->access = std::move(spec.access);
  node->scalar = std::move(spec.scalar);
  node->detail = spec.detail;
  if (spec.predicate.defined()) {
    node->predicate = ffi::Optional<ffi::ObjectRef>(spec.predicate);
  }
  node->predicate_free_until = spec.predicate_free_until;
  if (spec.predicate_free_from.has_value()) {
    node->predicate_free_from =
        ffi::Optional<int64_t>(*spec.predicate_free_from);
  }

  bool is_uniform = false;
  if (node->id == ValueRecipe::kLaneIndex || node->id == ValueRecipe::kVbrc) {
    is_uniform = false;
  } else if (IsUniformValueRecipe(node->id)) {
    is_uniform = true;
  } else if (spec.args.empty()) {
    // Other leaf recipes (vload / vload_strided / gather) carry per-lane
    // values.
    is_uniform = false;
  } else {
    is_uniform = true;
    for (const ValueNode &arg : spec.args)
      is_uniform = is_uniform && arg->is_uniform;
  }
  node->is_uniform = is_uniform;
  return ValueNode(std::move(node));
}

ValueNode MakeValue(const std::string &recipe, const std::string &dtype,
                    std::vector<ValueNode> args, std::string detail) {
  ValueSpec spec;
  spec.recipe = recipe;
  spec.dtype = dtype;
  spec.args = std::move(args);
  spec.detail = std::move(detail);
  return MakeValueNode(std::move(spec));
}

ValueNode PredicateOf(const ValueNode &node) {
  if (!node.defined())
    return ValueNode();
  return PredicateNode(node->predicate);
}

std::vector<ValueNode> ArgsOf(const ValueNode &node) {
  std::vector<ValueNode> args;
  if (!node.defined())
    return args;
  args.reserve(node->args.size());
  for (const ffi::ObjectRef &arg : node->args)
    args.push_back(Downcast<ValueNode>(arg));
  return args;
}

ValueContext WithGuard(const ValueContext &ctx, const ValueNode &guard,
                       int64_t free_until, std::optional<int64_t> free_from) {
  if (!guard.defined())
    return ctx;
  ValueContext narrowed;
  narrowed.axes = ctx.axes;
  narrowed.vector_axis = ctx.vector_axis;
  narrowed.analyzer = ctx.analyzer;
  narrowed.guard = guard;
  narrowed.guard_free_until = free_until;
  narrowed.guard_free_from = free_from;
  narrowed.lanes = ctx.lanes;
  return narrowed;
}

int64_t CombineFreeUntil(int64_t lhs, int64_t rhs) {
  return std::min(lhs, rhs);
}

std::optional<int64_t> CombineFreeFrom(std::optional<int64_t> lhs,
                                       std::optional<int64_t> rhs) {
  if (!lhs.has_value() || !rhs.has_value())
    return std::nullopt;
  return std::max(*lhs, *rhs);
}

std::pair<int64_t, std::optional<int64_t>>
GuardFreeBounds(const PrimExpr &expr, const ValueContext &ctx) {
  std::optional<int64_t> extent =
      ConstInt(ctx.vector_axis->extent, ctx.analyzer);
  int64_t lanes = ctx.lanes;
  if (!extent.has_value() || lanes <= 0 || *extent <= lanes) {
    return {0, std::nullopt};
  }
  int64_t chunks = (*extent + lanes - 1) / lanes;
  arith::Analyzer probe;
  // Fast path: a guard that holds over the whole axis holds in every chunk, so
  // one full-range proof replaces the per-chunk scan. This is the common case
  // (``i < extent``-style predicates) and the scan costs one Z3 query per
  // chunk.
  if (ProveGuard(expr, ctx, &probe, 0, *extent)) {
    return {chunks - 1, 1};
  }
  int64_t until = 0;
  while (until + 1 < chunks &&
         ProveGuard(expr, ctx, &probe, 0, (until + 1) * lanes)) {
    ++until;
  }
  std::optional<int64_t> from;
  for (int64_t chunk = chunks - 1; chunk > 0; --chunk) {
    // The tail chunk covers ``[chunk * lanes, extent)``.
    if (!ProveGuard(expr, ctx, &probe, chunk * lanes, *extent))
      break;
    from = chunk;
  }
  return {until, from};
}

bool AccessProvablyInBounds(const AccessInfo &info, const ValueContext &ctx,
                            int64_t lo, std::optional<int64_t> hi) {
  if (info->indirect)
    return false;
  std::optional<int64_t> numel = ElementCount(info->buffer);
  std::optional<int64_t> base =
      ConstInt(info->buffer->elem_offset, ctx.analyzer);
  if (!numel.has_value() || !base.has_value())
    return false;

  arith::Analyzer analyzer;
  for (const tirx::Var &var : FreeVars(info->offset)) {
    const Axis *axis = AxisOfVar(ctx.axes, var);
    if (axis == nullptr)
      return false;
    std::optional<int64_t> extent = ConstInt((*axis)->extent, ctx.analyzer);
    if (!extent.has_value())
      return false;
    if ((*axis)->var.same_as(ctx.vector_axis->var)) {
      int64_t start = lo > 0 ? lo : 0;
      int64_t end = hi.has_value() ? std::min(*hi, *extent) : *extent;
      if (end <= start)
        return false;
      analyzer.Bind(var, Range(IntImm32(start), IntImm32(end)));
    } else {
      analyzer.Bind(var, Range(IntImm32(0), IntImm32(*extent)));
    }
  }
  arith::ConstIntBound bound =
      analyzer.const_int_bound(analyzer.Simplify(info->offset));
  return bound->min_value >= *base && bound->max_value < *base + *numel;
}

/*!
 * \brief Keep the scalar-load guard check for a lane-uniform expression.
 *
 * A scalar load has no masked form, so a lane-uniform expression used under a
 * lane-varying lazy conditional must have provably in-bounds accesses.
 */
void RequireGuardableScalar(const PrimExpr &expr, const ValueContext &ctx) {
  if (!ctx.guard.defined())
    return;
  std::vector<std::string> unpredicable;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
    const auto *load = node.as<tirx::BufferLoadNode>();
    if (load == nullptr || !unpredicable.empty())
      return;
    AccessInfo info;
    if (ctx.load_infos != nullptr) {
      auto it = ctx.load_infos->find(load);
      if (it != ctx.load_infos->end())
        info = it->second;
    }
    if (!info.defined()) {
      info = ClassifyAccess(load->buffer, load->indices, ctx.axes, ctx.analyzer,
                            /*is_store=*/false);
    }
    if (!AccessProvablyInBounds(info, ctx)) {
      unpredicable.push_back(load->buffer->name.operator std::string());
    }
  });
  if (!unpredicable.empty()) {
    throw Unsupported(
        "lane-uniform load of '" + unpredicable[0] +
        "' under a lane-varying lazy conditional cannot be predicated: a "
        "scalar load has no masked form (design note §1 Case II / P0-3)");
  }
}

ValueNode UniformGuarded(const PrimExpr &expr, const ValueContext &ctx) {
  RequireGuardableScalar(expr, ctx);
  return Uniform(expr);
}

ValueNode Broadcast(const ValueNode &node, const ValueContext &ctx) {
  (void)ctx;
  return BroadcastImpl(node);
}

ValueNode NegatePredicate(const ValueNode &node) {
  if (node->id == ValueRecipe::kPnot) {
    return Downcast<ValueNode>(node->args[0]);
  }
  // A scalar predicate folds into one negated lane-uniform expression; a vector
  // mask (``vcmps``) has no scalar form and is negated with ``pnot``.
  if (node->is_uniform && node->scalar.has_value()) {
    ValueSpec spec;
    spec.recipe = "uniform_expr";
    spec.dtype = node->dtype;
    spec.scalar = ffi::Optional<PrimExpr>(logical_not(*node->scalar));
    spec.detail = "negated predicate";
    return MakeValueNode(std::move(spec));
  }
  ValueSpec spec;
  spec.recipe = "pnot";
  spec.dtype = MaskDtype();
  spec.args = {node};
  return MakeValueNode(std::move(spec));
}

ValueNode AndPredicate(const ValueNode &lhs, const ValueNode &rhs) {
  if (!lhs.defined())
    return rhs;
  if (!rhs.defined())
    return lhs;
  if (lhs->is_uniform && rhs->is_uniform && lhs->scalar.has_value() &&
      rhs->scalar.has_value()) {
    ValueSpec spec;
    spec.recipe = "uniform_expr";
    spec.dtype = lhs->dtype;
    spec.scalar = ffi::Optional<PrimExpr>(AndExpr(*lhs->scalar, *rhs->scalar));
    spec.detail = "predicate conjunction";
    return MakeValueNode(std::move(spec));
  }
  ValueSpec spec;
  spec.recipe = "pand";
  spec.dtype = MaskDtype();
  spec.args = {lhs, rhs};
  return MakeValueNode(std::move(spec));
}

bool ReadsBuffer(const ValueNode &node, const tirx::Buffer &buffer) {
  if (node->access.has_value() && SameBuffer((*node->access)->buffer, buffer)) {
    return true;
  }
  if (node->scalar.has_value()) {
    bool found = false;
    tirx::PostOrderVisit(*node->scalar, [&](const ffi::ObjectRef &sub) {
      const auto *load = sub.as<tirx::BufferLoadNode>();
      if (load != nullptr && SameBuffer(load->buffer, buffer))
        found = true;
    });
    if (found)
      return true;
  }
  for (const ffi::ObjectRef &arg : node->args) {
    if (ReadsBuffer(Downcast<ValueNode>(arg), buffer))
      return true;
  }
  return false;
}

bool ExprHasSameAddressLoad(const PrimExpr &expr, const tirx::Buffer &buffer,
                            const ffi::Array<PrimExpr> &indices) {
  bool found = false;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
    if (found)
      return;
    const auto *load = node.as<tirx::BufferLoadNode>();
    if (load != nullptr && SameBuffer(load->buffer, buffer) &&
        SameIndices(load->indices, indices)) {
      found = true;
    }
  });
  return found;
}

ValueNode SameAddressLoad(const ValueNode &node, const tirx::Buffer &buffer,
                          const ffi::Array<PrimExpr> &indices) {
  if ((node->id == ValueRecipe::kVLoad ||
       node->id == ValueRecipe::kUniformLoad) &&
      node->access.has_value() && SameBuffer((*node->access)->buffer, buffer) &&
      SameIndices((*node->access)->indices, indices)) {
    return node;
  }
  for (const ffi::ObjectRef &arg : node->args) {
    ValueNode found =
        SameAddressLoad(Downcast<ValueNode>(arg), buffer, indices);
    if (found.defined())
      return found;
  }
  return ValueNode();
}

bool ReadsOtherAddress(const ValueNode &node, const tirx::Buffer &buffer,
                       const ffi::Array<PrimExpr> &indices) {
  if ((node->id == ValueRecipe::kVLoad ||
       node->id == ValueRecipe::kUniformLoad) &&
      node->access.has_value() && SameBuffer((*node->access)->buffer, buffer) &&
      !SameIndices((*node->access)->indices, indices)) {
    return true;
  }
  for (const ffi::ObjectRef &arg : node->args) {
    if (ReadsOtherAddress(Downcast<ValueNode>(arg), buffer, indices))
      return true;
  }
  return false;
}

ValueNode BuildValue(const PrimExpr &expr, const ValueContext &ctx) {
  if (const auto *load = expr.as<tirx::BufferLoadNode>()) {
    AccessInfo info;
    if (ctx.load_infos != nullptr) {
      auto it = ctx.load_infos->find(load);
      if (it != ctx.load_infos->end())
        info = it->second;
    }
    if (!info.defined()) {
      info = ClassifyAccess(load->buffer, load->indices, ctx.axes, ctx.analyzer,
                            /*is_store=*/false);
    }
    return LoadNode(info, ctx);
  }
  if (const auto *var = expr.as<tirx::VarNode>()) {
    if (ctx.vector_axis->var.same_as(ffi::GetRef<tirx::Var>(var))) {
      // The vector axis variable *is* the lane id: vci.
      ValueSpec spec;
      spec.recipe = "lane_index";
      spec.dtype = DtypeOf(expr);
      spec.scalar = expr;
      spec.detail = "vci(offset=1)";
      return MakeValueNode(std::move(spec));
    }
    for (const Axis &axis : ctx.axes) {
      if (axis->var.same_as(ffi::GetRef<tirx::Var>(var))) {
        // A parallel variable of another axis is lane-uniform (outer loop var).
        ValueSpec spec;
        spec.recipe = "imm";
        spec.dtype = DtypeOf(expr);
        spec.scalar = expr;
        spec.detail = "outer-axis-var";
        return MakeValueNode(std::move(spec));
      }
    }
    ValueSpec spec;
    spec.recipe = "imm";
    spec.dtype = DtypeOf(expr);
    spec.scalar = expr;
    spec.detail = "capture";
    return MakeValueNode(std::move(spec));
  }
  if (expr.as<IntImmNode>() != nullptr || expr.as<FloatImmNode>() != nullptr) {
    ValueSpec spec;
    spec.recipe = "imm";
    spec.dtype = DtypeOf(expr);
    spec.scalar = expr;
    return MakeValueNode(std::move(spec));
  }
  if (const auto *call = expr.as<tirx::CallNode>()) {
    return BuildCall(ffi::GetRef<tirx::Call>(call), ctx);
  }

  if (const auto *floor_div = expr.as<tirx::FloorDivNode>()) {
    return FloorDivMod("FloorDiv", floor_div->a, floor_div->b, ctx, expr);
  }
  if (const auto *floor_mod = expr.as<tirx::FloorModNode>()) {
    return FloorDivMod("FloorMod", floor_mod->a, floor_mod->b, ctx, expr);
  }
  {
    // Comparison nodes: EQ/NE/LT/LE/GT/GE (all CmpOpNode<T> derivations).
    PrimExpr cmp_a;
    PrimExpr cmp_b;
    bool is_cmp = true;
    if (const auto *op = expr.as<tirx::EQNode>()) {
      cmp_a = op->a;
      cmp_b = op->b;
    } else if (const auto *op = expr.as<tirx::NENode>()) {
      cmp_a = op->a;
      cmp_b = op->b;
    } else if (const auto *op = expr.as<tirx::LTNode>()) {
      cmp_a = op->a;
      cmp_b = op->b;
    } else if (const auto *op = expr.as<tirx::LENode>()) {
      cmp_a = op->a;
      cmp_b = op->b;
    } else if (const auto *op = expr.as<tirx::GTNode>()) {
      cmp_a = op->a;
      cmp_b = op->b;
    } else if (const auto *op = expr.as<tirx::GENode>()) {
      cmp_a = op->a;
      cmp_b = op->b;
    } else {
      is_cmp = false;
    }
    if (is_cmp) {
      std::string pred;
      ComparePredFor(ToLower(TypeKeyTail(expr)), &pred);
      std::vector<ValueNode> args = {BuildValue(cmp_a, ctx),
                                     BuildValue(cmp_b, ctx)};
      return Compare(args, pred, ctx, expr);
    }
  }
  {
    std::string recipe;
    std::string node_l = ToLower(TypeKeyTail(expr));
    if (BinaryRecipeFor(node_l, &recipe)) {
      PrimExpr a;
      PrimExpr b;
      if (const auto *op = expr.as<tirx::AddNode>()) {
        a = op->a;
        b = op->b;
      } else if (const auto *op = expr.as<tirx::SubNode>()) {
        a = op->a;
        b = op->b;
      } else if (const auto *op = expr.as<tirx::MulNode>()) {
        a = op->a;
        b = op->b;
      } else if (const auto *op = expr.as<tirx::DivNode>()) {
        a = op->a;
        b = op->b;
      } else if (const auto *op = expr.as<tirx::MaxNode>()) {
        a = op->a;
        b = op->b;
      } else if (const auto *op = expr.as<tirx::MinNode>()) {
        a = op->a;
        b = op->b;
      } else {
        recipe.clear();
      }
      if (!recipe.empty()) {
        std::vector<ValueNode> args = {BuildValue(a, ctx), BuildValue(b, ctx)};
        return Elementwise(recipe, expr, std::move(args), ctx);
      }
    }
  }
  {
    std::string bool_recipe;
    std::string node_l = ToLower(TypeKeyTail(expr));
    if (BoolBinaryRecipeFor(node_l, &bool_recipe)) {
      if (const auto *op = expr.as<tirx::AndNode>()) {
        std::vector<ValueNode> args = {BuildValue(op->a, ctx),
                                       BuildValue(op->b, ctx)};
        return BoolNode(bool_recipe, expr, args, ctx);
      }
      if (const auto *op = expr.as<tirx::OrNode>()) {
        std::vector<ValueNode> args = {BuildValue(op->a, ctx),
                                       BuildValue(op->b, ctx)};
        return BoolNode(bool_recipe, expr, args, ctx);
      }
    }
    if (BoolUnaryRecipeFor(node_l, &bool_recipe)) {
      if (const auto *op = expr.as<tirx::NotNode>()) {
        std::vector<ValueNode> args = {BuildValue(op->a, ctx)};
        return BoolNode(bool_recipe, expr, args, ctx);
      }
    }
  }

  if (const auto *select = expr.as<tirx::SelectNode>()) {
    // ``Select`` is *eager*: both arms are evaluated, so the condition does not
    // restrict their validity; only the current guard propagates.
    std::vector<ValueNode> args = {BuildValue(select->condition, ctx),
                                   BuildValue(select->true_value, ctx),
                                   BuildValue(select->false_value, ctx)};
    bool all_uniform = true;
    for (const ValueNode &arg : args)
      all_uniform = all_uniform && arg->is_uniform;
    if (all_uniform)
      return UniformGuarded(expr, ctx);
    return SelectNode(args, ctx, expr);
  }
  if (const auto *cast = expr.as<tirx::CastNode>()) {
    ValueNode arg = BuildValue(cast->value, ctx);
    if (arg->is_uniform)
      return UniformGuarded(expr, ctx);
    // A same-width integer signedness change is bit-identical: the register is
    // reinterpreted, not converted. ``vcvt`` has no such pair (and should not:
    // it is a value conversion), so this is the ``vinterpret_cast`` analogue.
    const DataType src_dtype = cast->value.dtype();
    const DataType dst_dtype = expr.dtype();
    const bool integer_bits = (src_dtype.is_int() || src_dtype.is_uint()) &&
                              (dst_dtype.is_int() || dst_dtype.is_uint());
    const bool same_width = src_dtype.bits() == dst_dtype.bits() &&
                            src_dtype.lanes() == dst_dtype.lanes();
    // The target element type: vcvt decides the result lane count.
    ValueSpec spec;
    spec.recipe = integer_bits && same_width ? "reinterpret" : "vcvt";
    spec.dtype = VecOf(DtypeOf(expr));
    spec.args = {arg};
    spec.detail = ElemDtype(DtypeOf(expr));
    return MakeValueNode(std::move(spec));
  }
  throw Unsupported("unsupported value expression: " + TypeKeyTail(expr) +
                    " (" + PrintExpr(expr) + ")");
}

ValueNode IndexValue(const AccessInfo &info, const ValueContext &ctx) {
  ValueContext index_ctx;
  index_ctx.axes = ctx.axes;
  index_ctx.vector_axis = ctx.vector_axis;
  index_ctx.analyzer = ctx.analyzer;
  index_ctx.guard = ctx.guard;
  index_ctx.guard_free_until = ctx.guard_free_until;
  index_ctx.guard_free_from = ctx.guard_free_from;
  index_ctx.lanes = ctx.lanes;
  return BuildValue(info->offset, index_ctx);
}

ValueNode LoadNode(const AccessInfo &info, const ValueContext &ctx) {
  // A ``local.fragment`` buffer is a V-core register tile, not an addressable
  // memory operand: fragment vectorization is not supported, so the load is
  // rejected instead of emitting an un-compilable ``vld`` (P1 already keeps
  // fragment nests out of the vectorization path).
  if (StorageClassOf(info->buffer) == StorageClass::kFragment) {
    throw Unsupported("load from '" +
                      info->buffer->name.operator std::string() +
                      "' reads a local.fragment register tile; fragment "
                      "vectorization is not supported");
  }
  // Loads inside lazy conditionals are only legal when the condition holds.
  // Provably safe addresses keep a plain vld; otherwise the guard is attached
  // to the load. ``free_until`` / ``free_from`` bound the guard-free leading /
  // trailing chunk ranges, where the emitter can drop the mask again.
  ValueNode guard = ctx.guard;
  int64_t free_until = ctx.guard_free_until;
  std::optional<int64_t> free_from = ctx.guard_free_from;
  if (guard.defined()) {
    if (AccessProvablyInBounds(info, ctx)) {
      guard = ValueNode();
      free_until = 0;
      free_from = std::nullopt;
    } else {
      if (free_until <= 0 ||
          !AccessProvablyInBounds(info, ctx, /*lo=*/0,
                                  /*hi=*/free_until * ctx.lanes)) {
        free_until = 0;
      }
      if (!free_from.has_value() ||
          !AccessProvablyInBounds(info, ctx, /*lo=*/*free_from * ctx.lanes)) {
        free_from = std::nullopt;
      }
    }
  }
  std::string buffer_elem_dtype =
      ffi::DLDataTypeToString(info->buffer->dtype.element_of())
          .operator std::string();
  std::string vec_dtype_value = VecDtype(buffer_elem_dtype);

  AxisClass classified = AxisClassOf(info, ctx.vector_axis);
  switch (classified.kind) {
  case AccessClass::kContiguous:
    return ValueWithAccess("vload", vec_dtype_value, std::nullopt, info, guard,
                           free_until, free_from);
  case AccessClass::kUniform: {
    if (guard.defined()) {
      throw Unsupported(
          "lane-uniform load of '" + info->buffer->name.operator std::string() +
          "' under a lane-varying lazy conditional cannot be predicated: a "
          "scalar load has no masked form (design note §1 Case II / P0-3)");
    }
    std::string scalar_dtype =
        ffi::DLDataTypeToString(info->buffer->dtype).operator std::string();
    return ValueWithAccess("uniform_load", scalar_dtype,
                           std::string("lane-uniform"), info, ValueNode(), 0,
                           std::nullopt);
  }
  case AccessClass::kStrided: {
    if (classified.stride.has_value() && *classified.stride < 0) {
      // Negative stride (Delta <= -2) cannot be expressed as an unsigned lane
      // ramp; the index vector channel handles it (D8).
      return ValueWithAccess(
          "gather", vec_dtype_value,
          "stride=" + std::to_string(*classified.stride) + " (index vector)",
          info, guard, free_until, free_from, {IndexValue(info, ctx)});
    }
    return ValueWithAccess("vload_strided", vec_dtype_value,
                           "stride=" +
                               std::to_string(classified.stride.value_or(0)),
                           info, guard, free_until, free_from);
  }
  case AccessClass::kPermute:
  case AccessClass::kPiecewise:
  case AccessClass::kIndirect:
    return ValueWithAccess("gather", vec_dtype_value,
                           AccessClassName(classified.kind), info, guard,
                           free_until, free_from, {IndexValue(info, ctx)});
  case AccessClass::kTrueRandom:
    break;
  }
  std::string reason = info->reason.empty()
                           ? std::string("unknown reason")
                           : info->reason.operator std::string();
  throw Unsupported("non-affine access with respect to axis '" +
                    ctx.vector_axis->name.operator std::string() +
                    "': the per-lane index vector is not constructible (" +
                    reason + ")");
}

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm
