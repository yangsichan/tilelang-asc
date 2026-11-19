/*!
 * \file value.h
 * \brief Value DAG: lift scalar expressions into per-lane vector value nodes.
 *
 * C++ counterpart of ``parallel_vectorize/value.py``. The decision is again
 * purely the finite difference along the vector axis:
 *
 *   Delta_v == 1  -> vload             (contiguous)
 *   Delta_v == 0  -> uniform_load      (scalar read + vdup broadcast)
 *   Delta_v == k  -> vload_strided     (vgather2 with a stride ramp)
 *   indirect      -> gather            (per-lane index vector)
 *   loop var      -> lane_index        (vci)
 *
 * ``ValueNode`` is instruction-selection agnostic: ``recipe`` is a primitive
 * name, the concrete ``T.simd.*`` instruction is chosen by the emit layer.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_VALUE_H_
#define TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_VALUE_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/tirx/expr.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "access.h"

namespace tvm {
namespace tl {
namespace parallel_vectorize {

class ValueNodeNode;

/*!
 * \brief Typed form of ``ValueNodeNode::recipe``.
 *
 * ``recipe`` stays a string at the FFI boundary (tests read it by name), but
 * every consumer inside the compiler dispatches on this id: a ``switch`` over a
 * closed enum instead of a chain of ``std::string`` comparisons. Unknown
 * spellings map to ``kUnknown`` and are rejected by the emitter, exactly as the
 * string form was.
 */
enum class ValueRecipe : uint8_t {
  kUnknown = 0,
  // Leaves.
  kImm,
  kLaneIndex,
  kVLoad,
  kVLoadStrided,
  kGather,
  kUniformLoad,
  kUniformExpr,
  kVbrc,
  // Elementwise binary (vector, vector).
  kVadd,
  kVsub,
  kVmul,
  kVdiv,
  kVmax,
  kVmin,
  kVand,
  kVor,
  kVxor,
  kVshl,
  kVshr,
  // Scalar-vector binary (vector, scalar).
  kVadds,
  kVmuls,
  kVmaxs,
  kVmins,
  kVshrs,
  kVshls,
  // Elementwise unary.
  kVneg,
  kVabs,
  kVexp,
  kVsqrt,
  kVln,
  // Compare / select / convert.
  kVcmp,
  kVcmps,
  kVsel,
  kVcvt,
  /*! \brief Same-width integer signedness change: a register bit reinterpret.
   *
   * Bit-identical, so it must not go through ``vcvt`` (which would be a value
   * conversion). Emitted as ``tirx.reinterpret`` and costs no instruction.
   */
  kReinterpret,
  // Predicate algebra.
  kPand,
  kPor,
  kPxor,
  kPnot,
};

/*! \brief Number of recipe ids (the last enumerator plus one). */
inline constexpr size_t kValueRecipeCount =
    static_cast<size_t>(ValueRecipe::kPnot) + 1;

/*!
 * \brief How a recipe is lowered (one ``switch`` case per family, not per op).
 *
 * Recipes that share a lowering share a family, so adding e.g. a new
 * elementwise binary op needs no change here or in the emitter: only a row in
 * the recipe table (and the ``T.simd.*`` op table in emit.cc).
 */
enum class ValueRecipeFamily : uint8_t {
  kUnknown = 0,
  // Leaves with a bespoke lowering.
  kImm,
  kLaneIndex,
  kVLoad,
  kVLoadStrided,
  kGather,
  // Lane-uniform broadcast forms.
  kUniform,
  kVbrc,
  // Elementwise families (dispatched through the ``simd_*`` op table).
  kElementwiseBinary,
  kElementwiseUnary,
  kScalarVectorBinary, // the scalar is ``args[1]``
  kScalarVectorField,  // the scalar is ``node->scalar`` (shift recipes)
  // Structured value ops.
  kVcmp,
  kVcmps,
  kVsel,
  kVcvt,
  kReinterpret,
  // Predicate algebra.
  kPredicateBinary,
  kPredicateNot,
};

/*! \brief Static description of one recipe. */
struct ValueRecipeInfo {
  ValueRecipe id;
  const char *name;
  ValueRecipeFamily family;
};

/*! \brief The description of ``recipe`` (``kUnknown`` for unknown ids). */
const ValueRecipeInfo &ValueRecipeInfoOf(ValueRecipe recipe);

/*! \brief The FFI spelling of a typed recipe. */
const char *ValueRecipeName(ValueRecipe recipe);
/*! \brief Parse an FFI recipe name; ``kUnknown`` when it is not a recipe. */
ValueRecipe ValueRecipeFromName(const std::string &name);

/*! \brief Whether the recipe is a vector-vector elementwise op. */
bool IsElementwiseBinary(ValueRecipe recipe);
/*! \brief Whether the recipe is an elementwise unary op. */
bool IsElementwiseUnary(ValueRecipe recipe);
/*! \brief Whether the recipe is a vector-scalar elementwise op. */
bool IsScalarVectorBinary(ValueRecipe recipe);
/*! \brief Whether the recipe carries a lane-uniform (scalar) value. */
bool IsUniformValueRecipe(ValueRecipe recipe);
/*! \brief ``vadd -> vadds``; ``kUnknown`` when there is no scalar variant. */
ValueRecipe ScalarVariantOf(ValueRecipe recipe);
/*! \brief ``vadds -> vadd``; ``kUnknown`` when not a scalar-vector recipe. */
ValueRecipe VectorVariantOf(ValueRecipe recipe);

/*!
 * \brief One value node: a primitive recipe plus its operands.
 *
 * ``predicate`` is the *load guard* (P0-3): the value is only valid on lanes
 * where the predicate holds, which happens inside lazy ``if_then_else`` arms.
 * ``predicate_free_until`` / ``predicate_free_from`` give the leading/trailing
 * chunk ranges where the guard is provably true, so the emitter can drop the
 * masked gather in those blocks.
 */
class ValueNodeNode : public ffi::Object {
public:
  ffi::String recipe;
  /*! \brief Typed ``recipe``; not reflected (the string stays the FFI value).
   */
  ValueRecipe id{ValueRecipe::kUnknown};
  ffi::String dtype;
  ffi::Array<ffi::ObjectRef> args; // ValueNode elements
  ffi::Optional<AccessInfo> access;
  ffi::Optional<PrimExpr> scalar;
  ffi::String detail;
  ffi::Optional<ffi::ObjectRef> predicate; // ValueNode
  int64_t predicate_free_until{0};
  ffi::Optional<int64_t> predicate_free_from;
  bool is_uniform{false};

  static constexpr const bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.parallel_vectorize.ValueNode",
                                    ValueNodeNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ValueNodeNode>()
        .def_ro("recipe", &ValueNodeNode::recipe)
        .def_ro("dtype", &ValueNodeNode::dtype)
        .def_ro("args", &ValueNodeNode::args)
        .def_ro("access", &ValueNodeNode::access)
        .def_ro("scalar", &ValueNodeNode::scalar)
        .def_ro("detail", &ValueNodeNode::detail)
        .def_ro("predicate", &ValueNodeNode::predicate)
        .def_ro("predicate_free_until", &ValueNodeNode::predicate_free_until)
        .def_ro("predicate_free_from", &ValueNodeNode::predicate_free_from)
        .def_ro("is_uniform", &ValueNodeNode::is_uniform);
  }
};

class ValueNode : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ValueNode, ffi::ObjectRef,
                                             ValueNodeNode);
};

/*! \brief Value node fields that the constructors of Python set by keyword. */
struct ValueSpec {
  std::string recipe;
  std::string dtype;
  std::vector<ValueNode> args;
  ffi::Optional<AccessInfo> access;
  ffi::Optional<PrimExpr> scalar;
  std::string detail;
  ValueNode predicate; // undefined == None
  int64_t predicate_free_until{0};
  std::optional<int64_t> predicate_free_from;
};

/*! \brief Build a value node (``is_uniform`` is derived, exactly as Python). */
ValueNode MakeValueNode(ValueSpec spec);

/*! \brief Shorthand for a value node without access / guard fields. */
ValueNode MakeValue(const std::string &recipe, const std::string &dtype,
                    std::vector<ValueNode> args = {}, std::string detail = "");

/*! \brief The guard node of a value (undefined when there is none). */
ValueNode PredicateOf(const ValueNode &node);
/*! \brief Operands of a value node as typed value nodes. */
std::vector<ValueNode> ArgsOf(const ValueNode &node);

/*! \brief Load classification memo keyed by the statement's load nodes. */
using LoadInfoCache = std::map<const tirx::BufferLoadNode *, AccessInfo>;

/*! \brief Per-lane expression context (axes, vector axis, analyzer, guard). */
struct ValueContext {
  std::vector<Axis> axes;
  Axis vector_axis;
  arith::Analyzer *analyzer{nullptr};
  ValueNode guard; // undefined == no guard
  int64_t guard_free_until{0};
  std::optional<int64_t> guard_free_from;
  int64_t lanes{0};
  /*! \brief Load classification memo of the enclosing chunk.
   *
   * Keys are the ``BufferLoadNode`` pointers of the statements being planned,
   * so ``BuildValue`` reuses the classification already computed for axis
   * ranking instead of running ``ClassifyAccess`` per load (per candidate
   * axis). A miss falls back to classifying on the spot. */
  const LoadInfoCache *load_infos{nullptr};
};

/*! \brief Narrow the context to a new guard (no-op when ``guard`` is None). */
ValueContext WithGuard(const ValueContext &ctx, const ValueNode &guard,
                       int64_t free_until = 0,
                       std::optional<int64_t> free_from = std::nullopt);

/*! \brief Prefix conjunction of two guard-free ranges: the smaller one. */
int64_t CombineFreeUntil(int64_t lhs, int64_t rhs);
/*! \brief Suffix conjunction: ``None`` when either side cannot be proven. */
std::optional<int64_t> CombineFreeFrom(std::optional<int64_t> lhs,
                                       std::optional<int64_t> rhs);

/*! \brief ``[0, until)`` / ``[from, chunks)`` chunk ranges where ``expr``
 * holds. */
std::pair<int64_t, std::optional<int64_t>>
GuardFreeBounds(const PrimExpr &expr, const ValueContext &ctx);

/*! \brief Lift a scalar expression into a vector value node. */
ValueNode BuildValue(const PrimExpr &expr, const ValueContext &ctx);

/*! \brief Build the per-lane absolute element offset vector of an access. */
ValueNode IndexValue(const AccessInfo &info, const ValueContext &ctx);

/*! \brief Turn a classified access into a load node. */
ValueNode LoadNode(const AccessInfo &info, const ValueContext &ctx);

/*! \brief Whether a lane-uniform scalar is a safe operand under a lazy guard.
 */
ValueNode UniformGuarded(const PrimExpr &expr, const ValueContext &ctx);

/*!
 * \brief The guard check alone: throws when a lane-uniform expression used
 * under a lazy guard has a scalar load that cannot be masked.
 *
 * Used where the value itself must stay in the vector form (``vcmps``) instead
 * of folding to a scalar.
 */
void RequireGuardableScalar(const PrimExpr &expr, const ValueContext &ctx);

/*! \brief Broadcast a lane-uniform value to a vector (``vbrc``). */
ValueNode Broadcast(const ValueNode &node, const ValueContext &ctx);

/*! \brief ``pnot`` / scalar negate. */
ValueNode NegatePredicate(const ValueNode &node);
/*! \brief ``pand`` / scalar and; undefined operand means "no guard". */
ValueNode AndPredicate(const ValueNode &lhs, const ValueNode &rhs);

/*! \brief Whether the value DAG reads ``buffer`` anywhere. */
bool ReadsBuffer(const ValueNode &node, const tirx::Buffer &buffer);
/*! \brief Same-buffer, same-address load node inside the DAG (RMW old value).
 */
ValueNode SameAddressLoad(const ValueNode &node, const tirx::Buffer &buffer,
                          const ffi::Array<PrimExpr> &indices);
/*! \brief Whether the raw expression reads ``buffer`` at the same address.
 *
 * The expression-level counterpart of :func:`SameAddressLoad`, shared by the
 * planner's pre-normalization checks so the "reads its own destination"
 * predicate exists exactly once. */
bool ExprHasSameAddressLoad(const PrimExpr &expr, const tirx::Buffer &buffer,
                            const ffi::Array<PrimExpr> &indices);
/*! \brief Whether the DAG reads ``buffer`` at a *different* address. */
bool ReadsOtherAddress(const ValueNode &node, const tirx::Buffer &buffer,
                       const ffi::Array<PrimExpr> &indices);

/*!
 * \brief Whether the access provably stays inside the buffer for all lanes.
 *
 * Only used to decide whether a load inside a lazy conditional needs a guard:
 * provably safe loads keep emitting plain ``vld``.
 */
bool AccessProvablyInBounds(const AccessInfo &info, const ValueContext &ctx,
                            int64_t lo = 0,
                            std::optional<int64_t> hi = std::nullopt);

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_VALUE_H_
