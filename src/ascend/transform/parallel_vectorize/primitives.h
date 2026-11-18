/*!
 * \file primitives.h
 * \brief ParallelVectorize primitive fact tables, hardware constants and the
 * single rejection type used by P2 (plan) and P3 (emit).
 *
 * This is the C++ counterpart of ``tilelang/ascend/analysis/parallel_vectorize/
 * primitives.py``. Every table here is a fact about the SIMD ISA (recipe names,
 * reduction primitives, mask widths); P2/P3 must never invent a new one.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_PRIMITIVES_H_
#define TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_PRIMITIVES_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <exception>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace parallel_vectorize {

/*!
 * \brief The only rejection type of the parallel_vectorize vectorizer.
 *
 * Mirrors ``primitives.Unsupported``: ``message`` is the one-line summary that
 * tests match with ``pytest.raises(Unsupported, match=...)``, and
 * ``diagnostics`` carries *all* blockers collected during the run (ordered, the
 * list itself is part of the contract). Never replace it with ``LOG(FATAL)``.
 */
class Unsupported : public std::exception {
public:
  /*! \brief Code of a rejection that has not been classified yet. */
  static constexpr const char *kDefaultCode = "unsupported";

  explicit Unsupported(std::string message,
                       std::vector<std::string> diagnostics = {},
                       std::string code = kDefaultCode)
      : message_(std::move(message)), diagnostics_(std::move(diagnostics)),
        code_(std::move(code)) {}

  const char *what() const noexcept override { return message_.c_str(); }
  const std::string &message() const { return message_; }
  const std::vector<std::string> &diagnostics() const { return diagnostics_; }
  /*! \brief Stable reason code; free-form ``message`` must not be matched on.
   */
  const std::string &code() const { return code_; }

private:
  std::string message_;
  std::vector<std::string> diagnostics_;
  std::string code_;
};

/*! \brief Hardware predicate dtype: every ``T.simd.*`` predicate is boolx256.
 */
inline const std::string &MaskDtype() {
  static const std::string value = "boolx256";
  return value;
}

/*! \brief Execution mode of every ``T.simd.*`` data instruction. */
inline const std::string &ModeZeroing() {
  static const std::string value = "MODE_ZEROING";
  return value;
}

/*!
 * \brief Predicate-predicate ops (pand/por/pnot) use the finest b8 granularity.
 *
 * A fixed b32 (64 lanes) would silently truncate f16 (128 lanes) / int8 (256
 * lanes) predicates; extra predicate bits beyond the source comparison are zero
 * under MODE_ZEROING, so over-covering is harmless (Python R1).
 */
constexpr int kPredicateMaskBits = 8;

/*! \brief Reduction emission strategy threshold (see emit.cc). */
constexpr int kReduceUnrollLimit = 4;

/*! \brief Upper bound on guard chunk cuts (plan._GUARD_CUT_LIMIT). */
constexpr int kGuardCutLimit = 6;

/*! \brief One reduction primitive fact: TIR op -> element op -> lane reduce. */
struct ReducePrimitive {
  std::string tir_op; // Add / Max / Min
  std::string op;     // vadd / vmax / vmin
  std::string recipe; // vcadd / vcmax / vcmin
  std::string scalar; // vadds / vmaxs / vmins
  std::function<PrimExpr(const std::string &)>
      identity; // element dtype -> identity
};

/*! \brief The reduction table, the single source of truth (design note §2 Case
 * II). */
const std::vector<ReducePrimitive> &ReducePrimitives();

/*! \brief ``_REDUCE_OPS[tir_op]`` -> element op; false when unknown. */
bool ReduceOpFor(const std::string &tir_op, std::string *op);
/*! \brief ``_REDUCE_RECIPE[element_op]`` -> cross-lane recipe. */
bool ReduceRecipeFor(const std::string &op, std::string *recipe);
/*! \brief ``_REDUCE_ELEMENT[recipe]`` -> element op. */
bool ReduceElementFor(const std::string &recipe, std::string *op);
/*! \brief ``_REDUCE_SCALAR_RECIPE[element_op]`` -> scalar-vector op. */
bool ReduceScalarRecipeFor(const std::string &op, std::string *scalar);
/*!
 * \brief Unit element of a reduction recipe for a concrete element dtype.
 *
 * Deriving the identity from the element type (instead of hard-coding f32
 * extremes) is what keeps int32 / float16 min/max reductions emittable.
 */
PrimExpr ReduceIdentity(const std::string &recipe, const std::string &elem);

/*! \brief Elementwise binary recipes keyed by lower-cased TIR/Op tail. */
bool BinaryRecipeFor(const std::string &name_lower, std::string *recipe);
/*! \brief Elementwise unary recipes keyed by lower-cased TIR/Op tail. */
bool UnaryRecipeFor(const std::string &name_lower, std::string *recipe);
/*! \brief Comparison predicate keyed by lower-cased TIR/Op tail (lt/le/...). */
bool ComparePredFor(const std::string &name_lower, std::string *pred);
/*! \brief Boolean binary recipe keyed by lower-cased tail (pand/por). */
bool BoolBinaryRecipeFor(const std::string &name_lower, std::string *recipe);
/*! \brief Boolean unary recipe keyed by lower-cased tail (pnot). */
bool BoolUnaryRecipeFor(const std::string &name_lower, std::string *recipe);
/*! \brief Vector -> scalar-vector variant (vadd -> vadds); empty when none. */
std::string VecScalarBinary(const std::string &recipe);
/*! \brief Predicate flip when the operands of a comparison are swapped. */
std::string CompareFlip(const std::string &pred);
/*! \brief Whether the tail is an atomic op (``tl.atomic_add`` ...). */
bool IsAtomicName(const std::string &tail);
/*! \brief Whether ``node_name`` is an allowed index-vector operator. */
bool IsIndexValueOp(const std::string &node_name);

/*! \brief Element bits table (``_ELEM_BITS`` in language/simd.py). */
bool ElemBits(const std::string &elem, int *bits);
/*! \brief Vector register lanes for an element dtype (``_vreg_lanes``). */
int VRegLanes(const std::string &elem);
/*! \brief ``<elem>x<lanes>`` vector dtype (``_vec_dtype``). */
std::string VecDtype(const std::string &elem);
/*!
 * \brief Parse a dtype string, memoized.
 *
 * ``ffi::StringToDLDataType`` re-parses on every call and the emitter builds a
 * dozen vector dtypes per segment; the spelling set is tiny and closed.
 */
DataType DataTypeOf(const std::string &dtype_string);
/*! \brief Vector dtype of an element name, memoized. */
DataType VecDataType(const std::string &elem);
/*! \brief Element dtype of a DataType, memoized. */
DataType ElemDataType(const DataType &dtype);
/*!
 * \brief CCE predicate width for an element dtype (``_mask_bits``).
 *
 * Throws ``Unsupported`` with the ``ValueError`` text of the Python helper; the
 * plan layer re-wraps it (see ``RequireSupportedElement``) so the composed
 * message stays byte-identical.
 */
int MaskBits(const std::string &dtype_like);

/*! \brief Strip the lane suffix of a dtype string (``_elem_dtype``). */
std::string ElemDtype(const std::string &dtype_like);
/*! \brief Strip the lane suffix of a DataType. */
std::string ElemDtype(const DataType &dtype);
/*! \brief ``int32`` IntImm (``_int``). */
PrimExpr IntImm32(int64_t value);
/*! \brief ``lhs.data.same_as(rhs.data)`` (``_same_buffer``). */
bool SameBuffer(const tirx::Buffer &lhs, const tirx::Buffer &rhs);
/*!
 * \brief Structural index equality (``_same_indices``).
 *
 * Pointer identity would misreport ``IntImm(0)`` against ``IntImm(0)`` from a
 * different object, breaking the ``acc[0] = acc[0] + x`` self-read check.
 */
bool SameIndices(const ffi::Array<PrimExpr> &lhs,
                 const ffi::Array<PrimExpr> &rhs);

/*! \brief Fold statements into a SeqStmt (empty -> undefined, one -> itself).
 */
tirx::Stmt SeqOf(const ffi::Array<tirx::Stmt> &stmts);

/*! \brief Lower-cased tail of a Call's op name (``_call_tail``). */
std::string CallTail(const tirx::Call &call);

/*! \brief Lower-case for ASCII strings, mirroring Python ``str.lower``. */
std::string ToLower(std::string value);

/*!
 * \brief Tail of an object's type key (``tirx.For`` -> ``For``).
 *
 * Node names must match Python's ``type(node).__name__`` in diagnostics and in
 * the index-operator whitelist, so the type key is the only stable source.
 */
std::string TypeKeyTail(const ffi::ObjectRef &node);

/*! \brief ``Print`` an expression / statement the way Python ``str()`` does. */
std::string PrintExpr(const PrimExpr &expr);
std::string PrintStmt(const tirx::Stmt &stmt);

/*! \brief ForKind name used by diagnostics (``_for_kind_name``). */
std::string ForKindName(tirx::ForKind kind);

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_PRIMITIVES_H_
