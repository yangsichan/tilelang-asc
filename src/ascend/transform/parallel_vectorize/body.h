/*!
 * \file body.h
 * \brief Body normalization: multi-statement / branches -> ordered stores.
 *
 * C++ counterpart of ``parallel_vectorize/body.py``. The rules are
 * operator-agnostic:
 *
 *  * ``SeqStmt``       -> flatten in order (§3);
 *  * ``BufferStore``   -> one store;
 *  * single-arm ``if`` -> predicated store (``mask &= cond``);
 *  * two-arm ``if``    -> lazy ``if_then_else(cond, a, b)`` when both arms
 * write the same location;
 *  * everything else   -> explicit ``Unsupported`` (no silent downgrade).
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_BODY_H_
#define TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_BODY_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace parallel_vectorize {

/*! \brief Where to write, what to write, and an optional predicate. */
class StoreSpecNode : public ffi::Object {
public:
  tirx::Buffer buffer;
  ffi::Array<PrimExpr> indices;
  PrimExpr value;
  ffi::Optional<PrimExpr> mask;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.parallel_vectorize.StoreSpec",
                                    StoreSpecNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StoreSpecNode>()
        .def_ro("buffer", &StoreSpecNode::buffer)
        .def_ro("indices", &StoreSpecNode::indices)
        .def_ro("value", &StoreSpecNode::value)
        .def_ro("mask", &StoreSpecNode::mask);
  }
};

class StoreSpec : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StoreSpec, ffi::ObjectRef,
                                             StoreSpecNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StoreSpecNode);
};

/*! \brief Build a store spec. */
StoreSpec MakeStoreSpec(tirx::Buffer buffer, ffi::Array<PrimExpr> indices,
                        PrimExpr value,
                        ffi::Optional<PrimExpr> mask = std::nullopt);

/*!
 * \brief Scalar aliases introduced by ``Bind`` (``acc = a[i,j] * 2``).
 *
 * Only real value bindings participate; a Bind whose value is a Call (buffer
 * declaration / type constructor) is skipped, so §3 scalar temporaries fold
 * into the final expression.
 */
struct AliasMap {
  std::unordered_map<const tirx::VarNode *, PrimExpr> values;
  bool empty() const { return values.empty(); }
};

AliasMap CollectAliases(const tirx::Stmt &stmt);

/*! \brief Inline aliases (bounded passes to avoid a cyclic Bind chain). */
PrimExpr ApplyAliases(const PrimExpr &expr, const AliasMap &aliases);

/*!
 * \brief A scalar ``T.serial`` shell the plan keeps around its chunk program.
 *
 * A shell the planner can neither absorb as a vector/reduce axis nor unroll
 * (last-write-wins bodies, runtime trip counts, ...) is emitted as a real loop
 * with the vectorized chunk program inside it, which is the shape the scalar
 * source has anyway.
 */
struct ShellLoop {
  tirx::Var var;
  PrimExpr min;
  PrimExpr extent;
};

/*! \brief Would :func:`NormalizeBody` keep this serial loop as a real scalar
 * shell? */
TVM_DLL bool IsScalarShell(const tirx::ForNode *loop);

/*! \brief Normalize a body into an ordered list of store specs. */
std::vector<StoreSpec> NormalizeBody(const tirx::Stmt &stmt,
                                     const AliasMap &aliases,
                                     std::vector<ShellLoop> *shells = nullptr);

/*! \brief Normalize with an inherited predicate (recursion entry point). */
void NormalizeBodyInto(const tirx::Stmt &stmt, const AliasMap &aliases,
                       std::vector<StoreSpec> *out,
                       ffi::Optional<PrimExpr> mask,
                       std::vector<ShellLoop> *shells = nullptr);

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_BODY_H_
