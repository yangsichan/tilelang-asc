/*!
 * \file emit.h
 * \brief Emission layer: ``VectorPlan`` / ``VectorProgram`` -> ``T.simd.*``
 * TIR.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_EMIT_H_
#define TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_EMIT_H_

#include <tvm/tirx/stmt.h>

#include "model.h"

namespace tvm {
namespace tl {
namespace parallel_vectorize {

/*!
 * \brief Emit a plan (or program) as ``T.simd.*`` TIR.
 *
 * The plan is read-only; every node / ordering rule matches the Python emitter
 * node for node (SSA names included).
 */
TVM_DLL tirx::Stmt EmitPlan(const ffi::ObjectRef &plan);

/*!
 * \brief ``simd_inst::mem_bar(VST_VLD)``.
 *
 * Closes a vector store before the next load. Emitted at the end of a
 * materialized loop whose body reads a buffer it also writes, and when the
 * planned Parallel sits inside a serial loop, so a read-modify-write body never
 * reads a value from before its own store.
 */
TVM_DLL tirx::Stmt VstVldBarrier();

/*!
 * \brief Invariant I5 check: no scalar data instruction in an emitted region.
 *
 * A ``BufferStore`` to a tensor element and a ``BufferLoad`` used as a scalar
 * data operand are forbidden inside ``T.SimdVF`` (bisheng refuses them). Scalar
 * values are only allowed in loop bounds, branch conditions, ``tl.access_ptr``
 * addresses, chunk indices and ``Bind`` values. Returns true when clean; fills
 * ``violations`` with one line per offending statement otherwise.
 */
TVM_DLL bool VerifyNoScalarDataOps(const tirx::Stmt &stmt,
                                   std::vector<std::string> *violations);

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_EMIT_H_
