/*!
 * \file vsimd_memory_legality.h
 * \brief Read-only memory-legality analysis for SIMD_VF bodies (P1).
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_VSIMD_MEMORY_LEGALITY_H_
#define TVM_TL_ASCEND_TRANSFORM_VSIMD_MEMORY_LEGALITY_H_

#include <tvm/ffi/string.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <string>
#include <vector>

namespace tvm {
namespace tl {

/*!
 * \brief Analyze the memory legality of every SIMD_VF block in \p func.
 *
 * The analysis is intentionally read-only: it does not rewrite the function.
 * It reports, per SIMD_VF and per T.Parallel nest, whether the stores are
 * injective and whether cross-iteration dependences exist. Manual SIMD VFs
 * (tl.simd.*, local.var, nd2nz helpers) are reported as opaque.
 */
TVM_DLL ffi::String AnalyzeVSimdMemoryLegality(const tirx::PrimFunc &func);

/*! \brief Per-VF/per-nest legality verdict and diagnostic. */
struct VSimdMemoryRegionStatus {
  int vf_index{-1};
  int nest_index{-1};
  std::string status;
  /*! \brief Outermost Parallel loop of the nest; identity for matching. */
  const tirx::ForNode *root{nullptr};
  /*! \brief Explanation of the verdict, retained for annotation and reporting.
   */
  std::string reason;
};

/*! \brief One analysis run: the rendered report plus the per-nest statuses. */
struct VSimdMemoryAnalysis {
  std::string report;
  std::vector<VSimdMemoryRegionStatus> regions;
};

/*!
 * \brief Run the legality analysis once and return both outputs.
 *
 * The report renderer and the per-nest status list come from the same visitor,
 * so a debug dump does not need to analyze the function twice.
 */
TVM_DLL VSimdMemoryAnalysis AnalyzeVSimdMemory(const tirx::PrimFunc &func);

TVM_DLL std::vector<VSimdMemoryRegionStatus>
AnalyzeVSimdMemoryRegions(const tirx::PrimFunc &func);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VSIMD_MEMORY_LEGALITY_H_
