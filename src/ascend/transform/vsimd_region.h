/*!
 * \file vsimd_region.h
 * \brief Build a VSimdRegion from an analyzed Parallel nest, plus the reference
 *        scalar expander used as a structural self-check.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_VSIMD_REGION_H_
#define TVM_TL_ASCEND_TRANSFORM_VSIMD_REGION_H_

#include <tvm/ffi/string.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <string>
#include <vector>

#include "vsimd_capture_scope.h"
#include "vsimd_ir.h"
#include "vsimd_memory_legality.h"

namespace tvm {
namespace tl {

/*! \brief Result of the region lift decision for one Parallel nest. */
struct VSimdRegionLift {
  bool lifted{false};
  bool internal_error{false}; // builder self-check failed: compiler bug
  std::string reason;         // rejection/internal-error reason
  vsimd::VSimdRegion region;  // valid only when lifted=true
};

/*! \brief Shared fragment rejection for the annotation pass and region report.
 * Returns an empty string when the subtree does not touch a fragment.
 */
TVM_DLL std::string FragmentRejectionReason(const tirx::For &loop);

/*!
 * \brief Build a region for a proven, closed, rectangular nest.
 *
 * `memory_status` is the legality status of this nest (LEGAL is required).
 * `memory_reason` retains the analysis explanation when the nest is rejected.
 * All other cases return lifted=false with a diagnostic; this function never
 * mutates IR. The built region carries the collapsed Parallel axes, the
 * ordered store ops of the innermost body, and the captured context.
 *
 * With `verify_body` the builder also reconstructs the region and compares it
 * against the original Parallel subtree (see CompareRegionWithOriginal). That
 * is the structural oracle: a mismatch means the region does not faithfully
 * describe the source, i.e. a compiler bug, and is reported as an internal
 * error rather than a user fallback.
 */
TVM_DLL VSimdRegionLift BuildRegion(const VSimdCaptureInfo &info,
                                    const std::string &memory_status,
                                    bool verify_body = true,
                                    const std::string &memory_reason = "");

/*!
 * \brief Reconstruct serial scalar / structured IR from a region.
 *
 * The result is a structural oracle, not the P3 physical lowering: stores and
 * branches are rebuilt and axes become serial loops. An opaque op (a nest P1
 * could not model) has no reconstructible effect.
 *
 * When an op cannot be represented, the function returns an undefined Stmt and,
 * if `unsupported` is not null, fills it with a diagnostic instead of silently
 * dropping the semantic. Callers must treat that as a compiler-bug signal, not
 * a user fallback.
 */
TVM_DLL tirx::Stmt ExpandVSimdRegionToLoops(const vsimd::VSimdRegion &region,
                                            std::string *unsupported = nullptr);

/*! \brief Result of the cheap structural expander/lift consistency check. */
struct VSimdRegionVerify {
  bool ok{true};
  std::string reason;
};

/*!
 * \brief Check that the expanded statement declares every local view, uses only
 * captured/declared buffers, and reproduces the region axes and stores in
 * source order. This is a structural guard, not a numerical oracle.
 */
TVM_DLL VSimdRegionVerify VerifyExpandedRegion(const vsimd::VSimdRegion &region,
                                               const tirx::Stmt &expanded);

/*!
 * \brief Compare a region's reconstruction against the original Parallel loop.
 *
 * Both sides are normalized first: Parallel becomes Serial, For annotations are
 * dropped, nested SeqStmt are flattened, and DeclBuffer statements are compared
 * as a set (the reconstruction hoists them out of the loops). The comparison is
 * structural, not numerical: it proves that the region describes the same
 * loops, stores, values and tileop calls in the same order, so P3 cannot lose
 * or reorder anything. TIR spans are not compared.
 *
 * The result carries a path such as `body[1].value` on mismatch.
 */
struct VSimdRegionMatch {
  bool ok{true};
  std::string reason;
};

TVM_DLL VSimdRegionMatch CompareRegionWithOriginal(
    const vsimd::VSimdRegion &region, const tirx::Stmt &original);

/*! \brief Whether the structural oracle is enabled by the pass config. */
TVM_DLL bool RegionVerifyEnabled();

/*!
 * \brief Render the per-nest region report from precomputed analyses.
 *
 * Split out so a debug dump can share one capture/memory analysis instead of
 * re-running them per section.
 */
TVM_DLL ffi::String
RenderVSimdRegionReport(const std::vector<VSimdCaptureInfo> &infos,
                        const std::vector<VSimdMemoryRegionStatus> &memory,
                        bool verify_body = true);

/*! \brief Lift every eligible nest and render the region/expanded dump. */
TVM_DLL ffi::String AnalyzeVSimdRegionReport(const tirx::PrimFunc &func);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VSIMD_REGION_H_
