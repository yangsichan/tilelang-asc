/*!
 * \file vsimd_capture_scope.h
 * \brief Read-only capture/scope analysis for SIMD_VF bodies (P1, PR-B).
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_VSIMD_CAPTURE_SCOPE_H_
#define TVM_TL_ASCEND_TRANSFORM_VSIMD_CAPTURE_SCOPE_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/tirx/function.h>

#include <string>
#include <utility>
#include <vector>

#include "ascend/layout/ascend_layouts.h"
#include "vsimd_analysis_common.h"

namespace tvm {
namespace tl {

/*!
 * \brief Typed capture/scope result for one SIMD_VF T.Parallel nest.
 *
 * This is the primary PR-B output consumed by the later region/lift stages.
 * The string report is only a renderer over this structure.
 *
 * Deliberately absent:
 * - thread_bounds: blockIdx/threadIdx ranges are launch metadata; the
 *   outer_vars "thread" kind is enough for capture/scope, and PR-C/P2/P3 can
 *   read launch extents from the function/module.
 * - versions: a version is a leading buffer dimension; it is already visible
 *   in the captured buffer shape and access maps. A separate count would be
 *   redundant and would depend on unstable tl.buffer_version map parsing.
 */
struct VSimdCaptureInfo {
  int vf_index{-1};      // 0-based SIMD_VF order in the function; report key
  int nest_index{-1};    // 0-based T.Parallel nest in the VF; -1 = opaque
  std::string vf_status; // aggregate VF status: CLOSED / OPEN / OPAQUE
  std::string status;    // this nest's status: CLOSED / OPEN / OPAQUE
  std::string message;   // reason for OPAQUE / no nest; empty when analyzed

  // Outermost Parallel loop of this nest. Consumers that need to attach
  // something to the nest match on this pointer instead of the ordinal, which
  // keeps the mapping structural (no parallel-ordinal bookkeeping).
  const tirx::ForNode *root_for{nullptr};

  // Definition domain of the parallel nest. Required to build index maps and
  // to reproduce the loop in the reference expander / region carrier.
  std::vector<vsimd::LoopInfo> domain;

  // Serial loops enclosing the nest inside the same SIMD_VF. They stay around
  // the emitted vector code, so their counters are bound (not captured).
  std::vector<vsimd::LoopInfo> outer_shells;

  // Every load/store in the nest, including store values. This is the source
  // for the reference expander and later P2/P3 address/memory planning.
  std::vector<vsimd::AccessInfo> accesses;

  // Unique buffers used by the nest, paired with capture kind
  // (param / alloc / local / view). This is the region's buffer capture set.
  std::vector<std::pair<tirx::Buffer, std::string>> captures;

  // Non-domain vars referenced by the nest, paired with kind
  // (thread / outer_loop / bind). Carried into the region so scalar captures
  // remain defined after lift.
  std::vector<std::pair<tirx::Var, std::string>> outer_vars;

  // Free variables that occur in shapes/strides/extents but are not defined
  // in scope (dynamic shape symbols). Needed for symbolic-shape regions.
  std::vector<tirx::Var> shape_syms;

  // Local T.view-style aliases: view buffer -> base buffer. Required for
  // alias/address identity; these are not separate captures.
  std::vector<std::pair<tirx::Buffer, tirx::Buffer>> local_views;

  // Layout entries for captured buffers. Required for layout-aware store
  // decisions; P2/P3 use this to choose vsts vs vsstb.
  std::vector<std::pair<tirx::Buffer, Layout>> layouts;

  // Relevant, deduplicated tl.assume conditions. This is the arithmetic
  // contract available to PR-C / P2/P3; may be empty.
  std::vector<PrimExpr> assumptions;

  // Relevant, deduplicated Bind definitions (e.g. tail = min(...)). Kept so
  // extent/index expressions can be reproduced after lift; may be empty.
  std::vector<std::pair<tirx::Var, PrimExpr>> bindings;

  // Closure errors (unbound var/buffer, invalid view, ...). Non-empty means
  // status=OPEN; empty for CLOSED/OPAQUE.
  std::vector<std::string> errors;
};

/*! \brief Analyze every SIMD_VF and return typed capture/scope results. */
TVM_DLL std::vector<VSimdCaptureInfo>
AnalyzeVSimdCaptureInfo(const tirx::PrimFunc &func);

/*! \brief Render typed results as a human-readable debug report. */
TVM_DLL ffi::String
RenderVSimdCaptureReport(const std::vector<VSimdCaptureInfo> &infos);

/*! \brief Convenience: analyze + render. */
TVM_DLL ffi::String AnalyzeVSimdCaptureScope(const tirx::PrimFunc &func);

/*! \brief Serialize typed results for FFI/tests. */
TVM_DLL ffi::Array<ffi::Map<ffi::String, ffi::Any>>
CaptureInfoToFFI(const std::vector<VSimdCaptureInfo> &infos);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VSIMD_CAPTURE_SCOPE_H_
