/*!
 * \file analyze_vsimd.cc
 * \brief Read-only P1 debug entry points.
 *
 * This file is the single debug surface for the P1 analyses: one report per
 * analysis stage (memory legality, capture/scope, region lift) plus one dump
 * pass that prints all of them. The pass never rewrites the IR, so it can be
 * dropped into any pipeline position for investigation.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/transform.h>

#include <sstream>

#include "vsimd_capture_scope.h"
#include "vsimd_memory_legality.h"
#include "vsimd_region.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

/* ------------------------------------------------------------------------- */
/* Per-stage reports                                                         */
/* ------------------------------------------------------------------------- */

String AscendAnalyzeSimdVFMemoryReport(const PrimFunc &func) {
  return AnalyzeVSimdMemoryLegality(func);
}

String AscendAnalyzeSimdVFCaptureReport(const PrimFunc &func) {
  return AnalyzeVSimdCaptureScope(func);
}

Array<Map<String, Any>> AscendAnalyzeSimdVFCaptureInfo(const PrimFunc &func) {
  return CaptureInfoToFFI(AnalyzeVSimdCaptureInfo(func));
}

String AscendAnalyzeVSimdRegionReport(const PrimFunc &func) {
  return AnalyzeVSimdRegionReport(func);
}

/* ------------------------------------------------------------------------- */
/* Combined dump                                                             */
/* ------------------------------------------------------------------------- */

String AscendDumpVSimdReport(const PrimFunc &func) {
  // One analysis run per stage: the region section is rendered from the same
  // capture/memory results instead of re-running them.
  std::vector<VSimdCaptureInfo> infos = AnalyzeVSimdCaptureInfo(func);
  VSimdMemoryAnalysis memory = AnalyzeVSimdMemory(func);
  std::ostringstream out;
  out << "=== memory legality ===\n" << memory.report;
  out << "=== capture / scope ===\n" << RenderVSimdCaptureReport(infos);
  out << "=== region lift ===\n"
      << RenderVSimdRegionReport(infos, memory.regions, RegionVerifyEnabled());
  return String(out.str());
}

tvm::transform::Pass AscendDumpVSimd() {
  auto pass_func = [](PrimFunc f, const IRModule &mod, PassContext ctx) {
    LOG(INFO) << "\n" << AscendDumpVSimdReport(f);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendDumpVSimd", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.transform.AscendAnalyzeSimdVFMemoryReport",
           AscendAnalyzeSimdVFMemoryReport)
      .def("tl.transform.AscendAnalyzeSimdVFCaptureReport",
           AscendAnalyzeSimdVFCaptureReport)
      .def("tl.transform.AscendAnalyzeSimdVFCaptureInfo",
           AscendAnalyzeSimdVFCaptureInfo)
      .def("tl.transform.AscendAnalyzeVSimdRegionReport",
           AscendAnalyzeVSimdRegionReport)
      .def("tl.transform.AscendDumpVSimdReport", AscendDumpVSimdReport)
      .def("tl.transform.AscendDumpVSimd", []() { return AscendDumpVSimd(); });
}

} // namespace tl
} // namespace tvm
