// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file onchip_memory_limits.h
 * \brief Physical on-chip memory capacities (UB / L1 / L0A / L0B / L0C).
 */

#pragma once

#include <tvm/target/target.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <cstdint>
#include <string>

#include "ascend/target_utils.h"

namespace tvm {
namespace tl {
namespace ascend {

// Check whether a kernel body contains any SIMT_VF blocks. SIMT_VF
// (asc_vf_call) reserves 32 KB of the physical 248 KB Unified Buffer for
// thread context, so the usable shared memory depends on its presence.
// NOTE: vf_regions.h's IsVFRegion also matches SIMD_VF, which does NOT reserve
// the thread context, so it is not a substitute for this detector.
class SimtVFDetector : public tirx::StmtVisitor {
public:
  bool found = false;

  void VisitStmt_(const tirx::SBlockNode *op) override {
    if (op->name_hint == "SIMT_VF") {
      found = true;
    }
    tirx::StmtVisitor::VisitStmt_(op);
  }
};

inline bool HasSimtVF(const tirx::Stmt &stmt) {
  SimtVFDetector detector;
  detector(stmt);
  return detector.found;
}

inline int64_t GetSharedMemoryLimit(const tirx::Stmt &kernel_body) {
  constexpr int64_t kPhysicalUnifiedBufferSize = 248 * 1024;
  constexpr int64_t kSimtVFThreadContextSize = 32 * 1024;
  return HasSimtVF(kernel_body)
             ? kPhysicalUnifiedBufferSize - kSimtVFThreadContextSize
             : kPhysicalUnifiedBufferSize;
}

inline int64_t GetL1MemoryLimit(Target target) {
  if (TargetIsAscend(target)) {
    return 512 * 1024; // 512 KB L1 cache on Ascend
  }
  return 0;
}

inline int64_t GetL0CMemoryLimit(Target target) {
  if (TargetIsAscend(target)) {
    return 256 * 1024; // 256 KB L0C buffer on Ascend
  }
  return 0;
}

inline int64_t GetL0AMemoryLimit(Target target) {
  if (TargetIsAscend(target)) {
    return 64 * 1024; // 64 KB L0A buffer on Ascend
  }
  return 0;
}

inline int64_t GetL0BMemoryLimit(Target target) {
  if (TargetIsAscend(target)) {
    return 64 * 1024; // 64 KB L0B buffer on Ascend
  }
  return 0;
}

/*!
 * \brief Physical capacities of one kernel, keyed by storage scope.
 *
 * All-zero means "no known capacity" (no target attribute, or a non-Ascend
 * target), which disables every capacity check that consults it.
 */
struct OnChipLimits {
  int64_t shared{0}; // Unified Buffer, SIMT_VF thread context already removed
  int64_t l1{0};
  int64_t l0c{0};
  int64_t l0a{0};
  int64_t l0b{0};

  int64_t GetScopeLimit(const std::string &scope) const {
    // "shared.dyn" is the storage scope allocation planning sees for the UB;
    // "shared" is the key the scheduling budget uses for the same pool.
    if (scope == "shared" || scope == "shared.dyn")
      return shared;
    if (scope == "shared.l1")
      return l1;
    if (scope == "shared.l0c")
      return l0c;
    if (scope == "shared.l0a")
      return l0a;
    if (scope == "shared.l0b")
      return l0b;
    return 0; // unknown scope: unchecked
  }
};

/*!
 * \brief Capacities for \p target; all-zero when the target is not Ascend.
 * \param kernel_body the body the UB size depends on (SIMT_VF detection).
 *
 * A device function may contain several kernels; SIMT_VF is then detected for
 * the function as a whole, which can only make the UB limit smaller
 * (conservative for the SIMT_VF-free kernel).
 */
inline OnChipLimits GetOnChipLimits(Target target,
                                    const tirx::Stmt &kernel_body) {
  OnChipLimits limits;
  if (!TargetIsAscend(target))
    return limits;
  limits.shared = GetSharedMemoryLimit(kernel_body);
  limits.l1 = GetL1MemoryLimit(target);
  limits.l0c = GetL0CMemoryLimit(target);
  limits.l0a = GetL0AMemoryLimit(target);
  limits.l0b = GetL0BMemoryLimit(target);
  return limits;
}

} // namespace ascend
} // namespace tl
} // namespace tvm
