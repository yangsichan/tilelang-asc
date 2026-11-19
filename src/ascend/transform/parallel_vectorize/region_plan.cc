/*!
 * \file region_plan.cc
 * \brief Builder and diagnostic printer of the per-Parallel plan carrier.
 */
#include "region_plan.h"

#include <tvm/ffi/reflection/registry.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace parallel_vectorize {

VSimdRegionPlan MakeVSimdRegionPlan(int64_t vf_index, int64_t region_index,
                                    VectorProgram program) {
  auto node = ffi::make_object<VSimdRegionPlanNode>();
  node->version = kRegionPlanVersion;
  node->vf_index = vf_index;
  node->region_index = region_index;
  node->program = std::move(program);
  return VSimdRegionPlan(std::move(node));
}

namespace {

using script::printer::AccessPath;
using script::printer::DictDoc;
using script::printer::Doc;
using script::printer::ExprDoc;
using script::printer::IRDocsifier;
using script::printer::ListDoc;
using script::printer::LiteralDoc;

/*
 * Without this printer the script printer treats the carrier as an opaque
 * object and prints ``metadata["tl.parallel_vectorize.VSimdRegionPlan"][0]``,
 * so a rejected/accepted plan never shows up readably in an IR dump. Rendering
 * it as a plain dict is diagnostic only: P3 always consumes the typed object.
 */

ExprDoc StrDoc(const ffi::String &value) {
  return LiteralDoc::Str(value, std::nullopt);
}

ExprDoc IntDoc(int64_t value) { return LiteralDoc::Int(value, std::nullopt); }

ExprDoc DictOf(std::vector<std::pair<const char *, ExprDoc>> fields) {
  ffi::Array<ExprDoc> keys;
  ffi::Array<ExprDoc> values;
  for (auto &field : fields) {
    keys.push_back(StrDoc(field.first));
    values.push_back(std::move(field.second));
  }
  return DictDoc(keys, values);
}

ExprDoc StrListDoc(const ffi::Array<ffi::String> &items) {
  ffi::Array<ExprDoc> elements;
  for (const ffi::String &item : items)
    elements.push_back(StrDoc(item));
  return ListDoc(elements);
}

/*! \brief Compact, bounded rendering of a ``VectorProgram``. */
ExprDoc ProgramDoc(const VectorProgram &program) {
  if (!program.defined())
    return LiteralDoc::None(std::nullopt);
  ffi::Array<ExprDoc> segments;
  for (const VectorPlan &segment : program->segments) {
    std::string axis = segment->vector_axis.defined()
                           ? std::string(segment->vector_axis->name)
                           : std::string();
    segments.push_back(DictOf({
        {"lanes", IntDoc(segment->lanes)},
        {"vector_axis", StrDoc(axis)},
        {"stores", IntDoc(static_cast<int64_t>(segment->stores.size()))},
        {"pre_stores",
         IntDoc(static_cast<int64_t>(segment->pre_stores.size()))},
        {"post_stores",
         IntDoc(static_cast<int64_t>(segment->post_stores.size()))},
    }));
  }
  return DictOf({
      {"segments", ListDoc(segments)},
      {"diagnostics", StrListDoc(program->diagnostics)},
  });
}

ExprDoc RegionPlanDoc(const VSimdRegionPlan &plan, const AccessPath &path,
                      const IRDocsifier &d) {
  return DictOf({
      {"version", IntDoc(plan->version)},
      {"vf_index", IntDoc(plan->vf_index)},
      {"region_index", IntDoc(plan->region_index)},
      {"program", ProgramDoc(plan->program)},
  });
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<VSimdRegionPlan>("",
                                   [](VSimdRegionPlan plan, AccessPath path,
                                      IRDocsifier d) -> Doc {
                                     return RegionPlanDoc(plan, path, d);
                                   });

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() { VSimdRegionPlanNode::RegisterReflection(); }

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm
