/*!
 * \file analyze_parallel_vectorize.cc
 * \brief Offline analysis/planning FFI for the parallel_vectorize P2/P3
 * implementation.
 *
 * These entry points are the test-facing counterpart of the pipeline passes
 * (``AscendPlanVSimd`` / ``AscendEmitVSimd``): they plan and emit a raw
 * ``T.Parallel`` nest without P1 annotations, and expose the plan text and the
 * Δ-classification helpers used by the offline test suite.
 *
 * Every entry point converts :class:`parallel_vectorize::Unsupported` into the
 * ``Map{"ok": false, "message": ..., "diagnostics": [...]}`` shape so the
 * Python wrapper can re-raise the *identical* ``Unsupported`` (message +
 * diagnostics ordering are part of the contract, D3).
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>

#include <optional>
#include <string>
#include <vector>

#include "access.h"
#include "emit.h"
#include "model.h"
#include "plan.h"
#include "primitives.h"

namespace tvm {
namespace tl {
namespace parallel_vectorize {

namespace {

ffi::Map<ffi::String, ffi::Any> MakeError(const Unsupported &error) {
  ffi::Map<ffi::String, ffi::Any> result;
  result.Set("ok", ffi::Any(false));
  result.Set("code", ffi::Any(ffi::String(error.code())));
  result.Set("message", ffi::Any(ffi::String(error.message())));
  ffi::Array<ffi::String> diagnostics;
  for (const std::string &note : error.diagnostics()) {
    diagnostics.push_back(note);
  }
  result.Set("diagnostics", ffi::Any(diagnostics));
  return result;
}

/*! \brief Run ``body`` and bridge ``Unsupported`` into the Map protocol. */
template <typename Fn> ffi::Map<ffi::String, ffi::Any> Guarded(Fn body) {
  try {
    return body();
  } catch (const Unsupported &error) {
    return MakeError(error);
  }
}

/*! \brief ``{ok: true, key: value}`` — the success shape of every entry point.
 */
ffi::Map<ffi::String, ffi::Any> MakeOk(ffi::String key, ffi::Any value) {
  ffi::Map<ffi::String, ffi::Any> result;
  result.Set("ok", ffi::Any(true));
  result.Set(key, std::move(value));
  return result;
}

/*! \brief ``AxisClass`` -> ``[kind, stride?]`` (the shape Python reads). */
ffi::Array<ffi::Any> AxisClassToArray(const AxisClass &classified) {
  ffi::Array<ffi::Any> out;
  out.push_back(ffi::Any(ffi::String(AccessClassName(classified.kind))));
  out.push_back(classified.stride.has_value() ? ffi::Any(*classified.stride)
                                              : ffi::Any());
  return out;
}

/*! \brief FFI optional lanes -> ``std::optional<int64_t>`` (empty = derive). */
std::optional<int64_t> OptionalLanes(const ffi::Optional<int64_t> &lanes) {
  return lanes.has_value() ? std::optional<int64_t>(*lanes) : std::nullopt;
}

/*! \brief FFI plan inputs -> ``PlanOptions`` (empty optional = derive). */
PlanOptions
MakePlanOptions(const ffi::Optional<ffi::Map<tirx::Buffer, Layout>> &layouts) {
  PlanOptions options;
  if (layouts.has_value()) {
    for (const auto &entry : *layouts) {
      options.layouts.emplace_back(entry.first, entry.second);
    }
  }
  return options;
}

/* ------------------------------------------------------------------------- */
/* plan rendering (test oracle)                                              */
/* ------------------------------------------------------------------------- */

std::string OptionalIntText(const ffi::Optional<int64_t> &value) {
  return value.has_value() ? std::to_string(*value) : "None";
}

std::string RenderTree(const PlanNode &root);

/*! \brief Python ``f"{buffer.name}{list(indices)}"``. */
std::string SinkText(const StoreSpec &spec) {
  std::string out = spec->buffer->name.operator std::string() + "[";
  for (size_t i = 0; i < spec->indices.size(); ++i) {
    out += (i == 0 ? "" : ", ") + PrintExpr(spec->indices[i]);
  }
  return out + "]";
}

std::string PyStrList(const std::vector<std::string> &items) {
  std::string out = "[";
  for (size_t i = 0; i < items.size(); ++i)
    out += (i == 0 ? "'" : ", '") + items[i] + "'";
  return out + "]";
}

void RenderValue(const ValueNode &node, std::vector<std::string> *lines,
                 int depth) {
  std::string pad(static_cast<size_t>(depth) * 2, ' ');
  std::string detail = node->detail.empty()
                           ? std::string()
                           : " (" + node->detail.operator std::string() + ")";
  lines->push_back(pad + "- " + node->recipe.operator std::string() + ": " +
                   node->dtype.operator std::string() + detail);
  for (const ffi::ObjectRef &arg : node->args) {
    RenderValue(Downcast<ValueNode>(arg), lines, depth + 1);
  }
}

std::vector<std::string>
RenderSinglePlan(const VectorPlan &plan,
                 const std::vector<std::string> &extra_diagnostics) {
  std::vector<std::string> lines;
  lines.push_back(
      "vector_axis=" + plan->vector_axis->name.operator std::string() +
      " lanes=" + std::to_string(plan->lanes) + " chunks=" +
      OptionalIntText(plan->chunks) + " tail=" + OptionalIntText(plan->tail));
  ReduceInfo reduce = ReduceOf(plan);
  if (reduce.defined()) {
    lines.push_back("reduce(axis=" + reduce->axis->name.operator std::string() +
                    ", op=" + reduce->op.operator std::string() + " -> " +
                    reduce->recipe.operator std::string() + ")");
  }
  const std::pair<const char *, ffi::Array<StorePlan>> groups[] = {
      {"pre", plan->pre_stores},
      {"store", plan->stores},
      {"post", plan->post_stores}};
  for (const auto &group : groups) {
    for (const StorePlan &store : group.second) {
      // Mixed element widths / destination layouts: only non-default stores
      // carry an extra note, so the render of an ordinary plan stays identical.
      const std::string lane_mode =
          store->store_lane_mode.operator std::string();
      std::string note = lane_mode == "full"
                             ? ""
                             : " store_lanes=" + lane_mode + "/" +
                                   std::to_string(store->store_lanes);
      const std::string layout_kind = store->layout_kind.operator std::string();
      if (layout_kind != "nd") {
        note += " layout=" + layout_kind + "[" +
                std::to_string(store->nz_block_stride.has_value()
                                   ? *store->nz_block_stride
                                   : 0) +
                "]";
      }
      lines.push_back(std::string(group.first) + " " + SinkText(store->spec) +
                      " [" + AccessClassName(store->kind) + "] mask=" +
                      (store->mask.has_value() ? "yes" : "no") + note);
      RenderValue(store->value, &lines, 1);
    }
  }
  if (!plan->diagnostics.empty() || !extra_diagnostics.empty()) {
    lines.push_back("diagnostics:");
    for (const ffi::String &note : plan->diagnostics) {
      lines.push_back("  " + note.operator std::string());
    }
    for (const std::string &note : extra_diagnostics)
      lines.push_back("  " + note);
  }
  return lines;
}

std::string RenderPlanText(const ffi::ObjectRef &plan) {
  if (const auto *single = plan.as<VectorPlanNode>()) {
    std::vector<std::string> lines =
        RenderSinglePlan(ffi::GetRef<VectorPlan>(single), {});
    std::string out;
    for (size_t i = 0; i < lines.size(); ++i)
      out += (i == 0 ? "" : "\n") + lines[i];
    return out;
  }
  const auto *program = plan.as<VectorProgramNode>();
  TVM_FFI_ICHECK(program != nullptr)
      << "render_plan expects a VectorPlan or VectorProgram";
  VectorProgram node = ffi::GetRef<VectorProgram>(program);
  if (node->segments.size() == 1) {
    std::vector<std::string> extra;
    for (const ffi::String &note : node->diagnostics) {
      extra.push_back(note.operator std::string());
    }
    std::vector<std::string> lines = RenderSinglePlan(node->segments[0], extra);
    std::string out;
    for (size_t i = 0; i < lines.size(); ++i)
      out += (i == 0 ? "" : "\n") + lines[i];
    return out;
  }

  std::vector<std::string> names;
  for (const Axis &axis : node->axes)
    names.push_back(axis->name.operator std::string());
  std::vector<std::string> lines;
  std::string header = "program: axes=" + PyStrList(names);
  if (!node->shared_shells.empty()) {
    std::vector<std::string> shells;
    for (const Axis &shell : node->shared_shells)
      shells.push_back(shell->name.operator std::string());
    header += " shells=" + PyStrList(shells);
  }
  header += " segments=" + std::to_string(node->segments.size());
  lines.push_back(header);
  // The plan tree is the single emission source and ``build_plan`` always sets
  // it; a rootless program would be a construction bug, not a fallback shape.
  TVM_FFI_ICHECK(node->root.defined())
      << "render_plan expects a VectorProgram with a plan tree";
  // Structural tree (design A): the legacy header stays, every node is
  // indented by its depth. Chunk nodes still print the full plan detail.
  std::string tree = RenderTree(Downcast<PlanNode>(node->root));
  size_t begin = 0;
  while (begin < tree.size()) {
    size_t end = tree.find('\n', begin);
    if (end == std::string::npos)
      break;
    lines.push_back(tree.substr(begin, end - begin));
    begin = end + 1;
  }
  if (!node->diagnostics.empty()) {
    lines.push_back("diagnostics:");
    for (const ffi::String &note : node->diagnostics) {
      lines.push_back("  " + note.operator std::string());
    }
  }
  std::string out;
  for (size_t i = 0; i < lines.size(); ++i)
    out += (i == 0 ? "" : "\n") + lines[i];
  return out;
}

std::string RenderPlanNode(const PlanNode &node, int depth,
                           int64_t *chunk_index) {
  if (!node.defined())
    return "";
  const std::string pad(static_cast<size_t>(depth) * 2, ' ');
  switch (node->kind) {
  case PlanKind::kSeq: {
    std::string out = pad + "seq\n";
    for (const ffi::ObjectRef &child : node->children)
      out += RenderPlanNode(Downcast<PlanNode>(child), depth + 1, chunk_index);
    return out;
  }
  case PlanKind::kLoop: {
    std::string out = pad + "loop " +
                      node->loop_var->name_hint.operator std::string() + " [" +
                      PrintExpr(node->min) + ", " + PrintExpr(node->extent) +
                      ")" + (node->barrier_at_end ? " barrier" : "") + "\n";
    if (node->body.defined())
      out += RenderPlanNode(Downcast<PlanNode>(node->body), depth + 1,
                            chunk_index);
    return out;
  }
  case PlanKind::kChunk: {
    const VectorPlan &plan = node->chunk;
    std::string out =
        pad + "chunk #" + std::to_string(*chunk_index) +
        ": vector_axis=" + plan->vector_axis->name.operator std::string() +
        " lanes=" + std::to_string(plan->lanes) +
        " chunks=" + OptionalIntText(plan->chunks) +
        " tail=" + OptionalIntText(plan->tail) + "\n";
    ++*chunk_index;
    for (const std::string &line : RenderSinglePlan(plan, {}))
      out += pad + "  " + line + "\n";
    return out;
  }
  }
  return "";
}

/*! \brief Indented structural tree (the legacy header is kept above it). */
std::string RenderTree(const PlanNode &root) {
  int64_t chunk_index = 0;
  return RenderPlanNode(root, /*depth=*/0, &chunk_index);
}

} // namespace

ffi::Array<ffi::Any> AscendParallelVectorizeAxisClass(const AccessInfo &info,
                                                      const Axis &axis) {
  return AxisClassToArray(AxisClassOf(info, axis));
}

ffi::Map<ffi::String, ffi::Any> AscendParallelVectorizeBuildPlan(
    const tirx::Stmt &stmt, ffi::Optional<int64_t> lanes,
    ffi::Optional<ffi::Map<tirx::Buffer, Layout>> layouts) {
  return Guarded([&]() {
    VectorProgram program =
        BuildPlan(stmt, OptionalLanes(lanes), MakePlanOptions(layouts));
    return MakeOk("program", ffi::Any(program));
  });
}

ffi::Map<ffi::String, ffi::Any>
AscendParallelVectorizeEmitPlan(const ffi::ObjectRef &plan) {
  return Guarded([&]() { return MakeOk("stmt", ffi::Any(EmitPlan(plan))); });
}

ffi::Map<ffi::String, ffi::Any>
AscendParallelVectorizeVerifyNoScalarDataOps(const tirx::Stmt &stmt) {
  std::vector<std::string> violations;
  VerifyNoScalarDataOps(stmt, &violations);
  ffi::Array<ffi::String> notes;
  for (const std::string &note : violations)
    notes.push_back(note);
  ffi::Map<ffi::String, ffi::Any> result;
  result.Set("ok", ffi::Any(violations.empty()));
  result.Set("violations", ffi::Any(notes));
  return result;
}

ffi::Map<ffi::String, ffi::Any>
AscendParallelVectorizeRenderPlan(const ffi::ObjectRef &plan) {
  return Guarded([&]() {
    return MakeOk("text", ffi::Any(ffi::String(RenderPlanText(plan))));
  });
}

PrimExpr AscendParallelVectorizeFiniteDifference(const PrimExpr &offset,
                                                 const tirx::Var &var,
                                                 bool use_analyzer) {
  arith::Analyzer analyzer;
  return FiniteDifference(offset, var, use_analyzer ? &analyzer : nullptr);
}

ffi::Array<ffi::Any> AscendParallelVectorizeClassifyDelta(const PrimExpr &delta,
                                                          bool use_analyzer) {
  arith::Analyzer analyzer;
  return AxisClassToArray(
      ClassifyDelta(delta, use_analyzer ? &analyzer : nullptr));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendParallelVectorizeAxisClass",
                        AscendParallelVectorizeAxisClass);
  refl::GlobalDef().def("tl.transform.AscendParallelVectorizeBuildPlan",
                        AscendParallelVectorizeBuildPlan);
  refl::GlobalDef().def("tl.transform.AscendParallelVectorizeEmitPlan",
                        AscendParallelVectorizeEmitPlan);
  refl::GlobalDef().def(
      "tl.transform.AscendParallelVectorizeVerifyNoScalarDataOps",
      AscendParallelVectorizeVerifyNoScalarDataOps);
  refl::GlobalDef().def("tl.transform.AscendParallelVectorizeRenderPlan",
                        AscendParallelVectorizeRenderPlan);
  refl::GlobalDef().def("tl.transform.AscendParallelVectorizeFiniteDifference",
                        AscendParallelVectorizeFiniteDifference);
  refl::GlobalDef().def("tl.transform.AscendParallelVectorizeClassifyDelta",
                        AscendParallelVectorizeClassifyDelta);
}

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm
