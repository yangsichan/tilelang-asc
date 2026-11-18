/*!
 * \file vsimd_capture_scope.cc
 * \brief Read-only capture/scope analysis for SIMD_VF bodies (P1, PR-B).
 */
#include "vsimd_capture_scope.h"

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace vsimd;
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

using VarMap = std::unordered_map<Var, std::string, ffi::ObjectPtrHash,
                                  ffi::ObjectPtrEqual>;
using BufferMap =
    std::unordered_map<Var, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using LayoutMap =
    std::unordered_map<Var, Layout, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

namespace {

static String ExprString(const PrimExpr &expr) {
  std::ostringstream os;
  os << expr;
  return String(os.str());
}

static Array<String> DomainStrings(const std::vector<LoopInfo> &loops) {
  std::vector<String> out;
  for (const LoopInfo &loop : loops) {
    std::ostringstream os;
    os << loop.var->name_hint << " in [" << loop.min << ", "
       << (loop.min + loop.extent) << ")";
    out.push_back(String(os.str()));
  }
  return Array<String>(out.begin(), out.end());
}

static Array<String> AccessStrings(const std::vector<AccessInfo> &accesses) {
  std::vector<String> out;
  for (const AccessInfo &acc : accesses) {
    std::ostringstream os;
    os << (acc.is_write ? "W " : "R ") << acc.buffer->name << "[";
    for (size_t i = 0; i < acc.indices.size(); ++i) {
      if (i)
        os << ",";
      os << acc.indices[i];
    }
    os << "]";
    if (acc.is_write && acc.value.defined())
      os << " = " << acc.value;
    out.push_back(String(os.str()));
  }
  return Array<String>(out.begin(), out.end());
}

static Array<String>
BufferKindStrings(const std::vector<std::pair<Buffer, std::string>> &items) {
  std::vector<String> out;
  for (const auto &item : items) {
    out.push_back(
        String(std::string(item.first->name) + "[" + item.second + "]"));
  }
  return Array<String>(out.begin(), out.end());
}

static Array<String>
VarKindStrings(const std::vector<std::pair<Var, std::string>> &items) {
  std::vector<String> out;
  for (const auto &item : items) {
    out.push_back(
        String(std::string(item.first->name_hint) + "[" + item.second + "]"));
  }
  return Array<String>(out.begin(), out.end());
}

static Array<String> VarStrings(const std::vector<Var> &vars) {
  std::vector<String> out;
  for (const Var &var : vars)
    out.push_back(String(var->name_hint));
  return Array<String>(out.begin(), out.end());
}

static Array<String>
LocalViewStrings(const std::vector<std::pair<Buffer, Buffer>> &views) {
  std::vector<String> out;
  for (const auto &view : views) {
    out.push_back(String(std::string(view.first->name) + " -> " +
                         std::string(view.second->name)));
  }
  return Array<String>(out.begin(), out.end());
}

static Array<String>
LayoutStrings(const std::vector<std::pair<Buffer, Layout>> &layouts) {
  std::vector<String> out;
  for (const auto &layout : layouts)
    out.push_back(String(layout.first->name));
  return Array<String>(out.begin(), out.end());
}

static Array<String> AssumeStrings(const std::vector<PrimExpr> &assumptions) {
  std::vector<String> out;
  for (const PrimExpr &expr : assumptions)
    out.push_back(ExprString(expr));
  return Array<String>(out.begin(), out.end());
}

static Array<String>
BindingStrings(const std::vector<std::pair<Var, PrimExpr>> &bindings) {
  std::vector<String> out;
  for (const auto &binding : bindings) {
    std::ostringstream os;
    os << binding.first->name_hint << " = " << binding.second;
    out.push_back(String(os.str()));
  }
  return Array<String>(out.begin(), out.end());
}

static Array<String> ErrorStrings(const std::vector<std::string> &errors) {
  std::vector<String> out;
  for (const std::string &error : errors)
    out.push_back(String(error));
  return Array<String>(out.begin(), out.end());
}

static Map<String, Any> CaptureToMap(const VSimdCaptureInfo &info) {
  Map<String, Any> map;
  map.Set("vf_index", Any(info.vf_index));
  map.Set("nest_index", Any(info.nest_index));
  map.Set("vf_status", Any(String(info.vf_status)));
  map.Set("status", Any(String(info.status)));
  map.Set("message", Any(String(info.message)));
  map.Set("domain", Any(DomainStrings(info.domain)));
  map.Set("accesses", Any(AccessStrings(info.accesses)));
  map.Set("captures", Any(BufferKindStrings(info.captures)));
  map.Set("outer_vars", Any(VarKindStrings(info.outer_vars)));
  map.Set("shape_syms", Any(VarStrings(info.shape_syms)));
  map.Set("local_views", Any(LocalViewStrings(info.local_views)));
  map.Set("layouts", Any(LayoutStrings(info.layouts)));
  map.Set("assumptions", Any(AssumeStrings(info.assumptions)));
  map.Set("bindings", Any(BindingStrings(info.bindings)));
  map.Set("errors", Any(ErrorStrings(info.errors)));
  return map;
}

/* ------------------------------------------------------------------------- */
/* Scope-aware visitor                                                       */
/* ------------------------------------------------------------------------- */

class CaptureVisitor : public StmtExprVisitor {
public:
  std::vector<VSimdCaptureInfo> infos;

  void Run(const PrimFunc &func) {
    for (const auto &kv : func->buffer_map)
      AddBuffer(kv.second, "param");
    // Scalar function parameters can occur only in a guard or serial bound;
    // they are still bound captures, not unresolved free variables.
    for (const Var &param : func->params) {
      if (!param.dtype().is_handle())
        vars_[param] = "param";
    }
    VisitStmt(func->body);
  }

  void VisitStmt_(const SBlockNode *op) final {
    auto saved_vars = vars_;
    auto saved_buffers = buffers_;
    auto saved_layouts = layouts_;
    size_t saved_assumptions = assumptions_.size();
    size_t saved_bindings = bindings_.size();

    for (const Buffer &b : op->alloc_buffers)
      AddBuffer(b, "alloc");
    for (const auto &match : op->match_buffers)
      AddBuffer(match->buffer, "view");
    AddLayoutAnnotation(op->annotations);

    if (op->name_hint == "SIMD_VF") {
      VFAssumeCollector vf_assumes;
      vf_assumes(op->body);
      size_t local_saved_assumptions = assumptions_.size();
      assumptions_.insert(assumptions_.end(), vf_assumes.assumptions.begin(),
                          vf_assumes.assumptions.end());
      AnalyzeVF(op);
      assumptions_.resize(local_saved_assumptions);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }

    vars_ = std::move(saved_vars);
    buffers_ = std::move(saved_buffers);
    layouts_ = std::move(saved_layouts);
    assumptions_.resize(saved_assumptions);
    bindings_.resize(saved_bindings);
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    auto saved_vars = vars_;
    auto saved_buffers = buffers_;
    auto saved_layouts = layouts_;
    size_t saved_assumptions = assumptions_.size();
    size_t saved_bindings = bindings_.size();

    for (const Stmt &stmt : op->seq) {
      if (const auto *bind = stmt.as<BindNode>()) {
        vars_[bind->var] = "bind";
        bindings_.emplace_back(bind->var, bind->value);
      } else if (const auto *alloc = stmt.as<AllocBufferNode>()) {
        AddBuffer(alloc->buffer, "alloc");
      } else if (const auto *decl = stmt.as<DeclBufferNode>()) {
        AddBuffer(decl->buffer, "view");
      } else {
        VisitStmt(stmt);
      }
    }

    vars_ = std::move(saved_vars);
    buffers_ = std::move(saved_buffers);
    layouts_ = std::move(saved_layouts);
    assumptions_.resize(saved_assumptions);
    bindings_.resize(saved_bindings);
  }

  void VisitStmt_(const ForNode *op) final {
    auto saved_vars = vars_;

    vars_[op->loop_var] =
        op->kind == ForKind::kThreadBinding ? "thread" : "outer_loop";
    AddUndefinedVars(op->min, &shape_vars_);
    AddUndefinedVars(op->extent, &shape_vars_);
    StmtExprVisitor::VisitStmt_(op);

    vars_ = std::move(saved_vars);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    auto saved_vars = vars_;
    size_t saved_assumptions = assumptions_.size();

    if (op->attr_key == "thread_extent") {
      IterVar iv = Downcast<IterVar>(op->node);
      vars_[iv->var] = "thread";
      AddUndefinedVars(op->value, &shape_vars_);
    } else if (op->attr_key == tirx::attr::tilelang_assume) {
      assumptions_.push_back(Downcast<PrimExpr>(op->node));
    }

    StmtExprVisitor::VisitStmt_(op);

    vars_ = std::move(saved_vars);
    assumptions_.resize(saved_assumptions);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    AddBuffer(op->buffer, "alloc");
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    AddBuffer(op->buffer, "view");
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  Buffer Resolve(const Buffer &buffer) const {
    auto it = buffers_.find(buffer->data);
    if (it != buffers_.end())
      return it->second;
    return Buffer();
  }

  std::string KindOf(const Buffer &buffer) const {
    auto it = vars_.find(buffer->data);
    if (it != vars_.end())
      return it->second;
    return "unknown";
  }

  void AddBuffer(const Buffer &buffer, const std::string &kind) {
    buffers_[buffer->data] = buffer;
    vars_[buffer->data] = kind;
    for (const PrimExpr &shape : buffer->shape)
      AddUndefinedVars(shape, &shape_vars_);
    for (const PrimExpr &stride : buffer->strides)
      AddUndefinedVars(stride, &shape_vars_);
    AddUndefinedVars(buffer->elem_offset, &shape_vars_);
  }

  void AddLayoutAnnotation(const Map<String, Any> &annotations) {
    LayoutMap map;
    if (auto layout = annotations.Get("layout_map")) {
      if (auto bmap = layout.value().as<Map<Buffer, Layout>>()) {
        for (const auto &kv : bmap.value())
          map[kv.first->data] = kv.second;
      } else if (auto vmap = layout.value().as<Map<Var, Layout>>()) {
        for (const auto &kv : vmap.value())
          map[kv.first] = kv.second;
      }
    }
    layouts_.push_back(std::move(map));
  }

  const Layout *FindLayout(const Var &data) const {
    for (auto it = layouts_.rbegin(); it != layouts_.rend(); ++it) {
      auto found = it->find(data);
      if (found != it->end())
        return &found->second;
    }
    return nullptr;
  }

  void AnalyzeVF(const SBlockNode *vf) {
    ManualDetector manual;
    manual(ffi::GetRef<SBlock>(vf));
    if (manual.manual) {
      VSimdCaptureInfo info;
      info.vf_index = vf_index_;
      info.vf_status = "OPAQUE";
      info.status = "OPAQUE";
      info.message = "manual SIMD / nd2nz / local.var";
      infos.push_back(std::move(info));
      ++vf_index_;
      return;
    }

    NestCollector collector;
    collector(vf->body);
    if (collector.nests.empty()) {
      VSimdCaptureInfo info;
      info.vf_index = vf_index_;
      info.vf_status = "OPAQUE";
      info.status = "OPAQUE";
      info.message = "no T.Parallel nest";
      infos.push_back(std::move(info));
      ++vf_index_;
      return;
    }

    std::vector<VSimdCaptureInfo> local;
    bool all_closed = true;
    int nest_index = 0;
    for (const ParallelNest &nest : collector.nests) {
      VSimdCaptureInfo info;
      if (nest.unsupported) {
        // The verdict is per nest: this nest keeps its footprint (domain and
        // accesses) so an OPAQUE region can still report what it touches, while
        // its siblings are analysed normally.
        info.vf_index = vf_index_;
        info.nest_index = nest_index;
        info.root_for = nest.root;
        info.domain = nest.loops;
        info.outer_shells = nest.outer_shells;
        info.accesses = nest.accesses;
        info.status = "OPAQUE";
        info.message = nest.reason;
      } else {
        info = AnalyzeNest(nest, nest_index);
      }
      all_closed = all_closed && info.status == "CLOSED";
      local.push_back(std::move(info));
      ++nest_index;
    }
    for (VSimdCaptureInfo &info : local) {
      info.vf_status = all_closed ? "CLOSED" : "OPEN";
      infos.push_back(std::move(info));
    }
    ++vf_index_;
  }

  VSimdCaptureInfo AnalyzeNest(const ParallelNest &nest, int nest_index) {
    VSimdCaptureInfo info;
    info.vf_index = vf_index_;
    info.nest_index = nest_index;
    info.root_for = nest.root;
    info.domain = nest.loops;
    info.outer_shells = nest.outer_shells;
    info.accesses = nest.accesses;

    VarSet domain;
    for (const LoopInfo &loop : nest.loops)
      domain.insert(loop.var);
    // A serial shell counter is bound by the nest itself (the region carries it
    // as a ``serial`` op), so it is neither a capture nor an outer variable.
    VarSet shell;
    for (const Var &var : nest.shell_vars)
      shell.insert(var);
    VarSet relevant = domain;
    // The same holds for serial loops *enclosing* the nest: they stay in place
    // around the emitted vector code, so their counters are bound too.
    for (const vsimd::LoopInfo &loop : nest.outer_shells) {
      shell.insert(loop.var);
      AddUndefinedVars(loop.min, &relevant);
      AddUndefinedVars(loop.extent, &relevant);
    }

    VarSet seen_capture;
    VarSet seen_outer;
    VarSet seen_shape;
    VarSet seen_layout;
    VarSet unbound_vars;
    using BufferSet =
        std::unordered_set<Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
    BufferSet unbound_buffers;
    BufferSet unbound_local_buffers;
    BufferSet mismatched_views;
    std::unordered_set<Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
        seen_views;

    for (const Buffer &view : nest.local_views) {
      if (!buffers_.count(view->data))
        AddBuffer(view, "local");
    }

    auto classify_var = [&](const Var &var) {
      relevant.insert(var);
      if (domain.count(var) || shell.count(var))
        return;
      if (buffers_.count(var))
        return;
      auto known = vars_.find(var);
      if (known != vars_.end()) {
        if (seen_outer.insert(var).second)
          info.outer_vars.emplace_back(var, known->second);
        return;
      }
      if (shape_vars_.count(var)) {
        if (seen_shape.insert(var).second)
          info.shape_syms.push_back(var);
        return;
      }
      if (unbound_vars.insert(var).second)
        info.errors.push_back(std::string("unbound var ") +
                              std::string(var->name_hint));
    };

    for (const LoopInfo &loop : nest.loops) {
      for (const Var &var : tirx::UndefinedVars(loop.min))
        classify_var(var);
      for (const Var &var : tirx::UndefinedVars(loop.extent))
        classify_var(var);
    }

    for (const AccessInfo &acc : nest.accesses) {
      Buffer base = Resolve(acc.buffer);
      if (!base.defined()) {
        if (unbound_buffers.insert(acc.buffer).second)
          info.errors.push_back(std::string("unbound buffer ") +
                                std::string(acc.buffer->name));
        continue;
      }
      if (!acc.buffer.same_as(base) && seen_views.insert(acc.buffer).second)
        info.local_views.emplace_back(acc.buffer, base);
      if (seen_capture.insert(base->data).second) {
        info.captures.emplace_back(base, KindOf(base));
        for (const PrimExpr &shape : base->shape)
          AddUndefinedVars(shape, &relevant);
        for (const PrimExpr &stride : base->strides)
          AddUndefinedVars(stride, &relevant);
        AddUndefinedVars(base->elem_offset, &relevant);
      }
      if (const Layout *layout = FindLayout(base->data)) {
        (void)layout;
        if (seen_layout.insert(base->data).second)
          info.layouts.emplace_back(base, *layout);
      }

      std::vector<PrimExpr> exprs(acc.indices.begin(), acc.indices.end());
      if (acc.value.defined())
        exprs.push_back(acc.value);
      for (const GuardLiteral &guard : acc.guards)
        exprs.push_back(guard.condition);
      for (const For &loop : acc.serial_loops) {
        exprs.push_back(loop->min);
        exprs.push_back(loop->extent);
      }
      for (const PrimExpr &expr : exprs) {
        for (const Var &var : tirx::UndefinedVars(expr))
          classify_var(var);
      }
    }

    for (const Buffer &view : nest.local_views) {
      Buffer base = Resolve(view);
      if (!base.defined()) {
        if (unbound_local_buffers.insert(view).second)
          info.errors.push_back(std::string("unbound local buffer ") +
                                std::string(view->name));
        continue;
      }
      if (view.same_as(base))
        continue;
      if (!view->data.same_as(base->data)) {
        if (mismatched_views.insert(view).second)
          info.errors.push_back(std::string("local view ") +
                                std::string(view->name) +
                                " resolves to a different data var");
        continue;
      }
      bool duplicate = false;
      for (const auto &entry : info.local_views) {
        if (entry.first.same_as(view) && entry.second.same_as(base)) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate)
        info.local_views.emplace_back(view, base);
    }

    for (const auto &binding : bindings_) {
      if (relevant.count(binding.first))
        info.bindings.push_back(binding);
    }
    std::unordered_set<std::string> seen_assumptions;
    for (const PrimExpr &cond : assumptions_) {
      if (!IsSupportedAssumption(cond))
        continue;
      bool hit = false;
      for (const Var &var : tirx::UndefinedVars(cond)) {
        if (relevant.count(var)) {
          hit = true;
          break;
        }
      }
      if (!hit)
        continue;
      std::string key = std::string(ExprString(cond));
      if (seen_assumptions.insert(key).second)
        info.assumptions.push_back(cond);
    }

    info.status = info.errors.empty() ? "CLOSED" : "OPEN";
    return info;
  }

  VarMap vars_;
  BufferMap buffers_;
  std::vector<LayoutMap> layouts_;
  VarSet shape_vars_;
  std::vector<PrimExpr> assumptions_;
  std::vector<std::pair<Var, PrimExpr>> bindings_;
  int vf_index_{0};
};

} // namespace

std::vector<VSimdCaptureInfo> AnalyzeVSimdCaptureInfo(const PrimFunc &func) {
  CaptureVisitor visitor;
  visitor.Run(func);
  return visitor.infos;
}

String RenderVSimdCaptureReport(const std::vector<VSimdCaptureInfo> &infos) {
  std::ostringstream out;
  int current_vf = -1;
  for (const VSimdCaptureInfo &info : infos) {
    if (info.vf_index != current_vf) {
      current_vf = info.vf_index;
      out << "VF#" << info.vf_index << " status=" << info.vf_status << "\n";
    }
    if (info.nest_index < 0) {
      if (!info.message.empty())
        out << "  message: " << info.message << "\n";
      continue;
    }
    out << "  nest " << info.nest_index << ": status=" << info.status
        << " domain=";
    for (const LoopInfo &loop : info.domain) {
      out << std::string(loop.var->name_hint) << " in [" << loop.min << ", "
          << (loop.min + loop.extent) << ") ";
    }
    out << "\n";
    if (info.status != "CLOSED" && !info.message.empty())
      out << "    message: " << info.message << "\n";
    if (!info.captures.empty()) {
      out << "    captures:";
      for (const auto &capture : info.captures)
        out << " " << capture.first->name << "[" << capture.second << "]";
      out << "\n";
    }
    if (!info.outer_vars.empty()) {
      out << "    outer_vars:";
      for (const auto &var : info.outer_vars)
        out << " " << var.first->name_hint << "[" << var.second << "]";
      out << "\n";
    }
    if (!info.shape_syms.empty()) {
      out << "    shape_syms:";
      for (const Var &var : info.shape_syms)
        out << " " << var->name_hint;
      out << "\n";
    }
    if (!info.local_views.empty()) {
      out << "    local_views:";
      for (const auto &view : info.local_views)
        out << " " << view.first->name << " -> " << view.second->name;
      out << "\n";
    }
    if (!info.layouts.empty()) {
      out << "    layouts:";
      for (const auto &layout : info.layouts)
        out << " " << layout.first->name;
      out << "\n";
    }
    if (!info.assumptions.empty()) {
      out << "    assumptions:";
      for (const PrimExpr &expr : info.assumptions)
        out << " " << expr;
      out << "\n";
    }
    if (!info.errors.empty()) {
      out << "    errors:";
      for (const std::string &error : info.errors)
        out << " " << error;
      out << "\n";
    }
  }
  return String(out.str());
}

String AnalyzeVSimdCaptureScope(const PrimFunc &func) {
  auto infos = AnalyzeVSimdCaptureInfo(func);
  String report = RenderVSimdCaptureReport(infos);
  if (report.empty())
    return String("no SIMD_VF blocks\n");
  return report;
}

Array<Map<String, Any>>
CaptureInfoToFFI(const std::vector<VSimdCaptureInfo> &infos) {
  std::vector<Map<String, Any>> out;
  for (const VSimdCaptureInfo &info : infos)
    out.push_back(CaptureToMap(info));
  return Array<Map<String, Any>>(out.begin(), out.end());
}

} // namespace tl
} // namespace tvm
