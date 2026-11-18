/*!
 * \file model.h
 * \brief Instruction-selection-agnostic plan data structures.
 *
 * C++ counterpart of ``parallel_vectorize/model.py``: ``StorePlan`` /
 * ``ReduceInfo`` /
 * ``VectorPlan`` / ``VectorProgram`` are TVM FFI objects so Python can read the
 * exact same fields the Python dataclasses expose (D2). Dependency direction
 * stays strictly downward:
 *
 *   primitives -> access -> {body, value} -> model -> plan -> {emit, render}
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_MODEL_H_
#define TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_MODEL_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>

#include <optional>
#include <string>
#include <vector>

#include "access.h"
#include "body.h"
#include "value.h"

namespace tvm {
namespace tl {
namespace parallel_vectorize {

class ReduceInfoNode;

/*!
 * \brief Internal (non-reflected) views of the store's string fields.
 *
 * Every ``*_value`` / ``store_lane_mode`` / ``layout_kind`` field stays a
 * string because it is part of the FFI contract the Python tests read. These
 * shadow enums are what the planner and the emitter branch on, so a typo is
 * caught when the store is built instead of at emission time.
 */
enum class StoreLaneMode { kFull = 0, kPkHalf, kPkQuarter };
enum class StoreLayoutKind { kNd = 0, kNz };

/*!
 * \brief Non-default handling of one store relative to the loop skeleton.
 *
 * The array a store lives in already says whether it is a ``pre`` / ``main`` /
 * ``post`` store, so the role only records the deviations the emitter must
 * apply. Deciding them in the plan keeps P3 free of policy: it used to
 * re-derive both rules (folded identity init, write-back superseded by a post),
 * one of them twice.
 */
enum class StoreRole {
  kMain = 0,       // emitted in its array's normal position
  kPreFolded,      // pre store folded into the accumulator initial value
  kPostSuperseded, // main store whose write-back is taken over by a post store
};

/*! \brief ``StoreRole`` text used by reflection / diagnostics. */
const std::string &StoreRoleName(StoreRole role);
/*! \brief Inverse of :func:`StoreRoleName`. */
StoreRole StoreRoleFromName(const std::string &name);

/*!
 * \brief Full plan of one write: what, where, predicate / index vector, reduce.
 *
 * ``promote`` is decided once in the plan phase (same principle as the reduce
 * criterion): the emitter only consumes it and never re-derives it.
 */
class StorePlanNode : public ffi::Object {
public:
  StoreSpec spec;
  AccessInfo access;
  ValueNode value;
  ffi::Optional<ValueNode> mask;
  ffi::String kind_value; // AccessClass name; Python exposes ``kind``
  // Shadow enums (not reflected): the internal branching surface.
  AccessClass kind{AccessClass::kContiguous};
  StoreLaneMode lane_mode_id{StoreLaneMode::kFull};
  StoreLayoutKind layout_id{StoreLayoutKind::kNd};
  ffi::Optional<ValueNode> index;
  ffi::Optional<ffi::ObjectRef> reduce; // ReduceInfo
  ffi::Optional<ValueNode> promote;
  int64_t mask_free_until{0};
  ffi::Optional<int64_t> mask_free_from;
  // Mixed element widths in one segment (Stage 1): ``store_lanes`` is the
  // natural vector-register width of this store's element type, while the
  // segment's chunk width is ``VectorPlan.lanes``. ``store_lane_mode`` is
  // ``full`` when they match, otherwise the store writes ``plan.lanes`` of its
  // own register lanes and must be emitted in the packed form
  // (``PK_B*`` / ``PK4_B*``) whose suffix is ``store_pack_bits``.
  int64_t store_lanes{0};
  ffi::String store_lane_mode{"full"};
  ffi::Optional<int64_t> store_pack_bits;
  // Destination layout (Stage 2): ``nd`` keeps the plain contiguous store,
  // ``nz`` scatters 32B blocks straight from the register (``vsstb``) using
  // ``nz_block_stride`` (block distance, in 32B blocks).
  ffi::String layout_kind{"nd"};
  ffi::Optional<int64_t> nz_block_stride;
  // Plan-decided handling (see ``StoreRole``); ``main`` is the default.
  ffi::String role{"main"};

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.parallel_vectorize.StorePlan",
                                    StorePlanNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StorePlanNode>()
        .def_ro("spec", &StorePlanNode::spec)
        .def_ro("access", &StorePlanNode::access)
        .def_ro("value", &StorePlanNode::value)
        .def_ro("mask", &StorePlanNode::mask)
        .def_ro("kind_value", &StorePlanNode::kind_value)
        .def_ro("index", &StorePlanNode::index)
        .def_ro("reduce", &StorePlanNode::reduce)
        .def_ro("promote", &StorePlanNode::promote)
        .def_ro("mask_free_until", &StorePlanNode::mask_free_until)
        .def_ro("mask_free_from", &StorePlanNode::mask_free_from)
        .def_ro("store_lanes", &StorePlanNode::store_lanes)
        .def_ro("store_lane_mode", &StorePlanNode::store_lane_mode)
        .def_ro("store_pack_bits", &StorePlanNode::store_pack_bits)
        .def_ro("layout_kind", &StorePlanNode::layout_kind)
        .def_ro("nz_block_stride", &StorePlanNode::nz_block_stride)
        .def_ro("role", &StorePlanNode::role);
  }
};

class StorePlan : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StorePlan, ffi::ObjectRef,
                                             StorePlanNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StorePlanNode);
};

/*!
 * \brief Reduction description of one store (design note §2 Case II).
 *
 * ``VectorPlan.reduce`` only marks the segment as a reduction and names the
 * representative store; each store emits its own :attr:`StorePlan.reduce`
 * (a segment may hold several reductions).
 */
class ReduceInfoNode : public ffi::Object {
public:
  Axis axis;
  ffi::String op;
  ffi::String recipe;
  ffi::Optional<ValueNode> varying;
  bool accumulate{false};

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.parallel_vectorize.ReduceInfo",
                                    ReduceInfoNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReduceInfoNode>()
        .def_ro("axis", &ReduceInfoNode::axis)
        .def_ro("op", &ReduceInfoNode::op)
        .def_ro("recipe", &ReduceInfoNode::recipe)
        .def_ro("varying", &ReduceInfoNode::varying)
        .def_ro("accumulate", &ReduceInfoNode::accumulate);
  }
};

class ReduceInfo : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ReduceInfo, ffi::ObjectRef,
                                             ReduceInfoNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ReduceInfoNode);
};

/*!
 * \brief Segment-level plan (one body + optional ``pre``/``post`` siblings).
 */
class VectorPlanNode : public ffi::Object {
public:
  ffi::Array<Axis> axes;
  Axis vector_axis;
  int64_t lanes{0};
  ffi::Optional<int64_t> chunks;
  ffi::Optional<int64_t> tail;
  // Live chunk window ``[chunk_lo, chunk_hi)``: chunks outside it are provably
  // dead (every store's predicate is false for the whole chunk), so the emitter
  // may skip them entirely. ``chunk_hi < 0`` means "no upper clamp".
  int64_t chunk_lo{0};
  int64_t chunk_hi{-1};
  ffi::Array<StorePlan> stores;
  ffi::Array<StorePlan> pre_stores;
  ffi::Array<StorePlan> post_stores;
  ffi::Optional<ffi::ObjectRef> reduce; // ReduceInfo
  ffi::Array<int64_t> guard_boundaries;
  ffi::Array<ffi::String> diagnostics;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.parallel_vectorize.VectorPlan",
                                    VectorPlanNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VectorPlanNode>()
        .def_ro("axes", &VectorPlanNode::axes)
        .def_ro("vector_axis", &VectorPlanNode::vector_axis)
        .def_ro("lanes", &VectorPlanNode::lanes)
        .def_ro("chunks", &VectorPlanNode::chunks)
        .def_ro("tail", &VectorPlanNode::tail)
        .def_ro("chunk_lo", &VectorPlanNode::chunk_lo)
        .def_ro("chunk_hi", &VectorPlanNode::chunk_hi)
        .def_ro("stores", &VectorPlanNode::stores)
        .def_ro("pre_stores", &VectorPlanNode::pre_stores)
        .def_ro("post_stores", &VectorPlanNode::post_stores)
        .def_ro("reduce", &VectorPlanNode::reduce)
        .def_ro("guard_boundaries", &VectorPlanNode::guard_boundaries)
        .def_ro("diagnostics", &VectorPlanNode::diagnostics);
  }
};

class VectorPlan : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(VectorPlan, ffi::ObjectRef,
                                             VectorPlanNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VectorPlanNode);
};

class ReduceInfo;
class PlanNode;

/*!
 * \brief ``build_plan`` result: 1..N segments sharing the hoisted parallel
 * axes.
 *
 * ``shared_shells`` is the optional scalar chain that wraps *every* segment
 * (outermost first). It is empty unless the body held sibling scalar shells
 * whose common ancestors were factored out of the segments, which keeps the
 * source iteration order (one loop instead of one copy per segment) and leaves
 * each segment with a single shell chain. The emitter wraps the segment bodies
 * in it, inside ``axes`` and outside every segment.
 *
 * ``root`` is the structural plan tree (design A): the single source of truth
 * for emission. ``axes`` / ``shared_shells`` / ``segments`` remain as flat
 * compatibility views (tests and rendering read them); ``segments`` is the
 * in-order chunk list of the tree.
 */
class VectorProgramNode : public ffi::Object {
public:
  // ``root`` is a PlanNode at runtime; the base type keeps the header
  // order-independent (PlanNode is defined below).
  ffi::ObjectRef root;
  ffi::Array<Axis> axes;
  ffi::Array<Axis> shared_shells;
  ffi::Array<VectorPlan> segments;
  ffi::Array<ffi::String> diagnostics;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.parallel_vectorize.VectorProgram",
                                    VectorProgramNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VectorProgramNode>()
        .def_ro("root", &VectorProgramNode::root)
        .def_ro("axes", &VectorProgramNode::axes)
        .def_ro("shared_shells", &VectorProgramNode::shared_shells)
        .def_ro("segments", &VectorProgramNode::segments)
        .def_ro("diagnostics", &VectorProgramNode::diagnostics);
  }
};

class VectorProgram : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(VectorProgram, ffi::ObjectRef,
                                             VectorProgramNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VectorProgramNode);
};

/*!
 * \brief Structural plan node (design A, tree plan).
 *
 * The tree is isomorphic to the planned IR: ``kSeq`` holds ordered siblings,
 * ``kLoop`` a materialized loop, and ``kChunk`` the only node that carries
 * vector data instructions (a reused ``VectorPlan``). A lane-uniform ``if`` is
 * not a node: the planner predicates every leaf store of its arms, because the
 * VF contract only allows vector work plus masks (no scalar control flow).
 * Structure nodes only produce control flow; whether a leaf is vectorizable is
 * decided once, in the leaf.
 */
enum class PlanKind : int64_t { kSeq = 0, kLoop = 1, kChunk = 2 };

class PlanNode;

class PlanNodeNode : public ffi::Object {
public:
  // Shadow enum (not reflected); Python reads ``kind_value``.
  PlanKind kind{PlanKind::kSeq};
  ffi::String kind_value{"seq"};

  // kSeq
  // Recursive members use the base ObjectRef: PlanNode is both the tree node
  // and its own reference type, so it is only complete after this class. A
  // registered PlanNode child keeps its runtime type through reflection /
  // FFI; C++ call sites Downcast<PlanNode>.
  ffi::Array<ffi::ObjectRef> children;

  // kLoop (``for_kind`` keeps the source kind for diagnostics; emission is
  // always a serial loop)
  tirx::Var loop_var;
  PrimExpr min;
  PrimExpr extent;
  int64_t for_kind{1};
  ffi::ObjectRef body;
  bool barrier_at_end{false};

  // kChunk
  VectorPlan chunk;
  bool flattened{false};

  // Internal structural summary (not reflected): computed once by the builders
  // so the planner and the emitter never re-walk a subtree to count chunks.
  int64_t chunk_count{0};

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.parallel_vectorize.PlanNode",
                                    PlanNodeNode, ffi::Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PlanNodeNode>()
        .def_ro("kind_value", &PlanNodeNode::kind_value)
        .def_ro("children", &PlanNodeNode::children)
        .def_ro("loop_var", &PlanNodeNode::loop_var)
        .def_ro("min", &PlanNodeNode::min)
        .def_ro("extent", &PlanNodeNode::extent)
        .def_ro("for_kind", &PlanNodeNode::for_kind)
        .def_ro("body", &PlanNodeNode::body)
        .def_ro("barrier_at_end", &PlanNodeNode::barrier_at_end)
        .def_ro("chunk", &PlanNodeNode::chunk)
        .def_ro("flattened", &PlanNodeNode::flattened);
  }
};

class PlanNode : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PlanNode, ffi::ObjectRef,
                                             PlanNodeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(PlanNodeNode);
};

/*! \brief Builders (defaults mirror the Python dataclass defaults). */
StorePlan MakeStorePlan(StoreSpec spec, AccessInfo access, ValueNode value,
                        ffi::Optional<ValueNode> mask, AccessClass kind,
                        ffi::Optional<ValueNode> index = std::nullopt);
ReduceInfo MakeReduceInfo(Axis axis, std::string op, std::string recipe,
                          ValueNode varying = ValueNode(),
                          bool accumulate = false);
VectorPlan MakeVectorPlan(ffi::Array<Axis> axes, Axis vector_axis,
                          int64_t lanes, ffi::Optional<int64_t> chunks,
                          ffi::Optional<int64_t> tail);
VectorProgram MakeVectorProgram(ffi::Array<Axis> axes,
                                ffi::Array<Axis> shared_shells,
                                ffi::Array<VectorPlan> segments);

/*! \brief Structural plan builders (design A). */
PlanNode MakePlanSeq(ffi::Array<PlanNode> children);
PlanNode MakePlanLoop(tirx::Var var, PrimExpr min, PrimExpr extent,
                      int64_t for_kind, PlanNode body, bool barrier_at_end);
PlanNode MakePlanChunk(VectorPlan chunk, bool flattened = false);

/*! \brief The chunk plans of a tree in emission (source) order. */
TVM_DLL ffi::Array<VectorPlan> TreeSegments(const PlanNode &root);

/*! \brief The reduce info of a store (undefined when the store is not one). */
ReduceInfo ReduceOf(const StorePlan &store);
/*! \brief The segment-level reduce info (undefined when not a reduction). */
ReduceInfo ReduceOf(const VectorPlan &plan);

/*!
 * \brief Whether a buffer is per-iteration private storage
 * (``local``/``local.*``).
 *
 * Such storage has one instance per source iteration; vectorizing a parallel
 * axis would make all lanes share it, so it must become a vector register.
 */
bool IsPrivateScope(const tirx::Buffer &buffer);

/*!
 * \brief Whether a store can become a ``local.var`` vector accumulator.
 *
 * This is the single "read-modify-write invariant along the inner axes"
 * criterion, shared by the plan (which *allows* private scalars, S2) and the
 * emitter (which *promotes* shared/private RMW into a register).
 *
 * Returns the same-address old-value node, or an undefined ValueNode.
 */
ValueNode AccumulatorCandidate(const StorePlan &store, const VectorPlan &plan,
                               arith::Analyzer *analyzer);

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PARALLEL_VECTORIZE_MODEL_H_
