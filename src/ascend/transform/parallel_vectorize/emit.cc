/*!
 * \file emit.cc
 * \brief Emission layer implementation (strip-mining, masks, T.simd.* ops).
 */
#include "emit.h"

#include <tvm/ir/repr.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "access.h"
#include "ascend/op/builtin.h"
#include "primitives.h"
#include "value.h"

namespace tvm {
namespace tl {
namespace parallel_vectorize {

tirx::Stmt VstVldBarrier() {
  return tirx::Evaluate(tirx::Call(DataType::Void(), simd_mem_bar(),
                                   {tirx::StringImm("VST_VLD")}));
}

namespace {

using tirx::ForKind;
using tirx::Stmt;

/* ------------------------------------------------------------------------- */
/* T.simd.* call builders (mirror tilelang/ascend/language/simd.py)           */
/* ------------------------------------------------------------------------- */

PrimExpr SimdCall(const Op &op, DataType dtype,
                  const std::vector<PrimExpr> &args) {
  ffi::Array<PrimExpr> array;
  for (const PrimExpr &arg : args)
    array.push_back(arg);
  return tirx::Call(dtype, op, array);
}

PrimExpr StrImm(const std::string &value) { return tirx::StringImm(value); }

/*! \brief ``T.access_ptr(base, "r"/"w", extent)``. */
PrimExpr AccessPtr(const PrimExpr &base_load, const PrimExpr &extent,
                   int rw_mask) {
  return tirx::Call(DataType::Handle(), access_ptr(),
                    {base_load, extent, IntImm(DataType::Int(32), rw_mask)});
}

/*! \brief Literal-extent form (one register width or a packed fraction of it).
 */
PrimExpr AccessPtr(const PrimExpr &base_load, int64_t extent, int rw_mask) {
  return AccessPtr(base_load, IntImm(DataType::Int(32), extent), rw_mask);
}

/*!
 * \brief Elements in the whole buffer allocation (may be symbolic).
 *
 * Used as the declared extent of an *indexed* access: the runtime offset makes
 * the exact range unknowable at compile time, so the instruction conservatively
 * declares its allocation owner — the same may-access contract the PTO path
 * uses for ``vgather``/``vscatter``. Both channels build their index vector
 * from the buffer origin (``IndexAddr``), so the owner range covers every lane.
 */
PrimExpr OwnerElements(const tirx::Buffer &buffer) {
  // ``mul`` requires matching dtypes: a symbolic dimension may be int64 while a
  // static one is int32, and the access extent is int32 everywhere else.
  PrimExpr total = IntImm(DataType::Int(32), 1);
  for (const PrimExpr &dim : buffer->shape) {
    PrimExpr term = dim.dtype() == DataType::Int(32)
                        ? dim
                        : tirx::Cast(DataType::Int(32), dim);
    total = mul(total, term);
  }
  return total;
}

/*! \brief Elements spanned by a ``lanes``-long access with a constant stride.
 */
PrimExpr StridedExtent(int64_t lanes, int64_t stride) {
  int64_t span = stride < 0 ? -stride : stride;
  return IntImm(DataType::Int(32), (lanes - 1) * span + 1);
}

/*!
 * \brief ``vld`` dist that spreads a narrow source into wider containers.
 *
 * A widening ``vcvt`` reads every other (``PART_EVEN``) or every fourth
 * (``PART_P0``) sub-slot of the source register, so a plain ``NORM_*`` load
 * would hand it strided elements. The verified cast kernel therefore loads with
 * ``UNPK_*`` first: the source width picks the dist, the widening factor picks
 * the ``vcvt`` part. A 32-bit source (``i32 -> i64``) has a verified two-way
 * unpack (``UNPK_B32``) but no 64-bit ``NORM`` store on the ISA, so the whole
 * conversion stays unsupported and this returns an empty dist.
 */
std::string UnpkDist(const std::string &elem, int64_t factor) {
  int bits = 0;
  if (!ElemBits(elem, &bits))
    return "";
  if (factor == 4)
    return bits == 8 ? "UNPK4_B8" : "";
  if (factor != 2)
    return "";
  if (bits == 8)
    return "UNPK_B8";
  if (bits == 16)
    return "UNPK_B16";
  return "";
}

DataType BoolMaskDtype() { return DataType::Bool(256); }

PrimExpr Vld(const PrimExpr &load, const tirx::Buffer &buffer,
             const std::string &dist) {
  int lanes = VRegLanes(ElemDtype(buffer->dtype));
  int64_t extent = lanes;
  std::string elem = ElemDtype(buffer->dtype);
  // BRC_B* broadcasts one element of the suffix width.
  if (dist == "BRC_B8" || dist == "BRC_B16" || dist == "BRC_B32") {
    int brc_bits = dist == "BRC_B8" ? 8 : (dist == "BRC_B16" ? 16 : 32);
    int elem_bits = 0;
    ElemBits(elem, &elem_bits);
    extent = brc_bits / elem_bits;
  }
  PrimExpr ptr = AccessPtr(load, extent, /*rw=*/1);
  return SimdCall(simd_vld(), VecDataType(elem), {ptr, StrImm(dist)});
}

PrimExpr Vsts(const PrimExpr &load, const tirx::Buffer &buffer,
              const PrimExpr &src, const PrimExpr &mask,
              const std::string &dist) {
  int lanes = VRegLanes(ElemDtype(buffer->dtype));
  int64_t extent = lanes;
  if (dist.compare(0, 6, "ONEPT_") == 0) {
    extent = 1;
  } else if (dist.compare(0, 3, "PK_") == 0) {
    extent = lanes / 2;
  } else if (dist.compare(0, 4, "PK4_") == 0) {
    extent = lanes / 4;
  }
  PrimExpr ptr = AccessPtr(load, extent, /*rw=*/2);
  return SimdCall(simd_vsts(), DataType::Void(),
                  {ptr, src, mask, StrImm(dist)});
}

PrimExpr Vgather2(const PrimExpr &base_load, const tirx::Buffer &buffer,
                  const PrimExpr &index, const PrimExpr &mask,
                  const PrimExpr &extent) {
  std::string elem = ElemDtype(buffer->dtype);
  PrimExpr ptr = AccessPtr(base_load, extent, /*rw=*/1);
  std::string out_elem = elem;
  if (elem == "int8")
    out_elem = "int16";
  if (elem == "uint8")
    out_elem = "uint16";
  return SimdCall(simd_vgather2(), VecDataType(out_elem), {ptr, index, mask});
}

PrimExpr Vscatter(const PrimExpr &src, const PrimExpr &base_load,
                  const tirx::Buffer &buffer, const PrimExpr &index,
                  const PrimExpr &mask, const PrimExpr &extent) {
  PrimExpr ptr = AccessPtr(base_load, extent, /*rw=*/2);
  return SimdCall(simd_vscatter(), DataType::Void(), {src, ptr, index, mask});
}

/*! \brief ``tl.simd.vsstb``: scatter 32B blocks straight into an NZ
 * destination. */
PrimExpr Vsstb(const PrimExpr &src, const PrimExpr &base_load,
               const tirx::Buffer &buffer, const PrimExpr &mask,
               const PrimExpr &stride, const PrimExpr &extent) {
  PrimExpr ptr = AccessPtr(base_load, extent, /*rw=*/2);
  return SimdCall(simd_vsstb(), DataType::Void(), {src, ptr, stride, mask});
}

/*! \brief ``tl.simd.vdintlv``: de-interleave two registers into a pair. */
PrimExpr Vdintlv(const PrimExpr &low, const PrimExpr &high) {
  return SimdCall(simd_vdintlv(), low.dtype(), {low, high});
}

/*! \brief ``tl.simd.vintlv``: interleave two registers into a pair. */
PrimExpr Vintlv(const PrimExpr &low, const PrimExpr &high) {
  return SimdCall(simd_vintlv(), low.dtype(), {low, high});
}

/*! \brief ``tl.simd.vld2``: one dual-dest load, de-interleaved into a pair. */
PrimExpr Vld2(const PrimExpr &base_load, const tirx::Buffer &buffer) {
  const std::string elem = ElemDtype(buffer->dtype);
  int bits = 0;
  ElemBits(elem, &bits);
  PrimExpr ptr = AccessPtr(base_load, 2 * VRegLanes(elem), /*rw=*/1);
  return SimdCall(simd_vld2(), VecDataType(elem),
                  {ptr, StrImm("DINTLV_B" + std::to_string(bits))});
}

/*! \brief ``tl.simd.pair_get``: one register of a ``vdintlv``/``vintlv`` pair.
 */
PrimExpr PairGet(const PrimExpr &pair, int index) {
  return SimdCall(simd_pair_get(), pair.dtype(), {pair, IntImm32(index)});
}

/*! \brief ``tl.simd.vor``: bitwise or of two registers under ``mask``. */
PrimExpr Vor(const PrimExpr &lhs, const PrimExpr &rhs, const PrimExpr &mask) {
  return SimdCall(simd_vor(), lhs.dtype(),
                  {lhs, rhs, mask, StrImm(ModeZeroing())});
}

/*! \brief ``T.reinterpret`` one register to another element type (same lanes).
 */
PrimExpr ReinterpretAs(const PrimExpr &value, const std::string &elem) {
  return reinterpret(VecDataType(elem), value);
}

PrimExpr Vdup(const PrimExpr &scalar, const std::string &elem,
              const PrimExpr &mask) {
  return SimdCall(simd_vdup(), VecDataType(elem),
                  {scalar, mask, StrImm(ModeZeroing())});
}

/*!
 * \brief ``tl.simd.vci``: per-lane index ramp.
 *
 * CANN's ``asc_arange`` only overloads the signed integer element types
 * (``int8``/``int16``/``int32``/``half``/``float``); there is no unsigned
 * overload in any shipped CANN. The index channel of ``asc_gather`` /
 * ``asc_scatter`` is nevertheless ``vector_u16`` / ``vector_u32``, so an
 * unsigned ramp is built from the signed ramp of the same register width and
 * reinterpreted. Every index the emitter produces is non-negative, so the bit
 * pattern is identical.
 */
PrimExpr Vci(const PrimExpr &index, const std::string &elem) {
  std::string ramp_elem = elem;
  if (elem == "uint32") {
    ramp_elem = "int32";
  } else if (elem == "uint16") {
    ramp_elem = "int16";
  }
  PrimExpr ramp = SimdCall(simd_vci(), VecDataType(ramp_elem),
                           {index, StrImm("INC_ORDER")});
  if (ramp_elem == elem)
    return ramp;
  return ReinterpretAs(ramp, elem);
}

/*!
 * \brief Index element type ``asc_gather`` / ``asc_scatter`` require.
 *
 * CANN only offers the 8-bit-source gather with a ``vector_u16`` index; every
 * wider source uses ``vector_u32``.
 */
std::string IndexElemFor(const std::string &elem) {
  int bits = 0;
  if (ElemBits(elem, &bits) && bits == 8)
    return "uint16";
  return "uint32";
}

PrimExpr Pset(int bits, const std::string &dist) {
  return SimdCall(simd_pset(), BoolMaskDtype(),
                  {IntImm(DataType::Int(32), bits), StrImm(dist)});
}

PrimExpr UpdateMask(const PrimExpr &count, int bits) {
  return SimdCall(simd_update_mask(), BoolMaskDtype(),
                  {count, IntImm(DataType::Int(32), bits)});
}

PrimExpr MethodMask(const Op &op, const std::vector<PrimExpr> &args) {
  return SimdCall(op, BoolMaskDtype(), args);
}

/*!
 * \brief ``T.simd.*`` op of one recipe.
 *
 * This table is the only place a recipe is bound to an instruction: adding
 * e.g. ``vmod`` needs one row here plus one row in the recipe table
 * (``value.cc``), with no change to the emitter's dispatch.
 */
using SimdOpGetter = const Op &(*)();

const std::array<SimdOpGetter, kValueRecipeCount> &SimdOpTable() {
  static const auto table = [] {
    std::array<SimdOpGetter, kValueRecipeCount> ops{};
    ops[static_cast<size_t>(ValueRecipe::kVadd)] = simd_vadd;
    ops[static_cast<size_t>(ValueRecipe::kVsub)] = simd_vsub;
    ops[static_cast<size_t>(ValueRecipe::kVmul)] = simd_vmul;
    ops[static_cast<size_t>(ValueRecipe::kVdiv)] = simd_vdiv;
    ops[static_cast<size_t>(ValueRecipe::kVmax)] = simd_vmax;
    ops[static_cast<size_t>(ValueRecipe::kVmin)] = simd_vmin;
    ops[static_cast<size_t>(ValueRecipe::kVand)] = simd_vand;
    ops[static_cast<size_t>(ValueRecipe::kVor)] = simd_vor;
    ops[static_cast<size_t>(ValueRecipe::kVxor)] = simd_vxor;
    ops[static_cast<size_t>(ValueRecipe::kVshl)] = simd_vshl;
    ops[static_cast<size_t>(ValueRecipe::kVshr)] = simd_vshr;
    ops[static_cast<size_t>(ValueRecipe::kVadds)] = simd_vadds;
    ops[static_cast<size_t>(ValueRecipe::kVmuls)] = simd_vmuls;
    ops[static_cast<size_t>(ValueRecipe::kVmaxs)] = simd_vmaxs;
    ops[static_cast<size_t>(ValueRecipe::kVmins)] = simd_vmins;
    ops[static_cast<size_t>(ValueRecipe::kVshrs)] = simd_vshrs;
    ops[static_cast<size_t>(ValueRecipe::kVshls)] = simd_vshls;
    ops[static_cast<size_t>(ValueRecipe::kVneg)] = simd_vneg;
    ops[static_cast<size_t>(ValueRecipe::kVabs)] = simd_vabs;
    ops[static_cast<size_t>(ValueRecipe::kVexp)] = simd_vexp;
    ops[static_cast<size_t>(ValueRecipe::kVsqrt)] = simd_vsqrt;
    ops[static_cast<size_t>(ValueRecipe::kVln)] = simd_vln;
    return ops;
  }();
  return table;
}

/*! \brief The ``T.simd.*`` op of ``recipe`` (unknown recipes are a bug). */
const Op &SimdOpOf(ValueRecipe recipe) {
  size_t index = static_cast<size_t>(recipe);
  if (index < kValueRecipeCount && SimdOpTable()[index] != nullptr) {
    return SimdOpTable()[index]();
  }
  TVM_FFI_ICHECK(false) << "recipe '" << ValueRecipeName(recipe)
                        << "' has no T.simd.* instruction";
  return simd_vadd();
}

/*! \brief Cross-lane reduction op; ``vcadd`` widens 16-bit integer inputs. */
PrimExpr ReduceCall(const std::string &recipe, const PrimExpr &src,
                    const PrimExpr &mask) {
  std::string src_dtype =
      ffi::DLDataTypeToString(src.dtype()).operator std::string();
  std::string elem = ElemDtype(src_dtype);
  std::string out = src_dtype;
  if (recipe == "vcadd") {
    if (elem == "int16")
      out = VecDtype("int32");
    if (elem == "uint16")
      out = VecDtype("uint32");
  }
  const Op &op = recipe == "vcadd"
                     ? simd_vcadd()
                     : (recipe == "vcmax" ? simd_vcmax() : simd_vcmin());
  return SimdCall(op, DataTypeOf(out), {src, mask, StrImm(ModeZeroing())});
}

/*! \brief The 8-bit float element types ``vcvt`` accepts as a destination. */
bool IsFp8Elem(const std::string &elem) {
  return elem == "float8_e4m3" || elem == "float8_e4m3fn" ||
         elem == "float8_e5m2";
}

/*!
 * \brief ``T.simd.vcvt`` argument protocol for the default round/sat/part.
 *
 * Mirrors ``language/simd.py:vcvt`` with ``round="ROUND_R"``, ``sat=True`` and
 * ``part=0``. Narrowing steps use the packed form (``PART_EVEN``/``PART_P0``
 * plus a ``PK_B*`` store); widening steps are paired with the ``UNPK_*`` load
 * the emitter selects when the load register is wider than the chunk. Pairs
 * whose source is a 64-bit register (``i64->*``) and every 32-bit -> 64-bit
 * widening (``i32/f32->i64``) stay rejected: the ISA has no 64-bit ``NORM``
 * store, so the conversion has no usable consumer. They keep the Python
 * "Unsupported vcvt conversion" rejection, so the nest falls back to scalar.
 */
PrimExpr Vcvt(const PrimExpr &src, const std::string &target_elem,
              const PrimExpr &mask, const std::string &part = "PART_EVEN") {
  std::string src_elem =
      ElemDtype(ffi::DLDataTypeToString(src.dtype()).operator std::string());
  const std::string round = "ROUND_R";
  const std::string sat = "RS_ENABLE";
  const std::string part2 = part;
  const std::string part4 = "PART_P0";
  std::vector<PrimExpr> args = {src, mask};
  auto push = [&](const std::string &value) { args.push_back(StrImm(value)); };
  // ``PART_EVEN`` matches ``simd.vcvt(part=0)`` (the language default) and is
  // what the verified cast kernel emits before a packed ``PK_B*`` store: a
  // narrowing conversion fills the even 16-bit slots of the wider destination
  // register, and the packed store writes exactly those lanes.
  if (src_elem == "float32") {
    if (target_elem == "int32") {
      push(round);
      push(sat);
      push(ModeZeroing());
    } else if (target_elem == "int16") {
      push(round);
      push(sat);
      push(part2);
      push(ModeZeroing());
    } else if (target_elem == "float16" || target_elem == "bfloat16") {
      push(round);
      push(sat);
      push(part2);
      push(ModeZeroing());
    } else if (IsFp8Elem(target_elem)) {
      push(round);
      push(sat);
      push(part4);
      push(ModeZeroing());
    }
  } else if (src_elem == "float16") {
    if (target_elem == "int32") {
      push(round);
      push(part2);
      push(ModeZeroing());
    } else if (target_elem == "int16") {
      push(round);
      push(sat);
      push(ModeZeroing());
    } else if (target_elem == "int8" || target_elem == "uint8") {
      push(round);
      push(sat);
      push(part2);
      push(ModeZeroing());
    } else if (IsFp8Elem(target_elem)) {
      push(round);
      push(sat);
      push(part2);
      push(ModeZeroing());
    } else if (target_elem == "bfloat16") {
      push(round);
      push(ModeZeroing());
    } else if (target_elem == "float32") {
      push(part2);
      push(ModeZeroing());
    }
  } else if (src_elem == "bfloat16") {
    if (target_elem == "int32") {
      push(round);
      push(sat);
      push(part2);
      push(ModeZeroing());
    } else if (target_elem == "float16") {
      push(sat);
      push(round);
      push(ModeZeroing());
    } else if (target_elem == "float32") {
      push(part2);
      push(ModeZeroing());
    } else if (IsFp8Elem(target_elem)) {
      push(round);
      push(sat);
      push(part2);
      push(ModeZeroing());
    }
  } else if (src_elem == "int32" && target_elem == "float32") {
    push(round);
    push(ModeZeroing());
  } else if (src_elem == "int16" && target_elem == "float32") {
    push(part2);
    push(ModeZeroing());
  } else if (src_elem == "int16" && target_elem == "float16") {
    push(round);
    push(ModeZeroing());
  } else if (src_elem == "int32" &&
             (target_elem == "int16" || target_elem == "uint16")) {
    push(sat);
    push(part2);
    push(ModeZeroing());
  } else if (src_elem == "int16" &&
             (target_elem == "int32" || target_elem == "uint32")) {
    push(part2);
    push(ModeZeroing());
  } else if ((src_elem == "int32" || src_elem == "uint32") &&
             target_elem == "uint8") {
    // 4-way narrowing: ``PART_P0`` is the quarter selector ``simd.vcvt`` uses
    // for ``part=0`` on fp8/int8 destinations.
    push(sat);
    push(part4);
    push(ModeZeroing());
  } else if (src_elem == "uint32" &&
             (target_elem == "int16" || target_elem == "uint16")) {
    push(sat);
    push(part2);
    push(ModeZeroing());
  } else if ((src_elem == "int16" || src_elem == "uint16") &&
             target_elem == "uint8") {
    push(sat);
    push(part2);
    push(ModeZeroing());
  } else if (src_elem == "uint16" && target_elem == "uint32") {
    // 2-way widening from a 16-bit source: the load spreads with ``UNPK_B16``
    // and the even sub-slots hold the contiguous elements.
    push(part2);
    push(ModeZeroing());
  } else if ((src_elem == "int8" && target_elem == "int16") ||
             (src_elem == "uint8" && target_elem == "uint16")) {
    // 2-way widening from an 8-bit source: ``UNPK_B8`` load + ``PART_EVEN``.
    push(part2);
    push(ModeZeroing());
  } else if ((src_elem == "int8" || src_elem == "uint8") &&
             target_elem == "float16") {
    // Same recipe as the integer widening: ``UNPK_B8`` + ``PART_EVEN``.
    push(part2);
    push(ModeZeroing());
  } else if (src_elem == "int8" && target_elem == "int32") {
    // 4-way widening from an 8-bit source: ``UNPK4_B8`` load + ``PART_P0``.
    push(part4);
    push(ModeZeroing());
  } else if (src_elem == "uint8" && target_elem == "uint32") {
    push(part4);
    push(ModeZeroing());
  } else if (IsFp8Elem(src_elem) && target_elem == "float32") {
    // The verified cast-kernel recipe: ``UNPK4_B8`` load + ``PART_P0``.
    push(part4);
    push(ModeZeroing());
  }
  if (args.size() == 2) {
    throw Unsupported("Unsupported vcvt conversion " + src_elem + "->" +
                      target_elem);
  }
  return SimdCall(simd_vcvt(), VecDataType(target_elem), args);
}

/*! \brief ``tirx.reinterpret`` a signed index register as unsigned. */
PrimExpr UnsignedIndex(const PrimExpr &reg) {
  std::string dtype =
      ffi::DLDataTypeToString(reg.dtype()).operator std::string();
  if (dtype.compare(0, 3, "int") == 0) {
    dtype.replace(0, 3, "uint");
    return reinterpret(DataTypeOf(dtype), reg);
  }
  return reg;
}

/*!
 * \brief Coerce a per-lane index register to the element type CANN requires.
 *
 * ``asc_gather`` / ``asc_scatter`` take ``vector_u16`` for an 8-bit source and
 * ``vector_u32`` otherwise. A runtime index is usually ``int32``;
 * reinterpreting a 32-bit register as 16-bit would halve the lane count, so the
 * mismatch is rejected instead of silently miscompiled.
 */
PrimExpr IndexAs(const PrimExpr &reg, const std::string &index_elem) {
  PrimExpr index = UnsignedIndex(reg);
  std::string elem =
      ElemDtype(ffi::DLDataTypeToString(index.dtype()).operator std::string());
  if (elem == index_elem)
    return index;
  if (ElemDtype(index_elem) == "uint16" && elem == "uint32") {
    throw Unsupported(
        "a 32-bit runtime index cannot drive an 8-bit gather/scatter: CANN "
        "requires a "
        "vector_u16 index there and a reinterpret would halve the lanes");
  }
  return ReinterpretAs(index, index_elem);
}

/* ------------------------------------------------------------------------- */
/* structural cache keys                                                     */
/* ------------------------------------------------------------------------- */

void ExprKeyInto(const PrimExpr &expr, std::string *out);

/*!
 * \brief Append ``<TypeKeyTail>(a,b)`` when ``expr`` is a binary node.
 *
 * All ``tirx`` arithmetic / comparison / logical nodes share the same
 * ``a``/``b`` layout but have no common non-template base, so only the type
 * list is spelled out; the name comes from the node's own type key, never a
 * parallel table.
 */
template <typename Node>
bool ExprKeyBinary(const PrimExpr &expr, std::string *out) {
  const auto *op = expr.as<Node>();
  if (op == nullptr)
    return false;
  *out += TypeKeyTail(expr);
  *out += "(";
  ExprKeyInto(op->a, out);
  *out += ",";
  ExprKeyInto(op->b, out);
  *out += ")";
  return true;
}

/*!
 * \brief Fast structural key of an expression (stand-in for Python
 * ``str(expr)``).
 *
 * ``ffi::ReprPrint`` costs ~7 us per call, which dominates the whole emitter;
 * this serializer walks the same node shapes and appends to a string, keeping
 * the "structurally equal => same key" guarantee that SSA reuse relies on.
 */
void ExprKeyInto(const PrimExpr &expr, std::string *out) {
  if (!expr.defined()) {
    *out += "~";
    return;
  }
  if (const auto *var = expr.as<tirx::VarNode>()) {
    // Object identity, not the name: two distinct Vars may share a name_hint
    // (per-nest loop vars, rebuilt SSA names), and keying them together made
    // the forwarding / memo maps alias unrelated values. Every key lives only
    // while the plan owning the Var is alive.
    *out += "v:";
    *out += std::to_string(reinterpret_cast<uintptr_t>(var));
    return;
  }
  if (const auto *imm = expr.as<IntImmNode>()) {
    *out += "i:";
    *out += std::to_string(imm->value);
    *out += ":";
    *out += std::to_string(imm->dtype.bits());
    return;
  }
  if (const auto *imm = expr.as<FloatImmNode>()) {
    *out += "f:";
    *out += std::to_string(imm->value);
    *out += ":";
    *out += std::to_string(imm->dtype.bits());
    return;
  }
  if (const auto *load = expr.as<tirx::BufferLoadNode>()) {
    *out += "L:";
    *out += load->buffer->name.operator std::string();
    *out += "(";
    for (const PrimExpr &index : load->indices) {
      ExprKeyInto(index, out);
      *out += ",";
    }
    *out += ")";
    return;
  }
  // Binary / comparison nodes: one generic builder, only the type list below.
  if (ExprKeyBinary<tirx::AddNode>(expr, out) ||
      ExprKeyBinary<tirx::SubNode>(expr, out) ||
      ExprKeyBinary<tirx::MulNode>(expr, out) ||
      ExprKeyBinary<tirx::DivNode>(expr, out) ||
      ExprKeyBinary<tirx::ModNode>(expr, out) ||
      ExprKeyBinary<tirx::FloorDivNode>(expr, out) ||
      ExprKeyBinary<tirx::FloorModNode>(expr, out) ||
      ExprKeyBinary<tirx::MinNode>(expr, out) ||
      ExprKeyBinary<tirx::MaxNode>(expr, out) ||
      ExprKeyBinary<tirx::EQNode>(expr, out) ||
      ExprKeyBinary<tirx::NENode>(expr, out) ||
      ExprKeyBinary<tirx::LTNode>(expr, out) ||
      ExprKeyBinary<tirx::LENode>(expr, out) ||
      ExprKeyBinary<tirx::GTNode>(expr, out) ||
      ExprKeyBinary<tirx::GENode>(expr, out) ||
      ExprKeyBinary<tirx::AndNode>(expr, out) ||
      ExprKeyBinary<tirx::OrNode>(expr, out)) {
    return;
  }
  if (const auto *op = expr.as<tirx::NotNode>()) {
    *out += "Not(";
    ExprKeyInto(op->a, out);
    *out += ")";
    return;
  }
  if (const auto *op = expr.as<tirx::CastNode>()) {
    *out += "Cast:";
    *out += ffi::DLDataTypeToString(op->dtype).operator std::string();
    *out += "(";
    ExprKeyInto(op->value, out);
    *out += ")";
    return;
  }
  if (const auto *op = expr.as<tirx::SelectNode>()) {
    *out += "Select(";
    ExprKeyInto(op->condition, out);
    *out += ",";
    ExprKeyInto(op->true_value, out);
    *out += ",";
    ExprKeyInto(op->false_value, out);
    *out += ")";
    return;
  }
  if (const auto *op = expr.as<tirx::CallNode>()) {
    *out += "C:";
    if (const auto *opn = op->op.as<OpNode>()) {
      *out += opn->name;
    } else if (const auto *gvar = op->op.as<GlobalVarNode>()) {
      // A callee is part of the value identity: two calls with the same args
      // but different functions must never share a key.
      *out += "g:" + std::to_string(reinterpret_cast<uintptr_t>(gvar));
    }
    *out += "(";
    for (const PrimExpr &arg : op->args) {
      ExprKeyInto(arg, out);
      *out += ",";
    }
    *out += ")";
    return;
  }
  // Fallback: rare shapes keep the exact printed form.
  *out += PrintExpr(expr);
}

std::string ExprKey(const PrimExpr &expr) {
  std::string out;
  ExprKeyInto(expr, &out);
  return out;
}

/*! \brief Per-thread buffer-key memo shared by ``BufferKey`` and its reset. */
std::unordered_map<const tirx::BufferNode *, std::string> &BufferKeyCache() {
  thread_local std::unordered_map<const tirx::BufferNode *, std::string> cache;
  return cache;
}

/*!
 * \brief Identity of a buffer: its data Var plus the structural view fields.
 *
 * The data Var identity is mandatory (S0): two same-named allocations produce
 * two distinct data Vars but identical names, dtypes, scopes and shapes, so a
 * purely structural key made the P3 ``forward_`` map (and ``store_gen``) alias
 * them and miscompile the second write (analysis §2.3 / M6).
 */
std::string BufferKey(const tirx::Buffer &buffer) {
  // Memoized per Buffer node for the duration of one ``EmitPlan`` call: the key
  // walks shape / strides / elem_offset and is rebuilt for every load of every
  // hoisting decision (``SplitLoopInvariant`` runs once per chunk). The cache
  // is cleared at each entry point, while the plan's buffers are still alive.
  auto &cache = BufferKeyCache();
  auto cached = cache.find(buffer.get());
  if (cached != cache.end())
    return cached->second;
  // PrintExpr costs microseconds per call and this runs for every buffer of
  // every segment; the fast serializer gives the same equivalence relation.
  std::string out =
      "@" + std::to_string(reinterpret_cast<uintptr_t>(buffer->data.get())) +
      "|" + buffer->data->name_hint.operator std::string() + "|" +
      ffi::DLDataTypeToString(buffer->dtype).operator std::string() + "|" +
      buffer.scope().operator std::string() + "|(";
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    out += (i == 0 ? "" : ",") + ExprKey(buffer->shape[i]);
  }
  out += ")|(";
  for (size_t i = 0; i < buffer->strides.size(); ++i) {
    out += (i == 0 ? "" : ",") + ExprKey(buffer->strides[i]);
  }
  out += ")|" + ExprKey(buffer->elem_offset);
  cache.emplace(buffer.get(), out);
  return out;
}

/*!
 * \brief Drop the memoized buffer keys.
 *
 * Must run at the start of every emission entry point: the keys are keyed by
 * ``BufferNode`` address, which is only stable while the plan owning the buffer
 * is alive.
 */
void ResetBufferKeyCache() { BufferKeyCache().clear(); }

std::string IndexKey(const ffi::Array<PrimExpr> &indices) {
  std::string out;
  for (size_t i = 0; i < indices.size(); ++i) {
    out += (i == 0 ? "" : ",") + ExprKey(indices[i]);
  }
  return out;
}

std::string ValueKey(const ValueNode &node,
                     const std::map<std::string, int64_t> &store_gen);

std::string AccessKey(const AccessInfo &access,
                      const std::map<std::string, int64_t> &store_gen) {
  std::string buffer_key = BufferKey(access->buffer);
  int64_t generation = 0;
  if (!access->is_store) {
    auto it = store_gen.find(buffer_key);
    if (it != store_gen.end())
      generation = it->second;
  }
  return buffer_key + "#" + IndexKey(access->indices) + "#" +
         ExprKey(access->offset) + "#" + std::to_string(generation);
}

std::string ValueKey(const ValueNode &node,
                     const std::map<std::string, int64_t> &store_gen) {
  std::string out = node->recipe.operator std::string() + "|" +
                    node->dtype.operator std::string() + "|" +
                    node->detail.operator std::string() + "|";
  out += node->scalar.has_value() ? ExprKey(*node->scalar) : std::string("-");
  out += "|";
  if (node->access.has_value())
    out += AccessKey(*node->access, store_gen);
  out += "|(";
  for (size_t i = 0; i < node->args.size(); ++i) {
    out += (i == 0 ? "" : ";") +
           ValueKey(Downcast<ValueNode>(node->args[i]), store_gen);
  }
  out += ")|";
  ValueNode predicate = PredicateOf(node);
  if (predicate.defined())
    out += ValueKey(predicate, store_gen);
  return out;
}

std::vector<std::string> LoadBufferKeys(const PrimExpr &expr) {
  std::vector<std::string> keys;
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
    const auto *load = node.as<tirx::BufferLoadNode>();
    if (load != nullptr)
      keys.push_back(BufferKey(load->buffer));
  });
  return keys;
}

} // namespace

namespace {

/* ------------------------------------------------------------------------- */
/* program-shared prologue                                                   */
/* ------------------------------------------------------------------------- */

class ProgramPrologue {
public:
  std::vector<Stmt> statements;
  std::map<std::pair<std::string, int>, tirx::Var> masks;
  std::map<std::tuple<int64_t, const void *, std::string>, tirx::Var> ramps;
  /*! \brief Loop-invariant constants shared by every chunk of the program. */
  std::map<std::pair<std::string, std::string>, tirx::Var> dups;
};

/*! \brief Segment-to-segment register forwarding entry. */
struct CarryEntry {
  tirx::Var axis;
  PrimExpr value;
};

/*!
 * \brief Shared de-interleaved domain of the paired (pack2) emitter.
 *
 * In this domain a single ``vld2`` fetches a whole chunk pair and hands the two
 * parity streams to the even/odd emitters, so every value is a *strided* stream
 * instead of a contiguous register. Contiguous destinations re-interleave the
 * two streams at the store site (``vintlv``); the NZ destination consumes the
 * streams directly (``vcvt`` even/odd parts). Section 3 of the analysis: this
 * trades two ``vld`` + one ``vdintlv`` for one ``vld2`` + one ``vintlv``.
 */
struct PairDomain {
  std::map<std::string, PrimExpr> loads; // pair-base key -> vld2 pair value
  std::vector<Stmt> *lets{nullptr}; // vld2 binds, emitted before both halves
  int next{0};                      // unique name counter
};

/* ------------------------------------------------------------------------- */
/* _Emitter                                                                  */
/* ------------------------------------------------------------------------- */

class Emitter {
public:
  Emitter(const VectorPlan &plan, arith::Analyzer *analyzer, PrimExpr chunk,
          std::string prefix, bool masked_tail = false, int64_t chunk_min = 0,
          std::optional<int64_t> chunk_max = std::nullopt,
          ProgramPrologue *shared = nullptr,
          std::map<std::string, CarryEntry> *carry = nullptr,
          bool allow_carry = false, PairDomain *pair = nullptr,
          int pair_stream = 0)
      : plan_(plan), analyzer_(analyzer), chunk_(std::move(chunk)),
        prefix_(std::move(prefix)), masked_tail_(masked_tail),
        chunk_min_(chunk_min), chunk_max_(chunk_max), shared_(shared),
        carry_(carry != nullptr ? carry : &owned_carry_),
        allow_carry_(allow_carry), pair_(pair), pair_stream_(pair_stream) {
    lane_base_ = chunk_ * IntImm32(plan_->lanes);
    for (const StorePlan &store : plan_->pre_stores)
      written_buffers_.push_back(store->spec->buffer);
    for (const StorePlan &store : plan_->stores)
      written_buffers_.push_back(store->spec->buffer);
    for (const StorePlan &store : plan_->post_stores)
      written_buffers_.push_back(store->spec->buffer);
    for (const tirx::Buffer &buffer : written_buffers_)
      written_keys_.insert(BufferKey(buffer));
  }

  const VectorPlan &plan() const { return plan_; }
  const std::string &prefix() const { return prefix_; }
  int64_t counter() const { return counter_; }
  void set_counter(int64_t value) { counter_ = value; }
  std::vector<Stmt> &lets() { return lets_; }
  std::vector<Stmt> &prologue() { return prologue_; }
  PrimExpr lane_base() const { return lane_base_; }
  void set_lane_base(PrimExpr value) { lane_base_ = std::move(value); }
  bool masked_tail() const { return masked_tail_; }
  void set_masked_tail(bool value) { masked_tail_ = value; }
  void clear_forward() { forward_.clear(); }

  /*!
   * \brief Whether ``value`` is a masked 8-bit load emitted in the widened
   *        (one byte per 16-bit lane) gather form.
   *
   * Such a value must be written back with the packed ``PK_B16`` store: a plain
   * ``NORM_B8`` would reinterpret the 16-bit lanes as bytes and write
   * interleaved garbage. The same-width reinterpret chain is part of the value
   * (a register bit view), so it is looked through.
   *
   * Mirrors the acceptance rule in the ``kVLoad`` emitter: only a static tail
   * of at most half a byte register and no statement predicate can be gathered.
   */
  bool IsByteGatherValue(const ValueNode &value) const {
    const ValueNodeNode *node = value.get();
    while (node != nullptr && node->id == ValueRecipe::kReinterpret &&
           node->args.size() == 1) {
      node = Downcast<ValueNode>(node->args[0]).get();
    }
    if (node == nullptr || node->id != ValueRecipe::kVLoad)
      return false;
    int bits = 0;
    if (!ElemBits(ElemDtype(node->dtype.operator std::string()), &bits) ||
        bits != 8) {
      return false;
    }
    const bool predicated =
        node->predicate.has_value() && node->predicate.value().defined();
    return masked_tail_ && plan_->tail.has_value() && !predicated &&
           *plan_->tail <= plan_->lanes / 2;
  }

  /* -- SSA binding ------------------------------------------------------ */
  tirx::Var Bind(const PrimExpr &expr, std::vector<Stmt> *target) {
    tirx::Var var("v" + prefix_ + std::to_string(counter_), expr.dtype());
    ++counter_;
    target->push_back(tirx::Bind(var, expr));
    return var;
  }

  tirx::Var PrologueBind(const PrimExpr &expr) {
    tirx::Var var = Bind(expr, &prologue_);
    prologue_vars_.insert(var.get());
    return var;
  }

  /*!
   * \brief Fold a just-emitted SSA binding back into its consumer value.
   *
   * ``Build`` binds every instruction result to a fresh SSA var.  A store that
   * consumes that very binding can be emitted in place
   * (``buf = vadds(...)``) instead of copying through a temporary
   * (``auto t = vadds(...); buf = t``).  Only a binding that is the last
   * emitted statement is folded — it was created for this value and cannot be
   * referenced by earlier statements; memo entries for the removed var are
   * dropped so later uses rebuild instead of dangling.
   */
  PrimExpr InlineFreshBind(const PrimExpr &value) {
    if (!value.defined() || lets_.empty())
      return value;
    const auto *bind = lets_.back().as<tirx::BindNode>();
    if (bind == nullptr || !bind->var.same_as(value))
      return value;
    PrimExpr inlined = bind->value;
    lets_.pop_back();
    for (auto it = memo_.begin(); it != memo_.end();) {
      if (it->second.same_as(value)) {
        it = memo_.erase(it);
      } else {
        ++it;
      }
    }
    return inlined;
  }

  /* -- masks ------------------------------------------------------------ */
  std::optional<tirx::Var> SharedMask(const std::pair<std::string, int> &key,
                                      const std::string &name,
                                      const PrimExpr &expr) {
    if (shared_ == nullptr)
      return std::nullopt;
    tirx::Var var;
    auto it = shared_->masks.find(key);
    if (it == shared_->masks.end()) {
      var = tirx::Var(name, DataType::Bool(256));
      shared_->statements.push_back(tirx::Bind(var, expr));
      shared_->masks[key] = var;
    } else {
      var = it->second;
    }
    prologue_vars_.insert(var.get());
    shared_mask_ids_.insert(var.get());
    return var;
  }

  PrimExpr AllMask(int bits) {
    auto key = std::make_pair(std::string("all"), bits);
    auto it = mask_vars_.find(key);
    if (it != mask_vars_.end())
      return it->second;
    PrimExpr expr = Pset(bits, "PAT_ALL");
    std::optional<tirx::Var> shared_var =
        SharedMask(key, "mask_" + std::to_string(bits), expr);
    PrimExpr var;
    if (shared_var.has_value()) {
      var = *shared_var;
    } else {
      tirx::Var local_var("mask" + prefix_ + std::to_string(bits),
                          DataType::Bool(256));
      prologue_.push_back(tirx::Bind(local_var, expr));
      prologue_vars_.insert(local_var.get());
      var = local_var;
    }
    mask_vars_[key] = var;
    return var;
  }

  PrimExpr Lane0Mask(int bits) {
    auto key = std::make_pair(std::string("lane0"), bits);
    auto it = mask_vars_.find(key);
    if (it != mask_vars_.end())
      return it->second;
    PrimExpr expr = Pset(bits, "PAT_VL1");
    std::optional<tirx::Var> shared_var =
        SharedMask(key, "lane0_" + std::to_string(bits), expr);
    PrimExpr var;
    if (shared_var.has_value()) {
      var = *shared_var;
    } else {
      tirx::Var local_var("lane0" + prefix_ + std::to_string(bits),
                          DataType::Bool(256));
      prologue_.push_back(tirx::Bind(local_var, expr));
      prologue_vars_.insert(local_var.get());
      var = local_var;
    }
    mask_vars_[key] = var;
    return var;
  }

  PrimExpr DataMask(const std::string &dtype_like) {
    return AllMask(MaskBits(dtype_like));
  }

  /*!
   * \brief Predicate for an element-wise instruction.
   *
   * A tail chunk only computes ``tail`` lanes. Using the tail predicate keeps
   * the inactive lanes out of the operation: the store already discards them,
   * but a full-mask op would push gather garbage (possibly NaN/Inf) through the
   * datapath. A cross-lane reduction keeps the full mask, because its operands
   * feed a reduce tree whose own mask/fill already models the tail.
   */
  PrimExpr ArithMask(const std::string &dtype_like) {
    if (ReduceOf(plan_).defined())
      return DataMask(dtype_like);
    return RampMask(MaskBits(dtype_like));
  }

  PrimExpr TailMask(int bits) {
    if (!masked_tail_)
      return PrimExpr();
    auto key = std::make_pair(std::string("tail"), bits);
    auto it = mask_vars_.find(key);
    if (it != mask_vars_.end())
      return it->second;
    PrimExpr var;
    if (plan_->tail.has_value()) {
      // A static tail predicate does not depend on the segment, so one shared
      // program-level binding serves every segment that needs it.
      PrimExpr expr = UpdateMask(IntImm32(*plan_->tail), bits);
      std::optional<tirx::Var> shared_var = SharedMask(
          std::make_pair("tail" + std::to_string(*plan_->tail), bits),
          "tail" + std::to_string(*plan_->tail) + "_" + std::to_string(bits),
          expr);
      var = shared_var.has_value() ? *shared_var : PrologueBind(expr);
    } else {
      PrimExpr remaining =
          plan_->vector_axis->extent - chunk_ * IntImm32(plan_->lanes);
      PrimExpr count = min(remaining, IntImm32(plan_->lanes));
      var = Bind(UpdateMask(count, bits), &lets_);
    }
    mask_vars_[key] = var;
    return var;
  }

  PrimExpr PredicateCombine(const PrimExpr &lhs, const PrimExpr &rhs,
                            int bits) {
    return Bind(MethodMask(simd_pand(), {lhs, rhs, AllMask(bits)}), &lets_);
  }

  /*!
   * \brief Mask of a predicate that is a lane-uniform *scalar* (a leaf value).
   *
   * Materializing it as ``update_mask(cond ? lanes : 0)`` needs a scalar select
   * — and, for a UB-backed condition, a scalar load — inside ``__simd_vf__``,
   * which bisheng refuses ("Unsupported scalar instruction in AIV loop"). The
   * mask is therefore built from vector instructions instead: broadcast the
   * condition at the mask's element width and compare it against zero. A scalar
   * *operand* of a vector instruction (the ``vdup`` source) is the same form as
   * ``vadds(v, buf[i], …)``, which the AIV loop accepts.
   *
   * Lane-uniform *comparisons* take the ``vcmps`` path instead (see
   * ``value.cc::Compare``), so only a bare bool leaf (a uniform bool load or a
   * scalar bool parameter) reaches this recipe.
   */
  PrimExpr UniformLeafMask(const ValueNode &node, int bits) {
    PrimExpr cond = analyzer_->Simplify(Scalar(node));
    if (const auto *imm = cond.as<IntImmNode>()) {
      // A compile-time condition needs no comparison: "true" is the shared
      // all-ones mask, "false" the empty mask.
      return imm->value != 0 ? AllMask(bits)
                             : Bind(UpdateMask(IntImm32(0), bits), &lets_);
    }
    const std::string elem = "int" + std::to_string(bits);
    PrimExpr value = cond.dtype() == DataTypeOf(elem)
                         ? cond
                         : tirx::Cast(DataTypeOf(elem), cond);
    // ``vdup``'s scalar operand may read UB directly (a legal
    // vector-instruction operand), so no scalar load is emitted as its own
    // instruction.
    PrimExpr broadcast = Bind(Vdup(value, elem, DataMask(elem)), &lets_);
    return Bind(SimdCall(simd_vcmps(), BoolMaskDtype(),
                         {broadcast, IntImm(DataTypeOf(elem), 0),
                          DataMask(elem), StrImm("ne")}),
                &lets_);
  }

  /*!
   * \brief Element width of the data a predicate recipe compares.
   *
   * A vector mask's lane granularity follows the element type of the values it
   * compares (``vcmps`` compares one vector lane per element). ``update_mask``
   * was the only width-adaptive form, and its argument needs a scalar select,
   * which the AIV loop cannot execute — so a guarded write whose element width
   * differs from the condition's has no legal mask.
   */
  int PredicateDataBits(const ValueNode &node) {
    if (!node.defined())
      return 0;
    switch (node->id) {
    case ValueRecipe::kVcmp:
    case ValueRecipe::kVcmps:
      if (node->args.empty())
        return 0;
      return MaskBits(ElemDtype(
          Downcast<ValueNode>(node->args[0])->dtype.operator std::string()));
    case ValueRecipe::kPnot:
    case ValueRecipe::kPand:
    case ValueRecipe::kPor:
    case ValueRecipe::kPxor:
      for (const ffi::ObjectRef &arg : node->args) {
        int arg_bits = PredicateDataBits(Downcast<ValueNode>(arg));
        if (arg_bits != 0)
          return arg_bits;
      }
      return 0;
    default:
      return 0;
    }
  }

  PrimExpr Predicate(const ValueNode &node, int bits) {
    auto node_it = node_predicates_.find({node.get(), bits});
    if (node_it != node_predicates_.end())
      return node_it->second;
    std::string key = CachedValueKey(node) + "@" + std::to_string(bits);
    auto it = predicates_.find(key);
    if (it != predicates_.end()) {
      node_predicates_[{node.get(), bits}] = it->second;
      return it->second;
    }
    const int mask_width = PredicateDataBits(node);
    if (mask_width != 0 && mask_width != bits) {
      throw Unsupported(
          "a store at " + std::to_string(bits) + "-bit elements cannot use a " +
          std::to_string(mask_width) +
          "-bit predicate mask: the mask granularity follows the condition's "
          "element type "
          "(compare at the data width, e.g. through an explicit cast)");
    }
    PrimExpr mask;
    // Predicate recipes are dispatched before the lane-uniform leaf fallback: a
    // uniform comparison is a ``vcmps`` mask (``is_uniform`` still true, but it
    // has no scalar form), so it must keep the vector path.
    if (node->id == ValueRecipe::kPnot) {
      mask =
          Bind(MethodMask(simd_pnot(),
                          {Predicate(Downcast<ValueNode>(node->args[0]), bits),
                           AllMask(bits)}),
               &lets_);
    } else if (node->id == ValueRecipe::kPand) {
      mask = PredicateCombine(
          Predicate(Downcast<ValueNode>(node->args[0]), bits),
          Predicate(Downcast<ValueNode>(node->args[1]), bits), bits);
    } else if (node->id == ValueRecipe::kPor) {
      mask =
          Bind(MethodMask(simd_por(),
                          {Predicate(Downcast<ValueNode>(node->args[0]), bits),
                           Predicate(Downcast<ValueNode>(node->args[1]), bits),
                           AllMask(bits)}),
               &lets_);
    } else if (node->id == ValueRecipe::kPxor) {
      mask =
          Bind(MethodMask(simd_pxor(),
                          {Predicate(Downcast<ValueNode>(node->args[0]), bits),
                           Predicate(Downcast<ValueNode>(node->args[1]), bits),
                           AllMask(bits)}),
               &lets_);
    } else if (node->is_uniform && (node->id == ValueRecipe::kImm ||
                                    node->id == ValueRecipe::kUniformExpr ||
                                    node->id == ValueRecipe::kUniformLoad)) {
      mask = UniformLeafMask(node, bits);
    } else {
      mask = Value(node);
    }
    predicates_[key] = mask;
    node_predicates_[{node.get(), bits}] = mask;
    return mask;
  }

  PrimExpr RampMask(int bits) {
    PrimExpr tail = TailMask(bits);
    return tail.defined() ? tail : AllMask(bits);
  }

  bool GuardFree(int64_t free_until, const ffi::Optional<int64_t> &free_from) {
    if (free_from.has_value() && *free_from <= chunk_min_)
      return true;
    return chunk_max_.has_value() && free_until >= *chunk_max_;
  }

  PrimExpr LoadGuard(const ValueNode &node, int bits) {
    PrimExpr mask = TailMask(bits);
    ValueNode guard_node = PredicateOf(node);
    if (guard_node.defined() &&
        GuardFree(node->predicate_free_until, node->predicate_free_from)) {
      guard_node = ValueNode();
    }
    if (guard_node.defined()) {
      PrimExpr guard = Predicate(guard_node, bits);
      mask = mask.defined() ? PredicateCombine(mask, guard, bits) : guard;
    }
    return mask;
  }

  /* -- addresses -------------------------------------------------------- */
  ffi::Array<PrimExpr> AddrIndicesAt(const AccessInfo &access,
                                     const PrimExpr &base) {
    ffi::Array<PrimExpr> out;
    for (const PrimExpr &index : access->indices) {
      out.push_back(analyzer_->Simplify(tirx::Substitute(
          index, [&](const tirx::Var &var) -> ffi::Optional<PrimExpr> {
            if (var.same_as(plan_->vector_axis->var))
              return ffi::Optional<PrimExpr>(base);
            return ffi::Optional<PrimExpr>();
          })));
    }
    return out;
  }

  ffi::Array<PrimExpr> AddrIndices(const AccessInfo &access, bool uniform) {
    return AddrIndicesAt(access, uniform ? IntImm32(0) : lane_base_);
  }

  PrimExpr Addr(const AccessInfo &access, bool uniform = false) {
    return tirx::BufferLoad(access->buffer, AddrIndices(access, uniform));
  }

  /*!
   * \brief De-interleaved stream register of a contiguous load (``vld2``
   * domain).
   *
   * The pair base is the *even* chunk's address (the odd emitter subtracts its
   * own chunk back off ``lane_base_``), so both streams share one dual-dest
   * load. The bind goes to the shared list, which is emitted before both
   * halves.
   */
  PrimExpr PairStream(const AccessInfo &access) {
    PrimExpr base = lane_base_ - IntImm32(pair_stream_ * plan_->lanes);
    PrimExpr addr =
        tirx::BufferLoad(access->buffer, AddrIndicesAt(access, base));
    std::string key = ExprKey(addr);
    auto it = pair_->loads.find(key);
    if (it == pair_->loads.end()) {
      PrimExpr load = Vld2(addr, access->buffer);
      tirx::Var var("dl" + prefix_ + std::to_string(pair_->next++),
                    load.dtype());
      pair_->lets->push_back(tirx::Bind(var, load));
      it = pair_->loads.emplace(key, var).first;
    }
    return Bind(PairGet(it->second, pair_stream_), &lets_);
  }

  PrimExpr IndexAddr(const AccessInfo &access) {
    ffi::Array<PrimExpr> zeros;
    for (size_t i = 0; i < access->indices.size(); ++i)
      zeros.push_back(IntImm32(0));
    return tirx::BufferLoad(access->buffer, zeros);
  }

  /* -- scalar operands -------------------------------------------------- */
  PrimExpr Scalar(const ValueNode &node) {
    const std::string &recipe = node->recipe;
    if (node->id == ValueRecipe::kImm ||
        node->id == ValueRecipe::kUniformExpr) {
      if (!node->scalar.has_value()) {
        throw Unsupported("cannot scalarize vector value '" + recipe + "'");
      }
      return *node->scalar;
    }
    if (node->id == ValueRecipe::kUniformLoad) {
      return Addr(*node->access, /*uniform=*/true);
    }
    if (node->id == ValueRecipe::kVbrc) {
      return Scalar(Downcast<ValueNode>(node->args[0]));
    }
    throw Unsupported("cannot scalarize vector value '" + recipe + "'");
  }

  /* -- writes / forwarding ---------------------------------------------- */
  void NoteWrite(const tirx::Buffer &buffer) {
    const std::string &key = CachedBufferKey(buffer);
    store_gen_[key] = store_gen_.count(key) != 0 ? store_gen_[key] + 1 : 1;
    value_keys_.clear();
    if (!forward_.empty()) {
      for (auto it = forward_.begin(); it != forward_.end();) {
        if (it->first.rfind(key, 0) == 0) {
          it = forward_.erase(it);
        } else {
          ++it;
        }
      }
    }
    if (!carry_->empty()) {
      for (auto it = carry_->begin(); it != carry_->end();) {
        if (it->first.rfind(key, 0) == 0) {
          it = carry_->erase(it);
        } else {
          ++it;
        }
      }
    }
  }

  void NoteStore(const StorePlan &store, const PrimExpr &value) {
    // A predicated store leaves the old value in the masked-off lanes, so the
    // destination only equals ``value`` when the write covers every lane. A
    // tail-masked chunk is equally partial. Forwarding either would silently
    // drop the predicate from a later read of the same address.
    if (store->mask.has_value() || masked_tail_)
      return;
    if (store->kind == AccessClass::kContiguous) {
      forward_[CachedBufferKey(store->spec->buffer) + "^" +
               IndexKey(store->spec->indices)] = value;
    }
  }

  void Promote(const ValueNode &node, const PrimExpr &vector) {
    promoted_[CachedValueKey(node)] = vector;
  }

  /* -- values ------------------------------------------------------------ */
  arith::Analyzer *analyzer() { return analyzer_; }

  /*! \brief Record a write-back for segment-to-segment forwarding. */
  void CarryStore(const StorePlan &store, const PrimExpr &value,
                  const tirx::Var &axis) {
    carry_->insert_or_assign(CachedBufferKey(store->spec->buffer) + "^" +
                                 IndexKey(store->spec->indices),
                             CarryEntry{axis, value});
  }

  PrimExpr CarryEntryLookup(const std::string &fwd_key,
                            const std::string &dtype) {
    if (!allow_carry_ || carry_->empty())
      return PrimExpr();
    auto it = carry_->find(fwd_key);
    if (it == carry_->end())
      return PrimExpr();
    if (!it->second.axis.same_as(plan_->vector_axis->var))
      return PrimExpr();
    if (ffi::DLDataTypeToString(it->second.value.dtype())
            .operator std::string() != dtype) {
      return PrimExpr();
    }
    return it->second.value;
  }

  PrimExpr Value(const ValueNode &node) {
    const std::string &key = CachedValueKey(node);
    auto it = memo_.find(key);
    if (it != memo_.end())
      return it->second;
    PrimExpr expr = Build(node);
    memo_[key] = expr;
    return expr;
  }

  PrimExpr VectorOperand(const ValueNode &node) {
    const std::string &key = CachedValueKey(node);
    auto it = promoted_.find(key);
    if (it != promoted_.end())
      return it->second;
    if (node->id == ValueRecipe::kImm ||
        node->id == ValueRecipe::kUniformExpr) {
      return Dup(node);
    }
    return Value(node);
  }

  PrimExpr Dup(const ValueNode &node) {
    std::string elem = ElemDtype(node->dtype.operator std::string());
    PrimExpr scalar = Scalar(node);
    if (ScalarHoistable(node)) {
      auto key = std::make_pair(PrintExpr(scalar), elem);
      auto it = dups_.find(key);
      if (it != dups_.end())
        return it->second;
      PrimExpr expr = Vdup(scalar, elem, DataMask(elem));
      // A true constant (no free variables, no loads) is invariant everywhere,
      // so it can be bound once per program instead of once per chunk and once
      // per iteration of every enclosing materialized loop.
      const bool program_constant = shared_ != nullptr &&
                                    !ContainsBufferLoad(scalar) &&
                                    FreeVars(scalar).empty();
      if (program_constant) {
        auto shared_it = shared_->dups.find(key);
        if (shared_it == shared_->dups.end()) {
          tirx::Var var("dup_" + std::to_string(shared_->dups.size()),
                        expr.dtype());
          shared_->statements.push_back(tirx::Bind(var, expr));
          shared_it = shared_->dups.emplace(key, var).first;
        }
        dups_[key] = shared_it->second;
        return shared_it->second;
      }
      tirx::Var var = PrologueBind(expr);
      dups_[key] = var;
      return var;
    }
    return Bind(Vdup(scalar, elem, DataMask(elem)), &lets_);
  }

  bool ScalarHoistable(const ValueNode &node) {
    PrimExpr expr = Scalar(node);
    if (!expr.defined())
      return true;
    std::vector<tirx::Var> axis_vars;
    for (const Axis &axis : plan_->axes)
      axis_vars.push_back(axis->var);
    bool blocked = false;
    tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef &sub) {
      const auto *load = sub.as<tirx::BufferLoadNode>();
      if (load != nullptr && written_keys_.count(BufferKey(load->buffer)) != 0)
        blocked = true;
      const auto *var = sub.as<tirx::VarNode>();
      if (var != nullptr) {
        for (const tirx::Var &axis_var : axis_vars) {
          if (axis_var.same_as(ffi::GetRef<tirx::Var>(var)))
            blocked = true;
        }
      }
    });
    return !blocked;
  }

  /* -- ramp -------------------------------------------------------------- */
  PrimExpr Ramp(int64_t stride, const PrimExpr &mask,
                const std::string &index_elem) {
    auto key = std::make_tuple(stride, mask.get(), index_elem);
    auto it = ramps_.find(key);
    if (it != ramps_.end())
      return it->second;
    PrimExpr expr = SimdCall(simd_vmuls(), VecDataType(index_elem),
                             {Vci(IntImm32(0), index_elem), IntImm32(stride),
                              mask, StrImm(ModeZeroing())});
    PrimExpr cached;
    const tirx::VarNode *mask_var = mask.as<tirx::VarNode>();
    if (prologue_vars_.count(mask_var) != 0) {
      if (shared_ != nullptr && shared_mask_ids_.count(mask_var) != 0) {
        auto shared_it = shared_->ramps.find(key);
        if (shared_it == shared_->ramps.end()) {
          tirx::Var var("ramp_" + std::to_string(stride), expr.dtype());
          shared_->statements.push_back(tirx::Bind(var, expr));
          shared_->ramps[key] = var;
          cached = var;
        } else {
          cached = shared_it->second;
        }
        prologue_vars_.insert(cached.as<tirx::VarNode>());
      } else {
        cached = PrologueBind(expr);
      }
    } else {
      cached = Bind(expr, &lets_);
    }
    ramps_[key] = cached;
    return cached;
  }

  void SetMemo(const std::string &key, const PrimExpr &value) {
    memo_[key] = value;
  }

  void ClearChunkCaches() {
    memo_.clear();
    predicates_.clear();
    node_predicates_.clear();
  }

  /*!
   * \brief Structural value key, memoized per node.
   *
   * Printing scalars / access keys is by far the most expensive part of the
   * emitter. The plan DAG is immutable, and the key only depends on the write
   * generation of each buffer, so the cache is dropped whenever a write bumps a
   * generation (``NoteWrite``).
   */
  const std::string &CachedValueKey(const ValueNode &node) {
    auto it = value_keys_.find(node.get());
    if (it != value_keys_.end())
      return it->second;
    return value_keys_.emplace(node.get(), ValueKey(node, store_gen_))
        .first->second;
  }

  const std::string &CachedBufferKey(const tirx::Buffer &buffer) {
    auto it = buffer_keys_.find(buffer.get());
    if (it != buffer_keys_.end())
      return it->second;
    return buffer_keys_.emplace(buffer.get(), BufferKey(buffer)).first->second;
  }

  void AppendLets(std::vector<Stmt> stmts) {
    for (Stmt &stmt : stmts)
      lets_.push_back(std::move(stmt));
  }

  std::vector<Stmt> TakeLets() {
    std::vector<Stmt> out = std::move(lets_);
    lets_.clear();
    return out;
  }

  const std::map<std::string, int64_t> &store_gen() const { return store_gen_; }
  ProgramPrologue *shared() const { return shared_; }
  std::map<std::string, CarryEntry> *carry() const { return carry_; }
  bool allow_carry() const { return allow_carry_; }

  std::vector<Stmt> lets_;
  std::vector<Stmt> prologue_;

private:
  PrimExpr Build(const ValueNode &node);

  const VectorPlan &plan_;
  PrimExpr chunk_;
  std::string prefix_;
  PrimExpr lane_base_;
  bool masked_tail_;
  int64_t chunk_min_;
  std::optional<int64_t> chunk_max_;
  ProgramPrologue *shared_;
  std::map<std::string, CarryEntry> owned_carry_;
  std::map<std::string, CarryEntry> *carry_;
  bool allow_carry_;
  int64_t counter_{0};
  std::map<std::pair<std::string, int>, PrimExpr> mask_vars_;
  std::map<std::string, PrimExpr> predicates_;
  /*! \brief Predicate memo keyed by node identity (and mask width).
   *
   * ``predicates_`` uses the generation-sensitive structural key, so a store
   * write invalidates it. A guard shared by several stores of one chunk (and
   * the loads it guards) must be compared once, at the first store: the source
   * ``if`` evaluates the condition once, before its body. The plan DAG is
   * immutable and one chunk builds one guard node, so node identity is a sound
   * key; the memo is cleared per chunk. */
  std::map<std::pair<const ValueNodeNode *, int>, PrimExpr> node_predicates_;
  std::map<std::tuple<int64_t, const void *, std::string>, PrimExpr> ramps_;
  std::set<const tirx::VarNode *> prologue_vars_;
  std::set<const tirx::VarNode *> shared_mask_ids_;
  std::map<std::string, int64_t> store_gen_;
  std::vector<tirx::Buffer> written_buffers_;
  std::set<std::string> written_keys_;
  std::map<std::pair<std::string, std::string>, PrimExpr> dups_;
  std::map<std::pair<std::string, std::string>, PrimExpr> vci_;
  std::map<std::string, PrimExpr> forward_;
  std::map<std::string, PrimExpr> promoted_;
  std::map<std::string, PrimExpr> memo_;
  std::map<const ValueNodeNode *, std::string> value_keys_;
  std::map<const tirx::BufferNode *, std::string> buffer_keys_;
  PairDomain *pair_;
  int pair_stream_{0};
  // Owned by the emission entry point and shared by every chunk / segment
  // emitter: constructing one Analyzer costs ~6 us (five sub-analyzers plus the
  // Z3 client), which dominated the per-chunk emit cost.
  arith::Analyzer *analyzer_{nullptr};
};

PrimExpr Emitter::Build(const ValueNode &node) {
  const std::vector<ValueNode> args = ArgsOf(node);
  const std::string dtype = node->dtype.operator std::string();

  // Dispatch on the *family*, never on the individual recipe: a new op of an
  // existing family is a table row in value.cc / SimdOpTable(), not a new case.
  switch (ValueRecipeInfoOf(node->id).family) {
  case ValueRecipeFamily::kImm:
    if (!node->scalar.has_value()) {
      throw Unsupported("cannot scalarize vector value 'imm'");
    }
    return *node->scalar;
  case ValueRecipeFamily::kLaneIndex: {
    // Absolute lane value = chunk base + lane id (lane-varying compares).
    std::string elem = ElemDtype(dtype);
    auto key = std::make_pair(PrintExpr(lane_base_), elem);
    auto it = vci_.find(key);
    if (it != vci_.end())
      return it->second;
    // A chunk-independent ramp (a single-chunk nest, e.g. a row loop) is
    // loop-invariant: bind it once in the prologue instead of rebuilding it in
    // every iteration of the enclosing serial axis/shell.
    tirx::Var var = FreeVars(lane_base_).empty()
                        ? PrologueBind(Vci(lane_base_, elem))
                        : Bind(Vci(lane_base_, elem), &lets_);
    vci_[key] = var;
    return var;
  }
  case ValueRecipeFamily::kVLoad: {
    std::string fwd_key = CachedBufferKey((*node->access)->buffer) + "^" +
                          IndexKey((*node->access)->indices);
    auto fwd = forward_.find(fwd_key);
    if (fwd != forward_.end() &&
        ffi::DLDataTypeToString(fwd->second.dtype()).operator std::string() ==
            dtype) {
      return fwd->second;
    }
    if (!masked_tail_ && !PredicateOf(node).defined()) {
      PrimExpr carried = CarryEntryLookup(fwd_key, dtype);
      if (carried.defined())
        return carried;
    }
    // A load register wider than the chunk feeds a widening conversion: the
    // chunk only uses its low ``plan_->lanes`` elements, so the load must
    // spread them into the wider conversion containers (``UNPK_*``).
    const std::string load_elem = ElemDtype(dtype);
    const bool widening = VRegLanes(load_elem) > plan_->lanes;
    const int64_t widen_factor =
        widening ? VRegLanes(load_elem) / plan_->lanes : 1;
    int load_bits = 0;
    const bool byte_source = ElemBits(load_elem, &load_bits) && load_bits == 8;
    int bits = MaskBits(dtype);
    // A masked byte load is gathered into 16-bit lanes (``vgather2`` widens
    // one byte per lane), so its predicate is the b16 one: asking for the b8
    // tail mask would leave an unused ``update_mask_b8`` in the prologue and
    // count bytes where the gather counts lanes.
    PrimExpr mask = LoadGuard(node, byte_source ? 16 : bits);
    if (pair_ != nullptr && !mask.defined() && !widening) {
      // De-interleaved pair domain: one ``vld2`` replaces the two contiguous
      // loads (and the explicit ``vdintlv`` the merge would need).
      return PairStream(*node->access);
    }
    if (!mask.defined()) {
      std::string dist = "NORM";
      if (widening) {
        dist = UnpkDist(load_elem, widen_factor);
        if (dist.empty()) {
          throw Unsupported("a widening load of '" + load_elem +
                            "' has no verified unpack layout");
        }
      }
      return Bind(Vld(Addr(*node->access), (*node->access)->buffer, dist),
                  &lets_);
    }
    if (widening) {
      // A two-way widening from an 8-bit source is exactly what the masked
      // gather produces: ``vgather2`` widens every byte into one 16-bit lane,
      // so the masked chunk reaches the conversion already widened. Every
      // other widening needs the guard applied after the unpack, which has no
      // verified form.
      if (!(byte_source && widen_factor == 2)) {
        throw Unsupported(
            "a widening load cannot be emitted through the masked (vgather2) "
            "form; "
            "the guard would have to be applied after the unpack");
      }
      // The gather widens each byte into a 16-bit lane, so ``mask`` was already
      // built at the b16 width above.
      return Bind(Vgather2(Addr(*node->access), (*node->access)->buffer,
                           Vci(IntImm32(0), IndexElemFor(load_elem)), mask,
                           IntImm(DataType::Int(32), plan_->lanes)),
                  &lets_);
    }
    if (byte_source) {
      // Same-width byte chunk: the gather widens each byte into a 16-bit
      // lane, so the value can only feed a plain reinterpreting store (which
      // narrows it back with the packed ``PK_B16`` form). Anything else would
      // read a 256-lane byte register that only has 128 valid slots, and a
      // 256-lane masked gather does not exist (128 destination lanes).
      if (PredicateOf(node).defined() || !masked_tail_ ||
          !plan_->tail.has_value() || *plan_->tail > plan_->lanes / 2) {
        throw Unsupported(
            "an 8-bit masked load can only be gathered in the half-register "
            "form: it needs a static tail of at most " +
            std::to_string(plan_->lanes / 2) +
            " lanes and no statement predicate");
      }
    }
    // vld has no predicate form: the tail/guard case degrades to a masked
    // vgather2 with the relative lane offsets (D10 / P0-3). The index channel
    // is relative to the chunk address, so the declared range stays a chunk.
    // An 8-bit source is widened by the gather, so ``mask`` is already the b16
    // predicate (see the load guard above).
    return Bind(
        Vgather2(Addr(*node->access), (*node->access)->buffer,
                 Vci(IntImm32(0), IndexElemFor(ElemDtype(dtype))), mask,
                 IntImm(DataType::Int(32), VRegLanes(ElemDtype(dtype)))),
        &lets_);
  }
  case ValueRecipeFamily::kVLoadStrided: {
    AxisClass classified = AxisClassOf(*node->access, plan_->vector_axis);
    int bits = MaskBits(dtype);
    if (VRegLanes(ElemDtype(dtype)) > plan_->lanes) {
      throw Unsupported(
          "a widening load cannot be emitted through the strided form; "
          "the strided channel has no unpack layout");
    }
    PrimExpr guard = LoadGuard(node, bits);
    if (!guard.defined())
      guard = AllMask(bits);
    int64_t stride = classified.stride.value_or(0);
    if (stride <= 0) {
      throw Unsupported(
          "a non-positive stride cannot declare a covering access range "
          "(the gather ramp is relative to the chunk address)");
    }
    PrimExpr index =
        Ramp(stride, RampMask(bits), IndexElemFor(ElemDtype(dtype)));
    // The ramp is relative to the chunk address, so the true touched set spans
    // ``(lanes - 1) * stride + 1`` elements; declaring only ``lanes`` would
    // under-cover it for the passes that consume the declaration.
    return Bind(Vgather2(Addr(*node->access), (*node->access)->buffer, index,
                         guard, StridedExtent(plan_->lanes, stride)),
                &lets_);
  }
  case ValueRecipeFamily::kGather: {
    int bits = MaskBits(dtype);
    if (VRegLanes(ElemDtype(dtype)) > plan_->lanes) {
      throw Unsupported(
          "a widening load cannot be emitted through the gather form; "
          "the index channel has no unpack layout");
    }
    PrimExpr mask = LoadGuard(node, bits);
    if (!mask.defined())
      mask = AllMask(bits);
    PrimExpr index = IndexAs(Value(args[0]), IndexElemFor(ElemDtype(dtype)));
    // Runtime offsets: declare the whole allocation owner (may-access).
    return Bind(Vgather2(IndexAddr(*node->access), (*node->access)->buffer,
                         index, mask, OwnerElements((*node->access)->buffer)),
                &lets_);
  }
  case ValueRecipeFamily::kUniform:
    return Dup(node);
  case ValueRecipeFamily::kVbrc:
    return Dup(args[0]);
  case ValueRecipeFamily::kScalarVectorBinary: {
    if (args.size() != 2)
      break;
    // Scalar-vector variant: the "scalar" operand may already be a promoted
    // vector register (a local.var accumulator), in which case the
    // vector-vector form is required.
    PrimExpr vector = Value(args[0]);
    const std::string &operand_key = CachedValueKey(args[1]);
    auto promoted = memo_.find(operand_key);
    if (promoted != memo_.end()) {
      return Bind(SimdCall(SimdOpOf(VectorVariantOf(node->id)),
                           DataTypeOf(dtype),
                           {vector, promoted->second, ArithMask(dtype),
                            StrImm(ModeZeroing())}),
                  &lets_);
    }
    return Bind(SimdCall(SimdOpOf(node->id), DataTypeOf(dtype),
                         {vector, Scalar(args[1]), ArithMask(dtype),
                          StrImm(ModeZeroing())}),
                &lets_);
  }
  case ValueRecipeFamily::kScalarVectorField:
    return Bind(SimdCall(SimdOpOf(node->id), DataTypeOf(dtype),
                         {Value(args[0]), *node->scalar, ArithMask(dtype),
                          StrImm(ModeZeroing())}),
                &lets_);
  case ValueRecipeFamily::kElementwiseUnary:
    if (args.size() != 1)
      break;
    return Bind(
        SimdCall(SimdOpOf(node->id), DataTypeOf(dtype),
                 {Value(args[0]), ArithMask(dtype), StrImm(ModeZeroing())}),
        &lets_);
  case ValueRecipeFamily::kElementwiseBinary:
    if (args.size() != 2)
      break;
    return Bind(SimdCall(SimdOpOf(node->id), DataTypeOf(dtype),
                         {Value(args[0]), Value(args[1]), ArithMask(dtype),
                          StrImm(ModeZeroing())}),
                &lets_);
  case ValueRecipeFamily::kVcmp:
    return Bind(SimdCall(simd_vcmp(), BoolMaskDtype(),
                         {Value(args[0]), Value(args[1]),
                          DataMask(args[0]->dtype.operator std::string()),
                          StrImm(node->detail.operator std::string())}),
                &lets_);
  case ValueRecipeFamily::kVcmps:
    return Bind(SimdCall(simd_vcmps(), BoolMaskDtype(),
                         {Value(args[0]), Scalar(args[1]),
                          DataMask(args[0]->dtype.operator std::string()),
                          StrImm(node->detail.operator std::string())}),
                &lets_);
  case ValueRecipeFamily::kVsel: {
    // args = [cond, true, false]; vsel order is (t, f, mask).
    PrimExpr true_value = VectorOperand(args[1]);
    PrimExpr false_value = VectorOperand(args[2]);
    int bits = MaskBits(ElemDtype(dtype));
    return Bind(SimdCall(simd_vsel(), DataTypeOf(dtype),
                         {true_value, false_value, Predicate(args[0], bits)}),
                &lets_);
  }
  case ValueRecipeFamily::kPredicateBinary: {
    const Op &op = node->id == ValueRecipe::kPand  ? simd_pand()
                   : node->id == ValueRecipe::kPor ? simd_por()
                                                   : simd_pxor();
    return Bind(MethodMask(op, {Value(args[0]), Value(args[1]),
                                AllMask(kPredicateMaskBits)}),
                &lets_);
  }
  case ValueRecipeFamily::kPredicateNot:
    return Bind(
        MethodMask(simd_pnot(), {Value(args[0]), AllMask(kPredicateMaskBits)}),
        &lets_);
  case ValueRecipeFamily::kVcvt: {
    PrimExpr src = Value(args[0]);
    std::string target = node->detail.operator std::string();
    // A masked 8-bit load is gathered through the widening ``vgather2``: its
    // register already holds one 16-bit lane per source byte, so an 8 -> 16
    // conversion has nothing left to do and an 8 -> f16 conversion starts
    // from the widened integer register.
    const std::string dag_elem =
        node->args.empty() ? std::string()
                           : ElemDtype(Downcast<ValueNode>(node->args[0])
                                           ->dtype.operator std::string());
    const std::string reg_elem =
        ElemDtype(ffi::DLDataTypeToString(src.dtype()).operator std::string());
    int dag_bits = 0;
    int reg_bits = 0;
    const bool gather_widened = ElemBits(dag_elem, &dag_bits) &&
                                ElemBits(reg_elem, &reg_bits) &&
                                dag_bits == 8 && reg_bits == 16;
    if (gather_widened) {
      // ``vgather2`` zero-extends an 8-bit source into its 16-bit lanes (CANN
      // intrinsic contract, ``__clang_cce_vector_intrinsics.h`` "the result in
      // the destination vector register is zero-extended to 16-bit"), so only
      // ``uint8 -> uint16`` is already the converted value.
      if (dag_elem == "uint8" && target == "uint16") {
        return src;
      }
      // A signed byte still needs the sign extension. The gathered register
      // holds the raw byte in the low half of each 16-bit lane, i.e. exactly
      // the layout ``UNPK_B8`` produces, so the verified full-chunk recipe
      // (``vcvt`` with ``PART_EVEN``) applies unchanged.
      PrimExpr widened = src;
      if (dag_elem == "int8") {
        widened = Bind(
            Vcvt(ReinterpretAs(src, "int8"), "int16", AllMask(8), "PART_EVEN"),
            &lets_);
      }
      if (target == "int16") {
        return widened;
      }
      if (target == "float16") {
        // uint8 values are exactly representable in int16, so the unsigned
        // gather result converts through the signed same-width recipe.
        PrimExpr as_int16 = reg_elem == "int16" || dag_elem == "int8"
                                ? widened
                                : ReinterpretAs(widened, "int16");
        return Bind(Vcvt(as_int16, "float16", AllMask(MaskBits("int16"))),
                    &lets_);
      }
      throw Unsupported(
          "a masked 8-bit load only widens to int16/uint16/float16 (got " +
          target + ")");
    }
    // 16-bit float -> fp8 has no single-step CANN conversion. The verified
    // cast kernel widens through float32 first: ``UNPK_B16`` load (already
    // selected by the widening-load path), f16/bf16 -> f32 with ``PART_EVEN``
    // under a b16 mask, then f32 -> fp8 with ``PART_P0`` under a b8 mask, and
    // finally a ``PK4_B32`` store.
    if (IsFp8Elem(target) && reg_bits == 16) {
      PrimExpr as_f32 =
          Bind(Vcvt(src, "float32", AllMask(16), "PART_EVEN"), &lets_);
      return Bind(Vcvt(as_f32, target, AllMask(8), "PART_P0"), &lets_);
    }
    int bits = std::min(
        MaskBits(ffi::DLDataTypeToString(src.dtype()).operator std::string()),
        MaskBits(target));
    std::string part = "PART_EVEN";
    return Bind(Vcvt(src, target, AllMask(bits), part), &lets_);
  }
  case ValueRecipeFamily::kReinterpret:
    // Same-width integer signedness change: a register bit reinterpret, so no
    // instruction is emitted (the C++ view type changes only).
    return Bind(ReinterpretAs(Value(args[0]), ElemDtype(dtype)), &lets_);
  case ValueRecipeFamily::kUnknown:
    break;
  }
  throw Unsupported("no emitter for value recipe '" +
                    node->recipe.operator std::string() + "'");
}

/* ------------------------------------------------------------------------- */
/* store emission                                                            */
/* ------------------------------------------------------------------------- */

PrimExpr StoreMask(const StorePlan &store, Emitter *emitter, int bits) {
  PrimExpr mask;
  if (store->mask.has_value() &&
      !emitter->GuardFree(store->mask_free_until, store->mask_free_from)) {
    // The predicate is provably true in this block: use the all-lane mask and
    // skip the whole vcmps chain.
    mask = emitter->Predicate(*store->mask, bits);
  }
  PrimExpr tail = emitter->TailMask(bits);
  if (tail.defined()) {
    mask = mask.defined() ? emitter->PredicateCombine(mask, tail, bits) : tail;
  }
  if (!mask.defined())
    mask = emitter->AllMask(bits);
  return mask;
}

void EmitVectorWrite(const StorePlan &store, Emitter *emitter, int bits,
                     const PrimExpr &mask) {
  PrimExpr value = emitter->VectorOperand(store->value);
  bool uniform = store->kind == AccessClass::kUniform;
  const StoreLaneMode lane_mode = store->lane_mode_id;
  if (store->layout_id == StoreLayoutKind::kNz) {
    if (lane_mode != StoreLaneMode::kFull) {
      // Only the paired emitter widens a half-register source into a full one
      // (``EmitPackedNzStore``); scattering it here would write stale lanes.
      throw Unsupported(
          "a half-register NZ store must be emitted in the paired form");
    }
    // ND -> NZ: scatter the register as 32B blocks at the layout block stride,
    // i.e. store the value straight into the NZ destination (no ND staging).
    //
    // ``vsstb``'s stride is ``(block_stride << 16) | repeat_stride``, where
    // ``repeat_stride`` is the POST_UPDATE pointer advance in 32B blocks. A
    // plain store (the only form emitted here) must pass 0: a non-zero value
    // rewrites every block at ``base + repeat`` and scrambles the destination
    // (measured on 950PR for f32 and bf16).
    int64_t block =
        store->nz_block_stride.has_value() ? *store->nz_block_stride : 0;
    PrimExpr stride = IntImm(DataType::Int(32), block << 16);
    // The 32B-block scatter can span the whole row: declare the owner.
    emitter->lets_.push_back(tirx::Evaluate(
        Vsstb(value, emitter->Addr(store->access, /*uniform=*/false),
              store->spec->buffer, mask, stride,
              OwnerElements(store->spec->buffer))));
    emitter->NoteWrite(store->spec->buffer);
    emitter->NoteStore(store, value);
    return;
  }
  if (lane_mode == StoreLaneMode::kFull && bits == 8 &&
      emitter->IsByteGatherValue(store->value)) {
    // A masked 8-bit load is gathered into 16-bit lanes (``vgather2`` widens a
    // byte source), so the value must be written back with the packed store:
    // ``PK_B16`` writes the low byte of each 16-bit lane contiguously, exactly
    // ``tail`` elements. A plain ``NORM_B8`` would reinterpret the 16-bit lanes
    // as two interleaved bytes and overrun the buffer.
    //
    // The value register has one 16-bit lane per source byte, so the store
    // predicate must be the b16 mask: the b8 mask the load uses counts bytes
    // and would only activate half the lanes (writing half the tail).
    const std::string elem = ElemDtype(store->spec->buffer->dtype);
    PrimExpr bytes = ReinterpretAs(value, elem);
    // The byte-gather validation rejects a statement predicate here, so the
    // only mask is the tail, and it must be re-derived at the 16-bit width.
    PrimExpr tail16 = emitter->TailMask(16);
    PrimExpr store_mask = tail16.defined() ? tail16 : mask;
    emitter->lets_.push_back(
        tirx::Evaluate(Vsts(emitter->Addr(store->access, /*uniform=*/false),
                            store->spec->buffer, bytes, store_mask, "PK_B16")));
    emitter->NoteWrite(store->spec->buffer);
    emitter->NoteStore(store, value);
    return;
  }
  std::string dist;
  if (lane_mode == StoreLaneMode::kPkHalf ||
      lane_mode == StoreLaneMode::kPkQuarter) {
    // Mixed element widths: the value register is wider than the chunk, so the
    // store must write exactly ``plan.lanes`` elements through the packed form
    // (verified cast-kernel recipe: ``vcvt`` with ``PART_EVEN`` + ``PK_B*``).
    int64_t pack_bits =
        store->store_pack_bits.has_value() ? *store->store_pack_bits : bits;
    dist = (lane_mode == StoreLaneMode::kPkHalf ? "PK_B" : "PK4_B") +
           std::to_string(pack_bits);
  } else {
    dist = (uniform ? "ONEPT_B" : "NORM_B") + std::to_string(bits);
  }
  emitter->lets_.push_back(
      tirx::Evaluate(Vsts(emitter->Addr(store->access, uniform),
                          store->spec->buffer, value, mask, dist)));
  emitter->NoteWrite(store->spec->buffer);
  emitter->NoteStore(store, value);
}

/*!
 * \brief The one store of a segment that needs the paired (pack2) emitter.
 *
 * A ``pk_half`` NZ store hands ``lanes`` valid elements to a ``2 * lanes``-lane
 * register and ``vsstb`` has no packed variant, so the plan pairs two chunks
 * (Stage 2b); an empty result keeps the historical single-chunk emission.
 */
std::optional<StorePlan> PackedNzStore(const VectorPlan &plan) {
  for (const StorePlan &store : plan->stores) {
    if (store->layout_kind.operator std::string() != "nz")
      continue;
    if (store->store_lane_mode.operator std::string() != "pk_half")
      continue;
    return store;
  }
  return std::nullopt;
}

/*!
 * \brief Emit the paired form of a half-register NZ store.
 *
 * One loop iteration covers two chunks (columns ``[0, 2 * lanes)`` of the row).
 * The two value registers hold consecutive column halves, so the lanes are
 * de-interleaved first: after ``vdintlv`` the even stream carries columns
 * ``0, 2, 4, ...`` and the odd stream ``1, 3, 5, ...``; ``vcvt`` with
 * ``PART_EVEN`` / ``PART_ODD`` writes each stream into the matching 16-bit slot
 * of one destination register and ``vor`` merges both parts. The merged
 * register holds ``2 * lanes`` consecutive elements and is scattered with a
 * single
 * ``vsstb`` — the same atomic recipe as the hand-written FA softmax.
 *
 * ``even_val`` / ``odd_val`` are the (wider) values of both chunks, already
 * evaluated in their own emitters: the caller appends both ``lets_`` lists
 * before this function's own chain, so the merge always sees bound registers.
 */
void EmitPackedNzStore(const StorePlan &store, Emitter *even,
                       const PrimExpr &even_val, const PrimExpr &odd_val,
                       bool deinterleaved, std::vector<Stmt> *out) {
  const std::string elem = ElemDtype(store->spec->buffer->dtype);
  PrimExpr mask = even->AllMask(MaskBits(elem));
  auto bind = [&](const PrimExpr &expr) {
    tirx::Var var("pk" + even->prefix() + std::to_string(out->size()),
                  expr.dtype());
    out->push_back(tirx::Bind(var, expr));
    return var;
  };
  PrimExpr lo = even_val;
  PrimExpr hi = odd_val;
  if (!deinterleaved) {
    // Contiguous halves: split them into the even/odd column streams first.
    PrimExpr pair = bind(Vdintlv(even_val, odd_val));
    lo = bind(PairGet(pair, 0));
    hi = bind(PairGet(pair, 1));
  }
  PrimExpr lo_narrow = bind(Vcvt(lo, elem, mask, "PART_EVEN"));
  PrimExpr hi_narrow = bind(Vcvt(hi, elem, mask, "PART_ODD"));
  // ``vor`` is bitwise, so the halves are merged as raw 16-bit lanes.
  PrimExpr merged_bits = bind(Vor(ReinterpretAs(lo_narrow, "uint16"),
                                  ReinterpretAs(hi_narrow, "uint16"), mask));
  PrimExpr merged = bind(ReinterpretAs(merged_bits, elem));
  // Same encoding as the plain NZ store: a non-post-update ``vsstb`` carries
  // only the 32B block stride (low 16 bits are the POST_UPDATE advance = 0).
  int64_t block =
      store->nz_block_stride.has_value() ? *store->nz_block_stride : 0;
  PrimExpr stride = IntImm(DataType::Int(32), block << 16);
  out->push_back(tirx::Evaluate(Vsstb(
      merged, even->Addr(store->access, /*uniform=*/false), store->spec->buffer,
      mask, stride, OwnerElements(store->spec->buffer))));
  even->NoteWrite(store->spec->buffer);
  even->NoteStore(store, merged);
}

/*!
 * \brief Whether one value keeps its lanes positionally independent.
 *
 * The de-interleaved domain permutes lane positions (even/odd streams), so a
 * value may only live there when every recipe is lane-wise: contiguous loads,
 * lane-uniform leaves and element-wise arithmetic. Reductions, permutes,
 * gathers, index vectors and predicated values all break the permutation.
 */
bool LaneLocalValue(const ValueNode &node) {
  if (!node.defined() || node->predicate.has_value())
    return false;
  switch (ValueRecipeInfoOf(node->id).family) {
  case ValueRecipeFamily::kImm:
  case ValueRecipeFamily::kUniform:
  case ValueRecipeFamily::kVbrc:
  case ValueRecipeFamily::kVLoad:
    return true;
  case ValueRecipeFamily::kVcvt:
  case ValueRecipeFamily::kReinterpret:
    return node->args.size() == 1 &&
           LaneLocalValue(Downcast<ValueNode>(node->args[0]));
  default:
    break;
  }
  // Scalar-vector recipes carry the scalar as the second argument node.
  const bool unary = IsElementwiseUnary(node->id) && node->args.size() == 1;
  const bool binary = IsElementwiseBinary(node->id) && node->args.size() == 2;
  const bool scalar_binary =
      IsScalarVectorBinary(node->id) && node->args.size() == 2;
  if (!unary && !binary && !scalar_binary)
    return false;
  for (const ffi::ObjectRef &arg : node->args) {
    if (!LaneLocalValue(Downcast<ValueNode>(arg)))
      return false;
  }
  return true;
}

/*! \brief Whether a whole chunk pair can be emitted in the de-interleaved
 * domain. */
bool PairDomainAllowed(const VectorPlan &plan, const StorePlan &packed) {
  for (const StorePlan &store : plan->stores) {
    if (store->kind != AccessClass::kContiguous)
      return false;
    if (store->mask.has_value())
      return false;
    if (!store.same_as(packed) &&
        store->store_lane_mode.operator std::string() != "full") {
      // A packed ND store needs its own half-register handling.
      return false;
    }
    if (!LaneLocalValue(store->value))
      return false;
  }
  return true;
}

/*!
 * \brief Emit a contiguous (ND) store out of the de-interleaved pair domain.
 *
 * Both streams are re-interleaved into the original lane order at the store
 * site, so the destination sees the same two contiguous chunk registers as
 * before. The forward map keeps the *stream* values: a later load of the same
 * address inside the segment stays in the de-interleaved domain.
 */
void EmitPairContiguousStore(const StorePlan &store, Emitter *even,
                             Emitter *odd, const PrimExpr &even_val,
                             const PrimExpr &odd_val, std::vector<Stmt> *out) {
  const std::string elem = ElemDtype(store->spec->buffer->dtype);
  const int bits = MaskBits(elem);
  PrimExpr mask = even->AllMask(bits);
  PrimExpr pair = even->Bind(Vintlv(even_val, odd_val), out);
  const std::string dist = "NORM_B" + std::to_string(bits);
  out->push_back(
      tirx::Evaluate(Vsts(even->Addr(store->access), store->spec->buffer,
                          PairGet(pair, 0), mask, dist)));
  out->push_back(
      tirx::Evaluate(Vsts(odd->Addr(store->access), store->spec->buffer,
                          PairGet(pair, 1), mask, dist)));
  even->NoteWrite(store->spec->buffer);
  odd->NoteWrite(store->spec->buffer);
  even->NoteStore(store, even_val);
  odd->NoteStore(store, odd_val);
}

void EmitStore(const VectorPlan &plan, const StorePlan &store,
               Emitter *emitter) {
  std::string elem = ElemDtype(store->spec->buffer->dtype);
  int bits = MaskBits(elem);
  PrimExpr mask = StoreMask(store, emitter, bits);
  AccessClass kind = store->kind;

  if (kind == AccessClass::kContiguous || kind == AccessClass::kUniform) {
    EmitVectorWrite(store, emitter, bits, mask);
    return;
  }

  PrimExpr value = emitter->VectorOperand(store->value);
  if (kind == AccessClass::kStrided) {
    // Constant *positive* stride writes use the stride ramp; a negative stride
    // falls through to the index-vector channel (D8).
    AxisClass classified = AxisClassOf(store->access, plan->vector_axis);
    if (classified.stride.has_value() && *classified.stride > 0) {
      int64_t stride = *classified.stride;
      PrimExpr index =
          emitter->Ramp(stride, emitter->RampMask(bits),
                        IndexElemFor(ElemDtype(store->spec->buffer->dtype)));
      // The ramp is relative to the chunk address: declare the spanned
      // envelope.
      emitter->lets_.push_back(tirx::Evaluate(
          Vscatter(value, emitter->Addr(store->access), store->spec->buffer,
                   index, mask, StridedExtent(plan->lanes, stride))));
      emitter->NoteWrite(store->spec->buffer);
      return;
    }
  }

  if (!store->index.has_value()) {
    throw Unsupported("no per-lane index vector for a " +
                      AccessClassName(store->kind) + " store");
  }
  PrimExpr index = IndexAs(emitter->Value(*store->index),
                           IndexElemFor(ElemDtype(store->spec->buffer->dtype)));
  // The index vector is absolute (``IndexAddr`` is the buffer origin), so the
  // declared range is the whole allocation owner.
  emitter->lets_.push_back(tirx::Evaluate(
      Vscatter(value, emitter->IndexAddr(store->access), store->spec->buffer,
               index, mask, OwnerElements(store->spec->buffer))));
  emitter->NoteWrite(store->spec->buffer);
}

void EmitOuterStore(const StorePlan &store, Emitter *emitter) {
  AccessClass kind = store->kind;
  if (kind != AccessClass::kUniform) {
    throw Unsupported("a statement outside the serial loop must be uniform "
                      "along the vector axis (got " +
                      AccessClassName(store->kind) + ")");
  }
  std::string elem = ElemDtype(store->spec->buffer->dtype);
  int bits = MaskBits(elem);
  if (!store->mask.has_value() && store->value->is_uniform) {
    PrimExpr scalar;
    bool ok = true;
    try {
      scalar = emitter->Scalar(store->value);
    } catch (const Unsupported &) {
      ok = false;
    }
    if (ok && scalar.defined()) {
      emitter->lets_.push_back(tirx::BufferStore(
          store->spec->buffer, scalar,
          emitter->AddrIndices(store->access, /*uniform=*/true)));
      emitter->NoteWrite(store->spec->buffer);
      return;
    }
  }
  PrimExpr mask = store->mask.has_value()
                      ? emitter->Predicate(*store->mask, bits)
                      : emitter->AllMask(bits);
  EmitVectorWrite(store, emitter, bits, mask);
}

/* ------------------------------------------------------------------------- */
/* strip-mining                                                              */
/* ------------------------------------------------------------------------- */

std::vector<Stmt> SeqOfVec(const std::vector<Stmt> &stmts) {
  ffi::Array<Stmt> array;
  for (const Stmt &stmt : stmts)
    array.push_back(stmt);
  Stmt folded = SeqOf(array);
  if (!folded.defined())
    return {};
  return {folded};
}

struct Block {
  std::vector<Stmt> prologue;
  std::vector<Stmt> lets;
};

using MakeBlock = std::function<Block(
    const PrimExpr &chunk_index, bool masked_tail, const std::string &label,
    int64_t chunk_min, std::optional<int64_t> chunk_max)>;

std::pair<std::vector<Stmt>, std::vector<Stmt>>
StripMine(const VectorPlan &plan, const tirx::Var &chunk,
          const MakeBlock &make_block) {
  std::vector<Stmt> prologue;

  auto block = [&](const PrimExpr &index, bool masked_tail,
                   const std::string &label, int64_t chunk_min,
                   std::optional<int64_t> chunk_max) {
    Block result = make_block(index, masked_tail, label, chunk_min, chunk_max);
    for (Stmt &stmt : result.prologue)
      prologue.push_back(std::move(stmt));
    return result.lets;
  };

  std::function<std::vector<Stmt>(int64_t, int64_t, const std::string &)>
      range_blocks = [&](int64_t lo, int64_t hi,
                         const std::string &label) -> std::vector<Stmt> {
    if (hi <= lo)
      return {};
    if (hi - lo == 1) {
      return SeqOfVec(block(IntImm32(lo), false, label, lo, hi));
    }
    std::vector<Stmt> body = SeqOfVec(block(chunk, false, label, lo, hi));
    std::vector<Stmt> out;
    out.push_back(tirx::For(chunk, IntImm32(lo), IntImm32(hi), ForKind::kSerial,
                            body.empty() ? Stmt() : body[0]));
    return out;
  };

  if (!plan->chunks.has_value()) {
    PrimExpr extent = plan->vector_axis->extent;
    PrimExpr full = tirx::FloorDiv(extent, IntImm32(plan->lanes));
    std::vector<Stmt> loop_body =
        SeqOfVec(block(chunk, false, "", 0, std::nullopt));
    std::vector<Stmt> blocks;
    blocks.push_back(tirx::For(chunk, IntImm32(0), full, ForKind::kSerial,
                               loop_body.empty() ? Stmt() : loop_body[0]));
    PrimExpr tail_chunk = tirx::FloorDiv(extent, IntImm32(plan->lanes));
    std::vector<Stmt> tail_body =
        SeqOfVec(block(tail_chunk, true, "tail_", 0, std::nullopt));
    blocks.push_back(tirx::IfThenElse(
        tirx::GT(tirx::FloorMod(extent, IntImm32(plan->lanes)), IntImm32(0)),
        tail_body.empty() ? Stmt() : tail_body[0], std::nullopt));
    return {prologue, blocks};
  }

  std::vector<int64_t> cuts(plan->guard_boundaries.begin(),
                            plan->guard_boundaries.end());

  std::function<std::vector<Stmt>(int64_t, int64_t)> split_range =
      [&](int64_t lo, int64_t hi) -> std::vector<Stmt> {
    std::vector<int64_t> points = {lo};
    for (int64_t cut : cuts) {
      if (cut > lo && cut < hi)
        points.push_back(cut);
    }
    points.push_back(hi);
    std::vector<Stmt> blocks;
    for (size_t index = 0; index + 1 < points.size(); ++index) {
      std::string label =
          points.size() == 2 ? "" : "gr" + std::to_string(index) + "_";
      for (Stmt &stmt : range_blocks(points[index], points[index + 1], label)) {
        blocks.push_back(std::move(stmt));
      }
    }
    return blocks;
  };

  // The plan may have proven a prefix / suffix of chunks dead (every store's
  // predicate is false there), in which case only the live window is emitted.
  const int64_t chunks = *plan->chunks;
  const int64_t live_lo =
      std::max<int64_t>(0, std::min<int64_t>(plan->chunk_lo, chunks));
  const int64_t live_hi =
      plan->chunk_hi < 0
          ? chunks
          : std::max<int64_t>(live_lo,
                              std::min<int64_t>(plan->chunk_hi, chunks));

  bool has_tail = plan->tail.has_value() && *plan->tail != 0;
  if (!has_tail) {
    if (live_hi - live_lo == 1) {
      // A single live chunk is straight-line code: ``for chunk in range(1)``
      // only confuses the backend.
      return {prologue, block(IntImm32(live_lo), false, "", live_lo, live_hi)};
    }
    return {prologue, split_range(live_lo, live_hi)};
  }

  const int64_t full = chunks - 1;
  std::vector<Stmt> blocks = split_range(live_lo, std::min(live_hi, full));
  // The tail block covers the last chunk, which is live only if the window
  // still reaches it.
  if (live_hi > full) {
    for (Stmt &stmt :
         SeqOfVec(block(IntImm32(full), true, "tail_", full, full + 1))) {
      blocks.push_back(std::move(stmt));
    }
  }
  return {prologue, blocks};
}

std::pair<std::vector<Stmt>, std::vector<Stmt>> EmitElementwiseChunks(
    const VectorPlan &plan, arith::Analyzer *analyzer, const tirx::Var &chunk,
    const std::string &prefix, ProgramPrologue *shared,
    std::map<std::string, CarryEntry> *carry, bool allow_carry) {
  std::optional<StorePlan> packed = PackedNzStore(plan);
  if (!packed.has_value()) {
    MakeBlock make_block = [&](const PrimExpr &chunk_index, bool masked_tail,
                               const std::string &label, int64_t chunk_min,
                               std::optional<int64_t> chunk_max) -> Block {
      Emitter body(plan, analyzer, chunk_index, prefix + label, masked_tail,
                   chunk_min, chunk_max, shared, carry, allow_carry, nullptr,
                   0);
      for (const StorePlan &store : plan->stores)
        EmitStore(plan, store, &body);
      return Block{body.prologue_, body.lets_};
    };
    return StripMine(plan, chunk, make_block);
  }

  // Stage 2b: one loop iteration covers a chunk *pair*. Both halves are emitted
  // by their own emitter, the packed store is deferred until both value chains
  // exist, and only then is the pair merged.
  TVM_FFI_ICHECK(plan->chunks.has_value())
      << "the paired NZ emitter needs static chunk counts (plan contract)";
  int64_t pairs = *plan->chunks / 2;
  const StorePlan &packed_store = packed.value();
  // Lane-wise bodies run in the de-interleaved domain: one ``vld2`` instead of
  // two loads plus the explicit ``vdintlv`` the merge would otherwise need.
  const bool deinterleave = PairDomainAllowed(plan, packed_store);

  auto emit_pair = [&](const PrimExpr &pair_index) -> Block {
    const PrimExpr even_chunk = pair_index * IntImm32(2);
    const PrimExpr odd_chunk = even_chunk + IntImm32(1);
    std::vector<Stmt> shared_lets;
    PairDomain domain;
    domain.lets = &shared_lets;
    Emitter even(plan, analyzer, even_chunk, prefix + "e", false, 0,
                 std::nullopt, shared, carry, allow_carry,
                 deinterleave ? &domain : nullptr, /*pair_stream=*/0);
    Emitter odd(plan, analyzer, odd_chunk, prefix + "o", false, 0, std::nullopt,
                shared, carry, allow_carry, deinterleave ? &domain : nullptr,
                /*pair_stream=*/1);
    // Value binds and the write they feed stay in source order (a later store
    // may read what an earlier one wrote through the forward map).
    std::vector<Stmt> body;
    auto flush = [&]() {
      for (Stmt &stmt : even.TakeLets())
        body.push_back(std::move(stmt));
      for (Stmt &stmt : odd.TakeLets())
        body.push_back(std::move(stmt));
    };
    for (const StorePlan &store : plan->stores) {
      if (store.same_as(packed_store))
        continue;
      if (deinterleave) {
        PrimExpr even_val = even.VectorOperand(store->value);
        PrimExpr odd_val = odd.VectorOperand(store->value);
        flush();
        EmitPairContiguousStore(store, &even, &odd, even_val, odd_val, &body);
      } else {
        EmitStore(plan, store, &even);
        EmitStore(plan, store, &odd);
      }
    }
    const ValueNode &packed_value =
        Downcast<ValueNode>(packed_store->value->args[0]);
    PrimExpr even_val = even.VectorOperand(packed_value);
    PrimExpr odd_val = odd.VectorOperand(packed_value);
    flush();
    Block block;
    for (Stmt &stmt : even.prologue_)
      block.prologue.push_back(std::move(stmt));
    for (Stmt &stmt : odd.prologue_)
      block.prologue.push_back(std::move(stmt));
    for (Stmt &stmt : shared_lets)
      block.lets.push_back(std::move(stmt));
    for (Stmt &stmt : body)
      block.lets.push_back(std::move(stmt));
    EmitPackedNzStore(packed_store, &even, even_val, odd_val, deinterleave,
                      &block.lets);
    return block;
  };

  if (pairs == 1) {
    Block single = emit_pair(IntImm32(0));
    return {single.prologue, single.lets};
  }
  Block pair_loop = emit_pair(chunk);
  Stmt loop_body =
      SeqOfVec(pair_loop.lets).empty() ? Stmt() : SeqOfVec(pair_loop.lets)[0];
  return {pair_loop.prologue,
          {tirx::For(chunk, IntImm32(0), IntImm32(pairs), ForKind::kSerial,
                     loop_body)}};
}

/* ------------------------------------------------------------------------- */
/* LICM helpers                                                              */
/* ------------------------------------------------------------------------- */

std::pair<std::vector<Stmt>, std::vector<Stmt>>
SplitLoopInvariant(const std::vector<Stmt> &lets,
                   const std::vector<Axis> &inner,
                   const std::set<std::string> &written_keys,
                   const std::vector<tirx::Var> &pinned) {
  std::vector<tirx::Var> inner_vars;
  for (const Axis &axis : inner)
    inner_vars.push_back(axis->var);
  std::vector<tirx::Var> loop_defined = pinned;
  std::vector<Stmt> hoisted;
  std::vector<Stmt> kept;
  for (const Stmt &stmt : lets) {
    const auto *bind = stmt.as<tirx::BindNode>();
    if (bind == nullptr) {
      kept.push_back(stmt);
      continue;
    }
    bool blocked = false;
    for (const tirx::Var &var : FreeVars(bind->value)) {
      for (const tirx::Var &inner_var : inner_vars) {
        if (var.same_as(inner_var))
          blocked = true;
      }
      for (const tirx::Var &loop_var : loop_defined) {
        if (var.same_as(loop_var))
          blocked = true;
      }
    }
    for (const std::string &key : LoadBufferKeys(bind->value)) {
      if (written_keys.count(key) != 0)
        blocked = true;
    }
    if (!blocked) {
      hoisted.push_back(stmt);
      continue;
    }
    loop_defined.push_back(bind->var);
    kept.push_back(stmt);
  }
  return {hoisted, kept};
}

bool StoreIsLoopInvariant(const StorePlan &store, const VectorPlan &plan,
                          const std::set<std::string> &written_keys) {
  std::vector<tirx::Var> axis_vars;
  for (const Axis &axis : plan->axes)
    axis_vars.push_back(axis->var);
  std::vector<PrimExpr> exprs(store->spec->indices.begin(),
                              store->spec->indices.end());
  exprs.push_back(store->spec->value);
  if (store->spec->mask.has_value())
    exprs.push_back(*store->spec->mask);
  bool blocked = false;
  for (const PrimExpr &expr : exprs) {
    tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
      const auto *var = node.as<tirx::VarNode>();
      if (var != nullptr) {
        ffi::ObjectRef ref = ffi::GetRef<ffi::ObjectRef>(var);
        for (const tirx::Var &axis_var : axis_vars) {
          if (axis_var.same_as(ffi::GetRef<tirx::Var>(var)))
            blocked = true;
        }
        (void)ref;
      }
      const auto *load = node.as<tirx::BufferLoadNode>();
      if (load != nullptr && written_keys.count(BufferKey(load->buffer)) != 0)
        blocked = true;
    });
  }
  return !blocked;
}

std::vector<tirx::Var> StmtVars(const std::vector<Stmt> &stmts) {
  std::vector<tirx::Var> vars;
  for (const Stmt &stmt : stmts) {
    if (const auto *bind = stmt.as<tirx::BindNode>())
      vars.push_back(bind->var);
  }
  return vars;
}

bool HasHoisted(const std::vector<Axis> &hoisted, const Axis &axis) {
  for (const Axis &item : hoisted) {
    if (item->var.same_as(axis->var))
      return true;
  }
  return false;
}

/* ------------------------------------------------------------------------- */
/* reduction emission (forward declarations)                                 */
/* ------------------------------------------------------------------------- */

void EmitReduce(const VectorPlan &plan, Emitter *emitter);
PrimExpr EmitReduceChunk(const VectorPlan &plan, const StorePlan &store,
                         Emitter *body, const ReduceInfo &reduce, int bits,
                         const PrimExpr &mask, const PrimExpr &fill);

/* ------------------------------------------------------------------------- */
/* shared pre-store handling                                                 */
/* ------------------------------------------------------------------------- */

/*! \brief Plan-decided handling of one store. */
StoreRole RoleOf(const StorePlan &store) {
  return StoreRoleFromName(store->role.operator std::string());
}

/*!
 * \brief Every buffer this segment writes, as structural keys.
 *
 * Both the pre-store hoisting and ``SplitLoopInvariant`` ask "does this value
 * read something the loop writes?".
 */
std::set<std::string> WrittenStoreKeys(const VectorPlan &plan) {
  std::set<std::string> keys;
  for (const StorePlan &store : plan->pre_stores)
    keys.insert(BufferKey(store->spec->buffer));
  for (const StorePlan &store : plan->stores)
    keys.insert(BufferKey(store->spec->buffer));
  for (const StorePlan &store : plan->post_stores)
    keys.insert(BufferKey(store->spec->buffer));
  return keys;
}

/*! \brief Pre stores split by where they are emitted. */
struct PreStores {
  std::vector<StorePlan> invariant; // hoisted out of the chunk loop
  std::vector<StorePlan> loop;      // emitted once per chunk
};

PreStores SplitPreStores(const VectorPlan &plan, bool single,
                         const std::set<std::string> &written_keys) {
  PreStores split;
  for (const StorePlan &store : plan->pre_stores) {
    // A folded pre no longer writes memory: hoisting it would also hoist the
    // accumulator's initial value away from the loop that consumes it.
    const bool foldable = RoleOf(store) == StoreRole::kPreFolded;
    if (single && !foldable &&
        StoreIsLoopInvariant(store, plan, written_keys)) {
      split.invariant.push_back(store);
    } else {
      split.loop.push_back(store);
    }
  }
  return split;
}

/*! \brief Prologue / lets produced by the hoisted pre stores. */
struct HoistedPre {
  std::vector<Stmt> prologue;
  std::vector<Stmt> lets;
};

HoistedPre EmitHoistedPreStores(const VectorPlan &plan,
                                arith::Analyzer *analyzer,
                                const std::vector<StorePlan> &invariant,
                                const std::string &prefix,
                                ProgramPrologue *shared,
                                std::map<std::string, CarryEntry> *carry) {
  HoistedPre out;
  if (invariant.empty())
    return out;
  Emitter emitter(plan, analyzer, IntImm32(0), prefix + "hopre_", false, 0,
                  std::nullopt, shared, carry, false);
  for (const StorePlan &store : invariant)
    EmitOuterStore(store, &emitter);
  out.prologue = emitter.prologue_;
  out.lets = emitter.TakeLets();
  return out;
}

/* ------------------------------------------------------------------------- */
/* segment emission                                                          */
/* ------------------------------------------------------------------------- */

std::vector<Axis> InnerAxes(const VectorPlan &plan,
                            const std::vector<Axis> &hoisted) {
  std::vector<Axis> inner;
  for (const Axis &axis : plan->axes) {
    if (axis->index <= plan->vector_axis->index)
      continue;
    if (HasHoisted(hoisted, axis))
      continue;
    inner.push_back(axis);
  }
  return inner;
}

/*! \brief Per-segment axis counters that must be rebound with a fresh Var. */
using AxisRenames = std::map<const tirx::VarNode *, tirx::Var>;

/*!
 * \brief Wrap ``body`` in a serial loop over ``axis``.
 *
 * When ``renames`` maps the axis counter, the loop binds a fresh Var and the
 * body's uses are renamed with it: a program emits several segment bodies
 * sequentially, and one Var may only be bound once per statement.
 */
Stmt WrapSerialAxis(const Axis &axis, const AxisRenames &renames, Stmt body) {
  auto it = renames.find(axis->var.get());
  if (it == renames.end()) {
    return tirx::For(axis->var, IntImm32(0), axis->extent, ForKind::kSerial,
                     std::move(body));
  }
  const tirx::Var &fresh = it->second;
  if (body.defined()) {
    body = tirx::Substitute(
        body, [&](const tirx::Var &var) -> ffi::Optional<PrimExpr> {
          if (var.same_as(axis->var))
            return ffi::Optional<PrimExpr>(fresh);
          return ffi::Optional<PrimExpr>();
        });
  }
  return tirx::For(fresh, IntImm32(0), axis->extent, ForKind::kSerial,
                   std::move(body));
}

std::pair<std::vector<Stmt>, Stmt> EmitSegmentStmts(
    const VectorPlan &plan, arith::Analyzer *analyzer, const tirx::Var &chunk,
    const std::vector<Axis> &hoisted, const std::string &prefix, bool single,
    ProgramPrologue *shared, std::map<std::string, CarryEntry> *carry,
    bool allow_carry, const AxisRenames &renames);

std::pair<std::vector<Stmt>, Stmt> EmitParallelAxisParts(
    const VectorPlan &plan, arith::Analyzer *analyzer, const tirx::Var &chunk,
    const std::string &prefix, const std::vector<Axis> &hoisted, bool single,
    ProgramPrologue *shared, std::map<std::string, CarryEntry> *carry,
    bool allow_carry, const AxisRenames &renames);

std::pair<std::vector<Stmt>, Stmt> EmitSegmentStmts(
    const VectorPlan &plan, arith::Analyzer *analyzer, const tirx::Var &chunk,
    const std::vector<Axis> &hoisted, const std::string &prefix, bool single,
    ProgramPrologue *shared, std::map<std::string, CarryEntry> *carry,
    bool allow_carry, const AxisRenames &renames) {
  // A parallel vector axis with inner axes uses the hand-written three-part
  // structure (pre -> for inner: main -> post) so pre/post run once per chunk.
  std::vector<Axis> inner = InnerAxes(plan, hoisted);
  if (plan->vector_axis->is_parallel && !inner.empty() &&
      !ReduceOf(plan).defined()) {
    if (PackedNzStore(plan).has_value()) {
      throw Unsupported("a packed NZ store needs the vector axis to be the "
                        "innermost axis (split the nest)");
    }
    std::pair<std::vector<Stmt>, Stmt> parts =
        EmitParallelAxisParts(plan, analyzer, chunk, prefix, hoisted, single,
                              shared, carry, allow_carry, renames);
    // The three-part structure only loops the axes inside the vector axis; an
    // outer axis that the program does not hoist still needs its serial loop,
    // otherwise the emitted body references its variable unbound (a multi-
    // segment program with a lane axis of its own hoists nothing).
    Stmt body = parts.second;
    for (auto it = plan->axes.rbegin(); it != plan->axes.rend(); ++it) {
      const Axis &axis = *it;
      if (axis->index >= plan->vector_axis->index)
        continue;
      if (HasHoisted(hoisted, axis))
        continue;
      body = WrapSerialAxis(axis, renames, std::move(body));
    }
    return {parts.first, body};
  }

  Emitter emitter(plan, analyzer, chunk, prefix, false, 0, std::nullopt, shared,
                  carry, allow_carry, nullptr, 0);
  const std::set<std::string> written_keys = WrittenStoreKeys(plan);
  const PreStores pre = SplitPreStores(plan, single, written_keys);

  for (const StorePlan &store : pre.loop)
    EmitOuterStore(store, &emitter);
  std::vector<Stmt> pre_lets = emitter.TakeLets();
  emitter.clear_forward();

  std::vector<Stmt> main_prologue;
  std::vector<Stmt> main_lets;
  ReduceInfo reduce = ReduceOf(plan);
  if (reduce.defined()) {
    EmitReduce(plan, &emitter);
    main_lets = emitter.TakeLets();
  } else {
    auto chunks = EmitElementwiseChunks(plan, analyzer, chunk, prefix, shared,
                                        carry, allow_carry);
    main_prologue = chunks.first;
    main_lets = chunks.second;
  }

  for (const StorePlan &store : plan->post_stores)
    EmitOuterStore(store, &emitter);
  std::vector<Stmt> post_lets = emitter.TakeLets();

  std::vector<Axis> wrap_axes;
  for (const Axis &axis : plan->axes) {
    if (axis->var.same_as(plan->vector_axis->var))
      continue;
    if (HasHoisted(hoisted, axis))
      continue;
    wrap_axes.push_back(axis);
  }

  std::vector<Stmt> hoisted_body;
  std::vector<Stmt> flat_main;
  for (const Stmt &stmt : main_lets) {
    if (const auto *seq = stmt.as<tirx::SeqStmtNode>()) {
      for (const Stmt &child : seq->seq)
        flat_main.push_back(child);
    } else {
      flat_main.push_back(stmt);
    }
  }
  bool all_bind_or_eval = !flat_main.empty();
  for (const Stmt &stmt : flat_main) {
    if (stmt.as<tirx::BindNode>() == nullptr &&
        stmt.as<tirx::EvaluateNode>() == nullptr) {
      all_bind_or_eval = false;
    }
  }
  if (single && (!wrap_axes.empty() || !hoisted.empty()) && all_bind_or_eval) {
    std::vector<Axis> inner_and_hoisted = wrap_axes;
    inner_and_hoisted.insert(inner_and_hoisted.end(), hoisted.begin(),
                             hoisted.end());
    auto split = SplitLoopInvariant(flat_main, inner_and_hoisted, written_keys,
                                    StmtVars(pre_lets));
    hoisted_body = split.first;
    flat_main = split.second;
  }
  // A chunk is straight-line: no scalar loop wraps the chunk program.
  std::vector<Stmt> main_group = flat_main;

  std::vector<Stmt> body_stmts = pre_lets;
  body_stmts.insert(body_stmts.end(), main_group.begin(), main_group.end());
  body_stmts.insert(body_stmts.end(), post_lets.begin(), post_lets.end());
  Stmt body = SeqOfVec(body_stmts).empty() ? Stmt() : SeqOfVec(body_stmts)[0];
  for (auto it = plan->axes.rbegin(); it != plan->axes.rend(); ++it) {
    const Axis &axis = *it;
    if (axis->var.same_as(plan->vector_axis->var))
      continue;
    if (HasHoisted(hoisted, axis))
      continue;
    body = WrapSerialAxis(axis, renames, std::move(body));
  }

  HoistedPre hoisted_pre = EmitHoistedPreStores(plan, analyzer, pre.invariant,
                                                prefix, shared, carry);

  std::vector<Stmt> prologue = emitter.prologue_;
  prologue.insert(prologue.end(), hoisted_pre.prologue.begin(),
                  hoisted_pre.prologue.end());
  prologue.insert(prologue.end(), main_prologue.begin(), main_prologue.end());
  prologue.insert(prologue.end(), hoisted_body.begin(), hoisted_body.end());
  prologue.insert(prologue.end(), hoisted_pre.lets.begin(),
                  hoisted_pre.lets.end());
  return {prologue, body};
}

/* ------------------------------------------------------------------------- */
/* reduction emission                                                        */
/* ------------------------------------------------------------------------- */

/*! \brief Typed elementwise op that combines two registers in a reduction. */
ValueRecipe ReduceElementRecipe(const ReduceInfo &reduce) {
  std::string element_op;
  ReduceElementFor(reduce->recipe.operator std::string(), &element_op);
  return ValueRecipeFromName(element_op);
}

void EmitReduceTail(const StorePlan &store, Emitter *emitter,
                    const ReduceInfo &reduce, const std::string &elem, int bits,
                    PrimExpr acc) {
  PrimExpr lane0 = emitter->Lane0Mask(bits);
  if (reduce->accumulate) {
    const ValueRecipe element_op = ReduceElementRecipe(reduce);
    PrimExpr old = emitter->Addr(store->access, /*uniform=*/true);
    std::string acc_elem =
        ElemDtype(ffi::DLDataTypeToString(acc.dtype()).operator std::string());
    if (acc_elem == elem) {
      acc =
          emitter->Bind(SimdCall(SimdOpOf(ScalarVariantOf(element_op)),
                                 DataTypeOf(ffi::DLDataTypeToString(acc.dtype())
                                                .operator std::string()),
                                 {acc, old, lane0, StrImm(ModeZeroing())}),
                        &emitter->lets_);
    } else {
      PrimExpr old_vec = emitter->Bind(
          Vld(old, store->spec->buffer, "BRC_B" + std::to_string(bits)),
          &emitter->lets_);
      acc =
          emitter->Bind(SimdCall(SimdOpOf(element_op),
                                 DataTypeOf(ffi::DLDataTypeToString(acc.dtype())
                                                .operator std::string()),
                                 {acc, old_vec, lane0, StrImm(ModeZeroing())}),
                        &emitter->lets_);
    }
  }
  PrimExpr dst = emitter->Addr(store->access, /*uniform=*/true);
  emitter->lets_.push_back(tirx::Evaluate(Vsts(
      dst, store->spec->buffer, acc, lane0, "ONEPT_B" + std::to_string(bits))));
  emitter->NoteWrite(store->spec->buffer);
}

PrimExpr EmitReduceChunk(const VectorPlan &plan, const StorePlan &store,
                         Emitter *body, const ReduceInfo &reduce, int bits,
                         const PrimExpr &mask, const PrimExpr &fill) {
  PrimExpr loaded = body->Value(*reduce->varying);
  PrimExpr tail = body->masked_tail() ? body->TailMask(bits) : PrimExpr();
  if (tail.defined()) {
    if (reduce->recipe == "vcadd") {
      return body->Bind(ReduceCall("vcadd", loaded, tail), &body->lets_);
    }
    loaded =
        body->Bind(SimdCall(simd_vsel(),
                            DataTypeOf(ffi::DLDataTypeToString(loaded.dtype())
                                           .operator std::string()),
                            {loaded, fill, tail}),
                   &body->lets_);
  }
  return body->Bind(
      ReduceCall(reduce->recipe.operator std::string(), loaded, mask),
      &body->lets_);
}

void EmitReduceUnrolled(const VectorPlan &plan, const StorePlan &store,
                        Emitter *emitter, const ReduceInfo &reduce,
                        int64_t chunks) {
  std::string elem = ElemDtype(store->spec->buffer->dtype);
  int bits = MaskBits(elem);
  PrimExpr mask = emitter->AllMask(bits);
  const ValueRecipe element_op = ReduceElementRecipe(reduce);

  bool has_tail = plan->tail.has_value() && *plan->tail != 0;
  PrimExpr fill;
  if (has_tail && reduce->recipe != "vcadd") {
    fill = emitter->Bind(
        Vdup(ReduceIdentity(reduce->recipe.operator std::string(), elem), elem,
             mask),
        &emitter->lets_);
  }

  int64_t last_chunk = chunks - 1;
  std::vector<PrimExpr> chains;
  for (int64_t ck = 0; ck < chunks; ++ck) {
    emitter->set_lane_base(IntImm32(ck * plan->lanes));
    // Addresses change per chunk: the memo and predicate caches must follow.
    emitter->ClearChunkCaches();
    emitter->set_masked_tail(has_tail && ck == last_chunk);
    chains.push_back(
        EmitReduceChunk(plan, store, emitter, reduce, bits, mask, fill));
  }
  emitter->set_masked_tail(false);

  while (chains.size() > 1) {
    PrimExpr lane0 = emitter->Lane0Mask(bits);
    std::vector<PrimExpr> merged;
    for (size_t k = 0; k + 1 < chains.size(); k += 2) {
      merged.push_back(emitter->Bind(
          SimdCall(SimdOpOf(element_op),
                   DataTypeOf(ffi::DLDataTypeToString(chains[k].dtype())
                                  .operator std::string()),
                   {chains[k], chains[k + 1], lane0, StrImm(ModeZeroing())}),
          &emitter->lets_));
    }
    if (chains.size() % 2 != 0)
      merged.push_back(chains.back());
    chains = merged;
  }

  EmitReduceTail(store, emitter, reduce, elem, bits, chains[0]);
}

void EmitReduceLoop(const VectorPlan &plan, const StorePlan &store,
                    Emitter *emitter, const ReduceInfo &reduce) {
  std::string elem = ElemDtype(store->spec->buffer->dtype);
  int bits = MaskBits(elem);
  PrimExpr mask = emitter->AllMask(bits);
  const ValueRecipe element_op = ReduceElementRecipe(reduce);
  PrimExpr identity =
      ReduceIdentity(reduce->recipe.operator std::string(), elem);

  tirx::Buffer acc = tirx::decl_buffer({IntImm32(1)}, VecDataType(elem),
                                       "acc" + emitter->prefix() +
                                           std::to_string(emitter->counter()),
                                       "local.var");
  emitter->set_counter(emitter->counter() + 1);
  // The accumulator lives outside the chunk (and outer axis) loops: one
  // register per segment, re-initialized on every outer iteration.
  emitter->prologue_.push_back(tirx::AllocBuffer(acc));
  emitter->lets_.push_back(
      tirx::BufferStore(acc, Vdup(identity, elem, mask), {IntImm32(0)}));

  bool may_have_tail =
      !plan->chunks.has_value() || (plan->tail.has_value() && *plan->tail != 0);
  PrimExpr fill;
  if (may_have_tail && reduce->recipe != "vcadd") {
    fill = emitter->Bind(Vdup(identity, elem, mask), &emitter->lets_);
  }

  MakeBlock build_chunk = [&](const PrimExpr &chunk_index, bool masked_tail,
                              const std::string &label, int64_t chunk_min,
                              std::optional<int64_t> chunk_max) -> Block {
    Emitter body(plan, emitter->analyzer(), chunk_index,
                 emitter->prefix() + label, masked_tail, chunk_min, chunk_max,
                 emitter->shared(), emitter->carry(), false);
    PrimExpr reduced =
        EmitReduceChunk(plan, store, &body, reduce, bits, mask, fill);
    PrimExpr lane0 = emitter->Lane0Mask(bits);
    PrimExpr total =
        body.Bind(SimdCall(SimdOpOf(element_op),
                           DataTypeOf(ffi::DLDataTypeToString(reduced.dtype())
                                          .operator std::string()),
                           {tirx::BufferLoad(acc, {IntImm32(0)}), reduced,
                            lane0, StrImm(ModeZeroing())}),
                  &body.lets_);
    body.lets_.push_back(tirx::BufferStore(acc, total, {IntImm32(0)}));
    return Block{body.prologue_, body.lets_};
  };

  // Each reduction gets its own chunk loop (sibling reductions inside one
  // segment must not share the loop variable).
  tirx::Var loop_var(emitter->prefix() + "ck" +
                         std::to_string(emitter->counter()),
                     DataType::Int(32));
  emitter->set_counter(emitter->counter() + 1);
  auto blocks = StripMine(plan, loop_var, build_chunk);
  for (Stmt &stmt : blocks.first)
    emitter->prologue_.push_back(std::move(stmt));
  for (Stmt &stmt : blocks.second)
    emitter->lets_.push_back(std::move(stmt));

  PrimExpr final_value =
      emitter->Bind(tirx::BufferLoad(acc, {IntImm32(0)}), &emitter->lets_);
  EmitReduceTail(store, emitter, reduce, elem, bits, final_value);
}

void EmitOneReduce(const VectorPlan &plan, const StorePlan &store,
                   Emitter *emitter) {
  ReduceInfo reduce = ReduceOf(store);
  std::optional<int64_t> chunks = plan->chunks.has_value()
                                      ? std::optional<int64_t>(*plan->chunks)
                                      : std::nullopt;
  if (chunks.has_value() && *chunks == 0) {
    // Zero iterations: the destination keeps its value (and no empty chain).
    return;
  }
  if (chunks.has_value() && *chunks <= kReduceUnrollLimit) {
    EmitReduceUnrolled(plan, store, emitter, reduce, *chunks);
  } else {
    EmitReduceLoop(plan, store, emitter, reduce);
  }
}

void EmitReduce(const VectorPlan &plan, Emitter *emitter) {
  for (const StorePlan &store : plan->stores) {
    ReduceInfo reduce = ReduceOf(store);
    if (!reduce.defined()) {
      EmitOuterStore(store, emitter); // lane-uniform value: one broadcast write
      continue;
    }
    EmitOneReduce(plan, store, emitter);
  }
}

/* ------------------------------------------------------------------------- */
/* parallel-axis pre/main/post structure                                     */
/* ------------------------------------------------------------------------- */

std::pair<std::vector<Stmt>, Stmt> EmitParallelAxisParts(
    const VectorPlan &plan, arith::Analyzer *analyzer, const tirx::Var &chunk,
    const std::string &prefix, const std::vector<Axis> &hoisted, bool single,
    ProgramPrologue *shared, std::map<std::string, CarryEntry> *carry,
    bool allow_carry, const AxisRenames &renames) {
  TVM_FFI_ICHECK(!ReduceOf(plan).defined())
      << "a parallel vector axis with pre/post must go through the elementwise "
         "path";
  std::vector<Axis> inner = InnerAxes(plan, hoisted);
  std::vector<Stmt> prologue;
  const std::set<std::string> written_keys = WrittenStoreKeys(plan);
  const PreStores pre = SplitPreStores(plan, single, written_keys);

  std::vector<StorePlan> candidates;
  for (const StorePlan &store : plan->stores) {
    if (store->promote.has_value())
      candidates.push_back(store);
  }

  auto finish = [&](const Stmt &body) -> std::pair<std::vector<Stmt>, Stmt> {
    HoistedPre hoisted = EmitHoistedPreStores(plan, analyzer, pre.invariant,
                                              prefix, shared, carry);
    for (Stmt &stmt : hoisted.prologue)
      prologue.push_back(std::move(stmt));
    if (!hoisted.lets.empty()) {
      std::vector<Stmt> out = prologue;
      out.insert(out.end(), hoisted.lets.begin(), hoisted.lets.end());
      return {out, body};
    }
    return {prologue, body};
  };

  MakeBlock make_block = [&](const PrimExpr &chunk_var, bool masked_tail,
                             const std::string &label, int64_t chunk_min,
                             std::optional<int64_t> chunk_max) -> Block {
    // SSA names of a program segment are prefixed with ``s{i}_`` so sibling
    // segments never collide.
    const std::string body_prefix = prefix + label;
    Emitter body_emitter(plan, analyzer, chunk_var, body_prefix, masked_tail,
                         chunk_min, chunk_max, shared, carry, allow_carry,
                         nullptr, 0);
    // Plan-decided fold: a pre store marked ``pre_folded`` supplies the
    // accumulator's initial value instead of writing memory. The plan already
    // proved it is unmasked, uniform, at the accumulator's address and not read
    // by a sibling.
    std::map<const StorePlanNode *, StorePlan> folded;
    std::set<const StorePlanNode *> consumed_pre;
    for (const StorePlan &store : candidates) {
      for (const StorePlan &folded_pre : pre.loop) {
        if (consumed_pre.count(folded_pre.get()) != 0 ||
            RoleOf(folded_pre) != StoreRole::kPreFolded) {
          continue;
        }
        if (SameBuffer(folded_pre->spec->buffer, store->spec->buffer) &&
            SameIndices(folded_pre->spec->indices, store->spec->indices)) {
          folded[store.get()] = folded_pre;
          consumed_pre.insert(folded_pre.get());
          break;
        }
      }
    }
    for (const StorePlan &store : pre.loop) {
      if (consumed_pre.count(store.get()) == 0)
        EmitStore(plan, store, &body_emitter);
    }
    std::vector<Stmt> pre_lets = body_emitter.TakeLets();
    // pre sits outside the inner loop: forwarding must not cross the boundary.
    body_emitter.clear_forward();

    // Accumulators: address invariant along the inner axes -> load into a
    // local.var vector register, update in the loop, write back once.
    std::vector<std::tuple<StorePlan, ValueNode, tirx::Buffer>> acc_order;
    std::map<const StorePlanNode *, size_t> acc_map;
    for (const StorePlan &store : candidates) {
      ValueNode old = *store->promote;
      tirx::Buffer acc = tirx::decl_buffer(
          {IntImm32(1)},
          DataTypeOf(VecDtype(ElemDtype(store->spec->buffer->dtype))),
          "acc" + body_prefix + std::to_string(body_emitter.counter()),
          "local.var");
      body_emitter.set_counter(body_emitter.counter() + 1);
      // The accumulator is declared inside the segment body, not in the program
      // prologue: an enclosing per-iteration loop then gives every iteration
      // its own register. One prologue-level slot would be reused by all
      // iterations and add a false WAW/WAR dependency that blocks overlapping
      // otherwise independent iterations.
      body_emitter.lets_.push_back(tirx::AllocBuffer(acc));
      PrimExpr init;
      auto fold_it = folded.find(store.get());
      if (fold_it != folded.end()) {
        init = body_emitter.Dup(fold_it->second->value);
      } else if (store->kind == AccessClass::kContiguous) {
        init = body_emitter.Value(old);
      } else {
        init = body_emitter.Dup(old);
      }
      body_emitter.lets_.push_back(tirx::BufferStore(acc, init, {IntImm32(0)}));
      body_emitter.SetMemo(ValueKey(old, body_emitter.store_gen()),
                           tirx::BufferLoad(acc, {IntImm32(0)}));
      acc_map[store.get()] = acc_order.size();
      acc_order.push_back(std::make_tuple(store, old, acc));
    }
    std::vector<Stmt> init_lets = body_emitter.TakeLets();

    for (const StorePlan &store : plan->stores) {
      auto it = acc_map.find(store.get());
      if (it == acc_map.end()) {
        EmitStore(plan, store, &body_emitter);
        continue;
      }
      tirx::Buffer acc = std::get<2>(acc_order[it->second]);
      PrimExpr value = body_emitter.VectorOperand(store->value);
      body_emitter.lets_.push_back(
          tirx::BufferStore(acc, value, {IntImm32(0)}));
    }
    std::vector<Stmt> main_lets = body_emitter.TakeLets();
    body_emitter.clear_forward();

    // Write-back: the plan marks a store whose post overwrites the same address
    // and consumes the old value, so the post owns the single ``vsts``.
    std::set<const StorePlanNode *> superseded;
    for (const StorePlan &store : candidates) {
      if (RoleOf(store) == StoreRole::kPostSuperseded)
        superseded.insert(store.get());
    }
    for (const auto &entry : acc_order) {
      const StorePlan &store = std::get<0>(entry);
      ValueNode old = std::get<1>(entry);
      tirx::Buffer acc = std::get<2>(entry);
      PrimExpr value = tirx::BufferLoad(acc, {IntImm32(0)});
      if (superseded.count(store.get()) == 0) {
        int bits = MaskBits(ElemDtype(store->spec->buffer->dtype));
        if (store->kind == AccessClass::kContiguous) {
          PrimExpr mask = StoreMask(store, &body_emitter, bits);
          body_emitter.lets_.push_back(tirx::Evaluate(
              Vsts(body_emitter.Addr(store->access), store->spec->buffer, value,
                   mask, "NORM_B" + std::to_string(bits))));
        } else {
          body_emitter.lets_.push_back(tirx::Evaluate(
              Vsts(body_emitter.Addr(store->access, /*uniform=*/true),
                   store->spec->buffer, value, body_emitter.AllMask(bits),
                   "ONEPT_B" + std::to_string(bits))));
        }
        body_emitter.NoteWrite(store->spec->buffer);
      }
      body_emitter.Promote(old, value);
      // Segment-to-segment register forwarding: a single-chunk, tail-free,
      // predicate-free contiguous write-back whose address is invariant along
      // the inner axes *is* the memory content of the whole vector.
      bool before_axis_hoisted = true;
      for (const Axis &axis : plan->axes) {
        if (axis->index >= plan->vector_axis->index)
          continue;
        if (!HasHoisted(hoisted, axis))
          before_axis_hoisted = false;
      }
      if (carry != nullptr && superseded.count(store.get()) == 0 &&
          !store->mask.has_value() && store->kind == AccessClass::kContiguous &&
          plan->chunks.has_value() && *plan->chunks == 1 &&
          !(plan->tail.has_value() && *plan->tail != 0) &&
          before_axis_hoisted) {
        body_emitter.CarryStore(store, value, plan->vector_axis->var);
      }
    }
    std::vector<Stmt> final_lets = body_emitter.TakeLets();

    for (const StorePlan &store : plan->post_stores) {
      EmitStore(plan, store, &body_emitter);
    }
    std::vector<Stmt> post_lets = body_emitter.TakeLets();

    std::set<std::string> licm_written = written_keys;
    for (const auto &entry : acc_order) {
      licm_written.insert(BufferKey(std::get<2>(entry)));
    }
    std::vector<Stmt> hoisted_main;
    std::vector<Stmt> loop_main = main_lets;
    auto split = SplitLoopInvariant(main_lets, inner, licm_written, {});
    hoisted_main = split.first;
    loop_main = split.second;

    std::vector<Stmt> prefix_stmts = pre_lets;
    prefix_stmts.insert(prefix_stmts.end(), init_lets.begin(), init_lets.end());
    prefix_stmts.insert(prefix_stmts.end(), hoisted_main.begin(),
                        hoisted_main.end());
    if (loop_main.empty()) {
      std::vector<Stmt> out = prefix_stmts;
      out.insert(out.end(), final_lets.begin(), final_lets.end());
      out.insert(out.end(), post_lets.begin(), post_lets.end());
      return Block{body_emitter.prologue_, out};
    }
    Stmt main_stmt = SeqOfVec(loop_main)[0];
    for (auto it = inner.rbegin(); it != inner.rend(); ++it) {
      main_stmt = WrapSerialAxis(*it, renames, std::move(main_stmt));
    }
    std::vector<Stmt> out = prefix_stmts;
    out.push_back(main_stmt);
    out.insert(out.end(), final_lets.begin(), final_lets.end());
    out.insert(out.end(), post_lets.begin(), post_lets.end());
    return Block{body_emitter.prologue_, out};
  };

  auto blocks = StripMine(plan, chunk, make_block);
  for (Stmt &stmt : blocks.first)
    prologue.push_back(std::move(stmt));
  return finish(SeqOfVec(blocks.second)[0]);
}

/* ------------------------------------------------------------------------- */
/* program emission                                                          */
/* ------------------------------------------------------------------------- */

bool SegmentConsumesCarry(const VectorPlan &segment,
                          const std::vector<Axis> &hoisted) {
  if (!segment->chunks.has_value() || *segment->chunks != 1)
    return false;
  for (const Axis &axis : segment->axes) {
    if (axis->index >= segment->vector_axis->index)
      continue;
    if (!HasHoisted(hoisted, axis))
      return false;
  }
  return true;
}

/* ------------------------------------------------------------------------- */
/* structural tree emission (design A)                                       */
/* ------------------------------------------------------------------------- */

/*! \brief One chunk's emission result: prologue + vector program body. */
struct TreeEmitResult {
  std::vector<Stmt> prologue;
  Stmt body;
};

struct TreeEmitState {
  ProgramPrologue shared;
  std::map<std::string, CarryEntry> carry;
  std::map<const tirx::VarNode *, int> wrap_count;
  std::set<const tirx::VarNode *> bound_axes;
  std::vector<Axis> hoisted; // materialized ancestor loop axes
  bool single{false};
  int64_t chunk_count{0};
  int64_t chunk_index{0};
  arith::Analyzer analyzer;
};

/*!
 * \brief Pre-walk the tree: chunk count + how often each non-vector axis is
 * wrapped.
 *
 * A non-hoisted axis may be wrapped by several chunk bodies sequentially; one
 * ``Var`` can only be bound once per statement, so every repetition after the
 * first gets a fresh Var (same name, own identity). Axes bound by an ancestor
 * ``Loop`` node are skipped here, matching the legacy ``wrap_count`` that
 * ignored program-level hoisted axes.
 */
void CollectChunkStats(const PlanNode &node,
                       std::set<const tirx::VarNode *> *ancestor_loops,
                       int64_t *count,
                       std::map<const tirx::VarNode *, int> *wrap_count) {
  if (!node.defined())
    return;
  switch (node->kind) {
  case PlanKind::kChunk:
    ++*count;
    for (const Axis &axis : node->chunk->axes) {
      if (axis->var.same_as(node->chunk->vector_axis->var))
        continue;
      if (ancestor_loops->count(axis->var.get()) != 0)
        continue;
      ++(*wrap_count)[axis->var.get()];
    }
    return;
  case PlanKind::kSeq:
    for (const ffi::ObjectRef &child : node->children)
      CollectChunkStats(Downcast<PlanNode>(child), ancestor_loops, count,
                        wrap_count);
    return;
  case PlanKind::kLoop:
    ancestor_loops->insert(node->loop_var.get());
    if (node->body.defined())
      CollectChunkStats(Downcast<PlanNode>(node->body), ancestor_loops, count,
                        wrap_count);
    ancestor_loops->erase(node->loop_var.get());
    return;
  }
}

TreeEmitResult EmitTreeNode(const PlanNode &node, TreeEmitState *state,
                            const std::string &prefix) {
  if (!node.defined())
    return {{}, Stmt()};
  switch (node->kind) {
  case PlanKind::kChunk: {
    const VectorPlan &plan = node->chunk;
    tirx::Var chunk("chunk_s" + std::to_string(state->chunk_index),
                    DataType::Int(32));
    AxisRenames renames;
    for (const Axis &axis : plan->axes) {
      if (axis->var.same_as(plan->vector_axis->var))
        continue;
      if (HasHoisted(state->hoisted, axis))
        continue;
      if (state->wrap_count[axis->var.get()] <= 1)
        continue;
      if (state->bound_axes.insert(axis->var.get()).second)
        continue; // first binding keeps the original counter
      renames[axis->var.get()] =
          tirx::Var(axis->var->name_hint, axis->var->dtype);
    }
    auto emitted = EmitSegmentStmts(
        plan, &state->analyzer, chunk, state->hoisted,
        prefix + "s" + std::to_string(state->chunk_index) + "_", state->single,
        &state->shared, &state->carry,
        SegmentConsumesCarry(plan, state->hoisted), renames);
    ++state->chunk_index;
    return {std::move(emitted.first), std::move(emitted.second)};
  }
  case PlanKind::kLoop: {
    Axis axis = MakeAxis(node->loop_var, node->extent, 0,
                         static_cast<ForKind>(node->for_kind));
    state->hoisted.push_back(axis);
    TreeEmitResult inner =
        node->body.defined()
            ? EmitTreeNode(Downcast<PlanNode>(node->body), state, prefix)
            : TreeEmitResult{};
    state->hoisted.pop_back();

    std::vector<Stmt> body_stmts;
    for (Stmt &stmt : inner.prologue)
      body_stmts.push_back(std::move(stmt));
    if (inner.body.defined())
      body_stmts.push_back(std::move(inner.body));
    if (node->barrier_at_end)
      body_stmts.push_back(VstVldBarrier());
    Stmt body = body_stmts.empty() ? Stmt() : SeqOfVec(body_stmts)[0];
    Stmt loop = tirx::For(node->loop_var, node->min, node->extent,
                          ForKind::kSerial, body);
    return {{}, loop}; // the prologue stays inside the loop (I7)
  }
  case PlanKind::kSeq: {
    std::vector<Stmt> prologue;
    std::vector<Stmt> bodies;
    for (const ffi::ObjectRef &child_ref : node->children) {
      PlanNode child = Downcast<PlanNode>(child_ref);
      TreeEmitResult part = EmitTreeNode(child, state, prefix);
      for (Stmt &stmt : part.prologue)
        prologue.push_back(std::move(stmt));
      if (part.body.defined())
        bodies.push_back(std::move(part.body));
    }
    Stmt body = bodies.empty() ? Stmt() : SeqOfVec(bodies)[0];
    return {std::move(prologue), body};
  }
  }
  return {{}, Stmt()};
}

Stmt EmitTreeProgram(const PlanNode &root) {
  TreeEmitState state;
  std::set<const tirx::VarNode *> ancestor_loops;
  CollectChunkStats(root, &ancestor_loops, &state.chunk_count,
                    &state.wrap_count);
  state.single = state.chunk_count == 1;
  TreeEmitResult emitted = EmitTreeNode(root, &state, "");
  std::vector<Stmt> all;
  for (Stmt &stmt : state.shared.statements)
    all.push_back(std::move(stmt));
  for (Stmt &stmt : emitted.prologue)
    all.push_back(std::move(stmt));
  if (emitted.body.defined())
    all.push_back(std::move(emitted.body));
  if (all.empty())
    return Stmt();
  return SeqOfVec(all)[0];
}

/* ------------------------------------------------------------------------- */
/* invariant I5: no scalar data instructions                                 */
/* ------------------------------------------------------------------------- */

namespace {

/*!
 * \brief Collects scalar data instructions in an emitted region.
 *
 * Allowed scalar uses (I5): loop bounds, branch conditions, ``tl.access_ptr``
 * addresses, chunk indices and ``Bind`` values. Everything else that reads or
 * writes a tensor element lane-by-lane is a violation.
 */
class ScalarDataOpVisitor : public tirx::StmtExprVisitor {
public:
  std::vector<std::string> violations;

  void Run(const tirx::Stmt &stmt) { VisitStmt(stmt); }

protected:
  void VisitStmt_(const tirx::BufferStoreNode *op) final {
    violations.push_back("scalar BufferStore to '" +
                         op->buffer->name.operator std::string() + "'");
  }

  void VisitStmt_(const tirx::ForNode *op) final {
    ++allowed_;
    VisitExpr(op->min);
    VisitExpr(op->extent);
    --allowed_;
    VisitStmt(op->body);
  }

  void VisitStmt_(const tirx::IfThenElseNode *op) final {
    ++allowed_;
    VisitExpr(op->condition);
    --allowed_;
    VisitStmt(op->then_case);
    if (op->else_case.has_value())
      VisitStmt(*op->else_case);
  }

  void VisitStmt_(const tirx::BindNode *op) final {
    ++allowed_;
    VisitExpr(op->value);
    --allowed_;
  }

  void VisitExpr_(const tirx::BufferLoadNode *op) final {
    if (in_simd_ == 0 && allowed_ == 0) {
      violations.push_back("scalar BufferLoad from '" +
                           op->buffer->name.operator std::string() + "'");
    }
    for (const PrimExpr &index : op->indices)
      VisitExpr(index);
  }

  void VisitExpr_(const tirx::CallNode *op) final {
    std::string name;
    if (const auto *fn = op->op.as<OpNode>())
      name = fn->name;
    const bool vector_channel =
        name.rfind("tl.simd", 0) == 0 || name == "tl.access_ptr";
    if (vector_channel)
      ++in_simd_;
    for (const PrimExpr &arg : op->args)
      VisitExpr(arg);
    if (vector_channel)
      --in_simd_;
  }

private:
  int allowed_{0};
  int in_simd_{0};
};

} // namespace

} // namespace

bool VerifyNoScalarDataOps(const tirx::Stmt &stmt,
                           std::vector<std::string> *violations) {
  ScalarDataOpVisitor visitor;
  if (stmt.defined())
    visitor.Run(stmt);
  if (violations != nullptr)
    *violations = visitor.violations;
  return visitor.violations.empty();
}

tirx::Stmt EmitPlan(const ffi::ObjectRef &plan) {
  ResetBufferKeyCache();
  if (const auto *program = plan.as<VectorProgramNode>()) {
    VectorProgram node = ffi::GetRef<VectorProgram>(program);
    TVM_FFI_ICHECK(node->root.defined())
        << "the program has no structural plan tree";
    return EmitTreeProgram(Downcast<PlanNode>(node->root));
  }
  const auto *single = plan.as<VectorPlanNode>();
  TVM_FFI_ICHECK(single != nullptr)
      << "emit_plan expects a VectorPlan or VectorProgram";
  VectorPlan node = ffi::GetRef<VectorPlan>(single);
  tirx::Var chunk("chunk", DataType::Int(32));
  arith::Analyzer analyzer;
  auto emitted =
      EmitSegmentStmts(node, &analyzer, chunk, {}, "",
                       /*single=*/true, nullptr, nullptr, false, AxisRenames{});
  std::vector<Stmt> all;
  for (Stmt &stmt : emitted.first)
    all.push_back(std::move(stmt));
  all.push_back(emitted.second);
  return SeqOfVec(all)[0];
}

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm
