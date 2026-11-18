/*!
 * \file primitives.cc
 * \brief ParallelVectorize primitive fact tables and small shared helpers.
 */
#include "primitives.h"

#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ir/op.h>
#include <tvm/ir/repr.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace tl {
namespace parallel_vectorize {

namespace {

/*!
 * \brief Elementwise binary recipes (``_ELEMENTWISE_BINARY``).
 *
 * Keyed by the lower-cased TIR node name ("add") *and* the intrinsic op tail:
 * both spellings share this table in Python.
 */
const std::unordered_map<std::string, std::string> &BinaryRecipes() {
  static const std::unordered_map<std::string, std::string> table = {
      {"add", "vadd"},        {"sub", "vsub"},        {"mul", "vmul"},
      {"div", "vdiv"},        {"max", "vmax"},        {"min", "vmin"},
      {"bitwiseand", "vand"}, {"bitwiseor", "vor"},   {"bitwisexor", "vxor"},
      {"shiftleft", "vshl"},  {"shiftright", "vshr"},
  };
  return table;
}

/*! \brief Elementwise unary recipes (``_ELEMENTWISE_UNARY``). */
const std::unordered_map<std::string, std::string> &UnaryRecipes() {
  static const std::unordered_map<std::string, std::string> table = {
      {"neg", "vneg"},   {"abs", "vabs"}, {"exp", "vexp"},
      {"sqrt", "vsqrt"}, {"log", "vln"},
  };
  return table;
}

/*! \brief Comparison predicates (``_COMPARE``). */
const std::unordered_map<std::string, std::string> &ComparePreds() {
  static const std::unordered_map<std::string, std::string> table = {
      {"lt", "lt"}, {"le", "le"}, {"gt", "gt"},
      {"ge", "ge"}, {"eq", "eq"}, {"ne", "ne"},
  };
  return table;
}

/*! \brief Boolean combination recipes (``_BOOL_BINARY`` / ``_BOOL_UNARY``). */
const std::unordered_map<std::string, std::string> &BoolBinaryRecipes() {
  static const std::unordered_map<std::string, std::string> table = {
      {"and", "pand"}, {"or", "por"}};
  return table;
}

const std::unordered_map<std::string, std::string> &BoolUnaryRecipes() {
  static const std::unordered_map<std::string, std::string> table = {
      {"not", "pnot"}};
  return table;
}

/*! \brief ``_VEC_SCALAR_BINARY``: vector op -> scalar-vector variant. */
const std::unordered_map<std::string, std::string> &VecScalarBinaryTable() {
  static const std::unordered_map<std::string, std::string> table = {
      {"vadd", "vadds"},
      {"vmul", "vmuls"},
      {"vmax", "vmaxs"},
      {"vmin", "vmins"}};
  return table;
}

/*! \brief ``_COMPARE_FLIP``: swapping the operands flips the predicate. */
const std::unordered_map<std::string, std::string> &CompareFlips() {
  static const std::unordered_map<std::string, std::string> table = {
      {"lt", "gt"}, {"gt", "lt"}, {"le", "ge"},
      {"ge", "le"}, {"eq", "eq"}, {"ne", "ne"}};
  return table;
}

/*! \brief ``_ELEM_BITS``: element dtype -> bit width (language/simd.py). */
const std::unordered_map<std::string, int> &ElemBitsTable() {
  static const std::unordered_map<std::string, int> table = {
      {"float32", 32},      {"float16", 16},    {"bfloat16", 16},
      {"int32", 32},        {"uint32", 32},     {"int16", 16},
      {"uint16", 16},       {"int64", 64},      {"uint64", 64},
      {"int8", 8},          {"uint8", 8},       {"float8_e4m3", 8},
      {"float8_e4m3fn", 8}, {"float8_e5m2", 8}, {"float8_e8m0fnu", 8},
      {"float4_e2m1fn", 4},
  };
  return table;
}

/*! \brief ``_INDEX_VALUE_OPS``: integer operators the index-vector channel
 * builds. */
const std::unordered_set<std::string> &IndexValueOps() {
  static const std::unordered_set<std::string> table = {
      "Var",       "IntImm",     "Add",       "Sub",       "Mul", "FloorDiv",
      "FloorMod",  "Cast",       "Min",       "Max",       "Neg", "BitwiseAnd",
      "BitwiseOr", "BitwiseXor", "ShiftLeft", "ShiftRight"};
  return table;
}

PrimExpr ZeroIdentity(const std::string &elem) {
  // float* -> FloatImm(0.0), integer -> IntImm(0); ``elem`` is a dtype string.
  const char *prefix = "float";
  if (elem.compare(0, 5, prefix) == 0) {
    return FloatImm(DataType(ffi::StringToDLDataType(elem)), 0.0);
  }
  return IntImm(DataType(ffi::StringToDLDataType(elem)), 0);
}

PrimExpr MinIdentity(const std::string &elem) {
  // Max reduction identity: -inf for floats, the dtype minimum for integers.
  if (elem.compare(0, 5, "float") == 0) {
    return FloatImm(DataType(ffi::StringToDLDataType(elem)),
                    -std::numeric_limits<double>::infinity());
  }
  return min_value(DataType(ffi::StringToDLDataType(elem)));
}

PrimExpr MaxIdentity(const std::string &elem) {
  // Min reduction identity: +inf for floats, the dtype maximum for integers.
  if (elem.compare(0, 5, "float") == 0) {
    return FloatImm(DataType(ffi::StringToDLDataType(elem)),
                    std::numeric_limits<double>::infinity());
  }
  return max_value(DataType(ffi::StringToDLDataType(elem)));
}

} // namespace

const std::vector<ReducePrimitive> &ReducePrimitives() {
  static const std::vector<ReducePrimitive> table = {
      ReducePrimitive{"Add", "vadd", "vcadd", "vadds", ZeroIdentity},
      ReducePrimitive{"Max", "vmax", "vcmax", "vmaxs", MinIdentity},
      ReducePrimitive{"Min", "vmin", "vcmin", "vmins", MaxIdentity},
  };
  return table;
}

bool ReduceOpFor(const std::string &tir_op, std::string *op) {
  for (const ReducePrimitive &primitive : ReducePrimitives()) {
    if (primitive.tir_op == tir_op) {
      *op = primitive.op;
      return true;
    }
  }
  return false;
}

bool ReduceRecipeFor(const std::string &op, std::string *recipe) {
  for (const ReducePrimitive &primitive : ReducePrimitives()) {
    if (primitive.op == op) {
      *recipe = primitive.recipe;
      return true;
    }
  }
  return false;
}

bool ReduceElementFor(const std::string &recipe, std::string *op) {
  for (const ReducePrimitive &primitive : ReducePrimitives()) {
    if (primitive.recipe == recipe) {
      *op = primitive.op;
      return true;
    }
  }
  return false;
}

bool ReduceScalarRecipeFor(const std::string &op, std::string *scalar) {
  for (const ReducePrimitive &primitive : ReducePrimitives()) {
    if (primitive.op == op) {
      *scalar = primitive.scalar;
      return true;
    }
  }
  return false;
}

PrimExpr ReduceIdentity(const std::string &recipe, const std::string &elem) {
  for (const ReducePrimitive &primitive : ReducePrimitives()) {
    if (primitive.recipe == recipe) {
      return primitive.identity(elem);
    }
  }
  TVM_FFI_ICHECK(false) << "unknown reduction recipe '" << recipe << "'";
  return PrimExpr();
}

bool BinaryRecipeFor(const std::string &name_lower, std::string *recipe) {
  auto it = BinaryRecipes().find(name_lower);
  if (it == BinaryRecipes().end())
    return false;
  *recipe = it->second;
  return true;
}

bool UnaryRecipeFor(const std::string &name_lower, std::string *recipe) {
  auto it = UnaryRecipes().find(name_lower);
  if (it == UnaryRecipes().end())
    return false;
  *recipe = it->second;
  return true;
}

bool ComparePredFor(const std::string &name_lower, std::string *pred) {
  auto it = ComparePreds().find(name_lower);
  if (it == ComparePreds().end())
    return false;
  *pred = it->second;
  return true;
}

bool BoolBinaryRecipeFor(const std::string &name_lower, std::string *recipe) {
  auto it = BoolBinaryRecipes().find(name_lower);
  if (it == BoolBinaryRecipes().end())
    return false;
  *recipe = it->second;
  return true;
}

bool BoolUnaryRecipeFor(const std::string &name_lower, std::string *recipe) {
  auto it = BoolUnaryRecipes().find(name_lower);
  if (it == BoolUnaryRecipes().end())
    return false;
  *recipe = it->second;
  return true;
}

std::string VecScalarBinary(const std::string &recipe) {
  auto it = VecScalarBinaryTable().find(recipe);
  return it == VecScalarBinaryTable().end() ? std::string() : it->second;
}

std::string CompareFlip(const std::string &pred) {
  auto it = CompareFlips().find(pred);
  return it == CompareFlips().end() ? std::string() : it->second;
}

bool IsAtomicName(const std::string &tail) {
  return tail.find("atomic") != std::string::npos;
}

bool IsIndexValueOp(const std::string &node_name) {
  return IndexValueOps().count(node_name) != 0;
}

bool ElemBits(const std::string &elem, int *bits) {
  auto it = ElemBitsTable().find(elem);
  if (it == ElemBitsTable().end())
    return false;
  *bits = it->second;
  return true;
}

int VRegLanes(const std::string &elem) {
  if (elem == "bool")
    return 256;
  int bits = 0;
  if (!ElemBits(elem, &bits)) {
    std::string options;
    std::vector<std::string> names;
    for (const auto &kv : ElemBitsTable())
      names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    for (size_t i = 0; i < names.size(); ++i) {
      options += (i == 0 ? "" : ", ") + names[i];
    }
    throw Unsupported("Unknown element dtype '" + elem + "'. Must be one of [" +
                      options + "]");
  }
  return 2048 / bits;
}

std::string VecDtype(const std::string &elem) {
  return elem + "x" + std::to_string(VRegLanes(elem));
}

int MaskBits(const std::string &dtype_like) {
  std::string elem = ElemDtype(dtype_like);
  int bits = 0;
  if (!ElemBits(elem, &bits)) {
    // Same text as the ValueError raised by language/simd.py _mask_bits.
    throw Unsupported("Cannot infer default mask width for dtype '" +
                      dtype_like + "'");
  }
  return std::max(bits, 8);
}

std::string ElemDtype(const std::string &dtype_like) {
  size_t pos = dtype_like.find('x');
  return pos == std::string::npos ? dtype_like : dtype_like.substr(0, pos);
}

/*
 * The dtype spellings are a small closed set and these helpers sit on the
 * emitter's hottest path (every ``T.simd.*`` call materializes its dtype), so
 * the caches are thread-local: TVM compiles multiple PrimFuncs in parallel and
 * a process-wide mutex here serializes every compilation thread.
 */
DataType DataTypeOf(const std::string &dtype_string) {
  thread_local std::unordered_map<std::string, DataType> cache;
  auto it = cache.find(dtype_string);
  if (it != cache.end())
    return it->second;
  DataType parsed(ffi::StringToDLDataType(dtype_string));
  cache.emplace(dtype_string, parsed);
  return parsed;
}

DataType VecDataType(const std::string &elem) {
  return DataTypeOf(VecDtype(elem));
}

DataType ElemDataType(const DataType &dtype) {
  thread_local std::unordered_map<uint64_t, DataType> cache;
  const uint64_t key = static_cast<uint64_t>(dtype.code()) << 40 |
                       static_cast<uint64_t>(dtype.bits()) << 8;
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  DataType parsed = dtype.element_of();
  cache.emplace(key, parsed);
  return parsed;
}

std::string ElemDtype(const DataType &dtype) {
  thread_local std::unordered_map<uint64_t, std::string> cache;
  DataType elem = dtype.element_of();
  const uint64_t key = static_cast<uint64_t>(elem.code()) << 40 |
                       static_cast<uint64_t>(elem.bits()) << 8 |
                       static_cast<uint64_t>(elem.lanes());
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  std::string text = ffi::DLDataTypeToString(elem).operator std::string();
  cache.emplace(key, text);
  return text;
}

PrimExpr IntImm32(int64_t value) { return IntImm(DataType::Int(32), value); }

bool SameBuffer(const tirx::Buffer &lhs, const tirx::Buffer &rhs) {
  return lhs.defined() && rhs.defined() && lhs->data.same_as(rhs->data);
}

bool SameIndices(const ffi::Array<PrimExpr> &lhs,
                 const ffi::Array<PrimExpr> &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  tirx::ExprDeepEqual equal;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!equal(lhs[i], rhs[i]))
      return false;
  }
  return true;
}

tirx::Stmt SeqOf(const ffi::Array<tirx::Stmt> &stmts) {
  if (stmts.empty())
    return tirx::Stmt();
  if (stmts.size() == 1)
    return stmts[0];
  return tirx::SeqStmt(stmts);
}

std::string CallTail(const tirx::Call &call) {
  std::string name;
  if (const auto *op = call->op.as<OpNode>()) {
    name = op->name;
  }
  if (name.find("call_extern") != std::string::npos && !call->args.empty()) {
    // call_extern keeps the real symbol in its first argument (tl.atomic_add).
    const PrimExpr &arg = call->args[0];
    std::string extern_name;
    if (const auto *str_imm = arg.as<tirx::StringImmNode>()) {
      extern_name = str_imm->value;
    } else if (arg.dtype().is_handle()) {
      extern_name.clear();
    }
    if (!extern_name.empty()) {
      size_t pos = extern_name.rfind('.');
      return ToLower(pos == std::string::npos ? extern_name
                                              : extern_name.substr(pos + 1));
    }
  }
  size_t pos = name.rfind('.');
  return ToLower(pos == std::string::npos ? name : name.substr(pos + 1));
}

std::string ToLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string TypeKeyTail(const ffi::ObjectRef &node) {
  std::string key = node->GetTypeKey();
  size_t pos = key.rfind('.');
  return pos == std::string::npos ? key : key.substr(pos + 1);
}

std::string PrintExpr(const PrimExpr &expr) {
  std::ostringstream os;
  os << ffi::ReprPrint(ffi::Any(expr));
  return os.str();
}

std::string PrintStmt(const tirx::Stmt &stmt) {
  std::ostringstream os;
  os << ffi::ReprPrint(ffi::Any(stmt));
  return os.str();
}

std::string ForKindName(tirx::ForKind kind) {
  switch (kind) {
  case tirx::ForKind::kSerial:
    return "serial";
  case tirx::ForKind::kParallel:
    return "parallel";
  case tirx::ForKind::kVectorized:
    return "vectorized";
  case tirx::ForKind::kUnrolled:
    return "unrolled";
  case tirx::ForKind::kThreadBinding:
    return "thread_binding";
  }
  return "serial";
}

} // namespace parallel_vectorize
} // namespace tl
} // namespace tvm
