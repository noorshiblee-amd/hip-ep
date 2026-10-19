/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "PopulatePatterns.h"
#include "TorchFrontend.h"
#include "TorchToHipUtils.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Torch/IR/TorchOps.h"
#include "hip/datatype_abi.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

using namespace mlir;

namespace {

FailureOr<int64_t> readI64Scalar(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return failure();
  auto attr = dyn_cast<IntegerAttr>(constant.getValue());
  if (!attr || !attr.getType().isInteger(64))
    return failure();
  return attr.getInt();
}

FailureOr<float> readF64Scalar(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return failure();
  auto attr = dyn_cast<FloatAttr>(constant.getValue());
  if (!attr || !attr.getType().isF64())
    return failure();
  return static_cast<float>(attr.getValue().convertToDouble());
}

FailureOr<bool> readI1Scalar(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return failure();
  auto attr = dyn_cast<IntegerAttr>(constant.getValue());
  if (!attr || !attr.getType().isInteger(1))
    return failure();
  return attr.getInt() != 0;
}

FailureOr<SmallVector<int64_t>> readI64List(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return failure();
  auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue());
  if (!dense || !dense.getElementType().isInteger(64))
    return failure();
  SmallVector<int64_t> values;
  for (APInt element : dense.getValues<APInt>())
    values.push_back(element.getSExtValue());
  return values;
}

LogicalResult normalizeAxis(int64_t &axis, int64_t rank) {
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    return failure();
  return success();
}

FailureOr<Value> emptyStatic(OpBuilder &builder, Location loc,
                             RankedTensorType type) {
  if (!type.hasStaticShape())
    return failure();
  return Value(tensor::EmptyOp::create(builder, loc, type.getShape(),
                                       type.getElementType()));
}

FailureOr<RankedTensorType> resultTensor(Operation *op, unsigned index) {
  auto valueTensor =
      dyn_cast<torch::ValueTensorType>(op->getResult(index).getType());
  if (!valueTensor) {
    op->emitError("expected a ranked tensor result");
    return failure();
  }
  if (llvm::any_of(valueTensor.getSizes(), ShapedType::isDynamic)) {
    op->emitError("dynamic torch tensor result is not supported");
    return failure();
  }
  Type element = valueTensor.getDtype();
  if (auto integer = dyn_cast<IntegerType>(element))
    element = IntegerType::get(element.getContext(), integer.getWidth());
  return RankedTensorType::get(valueTensor.getSizes(), element);
}

// For the few ops whose result extent is data-dependent.
FailureOr<RankedTensorType> dynamicResultTensor(Operation *op, unsigned index) {
  auto valueTensor =
      dyn_cast<torch::ValueTensorType>(op->getResult(index).getType());
  if (!valueTensor) {
    op->emitError("expected a ranked tensor result");
    return failure();
  }
  Type element = valueTensor.getDtype();
  if (auto integer = dyn_cast<IntegerType>(element))
    element = IntegerType::get(element.getContext(), integer.getWidth());
  return RankedTensorType::get(valueTensor.getSizes(), element);
}

Value i64Vector(OpBuilder &builder, Location loc, ArrayRef<int64_t> values) {
  auto type = RankedTensorType::get({static_cast<int64_t>(values.size())},
                                    builder.getI64Type());
  return arith::ConstantOp::create(builder, loc,
                                   DenseIntElementsAttr::get(type, values));
}

int64_t onnxTypeCode(Type type) {
  if (type.isF32())
    return 1;
  if (type.isInteger(8))
    return 3;
  if (type.isInteger(16))
    return 5;
  if (type.isInteger(32))
    return 6;
  if (type.isInteger(64))
    return 7;
  if (type.isInteger(1))
    return 9;
  if (type.isF16())
    return 10;
  if (type.isF64())
    return 11;
  if (type.isBF16())
    return 16;
  return 0;
}

FailureOr<Value> hipCast(OpBuilder &builder, Location loc, Value context,
                         Value input, RankedTensorType resultType) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (inputType && inputType == resultType)
    return input;
  int64_t code = onnxTypeCode(resultType.getElementType());
  if (code == 0)
    return failure();
  FailureOr<Value> init = emptyStatic(builder, loc, resultType);
  if (failed(init))
    return failure();
  auto cast = hip::CastOp::create(builder, loc, context, input, *init,
                                  builder.getI64IntegerAttr(code));
  return cast->getResult(0);
}

FailureOr<Value> scalarTensor(OpBuilder &builder, Location loc, Value scalar,
                              Type elementType) {
  Value converted = scalar;
  Type source = scalar.getType();
  if (source != elementType) {
    if (isa<FloatType>(elementType)) {
      FailureOr<Value> materialized =
          hip::materializeScalarTensor(builder, loc, scalar, elementType);
      if (failed(materialized))
        return failure();
      return materialized;
    }
    if (!source.isIntOrIndex() || !elementType.isIntOrIndex())
      return failure();
    unsigned sourceBits = source.getIntOrFloatBitWidth();
    unsigned destBits = elementType.getIntOrFloatBitWidth();
    if (sourceBits > destBits)
      converted = arith::TruncIOp::create(builder, loc, elementType, scalar);
    else if (sourceBits < destBits)
      converted = arith::ExtUIOp::create(builder, loc, elementType, scalar);
  }
  auto type = RankedTensorType::get({}, elementType);
  return Value(tensor::SplatOp::create(builder, loc, type, converted));
}

template <typename HipOp>
FailureOr<Value> unaryHip(OpBuilder &builder, Location loc, Value context,
                          Value input, RankedTensorType resultType) {
  FailureOr<Value> init = emptyStatic(builder, loc, resultType);
  if (failed(init))
    return failure();
  return HipOp::create(builder, loc, TypeRange{resultType}, context, input,
                       *init)
      ->getResult(0);
}

FailureOr<Value> transposeTo(OpBuilder &builder, Location loc, Value context,
                             Value input, ArrayRef<int64_t> perm,
                             RankedTensorType resultType) {
  FailureOr<Value> init = emptyStatic(builder, loc, resultType);
  if (failed(init))
    return failure();
  auto transposed = hip::TransposeOp::create(
      builder, loc, context, input, *init, builder.getI64ArrayAttr(perm));
  return transposed->getResult(0);
}

bool elementsStillTorch(Operation *list) {
  return llvm::any_of(list->getOperands(), [](Value value) {
    return torch::isTorchType(value.getType());
  });
}

LogicalResult lowerExpand(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType ||
      inputType.getElementType() != resultType->getElementType()) {
    return op->emitError("aten.expand input element type does not match");
  }
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError("aten.expand requires a static result shape");
  auto expanded =
      hip::ExpandOp::create(builder, op->getLoc(), context, op->getOperand(0),
                            op->getOperand(1), *init);
  op->getResult(0).replaceAllUsesWith(expanded->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerBmm(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError("aten.bmm requires a static result shape");
  SmallVector<NamedAttribute> attrs = {
      builder.getNamedAttr("transA", builder.getI64IntegerAttr(0)),
      builder.getNamedAttr("transB", builder.getI64IntegerAttr(0))};
  SmallVector<Value> operands = {context, op->getOperand(0), op->getOperand(1),
                                 *init};
  auto matmul = hip::MatmulOp::create(builder, op->getLoc(), operands, attrs);
  op->getResult(0).replaceAllUsesWith(matmul.getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerVarMean(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> varType = resultTensor(op, 0);
  FailureOr<RankedTensorType> meanType = resultTensor(op, 1);
  if (failed(varType) || failed(meanType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType || !inputType.hasStaticShape())
    return op->emitError("aten.var_mean requires a static input");
  FailureOr<SmallVector<int64_t>> dims = readI64List(op->getOperand(1));
  FailureOr<int64_t> correction = readI64Scalar(op->getOperand(2));
  FailureOr<bool> keepdim = readI1Scalar(op->getOperand(3));
  if (failed(dims) || failed(correction) || failed(keepdim))
    return op->emitError("aten.var_mean axes must be constant");
  for (int64_t &axis : *dims) {
    if (failed(normalizeAxis(axis, inputType.getRank())))
      return op->emitError("aten.var_mean axis is out of range");
  }
  Location loc = op->getLoc();
  Value axes = i64Vector(builder, loc, *dims);
  FailureOr<Value> meanInit = emptyStatic(builder, loc, *meanType);
  if (failed(meanInit))
    return failure();
  auto mean = hip::ReduceMeanOp::create(
      builder, loc, context, op->getOperand(0), axes, *meanInit,
      builder.getI64IntegerAttr(*keepdim ? 1 : 0),
      builder.getI64IntegerAttr(0));
  FailureOr<Value> centered = hip::emitElementwise(
      builder, loc, context, op->getOperand(0), mean->getResult(0), inputType,
      hip::ElementwiseKind::Sub);
  if (failed(centered))
    return op->emitError("aten.var_mean cannot subtract the mean");
  FailureOr<Value> squared =
      hip::emitElementwise(builder, loc, context, *centered, *centered,
                           inputType, hip::ElementwiseKind::Mul);
  if (failed(squared))
    return op->emitError("aten.var_mean cannot square the residual");
  FailureOr<Value> varInit = emptyStatic(builder, loc, *varType);
  if (failed(varInit))
    return failure();
  auto variance =
      hip::ReduceMeanOp::create(builder, loc, context, *squared, axes, *varInit,
                                builder.getI64IntegerAttr(*keepdim ? 1 : 0),
                                builder.getI64IntegerAttr(0));
  Value var = variance->getResult(0);
  if (*correction != 0) {
    int64_t count = 1;
    for (int64_t axis : *dims) {
      if (inputType.isDynamicDim(axis))
        return op->emitError("aten.var_mean correction needs static axes");
      count *= inputType.getDimSize(axis);
    }
    if (count <= *correction)
      return op->emitError("aten.var_mean correction is not smaller than N");
    auto scaleAttr = builder.getFloatAttr(
        builder.getF32Type(),
        static_cast<double>(count) / static_cast<double>(count - *correction));
    Value scaleScalar = arith::ConstantOp::create(builder, loc, scaleAttr);
    FailureOr<Value> scale = hip::materializeScalarTensor(
        builder, loc, scaleScalar, varType->getElementType());
    if (failed(scale))
      return failure();
    FailureOr<Value> scaled =
        hip::emitElementwise(builder, loc, context, var, *scale, *varType,
                             hip::ElementwiseKind::Mul);
    if (failed(scaled))
      return failure();
    var = *scaled;
  }
  op->getResult(0).replaceAllUsesWith(var);
  op->getResult(1).replaceAllUsesWith(mean->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerWhere(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError("aten.where requires a static result shape");
  auto selected =
      hip::WhereOp::create(builder, op->getLoc(), context, op->getOperand(0),
                           op->getOperand(1), op->getOperand(2), *init);
  op->getResult(0).replaceAllUsesWith(selected->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerScalarTensor(Operation *op, OpBuilder &builder,
                                Value /*context*/) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType) || resultType->getRank() != 0)
    return op->emitError("aten.scalar_tensor requires a rank-0 result");
  FailureOr<Value> value = scalarTensor(
      builder, op->getLoc(), op->getOperand(0), resultType->getElementType());
  if (failed(value))
    return op->emitError("aten.scalar_tensor cannot materialize its value");
  op->getResult(0).replaceAllUsesWith(*value);
  op->erase();
  return success();
}

LogicalResult lowerFullLike(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> fill = scalarTensor(builder, op->getLoc(), op->getOperand(1),
                                       resultType->getElementType());
  if (failed(fill))
    return op->emitError("aten.full_like cannot materialize its fill value");
  Value result = *fill;
  if (resultType->getRank() != 0) {
    FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
    if (failed(init))
      return failure();
    Value shape = i64Vector(builder, op->getLoc(), resultType->getShape());
    result = hip::ExpandOp::create(builder, op->getLoc(), context, result,
                                   shape, *init)
                 ->getResult(0);
  }
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return success();
}

LogicalResult lowerClone(Operation *op, OpBuilder & /*builder*/,
                         Value /*context*/) {
  op->getResult(0).replaceAllUsesWith(op->getOperand(0));
  op->erase();
  return success();
}

LogicalResult lowerCopy(Operation *op, OpBuilder & /*builder*/,
                        Value /*context*/) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  Value src = op->getOperand(1);
  if (src.getType() != *resultType)
    return op->emitError(
        "aten.copy requires src to match the result shape and element type");
  op->getResult(0).replaceAllUsesWith(src);
  op->erase();
  return success();
}

LogicalResult lowerSoftmax(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType || !inputType.hasStaticShape() || inputType.getRank() == 0)
    return op->emitError("aten._softmax requires a static ranked input");
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(1));
  FailureOr<bool> halfToFloat = readI1Scalar(op->getOperand(2));
  if (failed(dim) || failed(halfToFloat))
    return op->emitError("aten._softmax dim must be a constant");
  if (*halfToFloat)
    return op->emitError("aten._softmax half_to_float is not supported");
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, inputType.getRank())))
    return op->emitError("aten._softmax dim is out of range");
  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  SmallVector<int64_t> forward;
  if (axis != inputType.getRank() - 1) {
    for (int64_t index : llvm::seq<int64_t>(0, inputType.getRank())) {
      if (index != axis)
        forward.push_back(index);
    }
    forward.push_back(axis);
    SmallVector<int64_t> moved;
    for (int64_t index : forward)
      moved.push_back(inputType.getDimSize(index));
    auto movedType = RankedTensorType::get(moved, inputType.getElementType());
    FailureOr<Value> transposed =
        transposeTo(builder, loc, context, input, forward, movedType);
    if (failed(transposed))
      return failure();
    input = *transposed;
    inputType = movedType;
  }
  FailureOr<Value> init = emptyStatic(builder, loc, inputType);
  if (failed(init))
    return failure();
  Value softmaxed =
      hip::MiopenSoftmaxOp::create(builder, loc, context, input, *init)
          ->getResult(0);
  if (axis != resultType->getRank() - 1) {
    SmallVector<int64_t> backward(forward.size());
    for (auto [index, source] : llvm::enumerate(forward))
      backward[source] = static_cast<int64_t>(index);
    FailureOr<Value> restored =
        transposeTo(builder, loc, context, softmaxed, backward, *resultType);
    if (failed(restored))
      return failure();
    softmaxed = *restored;
  }
  op->getResult(0).replaceAllUsesWith(softmaxed);
  op->erase();
  return success();
}

LogicalResult lowerEqScalar(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return op->emitError("aten.eq.Scalar requires a ranked input");
  FailureOr<Value> other = scalarTensor(
      builder, op->getLoc(), op->getOperand(1), inputType.getElementType());
  if (failed(other))
    return op->emitError("aten.eq.Scalar cannot materialize its scalar");
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return failure();
  auto equal = hip::EqualOp::create(builder, op->getLoc(), context,
                                    op->getOperand(0), *other, *init);
  op->getResult(0).replaceAllUsesWith(equal->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerAllDim(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return op->emitError("aten.all.dim requires a ranked input");
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(1));
  FailureOr<bool> keepdim = readI1Scalar(op->getOperand(2));
  if (failed(dim) || failed(keepdim))
    return op->emitError("aten.all.dim axis must be constant");
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, inputType.getRank())))
    return op->emitError("aten.all.dim axis is out of range");
  Value axes = i64Vector(builder, op->getLoc(), {axis});
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return failure();
  auto reduced = hip::ReduceMinOp::create(
      builder, op->getLoc(), context, op->getOperand(0), axes, *init,
      builder.getI64IntegerAttr(*keepdim ? 1 : 0),
      builder.getI64IntegerAttr(0));
  op->getResult(0).replaceAllUsesWith(reduced->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerAssert(Operation *op, OpBuilder & /*builder*/,
                          Value /*context*/) {
  op->erase();
  return success();
}

LogicalResult lowerSizeInt(Operation *op, OpBuilder &builder,
                           Value /*context*/) {
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(1));
  if (!inputType || failed(dim))
    return op->emitError(
        "aten.size.int requires a ranked input and constant dim");
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, inputType.getRank())))
    return op->emitError("aten.size.int dim is out of range");
  Location loc = op->getLoc();
  Type i64 = builder.getI64Type();
  Value size;
  if (inputType.isDynamicDim(axis)) {
    size = arith::IndexCastOp::create(
        builder, loc, i64,
        tensor::DimOp::create(builder, loc, op->getOperand(0), axis));
  } else {
    size = arith::ConstantOp::create(
        builder, loc, i64,
        builder.getI64IntegerAttr(inputType.getDimSize(axis)));
  }
  op->getResult(0).replaceAllUsesWith(size);
  op->erase();
  return success();
}

template <arith::CmpIPredicate Predicate>
LogicalResult lowerIntCompare(Operation *op, OpBuilder &builder,
                              Value /*context*/) {
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  if (!lhs.getType().isInteger(64) || !rhs.getType().isInteger(64))
    return op->emitError("integer scalar compare expects i64 operands");
  Value compared =
      arith::CmpIOp::create(builder, op->getLoc(), Predicate, lhs, rhs);
  op->getResult(0).replaceAllUsesWith(compared);
  op->erase();
  return success();
}

// A 1-D input with at most one element is already free of consecutive
// duplicates. Only that case is lowered; inverse and counts must be off.
// Before: unique_consecutive %x : tensor<1xsi64> -> tensor<?xsi64>, ...
// After:  tensor.cast %x : tensor<1xi64> to tensor<?xi64>
LogicalResult lowerUniqueConsecutive(Operation *op, OpBuilder &builder,
                                     Value /*context*/) {
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  FailureOr<bool> inverse = readI1Scalar(op->getOperand(1));
  FailureOr<bool> counts = readI1Scalar(op->getOperand(2));
  if (!inputType || !inputType.hasStaticShape() || inputType.getRank() != 1 ||
      inputType.getNumElements() > 1 || failed(inverse) || failed(counts) ||
      *inverse || *counts ||
      !isa<torch::TorchNoneType>(op->getOperand(3).getType())) {
    return op->emitError("aten.unique_consecutive only lowers a 1-D input "
                         "with at most one element");
  }
  Location loc = op->getLoc();
  SmallVector<Value> replacements;
  for (unsigned index : llvm::seq<unsigned>(0, op->getNumResults())) {
    FailureOr<RankedTensorType> type = dynamicResultTensor(op, index);
    if (failed(type))
      return failure();
    if (index == 0) {
      replacements.push_back(
          tensor::CastOp::create(builder, loc, *type, op->getOperand(0)));
      continue;
    }
    FailureOr<Value> empty = emptyStatic(builder, loc, *type);
    if (failed(empty))
      return op->emitError(
          "aten.unique_consecutive unused results must be static");
    replacements.push_back(*empty);
  }
  for (auto [result, replacement] :
       llvm::zip_equal(op->getResults(), replacements))
    result.replaceAllUsesWith(replacement);
  op->erase();
  return success();
}

LogicalResult lowerArange(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  Location loc = op->getLoc();
  Type element = resultType->getElementType();
  FailureOr<Value> start =
      scalarTensor(builder, loc, op->getOperand(0), element);
  FailureOr<Value> limit =
      scalarTensor(builder, loc, op->getOperand(1), element);
  FailureOr<Value> delta =
      scalarTensor(builder, loc, op->getOperand(2), element);
  FailureOr<Value> init = emptyStatic(builder, loc, *resultType);
  if (failed(start) || failed(limit) || failed(delta) || failed(init))
    return op->emitError("aten.arange cannot materialize its bounds");
  auto range = hip::RangeOp::create(builder, loc, context, *start, *limit,
                                    *delta, *init);
  op->getResult(0).replaceAllUsesWith(range->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerSlice(Operation *op, OpBuilder &builder, Value /*context*/) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType || !inputType.hasStaticShape() ||
      inputType.getRank() != resultType->getRank()) {
    return op->emitError("aten.slice.Tensor requires a static same-rank input");
  }
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(1));
  FailureOr<int64_t> start = readI64Scalar(op->getOperand(2));
  FailureOr<int64_t> end = readI64Scalar(op->getOperand(3));
  FailureOr<int64_t> step = readI64Scalar(op->getOperand(4));
  if (failed(dim) || failed(start) || failed(end) || failed(step) ||
      *step <= 0) {
    return op->emitError("aten.slice.Tensor bounds must be constant");
  }
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, inputType.getRank())))
    return op->emitError("aten.slice.Tensor dim is out of range");
  int64_t size = inputType.getDimSize(axis);
  int64_t begin = *start < 0 ? *start + size : *start;
  int64_t limit = *end < 0 ? *end + size : *end;
  if (begin < 0)
    begin = 0;
  if (begin > size)
    begin = size;
  if (limit > size)
    limit = size;
  int64_t count = limit > begin ? (limit - begin + *step - 1) / *step : 0;
  if (!resultType->isDynamicDim(axis) && resultType->getDimSize(axis) != count)
    return op->emitError("aten.slice.Tensor result size does not match");
  // tensor.extract_slice rejects an offset equal to the dim size, even
  // for an empty slice.
  // Before: slice [96:] of tensor<1x96xf32> -> tensor<1x0xf32>
  // After:  tensor.extract_slice offsets [0, 0] sizes [1, 0]
  if (count == 0)
    begin = 0;
  SmallVector<OpFoldResult> offsets(inputType.getRank(),
                                    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(inputType.getRank(),
                                    builder.getIndexAttr(1));
  for (int64_t index : llvm::seq<int64_t>(0, inputType.getRank()))
    sizes.push_back(builder.getIndexAttr(resultType->getDimSize(index)));
  offsets[axis] = builder.getIndexAttr(begin);
  strides[axis] = builder.getIndexAttr(*step);
  auto sliced = tensor::ExtractSliceOp::create(builder, op->getLoc(),
                                               *resultType, op->getOperand(0),
                                               offsets, sizes, strides);
  op->getResult(0).replaceAllUsesWith(sliced.getResult());
  op->erase();
  return success();
}

// slice_scatter writes src into one window of self. A huge end is the
// open bound, same as slice.Tensor.
// Before: slice_scatter dim 0, start 1, on tensor<197x197xsi64>
// After:  tensor.insert_slice offsets [1, 0] sizes [196, 197]
LogicalResult lowerSliceScatter(Operation *op, OpBuilder &builder,
                                Value /*context*/) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto selfType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto srcType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!selfType || !srcType || !selfType.hasStaticShape() ||
      !srcType.hasStaticShape() || selfType.getRank() != srcType.getRank() ||
      selfType.getShape() != resultType->getShape()) {
    return op->emitError(
        "aten.slice_scatter requires static same-rank tensors");
  }
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(2));
  FailureOr<int64_t> start = readI64Scalar(op->getOperand(3));
  FailureOr<int64_t> end = readI64Scalar(op->getOperand(4));
  FailureOr<int64_t> step = readI64Scalar(op->getOperand(5));
  if (failed(dim) || failed(start) || failed(end) || failed(step) ||
      *step <= 0) {
    return op->emitError("aten.slice_scatter bounds must be constant");
  }
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, selfType.getRank())))
    return op->emitError("aten.slice_scatter dim is out of range");
  int64_t size = selfType.getDimSize(axis);
  int64_t begin = *start < 0 ? *start + size : *start;
  int64_t limit = *end < 0 ? *end + size : *end;
  if (begin < 0)
    begin = 0;
  if (begin > size)
    begin = size;
  if (limit > size)
    limit = size;
  int64_t count = limit > begin ? (limit - begin + *step - 1) / *step : 0;
  if (count == 0)
    begin = 0;
  for (int64_t index : llvm::seq<int64_t>(0, selfType.getRank())) {
    int64_t expected = index == axis ? count : selfType.getDimSize(index);
    if (srcType.getDimSize(index) != expected)
      return op->emitError("aten.slice_scatter source shape does not match");
  }
  if (srcType.getElementType() != resultType->getElementType())
    return op->emitError("aten.slice_scatter source element type must match");
  SmallVector<OpFoldResult> offsets(selfType.getRank(),
                                    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(selfType.getRank(),
                                    builder.getIndexAttr(1));
  for (int64_t dimSize : srcType.getShape())
    sizes.push_back(builder.getIndexAttr(dimSize));
  offsets[axis] = builder.getIndexAttr(begin);
  strides[axis] = builder.getIndexAttr(*step);
  auto inserted =
      tensor::InsertSliceOp::create(builder, op->getLoc(), op->getOperand(1),
                                    op->getOperand(0), offsets, sizes, strides);
  op->getResult(0).replaceAllUsesWith(inserted.getResult());
  op->erase();
  return success();
}

LogicalResult lowerUpsample(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError("aten.upsample_nearest2d requires a static shape");
  // mode 0 = nearest, coord transform 1 = asymmetric, nearest mode 0 = floor.
  auto resized = hip::ResizeOp::create(
      builder, op->getLoc(), *resultType, context, op->getOperand(0), *init,
      builder.getI64IntegerAttr(0), builder.getI64IntegerAttr(1),
      builder.getI64IntegerAttr(0));
  op->getResult(0).replaceAllUsesWith(resized.getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerBitwiseAnd(Operation *op, OpBuilder &builder,
                              Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return failure();
  auto combined =
      hip::AndOp::create(builder, op->getLoc(), context, op->getOperand(0),
                         op->getOperand(1), *init);
  op->getResult(0).replaceAllUsesWith(combined->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerEmpty(Operation *op, OpBuilder &builder, Value /*context*/) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError("aten.empty_strided requires a static shape");
  op->getResult(0).replaceAllUsesWith(*init);
  op->erase();
  return success();
}

LogicalResult lowerConvert(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> casted =
      hipCast(builder, op->getLoc(), context, op->getOperand(0), *resultType);
  if (failed(casted))
    return op->emitError("prims.convert_element_type cannot cast");
  op->getResult(0).replaceAllUsesWith(*casted);
  op->erase();
  return success();
}

// A zero-element input contributes no values, and its rank may have
// collapsed. The concat axis is resolved against the result rank.
// Before: cat([tensor<0xf32>, %x : tensor<1x2x3x4xf32>], dim = -2)
// After:  %x
LogicalResult lowerCat(Operation *op, OpBuilder &builder, Value context) {
  Operation *list = op->getOperand(0).getDefiningOp();
  if (!list || list->getName().getStringRef() != "torch.prim.ListConstruct")
    return op->emitError("aten.cat expects a prim.ListConstruct");
  if (elementsStillTorch(list))
    return success();
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  if (list->getNumOperands() == 0)
    return op->emitError("aten.cat list is empty");
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(1));
  if (failed(dim))
    return op->emitError("aten.cat dim must be a constant");
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, resultType->getRank())))
    return op->emitError("aten.cat dim is out of range");
  // Torch promotes mixed-dtype inputs to the result element type.
  SmallVector<Value> inputs;
  for (Value input : list->getOperands()) {
    auto type = dyn_cast<RankedTensorType>(input.getType());
    if (!type || !type.hasStaticShape())
      return op->emitError("aten.cat inputs must be ranked tensors");
    if (type.getNumElements() == 0)
      continue;
    if (type.getRank() != resultType->getRank())
      return op->emitError("aten.cat inputs must match the result rank");
    if (type.getElementType() != resultType->getElementType()) {
      FailureOr<Value> casted = hipCast(
          builder, op->getLoc(), context, input,
          RankedTensorType::get(type.getShape(), resultType->getElementType()));
      if (failed(casted))
        return op->emitError("aten.cat cannot cast an input");
      input = *casted;
    }
    inputs.push_back(input);
  }
  if (inputs.empty())
    return op->emitError("aten.cat list is empty");
  if (inputs.size() == 1 && inputs[0].getType() == *resultType) {
    op->getResult(0).replaceAllUsesWith(inputs[0]);
    op->erase();
    return success();
  }
  auto concat = tensor::ConcatOp::create(builder, op->getLoc(), axis, inputs);
  if (concat.getResult().getType() != *resultType)
    return op->emitError("aten.cat result shape does not match");
  op->getResult(0).replaceAllUsesWith(concat.getResult());
  op->erase();
  return success();
}

LogicalResult lowerCumsum(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(1));
  if (!inputType || failed(dim))
    return op->emitError("aten.cumsum dim must be a constant");
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, inputType.getRank())))
    return op->emitError("aten.cumsum dim is out of range");
  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  if (inputType.getElementType() != resultType->getElementType()) {
    auto castType = RankedTensorType::get(inputType.getShape(),
                                          resultType->getElementType());
    FailureOr<Value> casted = hipCast(builder, loc, context, input, castType);
    if (failed(casted))
      return op->emitError("aten.cumsum cannot cast its input");
    input = *casted;
  }
  FailureOr<Value> axisTensor =
      scalarTensor(builder, loc, op->getOperand(1), builder.getI64Type());
  FailureOr<Value> init = emptyStatic(builder, loc, *resultType);
  if (failed(axisTensor) || failed(init))
    return failure();
  auto sum = hip::CumSumOp::create(builder, loc, context, input, *axisTensor,
                                   *init, builder.getI64IntegerAttr(0),
                                   builder.getI64IntegerAttr(0));
  op->getResult(0).replaceAllUsesWith(sum->getResult(0));
  op->erase();
  return success();
}

template <typename HipOp>
LogicalResult lowerUnaryTensor(Operation *op, OpBuilder &builder, Value context,
                               StringRef name) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> result = unaryHip<HipOp>(builder, op->getLoc(), context,
                                            op->getOperand(0), *resultType);
  if (failed(result))
    return op->emitError() << name << " requires a static result shape";
  op->getResult(0).replaceAllUsesWith(*result);
  op->erase();
  return success();
}

LogicalResult lowerSin(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryTensor<hip::SinOp>(op, builder, context, "aten.sin");
}

LogicalResult lowerExp(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryTensor<hip::ExpOp>(op, builder, context, "aten.exp");
}

LogicalResult lowerCos(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryTensor<hip::CosOp>(op, builder, context, "aten.cos");
}

LogicalResult lowerSigmoid(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryTensor<hip::SigmoidOp>(op, builder, context, "aten.sigmoid");
}

LogicalResult lowerTanh(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryTensor<hip::TanhOp>(op, builder, context, "aten.tanh");
}

template <typename HipOp>
LogicalResult lowerUnaryTyped(Operation *op, OpBuilder &builder, Value context,
                              StringRef name) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError() << name << " requires a static result shape";
  Value result = HipOp::create(builder, op->getLoc(), *resultType, context,
                               op->getOperand(0), *init)
                     ->getResult(0);
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return success();
}

template <typename HipOp>
LogicalResult lowerUnaryInferred(Operation *op, OpBuilder &builder,
                                 Value context, StringRef name) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError() << name << " requires a static result shape";
  Value result =
      HipOp::create(builder, op->getLoc(), context, op->getOperand(0), *init)
          ->getResult(0);
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return success();
}

LogicalResult lowerAbs(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryTyped<hip::AbsOp>(op, builder, context, "aten.abs");
}

LogicalResult lowerCeil(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryTyped<hip::CeilOp>(op, builder, context, "aten.ceil");
}

LogicalResult lowerFloor(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryTyped<hip::FloorOp>(op, builder, context, "aten.floor");
}

LogicalResult lowerLog(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryTyped<hip::LogOp>(op, builder, context, "aten.log");
}

LogicalResult lowerLog1p(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType) || !isa<FloatType>(resultType->getElementType()))
    return op->emitError("aten.log1p requires a float result");
  Location loc = op->getLoc();
  auto scalarType = RankedTensorType::get({}, resultType->getElementType());
  Value one = arith::ConstantOp::create(
      builder, loc,
      DenseElementsAttr::get(
          scalarType, builder.getFloatAttr(scalarType.getElementType(), 1.0)));
  FailureOr<Value> shifted =
      hip::emitElementwise(builder, loc, context, op->getOperand(0), one,
                           *resultType, hip::ElementwiseKind::Add);
  if (failed(shifted))
    return op->emitError("aten.log1p add failed");
  FailureOr<Value> logged =
      unaryHip<hip::LogOp>(builder, loc, context, *shifted, *resultType);
  if (failed(logged))
    return failure();
  op->getResult(0).replaceAllUsesWith(*logged);
  op->erase();
  return success();
}

LogicalResult lowerSign(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryInferred<hip::SignOp>(op, builder, context, "aten.sign");
}

LogicalResult lowerSqrt(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryInferred<hip::SqrtOp>(op, builder, context, "aten.sqrt");
}

LogicalResult lowerBitwiseNot(Operation *op, OpBuilder &builder,
                              Value context) {
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType || !inputType.getElementType().isInteger(1))
    return op->emitError("aten.bitwise_not only lowers an i1 tensor");
  return lowerUnaryInferred<hip::NotOp>(op, builder, context,
                                        "aten.bitwise_not");
}

// x > c is c < x. x < c is hip.less. x <= c is the complement of x > c.
LogicalResult lowerScalarOrder(Operation *op, OpBuilder &builder, Value context,
                               StringRef name, bool swap, bool invert) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return op->emitError() << name << " requires a ranked input";
  FailureOr<Value> other = scalarTensor(
      builder, op->getLoc(), op->getOperand(1), inputType.getElementType());
  if (failed(other))
    return op->emitError() << name << " cannot materialize its scalar";
  Location loc = op->getLoc();
  Value lhs = swap ? *other : op->getOperand(0);
  Value rhs = swap ? op->getOperand(0) : *other;
  FailureOr<Value> lessInit = emptyStatic(builder, loc, *resultType);
  if (failed(lessInit))
    return failure();
  Value less = hip::LessOp::create(builder, loc, context, lhs, rhs, *lessInit)
                   ->getResult(0);
  Value result = less;
  if (invert) {
    FailureOr<Value> notInit = emptyStatic(builder, loc, *resultType);
    if (failed(notInit))
      return failure();
    result =
        hip::NotOp::create(builder, loc, context, less, *notInit)->getResult(0);
  }
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return success();
}

LogicalResult lowerGtScalar(Operation *op, OpBuilder &builder, Value context) {
  return lowerScalarOrder(op, builder, context, "aten.gt.Scalar",
                          /*swap=*/true, /*invert=*/false);
}

LogicalResult lowerLtScalar(Operation *op, OpBuilder &builder, Value context) {
  return lowerScalarOrder(op, builder, context, "aten.lt.Scalar",
                          /*swap=*/false, /*invert=*/false);
}

LogicalResult lowerLeScalar(Operation *op, OpBuilder &builder, Value context) {
  return lowerScalarOrder(op, builder, context, "aten.le.Scalar",
                          /*swap=*/true, /*invert=*/true);
}

LogicalResult lowerGather(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  auto dataType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto indexType = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  if (failed(resultType) || !dataType || !indexType ||
      !indexType.getElementType().isIntOrIndex()) {
    return op->emitError(
        "aten.gather requires ranked data and integer indices");
  }
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(1));
  FailureOr<bool> sparse = readI1Scalar(op->getOperand(3));
  if (failed(dim) || failed(sparse) || *sparse) {
    return op->emitError(
        "aten.gather only lowers a constant dim with sparse_grad false");
  }
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, dataType.getRank())) ||
      indexType.getRank() != dataType.getRank() ||
      indexType.getShape() != resultType->getShape()) {
    return op->emitError("aten.gather axis or index shape is invalid");
  }
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return failure();
  Value gathered =
      hip::GatherElementsOp::create(builder, op->getLoc(), context,
                                    op->getOperand(0), op->getOperand(2), *init,
                                    builder.getI64IntegerAttr(axis))
          ->getResult(0);
  op->getResult(0).replaceAllUsesWith(gathered);
  op->erase();
  return success();
}

LogicalResult lowerGelu(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto approximate = op->getOperand(1).getDefiningOp<torch::ConstantStrOp>();
  if (!approximate ||
      (approximate.getValue() != "none" && approximate.getValue() != "tanh")) {
    return op->emitError("aten.gelu approximate must be \"none\" or \"tanh\"");
  }
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError("aten.gelu requires a static result shape");
  Value result =
      hip::GeluOp::create(builder, op->getLoc(), context, op->getOperand(0),
                          *init, approximate.getValueAttr())
          ->getResult(0);
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return success();
}

LogicalResult lowerEmbedding(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<int64_t> padding = readI64Scalar(op->getOperand(2));
  FailureOr<bool> scale = readI1Scalar(op->getOperand(3));
  FailureOr<bool> sparse = readI1Scalar(op->getOperand(4));
  if (failed(padding) || failed(scale) || failed(sparse) || *scale || *sparse) {
    return op->emitError(
        "aten.embedding only lowers padding_idx with scale/sparse false");
  }
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return failure();
  Value gathered = hip::GatherOp::create(builder, op->getLoc(), context,
                                         op->getOperand(0), op->getOperand(1),
                                         *init, builder.getI64IntegerAttr(0))
                       ->getResult(0);
  op->getResult(0).replaceAllUsesWith(gathered);
  op->erase();
  return success();
}

LogicalResult lowerAlias(Operation *op, OpBuilder & /*builder*/,
                         Value /*context*/) {
  op->getResult(0).replaceAllUsesWith(op->getOperand(0));
  op->erase();
  return success();
}

LogicalResult lowerEqTensor(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return failure();
  auto compared =
      hip::EqualOp::create(builder, op->getLoc(), context, op->getOperand(0),
                           op->getOperand(1), *init);
  op->getResult(0).replaceAllUsesWith(compared->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerLeTensor(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  Location loc = op->getLoc();
  FailureOr<Value> lessInit = emptyStatic(builder, loc, *resultType);
  FailureOr<Value> notInit = emptyStatic(builder, loc, *resultType);
  if (failed(lessInit) || failed(notInit))
    return failure();
  // A <= B  <=>  !(B < A)
  Value less = hip::LessOp::create(builder, loc, context, op->getOperand(1),
                                   op->getOperand(0), *lessInit)
                   ->getResult(0);
  Value le =
      hip::NotOp::create(builder, loc, context, less, *notInit)->getResult(0);
  op->getResult(0).replaceAllUsesWith(le);
  op->erase();
  return success();
}

LogicalResult lowerMeanDim(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  if (!isa<torch::TorchNoneType>(op->getOperand(3).getType()))
    return op->emitError("aten.mean.dim dtype must be none");
  FailureOr<SmallVector<int64_t>> dims = readI64List(op->getOperand(1));
  FailureOr<bool> keepdim = readI1Scalar(op->getOperand(2));
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType || failed(dims) || failed(keepdim))
    return op->emitError("aten.mean.dim axes must be constant");
  SmallVector<int64_t> normalized;
  for (int64_t axis : *dims) {
    if (failed(normalizeAxis(axis, inputType.getRank())))
      return op->emitError("aten.mean.dim axis is out of range");
    normalized.push_back(axis);
  }
  Value axes = i64Vector(builder, op->getLoc(), normalized);
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return failure();
  Value mean = hip::ReduceMeanOp::create(
                   builder, op->getLoc(), context, op->getOperand(0), axes,
                   *init, builder.getI64IntegerAttr(*keepdim ? 1 : 0),
                   builder.getI64IntegerAttr(0))
                   ->getResult(0);
  op->getResult(0).replaceAllUsesWith(mean);
  op->erase();
  return success();
}

LogicalResult lowerNeScalar(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return op->emitError("aten.ne.Scalar requires a ranked input");
  Location loc = op->getLoc();
  FailureOr<Value> other =
      scalarTensor(builder, loc, op->getOperand(1), inputType.getElementType());
  FailureOr<Value> equalInit = emptyStatic(builder, loc, *resultType);
  FailureOr<Value> notInit = emptyStatic(builder, loc, *resultType);
  if (failed(other) || failed(equalInit) || failed(notInit))
    return op->emitError("aten.ne.Scalar cannot materialize its scalar");
  auto compared = hip::EqualOp::create(builder, loc, context, op->getOperand(0),
                                       *other, *equalInit);
  auto inverted = hip::NotOp::create(builder, loc, context,
                                     compared->getResult(0), *notInit);
  op->getResult(0).replaceAllUsesWith(inverted->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerNeg(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryTensor<hip::NegOp>(op, builder, context, "aten.neg");
}

LogicalResult lowerPowTensorScalar(Operation *op, OpBuilder &builder,
                                   Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  // A float exponent that is an integer, such as 3.0, uses the same multiply
  // path. log/exp would be undefined for a negative base.
  std::optional<int64_t> exponent;
  if (FailureOr<int64_t> asInt = readI64Scalar(op->getOperand(1));
      succeeded(asInt)) {
    exponent = *asInt;
  } else if (auto constant =
                 op->getOperand(1).getDefiningOp<arith::ConstantOp>()) {
    if (auto floating = dyn_cast<FloatAttr>(constant.getValue())) {
      double value = floating.getValue().convertToDouble();
      if (std::isfinite(value) && value == std::trunc(value))
        exponent = static_cast<int64_t>(value);
    }
  }
  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  if (exponent && *exponent >= 0 && *exponent <= 16) {
    if (*exponent == 0) {
      Value one = arith::ConstantOp::create(
          builder, loc,
          builder.getFloatAttr(resultType->getElementType(), 1.0));
      op->getResult(0).replaceAllUsesWith(
          tensor::SplatOp::create(builder, loc, *resultType, one));
      op->erase();
      return success();
    }
    Value current = input;
    for (int64_t i = 1; i < *exponent; ++i) {
      FailureOr<Value> product =
          hip::emitElementwise(builder, loc, context, current, input,
                               *resultType, hip::ElementwiseKind::Mul);
      if (failed(product))
        return op->emitError("aten.pow.Tensor_Scalar product failed");
      current = *product;
    }
    op->getResult(0).replaceAllUsesWith(current);
    op->erase();
    return success();
  }
  if (!op->getOperand(1).getDefiningOp<arith::ConstantOp>())
    return op->emitError("aten.pow.Tensor_Scalar exponent must be constant");
  FailureOr<Value> logged =
      unaryHip<hip::LogOp>(builder, loc, context, input, *resultType);
  if (failed(logged))
    return failure();
  FailureOr<Value> scale = scalarTensor(builder, loc, op->getOperand(1),
                                        resultType->getElementType());
  if (failed(scale))
    return op->emitError("aten.pow.Tensor_Scalar cannot materialize exponent");
  FailureOr<Value> product =
      hip::emitElementwise(builder, loc, context, *logged, *scale, *resultType,
                           hip::ElementwiseKind::Mul);
  if (failed(product))
    return op->emitError("aten.pow.Tensor_Scalar scale failed");
  FailureOr<Value> raised =
      unaryHip<hip::ExpOp>(builder, loc, context, *product, *resultType);
  if (failed(raised))
    return failure();
  op->getResult(0).replaceAllUsesWith(*raised);
  op->erase();
  return success();
}

// Same exp/log form as aten.pow.Scalar, so a negative base is undefined.
// Before: pow.Tensor_Tensor %base : tensor<f32>, %e : tensor<16xsi32>
// After:  hip.exp(hip.mul(hip.cast(%e) : tensor<16xf32>, hip.log(%base)))
LogicalResult lowerPowTensorTensor(Operation *op, OpBuilder &builder,
                                   Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  Type element = resultType->getElementType();
  if (!isa<FloatType>(element))
    return op->emitError("aten.pow.Tensor_Tensor requires a float result");
  auto baseType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto exponentType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!baseType || !exponentType || !baseType.hasStaticShape() ||
      !exponentType.hasStaticShape())
    return op->emitError("aten.pow.Tensor_Tensor requires static operands");
  Location loc = op->getLoc();
  auto floatBaseType = RankedTensorType::get(baseType.getShape(), element);
  auto floatExponentType =
      RankedTensorType::get(exponentType.getShape(), element);
  FailureOr<Value> base =
      hipCast(builder, loc, context, op->getOperand(0), floatBaseType);
  FailureOr<Value> exponent =
      hipCast(builder, loc, context, op->getOperand(1), floatExponentType);
  if (failed(base) || failed(exponent))
    return op->emitError("aten.pow.Tensor_Tensor cannot cast its operands");
  FailureOr<Value> logged =
      unaryHip<hip::LogOp>(builder, loc, context, *base, floatBaseType);
  if (failed(logged))
    return failure();
  FailureOr<Value> product =
      hip::emitElementwise(builder, loc, context, *exponent, *logged,
                           *resultType, hip::ElementwiseKind::Mul);
  if (failed(product))
    return op->emitError("aten.pow.Tensor_Tensor product failed");
  FailureOr<Value> raised =
      unaryHip<hip::ExpOp>(builder, loc, context, *product, *resultType);
  if (failed(raised))
    return failure();
  op->getResult(0).replaceAllUsesWith(*raised);
  op->erase();
  return success();
}

LogicalResult lowerIntBool(Operation *op, OpBuilder &builder,
                           Value /*context*/) {
  Value extended = arith::ExtUIOp::create(
      builder, op->getLoc(), builder.getI64Type(), op->getOperand(0));
  op->getResult(0).replaceAllUsesWith(extended);
  op->erase();
  return success();
}

LogicalResult lowerGeScalar(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return op->emitError("aten.ge.Scalar requires a ranked input");
  FailureOr<Value> other = scalarTensor(
      builder, op->getLoc(), op->getOperand(1), inputType.getElementType());
  if (failed(other))
    return op->emitError("aten.ge.Scalar cannot materialize its scalar");
  Location loc = op->getLoc();
  FailureOr<Value> lessInit = emptyStatic(builder, loc, *resultType);
  FailureOr<Value> notInit = emptyStatic(builder, loc, *resultType);
  if (failed(lessInit) || failed(notInit))
    return failure();
  auto less = hip::LessOp::create(builder, loc, context, op->getOperand(0),
                                  *other, *lessInit);
  auto inverted =
      hip::NotOp::create(builder, loc, context, less->getResult(0), *notInit);
  op->getResult(0).replaceAllUsesWith(inverted->getResult(0));
  op->erase();
  return success();
}

// A bool mask over the leading dims keeps the rows where it is true. The
// row count is data-dependent: hip.nonzero counts it on device and
// hip.readback_dim sizes the destination, as for ONNX Compress.
// Before: index %x[%mask] : tensor<1x16x1024>, tensor<1x16xi1>
//           -> tensor<?x1024>
// After:  hip.compress axis 0 of tensor<16x1024> by tensor<16xi1>
LogicalResult lowerMaskIndex(Operation *op, OpBuilder &builder, Value context,
                             Value mask) {
  auto dataType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto maskType = cast<RankedTensorType>(mask.getType());
  FailureOr<RankedTensorType> resultType = dynamicResultTensor(op, 0);
  if (failed(resultType))
    return failure();
  int64_t maskRank = maskType.getRank();
  if (!dataType || !dataType.hasStaticShape() || !maskType.hasStaticShape() ||
      maskRank == 0 || maskRank > dataType.getRank() ||
      dataType.getShape().take_front(maskRank) != maskType.getShape())
    return op->emitError(
        "aten.index.Tensor bool mask must match the static leading dims");
  SmallVector<int64_t> suffix(dataType.getShape().drop_front(maskRank));
  SmallVector<int64_t> expected = {ShapedType::kDynamic};
  expected.append(suffix);
  if (resultType->getShape() != ArrayRef<int64_t>(expected) ||
      resultType->getElementType() != dataType.getElementType())
    return op->emitError("aten.index.Tensor bool mask result does not match");
  Location loc = op->getLoc();
  int64_t rows = maskType.getNumElements();
  SmallVector<int64_t> flatShape = {rows};
  flatShape.append(suffix);
  auto flatType = RankedTensorType::get(flatShape, dataType.getElementType());
  auto flatMaskType = RankedTensorType::get({rows}, maskType.getElementType());
  Value flatData =
      tensor::ReshapeOp::create(builder, loc, flatType, op->getOperand(0),
                                i64Vector(builder, loc, flatShape))
          .getResult();
  Value flatMask = tensor::ReshapeOp::create(builder, loc, flatMaskType, mask,
                                             i64Vector(builder, loc, {rows}))
                       .getResult();
  auto indicesType = RankedTensorType::get({1, rows}, builder.getI64Type());
  auto countType = RankedTensorType::get({}, builder.getI32Type());
  Value indicesInit = tensor::EmptyOp::create(
      builder, loc, indicesType.getShape(), indicesType.getElementType());
  Value countInit = tensor::EmptyOp::create(builder, loc, countType.getShape(),
                                            countType.getElementType());
  auto scan = hip::NonZeroOp::create(
      builder, loc, TypeRange{indicesType, countType}, context, flatMask,
      indicesInit, countInit,
      builder.getI64IntegerAttr(HIPDNN_EP_DATATYPE_INT8));
  Value count = hip::ReadbackDimOp::create(builder, loc, builder.getIndexType(),
                                           context, scan.getResult(1))
                    .getResult();
  Value init =
      tensor::EmptyOp::create(builder, loc, resultType->getShape(),
                              resultType->getElementType(), ValueRange{count});
  Value selected = hip::CompressOp::create(
                       builder, loc, context, flatData, flatMask, init,
                       builder.getI64IntegerAttr(0), builder.getBoolAttr(false))
                       ->getResult(0);
  op->getResult(0).replaceAllUsesWith(selected);
  op->erase();
  return success();
}

// Before: index_select %table, 0, %ids : tensor<1026x1024>, tensor<16xsi64>
// After:  hip.gather axis = 0 -> tensor<16x1024>
LogicalResult lowerIndexSelect(Operation *op, OpBuilder &builder,
                               Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto dataType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto indexType = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  if (!dataType || !indexType || !dataType.hasStaticShape() ||
      !indexType.hasStaticShape() || indexType.getRank() > 1)
    return op->emitError(
        "aten.index_select requires a static 0-D or 1-D index");
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(1));
  if (failed(dim))
    return op->emitError("aten.index_select dim must be constant");
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, dataType.getRank())))
    return op->emitError("aten.index_select dim is out of range");
  Location loc = op->getLoc();
  // A 0-D index still keeps the selected axis with size 1.
  auto gatherIndexType =
      RankedTensorType::get({indexType.getNumElements()}, builder.getI64Type());
  Value index = op->getOperand(2);
  if (indexType.getRank() == 0) {
    auto sameElement = RankedTensorType::get({1}, indexType.getElementType());
    index = tensor::ReshapeOp::create(builder, loc, sameElement, index,
                                      i64Vector(builder, loc, {1}))
                .getResult();
  }
  FailureOr<Value> casted =
      hipCast(builder, loc, context, index, gatherIndexType);
  if (failed(casted))
    return op->emitError("aten.index_select cannot cast its index");
  SmallVector<int64_t> expected(dataType.getShape());
  expected[axis] = gatherIndexType.getDimSize(0);
  if (expected != resultType->getShape())
    return op->emitError("aten.index_select result shape does not match");
  FailureOr<Value> init = emptyStatic(builder, loc, *resultType);
  if (failed(init))
    return failure();
  Value gathered =
      hip::GatherOp::create(builder, loc, context, op->getOperand(0), *casted,
                            *init, builder.getI64IntegerAttr(axis))
          ->getResult(0);
  op->getResult(0).replaceAllUsesWith(gathered);
  op->erase();
  return success();
}

// Integer remainder follows the divisor's sign, as hip.mod with fmod = 0.
// A float remainder would need floor semantics that fmod does not give.
// Before: remainder.Scalar %x, 16384 : tensor<1x16xsi64>
// After:  hip.mod %x, dense<16384> : tensor<i64> {fmod = 0}
LogicalResult lowerRemainderScalar(Operation *op, OpBuilder &builder,
                                   Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType || !isa<IntegerType>(resultType->getElementType()) ||
      inputType.getElementType() != resultType->getElementType())
    return op->emitError("aten.remainder.Scalar requires an integer tensor");
  Location loc = op->getLoc();
  FailureOr<Value> divisor = scalarTensor(builder, loc, op->getOperand(1),
                                          resultType->getElementType());
  FailureOr<Value> init = emptyStatic(builder, loc, *resultType);
  if (failed(divisor) || failed(init))
    return op->emitError(
        "aten.remainder.Scalar cannot materialize its divisor");
  Value remainder =
      hip::ModOp::create(builder, loc, context, op->getOperand(0), *divisor,
                         *init, builder.getI64IntegerAttr(0))
          ->getResult(0);
  op->getResult(0).replaceAllUsesWith(remainder);
  op->erase();
  return success();
}

LogicalResult lowerIndex(Operation *op, OpBuilder &builder, Value context) {
  Operation *list = op->getOperand(1).getDefiningOp();
  if (!list || list->getName().getStringRef() != "torch.prim.ListConstruct")
    return op->emitError("aten.index.Tensor expects a prim.ListConstruct");
  // `none` is a full slice. Any other Torch value is an index that is not
  // ready yet.
  if (llvm::any_of(list->getOperands(), [](Value value) {
        return !isa<torch::TorchNoneType>(value.getType()) &&
               torch::isTorchType(value.getType());
      }))
    return success();
  if (list->getNumOperands() == 1) {
    auto maskType = dyn_cast<RankedTensorType>(list->getOperand(0).getType());
    if (maskType && maskType.getElementType().isInteger(1))
      return lowerMaskIndex(op, builder, context, list->getOperand(0));
  }
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto dataType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!dataType || !dataType.hasStaticShape() || list->getNumOperands() == 0)
    return op->emitError("aten.index.Tensor requires static index tensors");

  SmallVector<Value> tensorIndices;
  SmallVector<int64_t> tensorAxes;
  for (auto [axis, index] : llvm::enumerate(list->getOperands())) {
    if (isa<torch::TorchNoneType>(index.getType()))
      continue;
    tensorAxes.push_back(static_cast<int64_t>(axis));
    tensorIndices.push_back(index);
  }
  if (tensorIndices.empty())
    return op->emitError("aten.index.Tensor has no index tensor");
  // One tensor index, with `none` meaning a full slice of that axis, is
  // ONNX Gather: the index shape replaces `axis`.
  // Before: index [none, none, ids<27>] on tensor<1x64x27>
  // After:  hip.gather axis = 2
  if (tensorIndices.size() == 1) {
    int64_t axis = tensorAxes.front();
    if (axis >= dataType.getRank())
      return op->emitError("aten.index.Tensor axis is out of range");
    Location loc = op->getLoc();
    Value index = tensorIndices.front();
    auto indexType = dyn_cast<RankedTensorType>(index.getType());
    if (!indexType || !indexType.hasStaticShape())
      return op->emitError("aten.index.Tensor index must be static");
    if (indexType.getElementType() != builder.getI64Type()) {
      auto castType =
          RankedTensorType::get(indexType.getShape(), builder.getI64Type());
      FailureOr<Value> casted = hipCast(builder, loc, context, index, castType);
      if (failed(casted))
        return op->emitError("aten.index.Tensor cannot cast an index");
      index = *casted;
      indexType = castType;
    }
    SmallVector<int64_t> expected(dataType.getShape().begin(),
                                  dataType.getShape().begin() + axis);
    expected.append(indexType.getShape().begin(), indexType.getShape().end());
    expected.append(dataType.getShape().begin() + axis + 1,
                    dataType.getShape().end());
    if (expected != resultType->getShape())
      return op->emitError("aten.index.Tensor result shape does not match");
    FailureOr<Value> init = emptyStatic(builder, loc, *resultType);
    if (failed(init))
      return failure();
    Value gathered =
        hip::GatherOp::create(builder, loc, context, op->getOperand(0), index,
                              *init, builder.getI64IntegerAttr(axis))
            ->getResult(0);
    op->getResult(0).replaceAllUsesWith(gathered);
    op->erase();
    return success();
  }
  // A contiguous block of tensor indices, with none as a full slice
  // around that block, is GatherND. Leading full slices are batch_dims.
  // Before: index [none, none, h<3x1>, w<3>] on tensor<1x2x4x4>
  // After:  hip.gather_nd {batch_dims = 2}
  int64_t firstAxis = tensorAxes.front();
  int64_t lastAxis = tensorAxes.back();
  if (lastAxis - firstAxis + 1 != static_cast<int64_t>(tensorAxes.size()) ||
      lastAxis >= dataType.getRank())
    return op->emitError(
        "aten.index.Tensor mixed None and multiple indices are not supported");

  int64_t rank = 0;
  SmallVector<RankedTensorType> indexTypes;
  for (Value index : tensorIndices) {
    auto indexType = dyn_cast<RankedTensorType>(index.getType());
    if (!indexType || !indexType.hasStaticShape())
      return op->emitError("aten.index.Tensor index must be static");
    rank = std::max(rank, indexType.getRank());
    indexTypes.push_back(indexType);
  }
  SmallVector<int64_t> broadcast(rank, 1);
  for (RankedTensorType indexType : indexTypes) {
    int64_t offset = rank - indexType.getRank();
    for (int64_t index : llvm::seq<int64_t>(0, indexType.getRank())) {
      int64_t dim = indexType.getDimSize(index);
      int64_t &slot = broadcast[offset + index];
      if (dim == 1)
        continue;
      if (slot != 1 && slot != dim)
        return op->emitError("aten.index.Tensor indices do not broadcast");
      slot = dim;
    }
  }

  Location loc = op->getLoc();
  SmallVector<Value> stacked;
  auto broadcastType = RankedTensorType::get(broadcast, builder.getI64Type());
  for (Value index : tensorIndices) {
    Value current = index;
    auto indexType = cast<RankedTensorType>(index.getType());
    if (indexType.getElementType() != builder.getI64Type()) {
      auto castType =
          RankedTensorType::get(indexType.getShape(), builder.getI64Type());
      FailureOr<Value> casted =
          hipCast(builder, loc, context, current, castType);
      if (failed(casted))
        return op->emitError("aten.index.Tensor cannot cast an index");
      current = *casted;
      indexType = castType;
    }
    if (indexType.getShape() != ArrayRef<int64_t>(broadcast)) {
      FailureOr<Value> init = emptyStatic(builder, loc, broadcastType);
      if (failed(init))
        return failure();
      current = hip::ExpandOp::create(builder, loc, context, current,
                                      i64Vector(builder, loc, broadcast), *init)
                    ->getResult(0);
    }
    SmallVector<int64_t> unsqueezed(broadcast);
    unsqueezed.push_back(1);
    auto unsqueezedType =
        RankedTensorType::get(unsqueezed, builder.getI64Type());
    SmallVector<ReassociationIndices> reassociation;
    for (int64_t indexDim : llvm::seq<int64_t>(0, rank - 1))
      reassociation.push_back({indexDim});
    reassociation.push_back({rank - 1, rank});
    SmallVector<OpFoldResult> outputShape;
    for (int64_t dim : unsqueezed)
      outputShape.push_back(builder.getIndexAttr(dim));
    current = tensor::ExpandShapeOp::create(builder, loc, unsqueezedType,
                                            current, reassociation, outputShape)
                  .getResult();
    stacked.push_back(current);
  }
  Value indices =
      tensor::ConcatOp::create(builder, loc, rank, stacked).getResult();
  SmallVector<int64_t> expected(dataType.getShape().begin(),
                                dataType.getShape().begin() + firstAxis);
  expected.append(broadcast.begin(), broadcast.end());
  expected.append(dataType.getShape().begin() + lastAxis + 1,
                  dataType.getShape().end());
  if (expected != resultType->getShape())
    return op->emitError("aten.index.Tensor result shape does not match");
  if (firstAxis > 0) {
    SmallVector<int64_t> unit(firstAxis, 1);
    unit.append(broadcast.begin(), broadcast.end());
    unit.push_back(static_cast<int64_t>(tensorIndices.size()));
    auto unitType = RankedTensorType::get(unit, builder.getI64Type());
    indices = tensor::ReshapeOp::create(builder, loc, unitType, indices,
                                        i64Vector(builder, loc, unit))
                  .getResult();
    SmallVector<int64_t> expanded(dataType.getShape().begin(),
                                  dataType.getShape().begin() + firstAxis);
    expanded.append(broadcast.begin(), broadcast.end());
    expanded.push_back(static_cast<int64_t>(tensorIndices.size()));
    if (expanded != unit) {
      auto expandedType = RankedTensorType::get(expanded, builder.getI64Type());
      FailureOr<Value> expandedInit = emptyStatic(builder, loc, expandedType);
      if (failed(expandedInit))
        return failure();
      indices = hip::ExpandOp::create(builder, loc, context, indices,
                                      i64Vector(builder, loc, expanded),
                                      *expandedInit)
                    ->getResult(0);
    }
  }
  FailureOr<Value> init = emptyStatic(builder, loc, *resultType);
  if (failed(init))
    return failure();
  auto gathered =
      hip::GatherNDOp::create(builder, loc, context, op->getOperand(0), indices,
                              *init, builder.getI64IntegerAttr(firstAxis));
  op->getResult(0).replaceAllUsesWith(gathered->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerMaxPoolIndices(Operation *op, OpBuilder &builder,
                                  Value context) {
  if (!op->getResult(1).use_empty())
    return op->emitError("aten.max_pool2d_with_indices index result is used");
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<SmallVector<int64_t>> kernel = readI64List(op->getOperand(1));
  FailureOr<SmallVector<int64_t>> strides = readI64List(op->getOperand(2));
  FailureOr<SmallVector<int64_t>> padding = readI64List(op->getOperand(3));
  FailureOr<SmallVector<int64_t>> dilation = readI64List(op->getOperand(4));
  FailureOr<bool> ceilMode = readI1Scalar(op->getOperand(5));
  if (failed(kernel) || failed(strides) || failed(padding) ||
      failed(dilation) || failed(ceilMode)) {
    return op->emitError(
        "aten.max_pool2d_with_indices window must be constant");
  }
  int64_t spatial = static_cast<int64_t>(kernel->size());
  SmallVector<int64_t> pads;
  if (spatial == 0 || static_cast<int64_t>(strides->size()) != spatial ||
      static_cast<int64_t>(dilation->size()) != spatial ||
      static_cast<int64_t>(padding->size()) != spatial) {
    return op->emitError(
        "aten.max_pool2d_with_indices window ranks do not match");
  }
  pads.append(padding->begin(), padding->end());
  pads.append(padding->begin(), padding->end());
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return failure();
  SmallVector<Value> operands = {context, op->getOperand(0), *init};
  SmallVector<NamedAttribute> attrs = {
      builder.getNamedAttr("pool_mode", builder.getI64IntegerAttr(1)),
      builder.getNamedAttr("kernel_shape", builder.getI64ArrayAttr(*kernel)),
      builder.getNamedAttr("strides", builder.getI64ArrayAttr(*strides)),
      builder.getNamedAttr("pads", builder.getI64ArrayAttr(pads)),
      builder.getNamedAttr("dilations", builder.getI64ArrayAttr(*dilation)),
      builder.getNamedAttr("ceil_mode",
                           builder.getI64IntegerAttr(*ceilMode ? 1 : 0)),
      builder.getNamedAttr("storage_order", builder.getI64IntegerAttr(0)),
      builder.getNamedAttr("count_include_pad", builder.getI64IntegerAttr(0)),
      builder.getNamedAttr("p", builder.getI64IntegerAttr(0))};
  auto pool = hip::PoolOp::create(builder, op->getLoc(), TypeRange{*resultType},
                                  operands, attrs);
  op->getResult(0).replaceAllUsesWith(pool.getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerSelect(Operation *op, OpBuilder &builder,
                          Value /*context*/) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return op->emitError("aten.select.int requires a ranked input");
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(1));
  FailureOr<int64_t> index = readI64Scalar(op->getOperand(2));
  if (failed(dim) || failed(index))
    return op->emitError("aten.select.int index must be constant");
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, inputType.getRank())))
    return op->emitError("aten.select.int dim is out of range");
  for (int64_t other : llvm::seq<int64_t>(0, inputType.getRank()))
    if (other != axis && inputType.isDynamicDim(other))
      return op->emitError("aten.select.int requires static unselected dims");
  Location loc = op->getLoc();
  OpFoldResult offset;
  if (inputType.isDynamicDim(axis)) {
    // A negative index counts back from the runtime extent.
    // Before: select %x, 1, -1 : tensor<1x?x8xf32>
    // After:  tensor.extract_slice %x[0, dim(%x, 1) - 1, 0] [1, 1, 8]
    if (*index >= 0) {
      offset = builder.getIndexAttr(*index);
    } else {
      Value extent =
          tensor::DimOp::create(builder, loc, op->getOperand(0), axis);
      Value back = arith::ConstantIndexOp::create(builder, loc, -*index);
      offset = arith::SubIOp::create(builder, loc, extent, back).getResult();
    }
  } else {
    int64_t size = inputType.getDimSize(axis);
    int64_t chosen = *index < 0 ? *index + size : *index;
    if (chosen < 0 || chosen >= size)
      return op->emitError("aten.select.int index is out of range");
    offset = builder.getIndexAttr(chosen);
  }
  SmallVector<int64_t> midShape(inputType.getShape());
  midShape[axis] = 1;
  auto midType = RankedTensorType::get(midShape, inputType.getElementType());
  SmallVector<OpFoldResult> offsets(inputType.getRank(),
                                    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(inputType.getRank(),
                                    builder.getIndexAttr(1));
  for (int64_t dimSize : midShape)
    sizes.push_back(builder.getIndexAttr(dimSize));
  offsets[axis] = offset;
  Value sliced =
      tensor::ExtractSliceOp::create(builder, loc, midType, op->getOperand(0),
                                     offsets, sizes, strides)
          .getResult();
  Value shape = i64Vector(builder, loc, resultType->getShape());
  auto reshaped =
      tensor::ReshapeOp::create(builder, loc, *resultType, sliced, shape);
  op->getResult(0).replaceAllUsesWith(reshaped.getResult());
  op->erase();
  return success();
}

LogicalResult lowerNumToTensor(Operation *op, OpBuilder &builder,
                               Value /*context*/) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> value = scalarTensor(
      builder, op->getLoc(), op->getOperand(0), resultType->getElementType());
  if (failed(value))
    return op->emitError("prim.NumToTensor cannot materialize its scalar");
  op->getResult(0).replaceAllUsesWith(*value);
  op->erase();
  return success();
}

LogicalResult lowerDivMode(Operation *op, OpBuilder &builder, Value context) {
  auto mode = op->getOperand(2).getDefiningOp<torch::ConstantStrOp>();
  if (!mode || mode.getValue() != "floor")
    return op->emitError("aten.div.Tensor_mode only lowers floor");
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  Value rhs = op->getOperand(1);
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!rhsType)
    return op->emitError("aten.div.Tensor_mode requires a ranked divisor");
  if (rhsType.getElementType() != resultType->getElementType()) {
    auto castType =
        RankedTensorType::get(rhsType.getShape(), resultType->getElementType());
    FailureOr<Value> casted =
        hipCast(builder, op->getLoc(), context, rhs, castType);
    if (failed(casted))
      return op->emitError("aten.div.Tensor_mode cannot cast its divisor");
    rhs = *casted;
  }
  FailureOr<Value> quotient =
      hip::emitElementwise(builder, op->getLoc(), context, op->getOperand(0),
                           rhs, *resultType, hip::ElementwiseKind::Div);
  if (failed(quotient))
    return op->emitError("aten.div.Tensor_mode division failed");
  FailureOr<Value> floored = unaryHip<hip::FloorOp>(
      builder, op->getLoc(), context, *quotient, *resultType);
  if (failed(floored))
    return failure();
  op->getResult(0).replaceAllUsesWith(*floored);
  op->erase();
  return success();
}

LogicalResult lowerPow(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  Location loc = op->getLoc();
  FailureOr<Value> base = scalarTensor(builder, loc, op->getOperand(0),
                                       resultType->getElementType());
  if (failed(base))
    return op->emitError("aten.pow.Scalar cannot materialize its base");
  auto baseType = RankedTensorType::get({}, resultType->getElementType());
  FailureOr<Value> logged =
      unaryHip<hip::LogOp>(builder, loc, context, *base, baseType);
  if (failed(logged))
    return failure();
  FailureOr<Value> product =
      hip::emitElementwise(builder, loc, context, op->getOperand(1), *logged,
                           *resultType, hip::ElementwiseKind::Mul);
  if (failed(product))
    return op->emitError("aten.pow.Scalar product failed");
  FailureOr<Value> raised =
      unaryHip<hip::ExpOp>(builder, loc, context, *product, *resultType);
  if (failed(raised))
    return failure();
  op->getResult(0).replaceAllUsesWith(*raised);
  op->erase();
  return success();
}

LogicalResult lowerRepeat(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  FailureOr<SmallVector<int64_t>> repeats = readI64List(op->getOperand(1));
  if (!inputType || failed(repeats) ||
      static_cast<int64_t>(repeats->size()) < inputType.getRank()) {
    return op->emitError("aten.repeat list must cover the input rank");
  }
  // Extra repeat factors prepend size-1 dimensions to the input.
  FailureOr<Value> input =
      hip::rightAlignTensor(builder, op->getLoc(), op->getOperand(0),
                            static_cast<int64_t>(repeats->size()));
  if (failed(input))
    return op->emitError("aten.repeat cannot align the input rank");
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return failure();
  auto tiled = hip::TileOp::create(builder, op->getLoc(), context, *input,
                                   op->getOperand(1), *init);
  op->getResult(0).replaceAllUsesWith(tiled->getResult(0));
  op->erase();
  return success();
}

// aten.flip reverses each listed axis independently.
// Before: flip(x, [0]) on tensor<3x2xf32>
// After:  hip.gather(x, [2, 1, 0] {axis = 0})
LogicalResult lowerFlip(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType || !inputType.hasStaticShape() || inputType != *resultType)
    return op->emitError("aten.flip input must match a static result type");
  FailureOr<SmallVector<int64_t>> dims = readI64List(op->getOperand(1));
  if (failed(dims))
    return op->emitError("aten.flip dims must be a constant list");
  Location loc = op->getLoc();
  Value current = op->getOperand(0);
  SmallVector<bool> seen(inputType.getRank(), false);
  for (int64_t axis : *dims) {
    if (failed(normalizeAxis(axis, inputType.getRank())) || seen[axis])
      return op->emitError("aten.flip dim is out of range or repeated");
    seen[axis] = true;
    int64_t extent = inputType.getDimSize(axis);
    SmallVector<int64_t> reversed(extent);
    for (int64_t index : llvm::seq<int64_t>(extent))
      reversed[index] = extent - 1 - index;
    auto indexType = RankedTensorType::get({extent}, builder.getI64Type());
    Value indices = arith::ConstantOp::create(
        builder, loc, DenseIntElementsAttr::get(indexType, reversed));
    FailureOr<Value> init = emptyStatic(builder, loc, *resultType);
    if (failed(init))
      return failure();
    current = hip::GatherOp::create(builder, loc, context, current, indices,
                                    *init, builder.getI64IntegerAttr(axis))
                  ->getResult(0);
  }
  op->getResult(0).replaceAllUsesWith(current);
  op->erase();
  return success();
}

// aten.clamp(x, min, max). A `none` bound is omitted.
// Before: clamp(x, 0.0, none) on f32
// After:  hip.max(x, splat(0.0))
// Before: clamp(x, none, 26) on i64
// After:  hip.min(x, splat(26))
LogicalResult lowerClamp(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType ||
      inputType.getElementType() != resultType->getElementType()) {
    return op->emitError("aten.clamp input element type must match the result");
  }
  Location loc = op->getLoc();
  Value current = op->getOperand(0);
  auto applyBound = [&](Value bound, bool lower) -> LogicalResult {
    if (isa<torch::TorchNoneType>(bound.getType()))
      return success();
    FailureOr<Value> splat =
        scalarTensor(builder, loc, bound, resultType->getElementType());
    if (failed(splat))
      return op->emitError("aten.clamp bound must be a scalar");
    FailureOr<Value> next = hip::emitElementwise(
        builder, loc, context, current, *splat, *resultType,
        lower ? hip::ElementwiseKind::Max : hip::ElementwiseKind::Min);
    if (failed(next))
      return op->emitError("aten.clamp failed");
    current = *next;
    return success();
  };
  if (failed(applyBound(op->getOperand(1), true)) ||
      failed(applyBound(op->getOperand(2), false)))
    return failure();
  op->getResult(0).replaceAllUsesWith(current);
  op->erase();
  return success();
}

// aten.constant_pad_nd counts (begin, end) pairs from the last axis.
// hip.pad stores every axis as begins followed by ends.
// Before: pad [1,64,64,768] with [0, 0, 0, 6, 0, 6]
// After:  hip.pad pads dense<[0, 0, 0, 0, 0, 6, 6, 0]>
LogicalResult lowerConstantPad(Operation *op, OpBuilder &builder,
                               Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  FailureOr<SmallVector<int64_t>> padding = readI64List(op->getOperand(1));
  if (!inputType || !inputType.hasStaticShape() || failed(padding) ||
      padding->size() % 2 != 0 ||
      static_cast<int64_t>(padding->size()) > inputType.getRank() * 2) {
    return op->emitError(
        "aten.constant_pad_nd padding must be a constant even-length list");
  }
  int64_t rank = inputType.getRank();
  SmallVector<int64_t> onnxPads(2 * rank, 0);
  int64_t pairs = static_cast<int64_t>(padding->size()) / 2;
  for (int64_t pair : llvm::seq<int64_t>(0, pairs)) {
    int64_t axis = rank - 1 - pair;
    int64_t begin = (*padding)[2 * pair];
    int64_t end = (*padding)[2 * pair + 1];
    if (begin < 0 || end < 0)
      return op->emitError(
          "aten.constant_pad_nd negative padding is not supported");
    onnxPads[axis] = begin;
    onnxPads[rank + axis] = end;
  }
  for (int64_t axis : llvm::seq<int64_t>(0, rank)) {
    if (inputType.getDimSize(axis) + onnxPads[axis] + onnxPads[rank + axis] !=
        resultType->getDimSize(axis)) {
      return op->emitError(
          "aten.constant_pad_nd result shape does not match the padding");
    }
  }
  Location loc = op->getLoc();
  FailureOr<Value> fill = scalarTensor(builder, loc, op->getOperand(2),
                                       resultType->getElementType());
  FailureOr<Value> init = emptyStatic(builder, loc, *resultType);
  if (failed(fill) || failed(init))
    return op->emitError("aten.constant_pad_nd requires a scalar fill value");
  auto padded = hip::PadOp::create(builder, loc, context, op->getOperand(0),
                                   i64Vector(builder, loc, onnxPads), *fill,
                                   /*axes=*/Value(), *init, "constant");
  op->getResult(0).replaceAllUsesWith(padded->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerSqueeze(Operation *op, OpBuilder &builder,
                           Value /*context*/) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  FailureOr<SmallVector<int64_t>> dims = readI64List(op->getOperand(1));
  if (!inputType || !inputType.hasStaticShape() || failed(dims) ||
      inputType.getElementType() != resultType->getElementType()) {
    return op->emitError(
        "prims.squeeze requires a static input and constant dims");
  }
  for (int64_t dim : *dims) {
    if (failed(normalizeAxis(dim, inputType.getRank())) ||
        inputType.getDimSize(dim) != 1) {
      return op->emitError("prims.squeeze dim must be a size-1 axis");
    }
  }
  int64_t inputCount = 1;
  int64_t resultCount = 1;
  for (int64_t dim : inputType.getShape())
    inputCount *= dim;
  for (int64_t dim : resultType->getShape())
    resultCount *= dim;
  if (inputCount != resultCount)
    return op->emitError("prims.squeeze result shape does not match the input");
  Value shape = i64Vector(builder, op->getLoc(), resultType->getShape());
  auto reshaped = tensor::ReshapeOp::create(builder, op->getLoc(), *resultType,
                                            op->getOperand(0), shape);
  op->getResult(0).replaceAllUsesWith(reshaped.getResult());
  op->erase();
  return success();
}

bool isOrdTwo(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
    return integer.getValue() == 2;
  if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
    return floating.getValue().isExactlyValue(2.0);
  return false;
}

// A dtype code is accepted when it names the result element type. Torch's
// float32 code is 6, float16 is 5, and bfloat16 is 15.
bool vectorNormDtypeMatches(Value dtype, Type element) {
  if (isa<torch::TorchNoneType>(dtype.getType()))
    return true;
  auto constant = dtype.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto attr = dyn_cast<IntegerAttr>(constant.getValue());
  if (!attr)
    return false;
  int64_t code = attr.getInt();
  if (element.isF32())
    return code == 6;
  if (element.isF16())
    return code == 5;
  if (element.isBF16())
    return code == 15;
  return false;
}

// ord 2 is sqrt(sum(x * x)). A missing dim list reduces every axis.
// Before: linalg_vector_norm %x, 2, [-1], keepdim = true, dtype = float32
// After:  hip.mul, hip.reduce_sum {keepdims = 1}, hip.sqrt
LogicalResult lowerVectorNorm(Operation *op, OpBuilder &builder,
                              Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  if (!isOrdTwo(op->getOperand(1)))
    return op->emitError("aten.linalg_vector_norm only lowers ord 2");
  if (!vectorNormDtypeMatches(op->getOperand(4), resultType->getElementType()))
    return op->emitError("aten.linalg_vector_norm dtype must be none or the "
                         "result element type");
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  FailureOr<bool> keepdim = readI1Scalar(op->getOperand(3));
  if (!inputType || !inputType.hasStaticShape() || failed(keepdim))
    return op->emitError("aten.linalg_vector_norm requires a static input");
  SmallVector<int64_t> axes;
  if (isa<torch::TorchNoneType>(op->getOperand(2).getType())) {
    for (int64_t axis : llvm::seq<int64_t>(0, inputType.getRank()))
      axes.push_back(axis);
  } else {
    FailureOr<SmallVector<int64_t>> dims = readI64List(op->getOperand(2));
    if (failed(dims))
      return op->emitError(
          "aten.linalg_vector_norm dim must be a constant list");
    for (int64_t axis : *dims) {
      if (failed(normalizeAxis(axis, inputType.getRank())))
        return op->emitError("aten.linalg_vector_norm dim is out of range");
      axes.push_back(axis);
    }
  }
  Location loc = op->getLoc();
  FailureOr<Value> squared = hip::emitElementwise(
      builder, loc, context, op->getOperand(0), op->getOperand(0), inputType,
      hip::ElementwiseKind::Mul);
  if (failed(squared))
    return op->emitError("aten.linalg_vector_norm square failed");
  auto reducedType =
      RankedTensorType::get(resultType->getShape(), inputType.getElementType());
  FailureOr<Value> sumInit = emptyStatic(builder, loc, reducedType);
  FailureOr<Value> sqrtInit = emptyStatic(builder, loc, *resultType);
  if (failed(sumInit) || failed(sqrtInit))
    return failure();
  Value summed =
      hip::ReduceSumOp::create(builder, loc, context, *squared,
                               i64Vector(builder, loc, axes), *sumInit,
                               builder.getI64IntegerAttr(*keepdim ? 1 : 0),
                               builder.getI64IntegerAttr(0))
          ->getResult(0);
  if (reducedType.getElementType() != resultType->getElementType()) {
    FailureOr<Value> casted =
        hipCast(builder, loc, context, summed, *resultType);
    if (failed(casted))
      return op->emitError("aten.linalg_vector_norm cannot cast its sum");
    summed = *casted;
  }
  Value rooted = hip::SqrtOp::create(builder, loc, context, summed, *sqrtInit)
                     ->getResult(0);
  op->getResult(0).replaceAllUsesWith(rooted);
  op->erase();
  return success();
}

// aten.minimum is an elementwise min. Both operands keep the result type.
// Before: minimum(%a, %b) on tensor<16x16xsi64>
// After:  hip.min(%a, %b)
LogicalResult lowerMinimum(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto selfType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto otherType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!selfType || !otherType ||
      selfType.getElementType() != resultType->getElementType() ||
      otherType.getElementType() != resultType->getElementType()) {
    return op->emitError(
        "aten.minimum operands must match the result element type");
  }
  FailureOr<Value> result = hip::emitElementwise(
      builder, op->getLoc(), context, op->getOperand(0), op->getOperand(1),
      *resultType, hip::ElementwiseKind::Min);
  if (failed(result))
    return op->emitError("aten.minimum failed");
  op->getResult(0).replaceAllUsesWith(*result);
  op->erase();
  return success();
}

LogicalResult lowerGtTensor(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return failure();
  // A > B  <=>  B < A. hip.less broadcasts the two inputs.
  Value less = hip::LessOp::create(builder, op->getLoc(), context,
                                   op->getOperand(1), op->getOperand(0), *init)
                   ->getResult(0);
  op->getResult(0).replaceAllUsesWith(less);
  op->erase();
  return success();
}

FailureOr<std::pair<Value, Value>>
emitTopK(OpBuilder &builder, Location loc, Value context, Value input,
         int64_t k, int64_t axis, bool largest, bool sorted,
         RankedTensorType valuesType, RankedTensorType indicesType) {
  FailureOr<Value> valuesInit = emptyStatic(builder, loc, valuesType);
  FailureOr<Value> indicesInit = emptyStatic(builder, loc, indicesType);
  if (failed(valuesInit) || failed(indicesInit))
    return failure();
  Value kScalar =
      arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(k));
  auto kType = RankedTensorType::get({}, builder.getI64Type());
  Value kTensor = tensor::SplatOp::create(builder, loc, kType, kScalar);
  auto top = hip::TopKOp::create(
      builder, loc, TypeRange{valuesType, indicesType}, context, input, kTensor,
      *valuesInit, *indicesInit, builder.getI64IntegerAttr(axis),
      builder.getBoolAttr(largest), builder.getBoolAttr(sorted));
  return std::make_pair(top.getResult(0), top.getResult(1));
}

LogicalResult lowerTopk(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> valuesType = resultTensor(op, 0);
  FailureOr<RankedTensorType> indicesType = resultTensor(op, 1);
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  FailureOr<int64_t> k = readI64Scalar(op->getOperand(1));
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(2));
  FailureOr<bool> largest = readI1Scalar(op->getOperand(3));
  FailureOr<bool> sorted = readI1Scalar(op->getOperand(4));
  if (failed(valuesType) || failed(indicesType) || !inputType || failed(k) ||
      failed(dim) || failed(largest) || failed(sorted) || *k <= 0) {
    return op->emitError("aten.topk requires static tensors and constant k");
  }
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, inputType.getRank())))
    return op->emitError("aten.topk axis is out of range");
  FailureOr<std::pair<Value, Value>> top =
      emitTopK(builder, op->getLoc(), context, op->getOperand(0), *k, axis,
               *largest, *sorted, *valuesType, *indicesType);
  if (failed(top))
    return op->emitError("aten.topk failed");
  op->getResult(0).replaceAllUsesWith(top->first);
  op->getResult(1).replaceAllUsesWith(top->second);
  op->erase();
  return success();
}

LogicalResult lowerMaxDim(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> valuesType = resultTensor(op, 0);
  FailureOr<RankedTensorType> indicesType = resultTensor(op, 1);
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(1));
  FailureOr<bool> keepdim = readI1Scalar(op->getOperand(2));
  if (failed(valuesType) || failed(indicesType) || !inputType || failed(dim) ||
      failed(keepdim)) {
    return op->emitError("aten.max.dim requires static tensors");
  }
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, inputType.getRank())))
    return op->emitError("aten.max.dim axis is out of range");
  SmallVector<int64_t> keptShape(inputType.getShape());
  keptShape[axis] = 1;
  auto keptValues =
      RankedTensorType::get(keptShape, valuesType->getElementType());
  auto keptIndices =
      RankedTensorType::get(keptShape, indicesType->getElementType());
  FailureOr<std::pair<Value, Value>> top =
      emitTopK(builder, op->getLoc(), context, op->getOperand(0), /*k=*/1, axis,
               /*largest=*/true, /*sorted=*/true, keptValues, keptIndices);
  if (failed(top))
    return op->emitError("aten.max.dim failed");
  Value values = top->first;
  Value indices = top->second;
  if (!*keepdim) {
    Value shape = i64Vector(builder, op->getLoc(), valuesType->getShape());
    values = tensor::ReshapeOp::create(builder, op->getLoc(), *valuesType,
                                       values, shape)
                 .getResult();
    Value indexShape =
        i64Vector(builder, op->getLoc(), indicesType->getShape());
    indices = tensor::ReshapeOp::create(builder, op->getLoc(), *indicesType,
                                        indices, indexShape)
                  .getResult();
  }
  op->getResult(0).replaceAllUsesWith(values);
  op->getResult(1).replaceAllUsesWith(indices);
  op->erase();
  return success();
}

// aten.argmax keeps only the index from a top-1 reduction.
// Before: argmax(%x : tensor<1x16xsi32>, dim = -1, keepdim = false)
// After:  hip.top_k {axis = 1, largest = true}, then drop the size-1 axis
LogicalResult lowerArgmax(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> indicesType = resultTensor(op, 0);
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (isa<torch::TorchNoneType>(op->getOperand(1).getType()))
    return op->emitError("aten.argmax dim must be a constant");
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(1));
  FailureOr<bool> keepdim = readI1Scalar(op->getOperand(2));
  if (failed(indicesType) || !inputType || !inputType.hasStaticShape() ||
      failed(dim) || failed(keepdim)) {
    return op->emitError(
        "aten.argmax requires a static tensor and a constant dim");
  }
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, inputType.getRank())))
    return op->emitError("aten.argmax axis is out of range");
  SmallVector<int64_t> keptShape(inputType.getShape());
  keptShape[axis] = 1;
  auto keptValues =
      RankedTensorType::get(keptShape, inputType.getElementType());
  auto keptIndices =
      RankedTensorType::get(keptShape, indicesType->getElementType());
  FailureOr<std::pair<Value, Value>> top =
      emitTopK(builder, op->getLoc(), context, op->getOperand(0), /*k=*/1, axis,
               /*largest=*/true, /*sorted=*/true, keptValues, keptIndices);
  if (failed(top))
    return op->emitError("aten.argmax failed");
  Value indices = top->second;
  if (!*keepdim) {
    Value indexShape =
        i64Vector(builder, op->getLoc(), indicesType->getShape());
    indices = tensor::ReshapeOp::create(builder, op->getLoc(), *indicesType,
                                        indices, indexShape)
                  .getResult();
  }
  op->getResult(0).replaceAllUsesWith(indices);
  op->erase();
  return success();
}

LogicalResult lowerListUnpack(Operation *op, OpBuilder &builder,
                              Value /*context*/) {
  Operation *split = op->getOperand(0).getDefiningOp();
  if (!split ||
      split->getName().getStringRef() != "torch.aten.split_with_sizes") {
    return op->emitError("prim.ListUnpack only lowers a split_with_sizes list");
  }
  auto inputType = dyn_cast<RankedTensorType>(split->getOperand(0).getType());
  FailureOr<SmallVector<int64_t>> sizes = readI64List(split->getOperand(1));
  FailureOr<int64_t> dim = readI64Scalar(split->getOperand(2));
  if (!inputType || !inputType.hasStaticShape() || failed(sizes) ||
      failed(dim) || sizes->size() != op->getNumResults()) {
    return op->emitError("aten.split_with_sizes requires constant sizes");
  }
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, inputType.getRank())))
    return op->emitError("aten.split_with_sizes axis is out of range");
  int64_t cursor = 0;
  for (auto [index, section] : llvm::enumerate(*sizes)) {
    if (section < 0)
      return op->emitError(
          "aten.split_with_sizes section must be non-negative");
    FailureOr<RankedTensorType> resultType = resultTensor(op, index);
    if (failed(resultType) || resultType->getDimSize(axis) != section)
      return op->emitError("aten.split_with_sizes result size does not match");
    SmallVector<OpFoldResult> offsets(inputType.getRank(),
                                      builder.getIndexAttr(0));
    SmallVector<OpFoldResult> sliceSizes;
    SmallVector<OpFoldResult> strides(inputType.getRank(),
                                      builder.getIndexAttr(1));
    for (int64_t rank = 0; rank < inputType.getRank(); ++rank)
      sliceSizes.push_back(builder.getIndexAttr(inputType.getDimSize(rank)));
    offsets[axis] = builder.getIndexAttr(cursor);
    sliceSizes[axis] = builder.getIndexAttr(section);
    auto sliced = tensor::ExtractSliceOp::create(
        builder, op->getLoc(), *resultType, split->getOperand(0), offsets,
        sliceSizes, strides);
    op->getResult(index).replaceAllUsesWith(sliced.getResult());
    cursor += section;
  }
  if (cursor != inputType.getDimSize(axis))
    return op->emitError(
        "aten.split_with_sizes sections do not cover the axis");
  op->erase();
  split->erase();
  return success();
}

// [B, H, S, D] -> [B, S, H*D]. hip.gqa consumes the packed layout.
FailureOr<Value> packAttentionHeads(OpBuilder &builder, Location loc,
                                    Value context, Value input) {
  auto type = dyn_cast<RankedTensorType>(input.getType());
  if (!type || type.getRank() != 4 || !type.hasStaticShape())
    return failure();
  int64_t batch = type.getDimSize(0);
  int64_t heads = type.getDimSize(1);
  int64_t seq = type.getDimSize(2);
  int64_t dim = type.getDimSize(3);
  auto transposedType =
      RankedTensorType::get({batch, seq, heads, dim}, type.getElementType());
  FailureOr<Value> transposed =
      transposeTo(builder, loc, context, input, {0, 2, 1, 3}, transposedType);
  if (failed(transposed))
    return failure();
  auto packedType =
      RankedTensorType::get({batch, seq, heads * dim}, type.getElementType());
  SmallVector<ReassociationIndices> reassociation = {{0}, {1}, {2, 3}};
  return Value(tensor::CollapseShapeOp::create(builder, loc, packedType,
                                               *transposed, reassociation));
}

// [B, S, H*D] -> [B, H, S, D].
FailureOr<Value> unpackAttentionHeads(OpBuilder &builder, Location loc,
                                      Value context, Value packed,
                                      int64_t heads) {
  auto type = dyn_cast<RankedTensorType>(packed.getType());
  if (!type || type.getRank() != 3 || !type.hasStaticShape() || heads < 1 ||
      type.getDimSize(2) % heads != 0)
    return failure();
  int64_t batch = type.getDimSize(0);
  int64_t seq = type.getDimSize(1);
  int64_t dim = type.getDimSize(2) / heads;
  auto expandedType =
      RankedTensorType::get({batch, seq, heads, dim}, type.getElementType());
  SmallVector<ReassociationIndices> reassociation = {{0}, {1}, {2, 3}};
  SmallVector<OpFoldResult> outputShape = {
      builder.getIndexAttr(batch), builder.getIndexAttr(seq),
      builder.getIndexAttr(heads), builder.getIndexAttr(dim)};
  Value expanded =
      tensor::ExpandShapeOp::create(builder, loc, expandedType, packed,
                                    reassociation, outputShape)
          .getResult();
  auto resultType =
      RankedTensorType::get({batch, heads, seq, dim}, type.getElementType());
  return transposeTo(builder, loc, context, expanded, {0, 2, 1, 3}, resultType);
}

// Scaled dot-product attention, including its flash-attention kernel form and
// grouped-query attention (fewer key/value heads than query heads).
// Before: sdpa(q[1,4,2,8], k[1,2,2,8], v[1,2,2,8], enable_gqa = true)
// After:  hip.gqa {num_heads = 4, kv_num_heads = 2}
LogicalResult emitFusedAttention(Operation *op, OpBuilder &builder,
                                 Value context, Value query, Value key,
                                 Value value, Value attnMask, Value dropout,
                                 Value isCausal, Value scale, Value enableGqa,
                                 unsigned outputIndex) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, outputIndex);
  if (failed(resultType))
    return failure();
  auto queryType = dyn_cast<RankedTensorType>(query.getType());
  auto keyType = dyn_cast<RankedTensorType>(key.getType());
  auto valueType = dyn_cast<RankedTensorType>(value.getType());
  if (!queryType || !keyType || !valueType || !queryType.hasStaticShape() ||
      !keyType.hasStaticShape() || !valueType.hasStaticShape() ||
      queryType.getRank() != 4 || keyType.getRank() != 4 ||
      valueType.getRank() != 4 ||
      queryType.getElementType() != keyType.getElementType() ||
      queryType.getElementType() != valueType.getElementType()) {
    return op->emitError(
        "attention requires static rank-4 query, key, and value");
  }
  int64_t batch = queryType.getDimSize(0);
  int64_t queryHeads = queryType.getDimSize(1);
  int64_t querySeq = queryType.getDimSize(2);
  int64_t headDim = queryType.getDimSize(3);
  int64_t kvHeads = keyType.getDimSize(1);
  int64_t kvSeq = keyType.getDimSize(2);
  if (queryHeads < 1 || kvHeads < 1 || queryHeads % kvHeads != 0 ||
      keyType.getDimSize(0) != batch || valueType.getDimSize(0) != batch ||
      valueType.getDimSize(1) != kvHeads || valueType.getDimSize(2) != kvSeq ||
      keyType.getDimSize(3) != headDim || valueType.getDimSize(3) != headDim ||
      resultType->getShape() != queryType.getShape()) {
    return op->emitError("attention head layout does not match the result");
  }
  auto dropoutConstant = dropout.getDefiningOp<arith::ConstantOp>();
  auto dropoutAttr = dropoutConstant
                         ? dyn_cast<FloatAttr>(dropoutConstant.getValue())
                         : FloatAttr();
  if (!dropoutAttr || !dropoutAttr.getValue().isExactlyValue(0.0))
    return op->emitError("attention dropout must be 0");
  FailureOr<bool> causal = readI1Scalar(isCausal);
  FailureOr<bool> grouped = readI1Scalar(enableGqa);
  if (failed(causal) || failed(grouped))
    return op->emitError("attention is_causal and enable_gqa must be constant");
  if (*grouped == false && queryHeads != kvHeads)
    return op->emitError("grouped-query attention requires enable_gqa");
  bool hasMask = !isa<torch::TorchNoneType>(attnMask.getType());
  if (hasMask && *causal)
    return op->emitError("attention cannot combine is_causal with a mask");
  // A bool mask keeps True positions. False positions become -inf, matching
  // the scaled-dot-product rule, so hip.gqa can add the bias to the scores.
  if (hasMask) {
    auto maskType = dyn_cast<RankedTensorType>(attnMask.getType());
    if (!maskType)
      return op->emitError("attention mask must be a ranked tensor or none");
    Type element = queryType.getElementType();
    if (maskType.getElementType().isInteger(1)) {
      auto biasType = RankedTensorType::get(maskType.getShape(), element);
      Value keepScalar = arith::ConstantOp::create(
          builder, op->getLoc(), builder.getFloatAttr(element, 0.0));
      Value blockScalar = arith::ConstantOp::create(
          builder, op->getLoc(),
          builder.getFloatAttr(element,
                               -std::numeric_limits<double>::infinity()));
      Value keep =
          tensor::SplatOp::create(builder, op->getLoc(), biasType, keepScalar);
      Value block =
          tensor::SplatOp::create(builder, op->getLoc(), biasType, blockScalar);
      FailureOr<Value> biasInit = emptyStatic(builder, op->getLoc(), biasType);
      if (failed(biasInit))
        return failure();
      attnMask = hip::WhereOp::create(builder, op->getLoc(), context, attnMask,
                                      keep, block, *biasInit)
                     ->getResult(0);
    } else if (maskType.getElementType() != element) {
      return op->emitError(
          "attention mask element type must match the query or be i1");
    }
  }
  float scaleValue = 0.f;
  if (!isa<torch::TorchNoneType>(scale.getType())) {
    auto scaleConstant = scale.getDefiningOp<arith::ConstantOp>();
    auto scaleAttr = scaleConstant
                         ? dyn_cast<FloatAttr>(scaleConstant.getValue())
                         : FloatAttr();
    if (!scaleAttr)
      return op->emitError("attention scale must be a constant or none");
    scaleValue = static_cast<float>(scaleAttr.getValue().convertToDouble());
  }

  Location loc = op->getLoc();
  FailureOr<Value> packedQuery =
      packAttentionHeads(builder, loc, context, query);
  FailureOr<Value> packedKey = packAttentionHeads(builder, loc, context, key);
  FailureOr<Value> packedValue =
      packAttentionHeads(builder, loc, context, value);
  if (failed(packedQuery) || failed(packedKey) || failed(packedValue))
    return op->emitError("attention cannot pack heads");

  auto seqLenType = RankedTensorType::get({batch}, builder.getI32Type());
  Value seqLens = arith::ConstantOp::create(
      builder, loc,
      DenseElementsAttr::get(seqLenType, builder.getI32IntegerAttr(kvSeq)));
  auto totalType = RankedTensorType::get({}, builder.getI32Type());
  Value totalSeq = arith::ConstantOp::create(
      builder, loc,
      DenseElementsAttr::get(totalType, builder.getI32IntegerAttr(kvSeq)));

  auto outputType = RankedTensorType::get(
      {batch, querySeq, queryHeads * headDim}, queryType.getElementType());
  FailureOr<Value> outputInit = emptyStatic(builder, loc, outputType);
  FailureOr<Value> presentKey = emptyStatic(builder, loc, keyType);
  FailureOr<Value> presentValue = emptyStatic(builder, loc, valueType);
  if (failed(outputInit) || failed(presentKey) || failed(presentValue))
    return failure();

  SmallVector<Value> operands = {context,      *packedQuery, *packedKey,
                                 *packedValue, seqLens,      totalSeq};
  if (hasMask)
    operands.push_back(attnMask);
  operands.push_back(*outputInit);
  operands.push_back(*presentKey);
  operands.push_back(*presentValue);

  SmallVector<NamedAttribute> attrs = {
      builder.getNamedAttr("num_heads", builder.getI64IntegerAttr(queryHeads)),
      builder.getNamedAttr("kv_num_heads", builder.getI64IntegerAttr(kvHeads)),
      builder.getNamedAttr("scale", builder.getF32FloatAttr(scaleValue)),
      builder.getNamedAttr("do_rotary", builder.getI64IntegerAttr(0)),
      builder.getNamedAttr("rotary_interleaved", builder.getI64IntegerAttr(0)),
      builder.getNamedAttr("softcap", builder.getF32FloatAttr(0.f)),
      builder.getNamedAttr("local_window_size", builder.getI64IntegerAttr(-1)),
      builder.getNamedAttr("smooth_softmax", builder.getI64IntegerAttr(0)),
      builder.getNamedAttr("qk_output", builder.getI64IntegerAttr(0)),
      builder.getNamedAttr("k_quant_type", builder.getStringAttr("NONE")),
      builder.getNamedAttr("v_quant_type", builder.getStringAttr("NONE")),
      builder.getNamedAttr("kv_cache_bit_width", builder.getI64IntegerAttr(8)),
      builder.getNamedAttr("no_causal", builder.getBoolAttr(!*causal))};
  SmallVector<int32_t> segments = {
      1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, hasMask ? 1 : 0, 0, 0, 0, 1, 1, 1, 0};

  OperationState state(loc, "hip.gqa");
  state.addOperands(operands);
  state.addAttributes(attrs);
  state.addTypes({outputType, keyType, valueType});
  state.addAttribute("operand_segment_sizes",
                     builder.getDenseI32ArrayAttr(segments));
  Value packed = builder.create(state)->getResult(0);
  FailureOr<Value> unpacked =
      unpackAttentionHeads(builder, loc, context, packed, queryHeads);
  if (failed(unpacked))
    return op->emitError("attention cannot unpack heads");
  op->getResult(outputIndex).replaceAllUsesWith(*unpacked);
  return success();
}

LogicalResult lowerScaledDotProduct(Operation *op, OpBuilder &builder,
                                    Value context) {
  if (op->getNumOperands() != 8)
    return op->emitError(
        "aten.scaled_dot_product_attention expects 8 operands");
  if (failed(emitFusedAttention(
          op, builder, context, op->getOperand(0), op->getOperand(1),
          op->getOperand(2), op->getOperand(3), op->getOperand(4),
          op->getOperand(5), op->getOperand(6), op->getOperand(7), 0)))
    return failure();
  op->erase();
  return success();
}

// torch.hipdnn.qmoe operands:
//   activation [B, S, H], router probabilities [B*S, E],
//   fc1 packed weights [E, 2I, H/2], fc1 scales [E, 2I, H/block], fc1 bias,
//   fc2 packed weights [E, H, I/2], fc2 scales [E, H, I/block], fc2 bias,
//   then the scalars k, activation alpha, activation beta, and swiglu limit.
// The last dimension of each scale tensor is the reduction length divided by
// the quantization block. Two packed bytes cover four bits each.
//
// Before:
//   %y = torch.operator "torch.hipdnn.qmoe"(
//            %x, %probs, %fc1w, %fc1s, %fc1b, %fc2w, %fc2s, %fc2b,
//            %k, %alpha, %beta, %limit) : ... -> !torch.vtensor<[B,S,H],f32>
// After:
//   %y = hip.qmoe(%ctx) ins(%x, %probs, %fc1w, %fc1s, %fc2w, %fc2s : ...)
//          fc1_bias(%fc1b : ...) fc2_bias(%fc2b : ...)
//          outs(%empty : tensor<B x S x H x f32>)
//          {expert_weight_bits = 4, block_size = H / scale_groups,
//           swiglu_fusion = 1, ...}
LogicalResult lowerQuantizedMoe(Operation *op, OpBuilder &builder,
                                Value context) {
  if (op->getNumOperands() != 12 || op->getNumResults() != 1)
    return op->emitError("qmoe expects 12 operands and 1 result");

  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto routerType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto fc1WeightType = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  auto fc1ScaleType = dyn_cast<RankedTensorType>(op->getOperand(3).getType());
  auto fc1BiasType = dyn_cast<RankedTensorType>(op->getOperand(4).getType());
  auto fc2WeightType = dyn_cast<RankedTensorType>(op->getOperand(5).getType());
  auto fc2ScaleType = dyn_cast<RankedTensorType>(op->getOperand(6).getType());
  auto fc2BiasType = dyn_cast<RankedTensorType>(op->getOperand(7).getType());
  if (!inputType || !routerType || !fc1WeightType || !fc1ScaleType ||
      !fc1BiasType || !fc2WeightType || !fc2ScaleType || !fc2BiasType ||
      !inputType.hasStaticShape() || !routerType.hasStaticShape() ||
      !fc1WeightType.hasStaticShape() || !fc1ScaleType.hasStaticShape() ||
      !fc1BiasType.hasStaticShape() || !fc2WeightType.hasStaticShape() ||
      !fc2ScaleType.hasStaticShape() || !fc2BiasType.hasStaticShape())
    return op->emitError("qmoe tensors must have static shapes");
  if (inputType.getRank() != 3 || routerType.getRank() != 2 ||
      fc1WeightType.getRank() != 3 || fc1ScaleType.getRank() != 3 ||
      fc1BiasType.getRank() != 2 || fc2WeightType.getRank() != 3 ||
      fc2ScaleType.getRank() != 3 || fc2BiasType.getRank() != 2)
    return op->emitError("qmoe tensor ranks do not match the packed layout");

  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  if (*resultType != inputType)
    return op->emitError("qmoe result shape must match the activation");

  int64_t batch = inputType.getDimSize(0);
  int64_t seq = inputType.getDimSize(1);
  int64_t hidden = inputType.getDimSize(2);
  int64_t tokens = batch * seq;
  int64_t experts = fc1WeightType.getDimSize(0);
  int64_t fusedInter = fc1WeightType.getDimSize(1);
  if (hidden <= 0 || experts <= 0 || fusedInter <= 0 || fusedInter % 2 != 0)
    return op->emitError("qmoe activation and fc1 weight shapes are invalid");
  int64_t inter = fusedInter / 2;
  if (routerType.getDimSize(0) != tokens || routerType.getDimSize(1) != experts)
    return op->emitError("qmoe router probabilities must be [B*S, experts]");
  if (routerType.getElementType() != inputType.getElementType() ||
      fc1BiasType.getElementType() != inputType.getElementType() ||
      fc2BiasType.getElementType() != inputType.getElementType())
    return op->emitError(
        "qmoe router and bias element types must match the activation");
  if (!isa<FloatType>(fc1ScaleType.getElementType()) ||
      !isa<FloatType>(fc2ScaleType.getElementType()))
    return op->emitError("qmoe scales must be floating point");

  auto packedBytes = [](RankedTensorType type) -> bool {
    auto integer = dyn_cast<IntegerType>(type.getElementType());
    return integer && integer.getWidth() == 8;
  };
  if (!packedBytes(fc1WeightType) || !packedBytes(fc2WeightType))
    return op->emitError("qmoe expert weights must be 8-bit packed values");
  if (fc1WeightType.getDimSize(2) * 2 != hidden ||
      fc2WeightType.getDimSize(0) != experts ||
      fc2WeightType.getDimSize(1) != hidden ||
      fc2WeightType.getDimSize(2) * 2 != inter)
    return op->emitError("qmoe packed weight shapes must be 4-bit");
  if (fc1ScaleType.getDimSize(0) != experts ||
      fc1ScaleType.getDimSize(1) != fusedInter ||
      fc2ScaleType.getDimSize(0) != experts ||
      fc2ScaleType.getDimSize(1) != hidden ||
      fc1BiasType.getDimSize(0) != experts ||
      fc1BiasType.getDimSize(1) != fusedInter ||
      fc2BiasType.getDimSize(0) != experts ||
      fc2BiasType.getDimSize(1) != hidden)
    return op->emitError(
        "qmoe scale and bias shapes must match the expert weights");

  int64_t fc1Groups = fc1ScaleType.getDimSize(2);
  int64_t fc2Groups = fc2ScaleType.getDimSize(2);
  if (fc1Groups <= 0 || fc2Groups <= 0 || hidden % fc1Groups != 0 ||
      inter % fc2Groups != 0)
    return op->emitError("qmoe scale groups must divide the reduction length");
  int64_t blockSize = hidden / fc1Groups;
  if (blockSize != inter / fc2Groups || blockSize <= 0 || (blockSize % 2) != 0)
    return op->emitError("qmoe block size must be a positive even value");

  FailureOr<int64_t> k = readI64Scalar(op->getOperand(8));
  FailureOr<float> alpha = readF64Scalar(op->getOperand(9));
  FailureOr<float> beta = readF64Scalar(op->getOperand(10));
  FailureOr<float> limit = readF64Scalar(op->getOperand(11));
  if (failed(k) || failed(alpha) || failed(beta) || failed(limit))
    return op->emitError("qmoe k, alpha, beta, and limit must be constants");
  if (*k <= 0 || *k > experts)
    return op->emitError("qmoe k must be in range of the expert count");

  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return failure();
  Value none;
  auto hipOp = hip::QMoEOp::create(
      builder, op->getLoc(), context, op->getOperand(0), op->getOperand(1),
      op->getOperand(2), op->getOperand(3), op->getOperand(5),
      op->getOperand(6), op->getOperand(4), op->getOperand(7), none, none, none,
      none, none, none, none, *init, builder.getI64IntegerAttr(4),
      builder.getI64IntegerAttr(*k), builder.getI64IntegerAttr(blockSize),
      builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(1),
      builder.getI64IntegerAttr(0), builder.getF32FloatAttr(*alpha),
      builder.getF32FloatAttr(*beta), builder.getF32FloatAttr(*limit),
      builder.getStringAttr("swiglu"));
  op->getResult(0).replaceAllUsesWith(hipOp->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerFlashAttention(Operation *op, OpBuilder &builder,
                                  Value context) {
  auto name = op->getAttrOfType<StringAttr>("name");
  if (!name || !name.getValue().contains("scaled_dot_product_flash_attention"))
    return op->emitError("unsupported torch.operator");
  if (op->getNumOperands() != 7 || op->getNumResults() != 2)
    return op->emitError("flash attention expects 7 operands and 2 results");
  if (!op->getResult(1).use_empty())
    return op->emitError("flash attention logsumexp is not lowered");
  // The kernel has no enable_gqa flag; head counts come from the tensors.
  Value grouped = arith::ConstantOp::create(builder, op->getLoc(),
                                            builder.getBoolAttr(true));
  if (failed(emitFusedAttention(
          op, builder, context, op->getOperand(0), op->getOperand(1),
          op->getOperand(2), op->getOperand(5), op->getOperand(3),
          op->getOperand(4), op->getOperand(6), grouped, 0)))
    return failure();
  op->erase();
  return success();
}

LogicalResult lowerTorchOperator(Operation *op, OpBuilder &builder,
                                 Value context) {
  auto name = op->getAttrOfType<StringAttr>("name");
  if (name && name.getValue() == "torch.hipdnn.qmoe")
    return lowerQuantizedMoe(op, builder, context);
  return lowerFlashAttention(op, builder, context);
}

} // namespace

void mlir::torch::populateTorchModelRewrites(
    SmallVectorImpl<TorchRewrite> &rewrites) {
  rewrites.push_back({"torch.aten.expand", lowerExpand});
  rewrites.push_back({"torch.aten.bmm", lowerBmm});
  rewrites.push_back({"torch.aten.var_mean.correction", lowerVarMean});
  rewrites.push_back({"torch.aten.where.self", lowerWhere});
  rewrites.push_back({"torch.aten.scalar_tensor", lowerScalarTensor});
  rewrites.push_back({"torch.aten.full_like", lowerFullLike});
  rewrites.push_back({"torch.aten.clone", lowerClone});
  rewrites.push_back({"torch.aten.copy", lowerCopy});
  rewrites.push_back({"torch.aten._softmax", lowerSoftmax});
  rewrites.push_back(
      {"torch.aten.scaled_dot_product_attention", lowerScaledDotProduct});
  rewrites.push_back({"torch.operator", lowerTorchOperator});
  rewrites.push_back({"torch.aten.eq.Scalar", lowerEqScalar});
  rewrites.push_back({"torch.aten.all.dim", lowerAllDim});
  rewrites.push_back({"torch.aten._assert_tensor_metadata", lowerAssert});
  rewrites.push_back({"torch.aten.arange.start_step", lowerArange});
  rewrites.push_back({"torch.aten.slice.Tensor", lowerSlice});
  rewrites.push_back({"torch.aten.slice_scatter", lowerSliceScatter});
  rewrites.push_back({"torch.aten.upsample_nearest2d.vec", lowerUpsample});
  rewrites.push_back({"torch.aten.bitwise_and.Tensor", lowerBitwiseAnd});
  rewrites.push_back({"torch.aten.empty_strided", lowerEmpty});
  rewrites.push_back({"torch.prims.convert_element_type", lowerConvert});
  rewrites.push_back({"torch.aten.cat", lowerCat});
  rewrites.push_back({"torch.aten.cumsum", lowerCumsum});
  rewrites.push_back({"torch.aten.sin", lowerSin});
  rewrites.push_back({"torch.aten.exp", lowerExp});
  rewrites.push_back({"torch.aten.cos", lowerCos});
  rewrites.push_back({"torch.aten.Int.bool", lowerIntBool});
  rewrites.push_back({"torch.aten.ge.Scalar", lowerGeScalar});
  rewrites.push_back({"torch.aten.index.Tensor", lowerIndex});
  rewrites.push_back(
      {"torch.aten.max_pool2d_with_indices", lowerMaxPoolIndices});
  rewrites.push_back({"torch.aten.select.int", lowerSelect});
  rewrites.push_back({"torch.prim.NumToTensor.Scalar", lowerNumToTensor});
  rewrites.push_back({"torch.aten.div.Tensor_mode", lowerDivMode});
  rewrites.push_back({"torch.aten.pow.Scalar", lowerPow});
  rewrites.push_back({"torch.aten.repeat", lowerRepeat});
  rewrites.push_back({"torch.aten.flip", lowerFlip});
  rewrites.push_back({"torch.aten.sigmoid", lowerSigmoid});
  rewrites.push_back({"torch.aten.tanh", lowerTanh});
  rewrites.push_back({"torch.aten.gelu", lowerGelu});
  rewrites.push_back({"torch.aten.embedding", lowerEmbedding});
  rewrites.push_back({"torch.aten.alias", lowerAlias});
  rewrites.push_back({"torch.aten.eq.Tensor", lowerEqTensor});
  rewrites.push_back({"torch.aten.le.Tensor", lowerLeTensor});
  rewrites.push_back({"torch.aten.mean.dim", lowerMeanDim});
  rewrites.push_back({"torch.aten.ne.Scalar", lowerNeScalar});
  rewrites.push_back({"torch.aten.neg", lowerNeg});
  rewrites.push_back({"torch.aten.pow.Tensor_Scalar", lowerPowTensorScalar});
  rewrites.push_back({"torch.aten.pow.Tensor_Tensor", lowerPowTensorTensor});
  rewrites.push_back({"torch.aten.log1p", lowerLog1p});
  rewrites.push_back({"torch.aten.remainder.Scalar", lowerRemainderScalar});
  rewrites.push_back({"torch.aten._assert_scalar", lowerAssert});
  rewrites.push_back({"torch.aten.size.int", lowerSizeInt});
  rewrites.push_back(
      {"torch.aten.ge.int", lowerIntCompare<arith::CmpIPredicate::sge>});
  rewrites.push_back(
      {"torch.aten.le.int", lowerIntCompare<arith::CmpIPredicate::sle>});
  rewrites.push_back(
      {"torch.aten.eq.int", lowerIntCompare<arith::CmpIPredicate::eq>});
  rewrites.push_back({"torch.aten.unique_consecutive", lowerUniqueConsecutive});
  rewrites.push_back({"torch.aten.index_select", lowerIndexSelect});
  rewrites.push_back({"torch.aten.clamp", lowerClamp});
  rewrites.push_back({"torch.aten.constant_pad_nd", lowerConstantPad});
  rewrites.push_back({"torch.prims.squeeze", lowerSqueeze});
  rewrites.push_back({"torch.aten.linalg_vector_norm", lowerVectorNorm});
  rewrites.push_back({"torch.aten.abs", lowerAbs});
  rewrites.push_back({"torch.aten.sign", lowerSign});
  rewrites.push_back({"torch.aten.ceil", lowerCeil});
  rewrites.push_back({"torch.aten.floor", lowerFloor});
  rewrites.push_back({"torch.aten.log", lowerLog});
  rewrites.push_back({"torch.aten.sqrt", lowerSqrt});
  rewrites.push_back({"torch.aten.bitwise_not", lowerBitwiseNot});
  rewrites.push_back({"torch.aten.gt.Scalar", lowerGtScalar});
  rewrites.push_back({"torch.aten.lt.Scalar", lowerLtScalar});
  rewrites.push_back({"torch.aten.le.Scalar", lowerLeScalar});
  rewrites.push_back({"torch.aten.gather", lowerGather});
  rewrites.push_back({"torch.aten.gt.Tensor", lowerGtTensor});
  rewrites.push_back({"torch.aten.minimum", lowerMinimum});
  rewrites.push_back({"torch.aten.max.dim", lowerMaxDim});
  rewrites.push_back({"torch.aten.argmax", lowerArgmax});
  rewrites.push_back({"torch.aten.topk", lowerTopk});
  rewrites.push_back({"torch.prim.ListUnpack", lowerListUnpack});
}
