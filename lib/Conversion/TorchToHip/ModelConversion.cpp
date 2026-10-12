/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "PopulatePatterns.h"
#include "TorchFrontend.h"
#include "TorchToHipUtils.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Torch/IR/TorchOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

#include <algorithm>

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
      FailureOr<Value> materialized = hip::materializeScalarTensor(
          builder, loc, scalar, elementType);
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
      converted =
          arith::ExtUIOp::create(builder, loc, elementType, scalar);
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
  auto expanded = hip::ExpandOp::create(builder, op->getLoc(), context,
                                        op->getOperand(0), op->getOperand(1),
                                        *init);
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
  auto matmul =
      hip::MatmulOp::create(builder, op->getLoc(), operands, attrs);
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
  FailureOr<Value> squared = hip::emitElementwise(
      builder, loc, context, *centered, *centered, inputType,
      hip::ElementwiseKind::Mul);
  if (failed(squared))
    return op->emitError("aten.var_mean cannot square the residual");
  FailureOr<Value> varInit = emptyStatic(builder, loc, *varType);
  if (failed(varInit))
    return failure();
  auto variance = hip::ReduceMeanOp::create(
      builder, loc, context, *squared, axes, *varInit,
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
    Value scaleScalar =
        arith::ConstantOp::create(builder, loc, scaleAttr);
    FailureOr<Value> scale = hip::materializeScalarTensor(
        builder, loc, scaleScalar, varType->getElementType());
    if (failed(scale))
      return failure();
    FailureOr<Value> scaled = hip::emitElementwise(
        builder, loc, context, var, *scale, *varType, hip::ElementwiseKind::Mul);
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
  auto selected = hip::WhereOp::create(
      builder, op->getLoc(), context, op->getOperand(0), op->getOperand(1),
      op->getOperand(2), *init);
  op->getResult(0).replaceAllUsesWith(selected->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerScalarTensor(Operation *op, OpBuilder &builder,
                                Value /*context*/) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType) || resultType->getRank() != 0)
    return op->emitError("aten.scalar_tensor requires a rank-0 result");
  FailureOr<Value> value = scalarTensor(builder, op->getLoc(),
                                        op->getOperand(0),
                                        resultType->getElementType());
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
    auto movedType =
        RankedTensorType::get(moved, inputType.getElementType());
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
    FailureOr<Value> restored = transposeTo(builder, loc, context, softmaxed,
                                            backward, *resultType);
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
  FailureOr<Value> other = scalarTensor(builder, op->getLoc(),
                                        op->getOperand(1),
                                        inputType.getElementType());
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

LogicalResult lowerArange(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  Location loc = op->getLoc();
  Type element = resultType->getElementType();
  FailureOr<Value> start = scalarTensor(builder, loc, op->getOperand(0), element);
  FailureOr<Value> limit = scalarTensor(builder, loc, op->getOperand(1), element);
  FailureOr<Value> delta = scalarTensor(builder, loc, op->getOperand(2), element);
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
  SmallVector<OpFoldResult> offsets(inputType.getRank(),
                                    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(inputType.getRank(),
                                    builder.getIndexAttr(1));
  for (int64_t index : llvm::seq<int64_t>(0, inputType.getRank()))
    sizes.push_back(builder.getIndexAttr(resultType->getDimSize(index)));
  offsets[axis] = builder.getIndexAttr(begin);
  strides[axis] = builder.getIndexAttr(*step);
  auto sliced = tensor::ExtractSliceOp::create(
      builder, op->getLoc(), *resultType, op->getOperand(0), offsets, sizes,
      strides);
  op->getResult(0).replaceAllUsesWith(sliced.getResult());
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
  auto combined = hip::AndOp::create(builder, op->getLoc(), context,
                                     op->getOperand(0), op->getOperand(1),
                                     *init);
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
  FailureOr<Value> casted = hipCast(builder, op->getLoc(), context,
                                    op->getOperand(0), *resultType);
  if (failed(casted))
    return op->emitError("prims.convert_element_type cannot cast");
  op->getResult(0).replaceAllUsesWith(*casted);
  op->erase();
  return success();
}

LogicalResult lowerCat(Operation *op, OpBuilder &builder, Value /*context*/) {
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
  auto inputType = dyn_cast<RankedTensorType>(list->getOperand(0).getType());
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(1));
  if (!inputType || failed(dim))
    return op->emitError("aten.cat dim must be a constant");
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, inputType.getRank())))
    return op->emitError("aten.cat dim is out of range");
  auto concat = tensor::ConcatOp::create(builder, op->getLoc(), axis,
                                         list->getOperands());
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
  FailureOr<Value> axisTensor =
      scalarTensor(builder, loc, op->getOperand(1), builder.getI64Type());
  FailureOr<Value> init = emptyStatic(builder, loc, *resultType);
  if (failed(axisTensor) || failed(init))
    return failure();
  auto sum = hip::CumSumOp::create(builder, loc, context, op->getOperand(0),
                                   *axisTensor, *init,
                                   builder.getI64IntegerAttr(0),
                                   builder.getI64IntegerAttr(0));
  op->getResult(0).replaceAllUsesWith(sum->getResult(0));
  op->erase();
  return success();
}

template <typename HipOp>
LogicalResult lowerUnaryTensor(Operation *op, OpBuilder &builder,
                               Value context, StringRef name) {
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

LogicalResult lowerCos(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryTensor<hip::CosOp>(op, builder, context, "aten.cos");
}

LogicalResult lowerSigmoid(Operation *op, OpBuilder &builder, Value context) {
  return lowerUnaryTensor<hip::SigmoidOp>(op, builder, context, "aten.sigmoid");
}

LogicalResult lowerGelu(Operation *op, OpBuilder &builder, Value context) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto approximate = op->getOperand(1).getDefiningOp<torch::ConstantStrOp>();
  if (!approximate || (approximate.getValue() != "none" &&
                       approximate.getValue() != "tanh")) {
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

LogicalResult lowerIntBool(Operation *op, OpBuilder &builder,
                           Value /*context*/) {
  Value extended = arith::ExtUIOp::create(builder, op->getLoc(),
                                          builder.getI64Type(),
                                          op->getOperand(0));
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
  FailureOr<Value> other = scalarTensor(builder, op->getLoc(),
                                        op->getOperand(1),
                                        inputType.getElementType());
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

LogicalResult lowerIndex(Operation *op, OpBuilder &builder, Value context) {
  Operation *list = op->getOperand(1).getDefiningOp();
  if (!list || list->getName().getStringRef() != "torch.prim.ListConstruct")
    return op->emitError("aten.index.Tensor expects a prim.ListConstruct");
  if (elementsStillTorch(list))
    return success();
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  auto dataType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!dataType || !dataType.hasStaticShape() || list->getNumOperands() == 0)
    return op->emitError("aten.index.Tensor requires static index tensors");

  int64_t rank = 0;
  SmallVector<RankedTensorType> indexTypes;
  for (Value index : list->getOperands()) {
    if (isa<torch::TorchNoneType>(index.getType()))
      return op->emitError("aten.index.Tensor None index is not supported");
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
  auto broadcastType =
      RankedTensorType::get(broadcast, builder.getI64Type());
  for (Value index : list->getOperands()) {
    Value current = index;
    auto indexType = cast<RankedTensorType>(index.getType());
    if (indexType.getElementType() != builder.getI64Type()) {
      auto castType =
          RankedTensorType::get(indexType.getShape(), builder.getI64Type());
      FailureOr<Value> casted = hipCast(builder, loc, context, current, castType);
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
  FailureOr<Value> init = emptyStatic(builder, loc, *resultType);
  if (failed(init))
    return failure();
  auto gathered = hip::GatherNDOp::create(builder, loc, context,
                                          op->getOperand(0), indices, *init,
                                          builder.getI64IntegerAttr(0));
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
    return op->emitError("aten.max_pool2d_with_indices window must be constant");
  }
  int64_t spatial = static_cast<int64_t>(kernel->size());
  SmallVector<int64_t> pads;
  if (spatial == 0 || static_cast<int64_t>(strides->size()) != spatial ||
      static_cast<int64_t>(dilation->size()) != spatial ||
      static_cast<int64_t>(padding->size()) != spatial) {
    return op->emitError("aten.max_pool2d_with_indices window ranks do not match");
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
  if (!inputType || !inputType.hasStaticShape())
    return op->emitError("aten.select.int requires a static input");
  FailureOr<int64_t> dim = readI64Scalar(op->getOperand(1));
  FailureOr<int64_t> index = readI64Scalar(op->getOperand(2));
  if (failed(dim) || failed(index))
    return op->emitError("aten.select.int index must be constant");
  int64_t axis = *dim;
  if (failed(normalizeAxis(axis, inputType.getRank())))
    return op->emitError("aten.select.int dim is out of range");
  int64_t size = inputType.getDimSize(axis);
  int64_t chosen = *index < 0 ? *index + size : *index;
  if (chosen < 0 || chosen >= size)
    return op->emitError("aten.select.int index is out of range");
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
  offsets[axis] = builder.getIndexAttr(chosen);
  Location loc = op->getLoc();
  Value sliced = tensor::ExtractSliceOp::create(builder, loc, midType,
                                                op->getOperand(0), offsets,
                                                sizes, strides)
                     .getResult();
  Value shape = i64Vector(builder, loc, resultType->getShape());
  auto reshaped = tensor::ReshapeOp::create(builder, loc, *resultType, sliced,
                                            shape);
  op->getResult(0).replaceAllUsesWith(reshaped.getResult());
  op->erase();
  return success();
}

LogicalResult lowerNumToTensor(Operation *op, OpBuilder &builder,
                               Value /*context*/) {
  FailureOr<RankedTensorType> resultType = resultTensor(op, 0);
  if (failed(resultType))
    return failure();
  FailureOr<Value> value = scalarTensor(builder, op->getLoc(),
                                        op->getOperand(0),
                                        resultType->getElementType());
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
    auto castType = RankedTensorType::get(rhsType.getShape(),
                                          resultType->getElementType());
    FailureOr<Value> casted =
        hipCast(builder, op->getLoc(), context, rhs, castType);
    if (failed(casted))
      return op->emitError("aten.div.Tensor_mode cannot cast its divisor");
    rhs = *casted;
  }
  FailureOr<Value> quotient = hip::emitElementwise(
      builder, op->getLoc(), context, op->getOperand(0), rhs, *resultType,
      hip::ElementwiseKind::Div);
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
  FailureOr<Value> product = hip::emitElementwise(
      builder, loc, context, op->getOperand(1), *logged, *resultType,
      hip::ElementwiseKind::Mul);
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
      static_cast<int64_t>(repeats->size()) != inputType.getRank()) {
    return op->emitError("aten.repeat list must match the input rank");
  }
  FailureOr<Value> init = emptyStatic(builder, op->getLoc(), *resultType);
  if (failed(init))
    return failure();
  auto tiled = hip::TileOp::create(builder, op->getLoc(), context,
                                   op->getOperand(0), op->getOperand(1), *init);
  op->getResult(0).replaceAllUsesWith(tiled->getResult(0));
  op->erase();
  return success();
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
  rewrites.push_back({"torch.aten._softmax", lowerSoftmax});
  rewrites.push_back({"torch.aten.eq.Scalar", lowerEqScalar});
  rewrites.push_back({"torch.aten.all.dim", lowerAllDim});
  rewrites.push_back({"torch.aten._assert_tensor_metadata", lowerAssert});
  rewrites.push_back({"torch.aten.arange.start_step", lowerArange});
  rewrites.push_back({"torch.aten.slice.Tensor", lowerSlice});
  rewrites.push_back({"torch.aten.upsample_nearest2d.vec", lowerUpsample});
  rewrites.push_back({"torch.aten.bitwise_and.Tensor", lowerBitwiseAnd});
  rewrites.push_back({"torch.aten.empty_strided", lowerEmpty});
  rewrites.push_back({"torch.prims.convert_element_type", lowerConvert});
  rewrites.push_back({"torch.aten.cat", lowerCat});
  rewrites.push_back({"torch.aten.cumsum", lowerCumsum});
  rewrites.push_back({"torch.aten.sin", lowerSin});
  rewrites.push_back({"torch.aten.cos", lowerCos});
  rewrites.push_back({"torch.aten.Int.bool", lowerIntBool});
  rewrites.push_back({"torch.aten.ge.Scalar", lowerGeScalar});
  rewrites.push_back({"torch.aten.index.Tensor", lowerIndex});
  rewrites.push_back({"torch.aten.max_pool2d_with_indices", lowerMaxPoolIndices});
  rewrites.push_back({"torch.aten.select.int", lowerSelect});
  rewrites.push_back({"torch.prim.NumToTensor.Scalar", lowerNumToTensor});
  rewrites.push_back({"torch.aten.div.Tensor_mode", lowerDivMode});
  rewrites.push_back({"torch.aten.pow.Scalar", lowerPow});
  rewrites.push_back({"torch.aten.repeat", lowerRepeat});
  rewrites.push_back({"torch.aten.sigmoid", lowerSigmoid});
  rewrites.push_back({"torch.aten.gelu", lowerGelu});
}
