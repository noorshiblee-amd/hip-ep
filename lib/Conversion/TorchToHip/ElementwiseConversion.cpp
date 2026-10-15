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
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

namespace {

FailureOr<int64_t> readI64Scalar(Value value);

LogicalResult lowerAlphaBinary(Operation *op, OpBuilder &builder, Value context,
                               bool scalarOther, hip::ElementwiseKind kind,
                               StringRef name) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return op->emitError() << name << " expects 3 operands and 1 result";

  FailureOr<RankedTensorType> resultType =
      torch::toBuiltinTensorType(op->getResult(0).getType());
  if (failed(resultType)) {
    return op->emitError() << name
                           << " result element type must be f16, f32, or bf16";
  }

  Value self = op->getOperand(0);
  Value other = op->getOperand(1);
  Value alpha = op->getOperand(2);
  Location loc = op->getLoc();
  Type elementType = resultType->getElementType();

  auto selfType = dyn_cast<RankedTensorType>(self.getType());
  if (!selfType ||
      !torch::isAcceptedFloatElementType(selfType.getElementType())) {
    return op->emitError() << name
                           << " self must be an f16, f32, or bf16 tensor";
  }

  if (scalarOther) {
    if (isa<ShapedType>(other.getType()))
      return op->emitError() << name << " other must be a scalar";
    FailureOr<Value> otherTensor =
        hip::materializeScalarTensor(builder, loc, other, elementType);
    if (failed(otherTensor))
      return op->emitError() << "cannot materialize " << name << " other";
    other = *otherTensor;
  }

  auto otherType = dyn_cast<RankedTensorType>(other.getType());
  if (!otherType)
    return op->emitError() << name << " other must be a tensor";
  // A float result with an integer tensor other is a cast, then the
  // float kernel.
  if (!torch::isAcceptedFloatElementType(otherType.getElementType())) {
    FailureOr<Value> promoted =
        hip::castToElementType(builder, loc, context, other, elementType);
    if (failed(promoted))
      return op->emitError()
             << name << " other must be an f16, f32, or bf16 tensor";
    other = *promoted;
  }

  FailureOr<Value> castSelf =
      hip::castToElementType(builder, loc, context, self, elementType);
  FailureOr<Value> castOther =
      hip::castToElementType(builder, loc, context, other, elementType);
  if (failed(castSelf) || failed(castOther))
    return op->emitError() << name
                           << " cannot cast operands to the result type";

  FailureOr<hip::PackedBinary> packed = hip::packBinaryBroadcast(
      builder, loc, *castSelf, *castOther, *resultType);
  if (failed(packed))
    return op->emitError() << "cannot pack " << name << " into 4 dimensions";

  Value rhs = packed->rhs;
  if (!hip::isUnitScalar(alpha)) {
    FailureOr<Value> alphaTensor =
        hip::materializeScalarTensor(builder, loc, alpha, elementType);
    if (failed(alphaTensor))
      return op->emitError() << "cannot materialize " << name << " alpha";
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    FailureOr<Value> scaled =
        hip::emitElementwise(builder, loc, context, rhs, *alphaTensor, rhsType,
                             hip::ElementwiseKind::Mul);
    if (failed(scaled))
      return op->emitError() << name << " alpha multiply failed";
    rhs = *scaled;
  }

  FailureOr<Value> combined = hip::emitElementwise(
      builder, loc, context, packed->lhs, rhs, packed->resultType, kind);
  if (failed(combined))
    return op->emitError() << name << " failed";

  Value result = *combined;
  if (packed->needsExpand) {
    SmallVector<OpFoldResult> outputShape;
    for (int64_t dim : resultType->getShape())
      outputShape.push_back(builder.getIndexAttr(dim));
    result = tensor::ExpandShapeOp::create(builder, loc, *resultType, result,
                                           packed->reassociation, outputShape)
                 .getResult();
  }
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return success();
}

bool supportedIntegerWidth(unsigned width) {
  return width == 16 || width == 32 || width == 64;
}

LogicalResult integerScalarFits(Operation *op, int64_t value, unsigned width) {
  if (width >= 64)
    return success();
  int64_t minValue = -(int64_t{1} << (width - 1));
  int64_t maxValue = (int64_t{1} << (width - 1)) - 1;
  if (value < minValue || value > maxValue)
    return op->emitError(
        "integer scalar does not fit in the tensor element type");
  return success();
}

LogicalResult lowerIntegerAlphaBinary(Operation *op, OpBuilder &builder,
                                      Value context, bool scalarOther,
                                      hip::ElementwiseKind kind,
                                      StringRef name) {
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto elementType = inputType
                         ? dyn_cast<IntegerType>(inputType.getElementType())
                         : IntegerType();
  if (!inputType || !inputType.hasStaticShape() || !elementType ||
      !supportedIntegerWidth(elementType.getWidth())) {
    return op->emitError() << "integer " << name
                           << " requires a static i16, i32, or i64 tensor";
  }
  FailureOr<int64_t> alpha = readI64Scalar(op->getOperand(2));
  if (failed(alpha))
    return op->emitError() << "integer " << name << " alpha must be constant";
  Location loc = op->getLoc();
  Value rhs = op->getOperand(1);
  if (scalarOther) {
    FailureOr<int64_t> other = readI64Scalar(rhs);
    if (failed(other))
      return op->emitError()
             << "integer " << name << " other must be a constant";
    // self ± alpha * other. Either factor being zero leaves self unchanged.
    if (*other == 0 || *alpha == 0) {
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
      return success();
    }
    int64_t delta = *alpha * *other;
    if (failed(integerScalarFits(op, delta, elementType.getWidth())))
      return failure();
    Value deltaValue = arith::ConstantOp::create(
        builder, loc, IntegerAttr::get(elementType, delta));
    auto scalarType = RankedTensorType::get({}, elementType);
    rhs = tensor::SplatOp::create(builder, loc, scalarType, deltaValue);
  } else {
    auto otherType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!otherType || otherType.getElementType() != elementType)
      return op->emitError() << "integer " << name
                             << " other must match the input element type";
    if (*alpha != 1) {
      if (failed(integerScalarFits(op, *alpha, elementType.getWidth())))
        return failure();
      Value alphaValue = arith::ConstantOp::create(
          builder, loc, IntegerAttr::get(elementType, *alpha));
      auto scalarType = RankedTensorType::get({}, elementType);
      Value alphaTensor =
          tensor::SplatOp::create(builder, loc, scalarType, alphaValue);
      FailureOr<Value> scaled =
          hip::emitElementwise(builder, loc, context, rhs, alphaTensor,
                               otherType, hip::ElementwiseKind::Mul);
      if (failed(scaled))
        return op->emitError()
               << "integer " << name << " alpha multiply failed";
      rhs = *scaled;
    }
  }
  FailureOr<Value> result = hip::emitElementwise(
      builder, loc, context, op->getOperand(0), rhs, inputType, kind);
  if (failed(result))
    return op->emitError() << "integer " << name << " failed";
  op->getResult(0).replaceAllUsesWith(*result);
  op->erase();
  return success();
}

bool isIntegerTensorResult(Operation *op) {
  auto valueTensor =
      dyn_cast<torch::ValueTensorType>(op->getResult(0).getType());
  return valueTensor && isa<IntegerType>(valueTensor.getDtype());
}

LogicalResult lowerAtenAdd(Operation *op, OpBuilder &builder, Value context) {
  bool scalarOther = op->getName().getStringRef() == "torch.aten.add.Scalar";
  if (isIntegerTensorResult(op))
    return lowerIntegerAlphaBinary(op, builder, context, scalarOther,
                                   hip::ElementwiseKind::Add, "aten.add");
  return lowerAlphaBinary(op, builder, context, scalarOther,
                          hip::ElementwiseKind::Add, "aten.add");
}

LogicalResult lowerAtenSub(Operation *op, OpBuilder &builder, Value context) {
  bool scalarOther = op->getName().getStringRef() == "torch.aten.sub.Scalar";
  if (isIntegerTensorResult(op))
    return lowerIntegerAlphaBinary(op, builder, context, scalarOther,
                                   hip::ElementwiseKind::Sub, "aten.sub");
  return lowerAlphaBinary(op, builder, context, scalarOther,
                          hip::ElementwiseKind::Sub, "aten.sub");
}

// Shape ops keep the input element type, so an i1 or i64 tensor can be
// viewed or unsqueezed. Arithmetic ops stay on the float helper below.
FailureOr<RankedTensorType> shapedResultType(Operation *op, StringRef name) {
  if (op->getNumResults() != 1) {
    op->emitError() << name << " expects 1 result";
    return failure();
  }
  auto valueTensor =
      dyn_cast<torch::ValueTensorType>(op->getResult(0).getType());
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!valueTensor || !inputType) {
    op->emitError() << name << " requires a ranked tensor";
    return failure();
  }
  return RankedTensorType::get(valueTensor.getSizes(),
                               inputType.getElementType());
}

FailureOr<RankedTensorType> floatResultType(Operation *op, StringRef name) {
  if (op->getNumResults() != 1) {
    op->emitError() << name << " expects 1 result";
    return failure();
  }
  FailureOr<RankedTensorType> resultType =
      torch::toBuiltinTensorType(op->getResult(0).getType());
  if (failed(resultType)) {
    op->emitError() << name << " result element type must be f16, f32, or bf16";
    return failure();
  }
  return resultType;
}

LogicalResult lowerTensorBinary(Operation *op, OpBuilder &builder,
                                Value context, hip::ElementwiseKind kind,
                                StringRef name) {
  if (op->getNumOperands() != 2)
    return op->emitError() << name << " expects 2 operands";
  FailureOr<RankedTensorType> resultType = floatResultType(op, name);
  if (failed(resultType))
    return failure();
  Location loc = op->getLoc();
  Type elementType = resultType->getElementType();
  FailureOr<Value> lhs = hip::castToElementType(builder, loc, context,
                                                op->getOperand(0), elementType);
  FailureOr<Value> rhs = hip::castToElementType(builder, loc, context,
                                                op->getOperand(1), elementType);
  if (failed(lhs) || failed(rhs))
    return op->emitError() << name
                           << " cannot cast operands to the result type";
  FailureOr<Value> result = hip::emitElementwise(builder, loc, context, *lhs,
                                                 *rhs, *resultType, kind);
  if (failed(result))
    return op->emitError() << name << " failed";
  op->getResult(0).replaceAllUsesWith(*result);
  op->erase();
  return success();
}

LogicalResult lowerScalarBinary(Operation *op, OpBuilder &builder,
                                Value context, hip::ElementwiseKind kind,
                                StringRef name) {
  if (op->getNumOperands() != 2)
    return op->emitError() << name << " expects 2 operands";
  FailureOr<RankedTensorType> resultType = floatResultType(op, name);
  if (failed(resultType))
    return failure();
  Location loc = op->getLoc();
  Type elementType = resultType->getElementType();
  Value scalar = op->getOperand(1);
  if (isa<ShapedType>(scalar.getType()))
    return op->emitError() << name << " other must be a scalar";
  FailureOr<Value> other =
      hip::materializeScalarTensor(builder, loc, scalar, elementType);
  if (failed(other))
    return op->emitError() << "cannot materialize " << name << " other";
  FailureOr<Value> self = hip::castToElementType(
      builder, loc, context, op->getOperand(0), elementType);
  if (failed(self))
    return op->emitError() << name << " cannot cast self to the result type";
  FailureOr<Value> result = hip::emitElementwise(builder, loc, context, *self,
                                                 *other, *resultType, kind);
  if (failed(result))
    return op->emitError() << name << " failed";
  op->getResult(0).replaceAllUsesWith(*result);
  op->erase();
  return success();
}

LogicalResult lowerAtenMulTensor(Operation *op, OpBuilder &builder,
                                 Value context) {
  if (!isIntegerTensorResult(op)) {
    return lowerTensorBinary(op, builder, context, hip::ElementwiseKind::Mul,
                             "aten.mul");
  }
  auto valueTensor =
      dyn_cast<torch::ValueTensorType>(op->getResult(0).getType());
  auto lhsType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto rhsType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto elementType =
      lhsType ? dyn_cast<IntegerType>(lhsType.getElementType()) : IntegerType();
  if (!valueTensor || !lhsType || !rhsType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || !elementType ||
      !supportedIntegerWidth(elementType.getWidth()) ||
      rhsType.getElementType() != elementType ||
      llvm::any_of(valueTensor.getSizes(), ShapedType::isDynamic)) {
    return op->emitError(
        "integer aten.mul requires static i16, i32, or i64 tensors");
  }
  auto resultType =
      RankedTensorType::get(valueTensor.getSizes(), lhsType.getElementType());
  FailureOr<Value> result = hip::emitElementwise(
      builder, op->getLoc(), context, op->getOperand(0), op->getOperand(1),
      resultType, hip::ElementwiseKind::Mul);
  if (failed(result))
    return op->emitError("integer aten.mul failed");
  op->getResult(0).replaceAllUsesWith(*result);
  op->erase();
  return success();
}

LogicalResult lowerAtenDivTensor(Operation *op, OpBuilder &builder,
                                 Value context) {
  return lowerTensorBinary(op, builder, context, hip::ElementwiseKind::Div,
                           "aten.div");
}

LogicalResult lowerAtenMulScalar(Operation *op, OpBuilder &builder,
                                 Value context) {
  if (!isIntegerTensorResult(op)) {
    return lowerScalarBinary(op, builder, context, hip::ElementwiseKind::Mul,
                             "aten.mul");
  }
  auto valueTensor =
      dyn_cast<torch::ValueTensorType>(op->getResult(0).getType());
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto elementType = inputType
                         ? dyn_cast<IntegerType>(inputType.getElementType())
                         : IntegerType();
  if (!valueTensor || !inputType || !inputType.hasStaticShape() ||
      !elementType || !supportedIntegerWidth(elementType.getWidth()) ||
      llvm::any_of(valueTensor.getSizes(), ShapedType::isDynamic)) {
    return op->emitError(
        "integer aten.mul.Scalar requires a static i16, i32, or i64 tensor");
  }
  FailureOr<int64_t> other = readI64Scalar(op->getOperand(1));
  if (failed(other))
    return op->emitError("integer aten.mul.Scalar other must be a constant");
  if (failed(integerScalarFits(op, *other, elementType.getWidth())))
    return failure();
  Location loc = op->getLoc();
  Value scalar = arith::ConstantOp::create(
      builder, loc, IntegerAttr::get(elementType, *other));
  auto scalarType = RankedTensorType::get({}, elementType);
  Value rhs = tensor::SplatOp::create(builder, loc, scalarType, scalar);
  auto resultType = RankedTensorType::get(valueTensor.getSizes(), elementType);
  FailureOr<Value> result =
      hip::emitElementwise(builder, loc, context, op->getOperand(0), rhs,
                           resultType, hip::ElementwiseKind::Mul);
  if (failed(result))
    return op->emitError("integer aten.mul.Scalar failed");
  op->getResult(0).replaceAllUsesWith(*result);
  op->erase();
  return success();
}

LogicalResult lowerAtenDivScalar(Operation *op, OpBuilder &builder,
                                 Value context) {
  return lowerScalarBinary(op, builder, context, hip::ElementwiseKind::Div,
                           "aten.div");
}

FailureOr<Value> emptyStaticTensor(OpBuilder &builder, Location loc,
                                   RankedTensorType type) {
  if (!type.hasStaticShape())
    return failure();
  return Value(tensor::EmptyOp::create(builder, loc, type.getShape(),
                                       type.getElementType()));
}

LogicalResult lowerAtenRsqrt(Operation *op, OpBuilder &builder, Value context) {
  if (op->getNumOperands() != 1)
    return op->emitError("aten.rsqrt expects 1 operand");
  FailureOr<RankedTensorType> resultType = floatResultType(op, "aten.rsqrt");
  if (failed(resultType))
    return failure();
  Location loc = op->getLoc();
  FailureOr<Value> input = hip::castToElementType(
      builder, loc, context, op->getOperand(0), resultType->getElementType());
  if (failed(input))
    return op->emitError("aten.rsqrt cannot cast its input");
  FailureOr<Value> sqrtInit = emptyStaticTensor(builder, loc, *resultType);
  FailureOr<Value> recipInit = emptyStaticTensor(builder, loc, *resultType);
  if (failed(sqrtInit) || failed(recipInit))
    return op->emitError("aten.rsqrt requires a static result shape");
  auto sqrt = hip::SqrtOp::create(builder, loc, context, *input, *sqrtInit);
  auto recip = hip::ReciprocalOp::create(builder, loc, context,
                                         sqrt->getResult(0), *recipInit);
  op->getResult(0).replaceAllUsesWith(recip->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerAtenRelu(Operation *op, OpBuilder &builder, Value context) {
  if (op->getNumOperands() != 1)
    return op->emitError("aten.relu expects 1 operand");
  FailureOr<RankedTensorType> resultType = floatResultType(op, "aten.relu");
  if (failed(resultType))
    return failure();
  Location loc = op->getLoc();
  Type elementType = resultType->getElementType();
  auto floatType = dyn_cast<FloatType>(elementType);
  if (!floatType)
    return op->emitError("aten.relu requires a float result");
  auto zeroType = RankedTensorType::get({}, elementType);
  auto zeroAttr = FloatAttr::get(
      floatType, APFloat::getZero(floatType.getFloatSemantics()));
  Value zero = arith::ConstantOp::create(
      builder, loc, SplatElementsAttr::get(zeroType, zeroAttr));
  FailureOr<Value> input = hip::castToElementType(
      builder, loc, context, op->getOperand(0), elementType);
  if (failed(input))
    return op->emitError("aten.relu cannot cast its input");
  // Before: aten.relu %x : tensor<...xT>
  // After:  hip.max(%x, dense<0> : tensor<T>)
  FailureOr<Value> result =
      hip::emitElementwise(builder, loc, context, *input, zero, *resultType,
                           hip::ElementwiseKind::Max);
  if (failed(result))
    return op->emitError("aten.relu failed");
  op->getResult(0).replaceAllUsesWith(*result);
  op->erase();
  return success();
}

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

LogicalResult lowerListConstruct(Operation *op, OpBuilder &builder,
                                 Value /*context*/) {
  // Tensor lists are read by cat and index. Leave the op; the driver erases
  // it once those users are gone.
  if (auto listType = dyn_cast<torch::ListType>(op->getResult(0).getType())) {
    if (!isa<torch::IntType>(listType.getElementType()))
      return success();
  }
  SmallVector<int64_t> values;
  values.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    FailureOr<int64_t> value = readI64Scalar(operand);
    if (failed(value)) {
      return op->emitError(
          "torch.prim.ListConstruct element must be a constant i64");
    }
    values.push_back(*value);
  }
  auto type = RankedTensorType::get({static_cast<int64_t>(values.size())},
                                    builder.getI64Type());
  auto constant = arith::ConstantOp::create(
      builder, op->getLoc(), DenseIntElementsAttr::get(type, values));
  op->getResult(0).replaceAllUsesWith(constant.getResult());
  op->erase();
  return success();
}

// Before: aten.unsqueeze %x, %c-1 : tensor<64xf32> -> tensor<64x1xf32>
// After:  tensor.expand_shape %x [[0, 1]] output_shape [64, 1]
// The inserted size-1 axis is grouped with its neighbor so the group's
// product stays equal to that input dimension.
LogicalResult lowerAtenUnsqueeze(Operation *op, OpBuilder &builder,
                                 Value /*context*/) {
  if (op->getNumOperands() != 2)
    return op->emitError("aten.unsqueeze expects 2 operands");
  FailureOr<RankedTensorType> resultType =
      shapedResultType(op, "aten.unsqueeze");
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType || !inputType.hasStaticShape() || inputType.getRank() == 0)
    return op->emitError("aten.unsqueeze requires a static ranked input");
  FailureOr<int64_t> axis = readI64Scalar(op->getOperand(1));
  if (failed(axis))
    return op->emitError("aten.unsqueeze dim must be a constant i64");
  int64_t normalized = *axis;
  if (failed(normalizeAxis(normalized, inputType.getRank() + 1)))
    return op->emitError("aten.unsqueeze dim is out of range");

  SmallVector<ReassociationIndices> reassociation;
  if (normalized == 0) {
    reassociation.push_back({0, 1});
    for (int64_t dim : llvm::seq<int64_t>(1, inputType.getRank()))
      reassociation.push_back({dim + 1});
  } else {
    for (int64_t dim : llvm::seq<int64_t>(0, normalized - 1))
      reassociation.push_back({dim});
    reassociation.push_back({normalized - 1, normalized});
    for (int64_t dim : llvm::seq<int64_t>(normalized, inputType.getRank()))
      reassociation.push_back({dim + 1});
  }

  SmallVector<int64_t> expected(inputType.getShape());
  expected.insert(expected.begin() + normalized, 1);
  if (expected != resultType->getShape())
    return op->emitError("aten.unsqueeze result shape does not match dim");

  SmallVector<OpFoldResult> outputShape;
  for (int64_t dim : resultType->getShape())
    outputShape.push_back(builder.getIndexAttr(dim));
  auto expanded = tensor::ExpandShapeOp::create(builder, op->getLoc(),
                                                *resultType, op->getOperand(0),
                                                reassociation, outputShape);
  op->getResult(0).replaceAllUsesWith(expanded.getResult());
  op->erase();
  return success();
}

LogicalResult lowerAtenView(Operation *op, OpBuilder &builder,
                            Value /*context*/) {
  if (op->getNumOperands() != 2)
    return op->emitError("aten.view expects 2 operands");
  FailureOr<RankedTensorType> resultType = shapedResultType(op, "aten.view");
  if (failed(resultType) || !resultType->hasStaticShape())
    return failed(resultType)
               ? failure()
               : op->emitError("aten.view requires a static shape");
  auto reshaped = tensor::ReshapeOp::create(
      builder, op->getLoc(), *resultType, op->getOperand(0), op->getOperand(1));
  op->getResult(0).replaceAllUsesWith(reshaped.getResult());
  op->erase();
  return success();
}

LogicalResult lowerAtenPermute(Operation *op, OpBuilder &builder,
                               Value context) {
  if (op->getNumOperands() != 2)
    return op->emitError("aten.permute expects 2 operands");
  FailureOr<RankedTensorType> resultType = floatResultType(op, "aten.permute");
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType || !inputType.hasStaticShape())
    return op->emitError("aten.permute requires a static ranked input");
  FailureOr<SmallVector<int64_t>> perm = readI64List(op->getOperand(1));
  if (failed(perm) ||
      static_cast<int64_t>(perm->size()) != inputType.getRank()) {
    return op->emitError("aten.permute dims must be a constant permutation");
  }
  for (int64_t &axis : *perm)
    if (failed(normalizeAxis(axis, inputType.getRank())))
      return op->emitError("aten.permute dim is out of range");
  FailureOr<Value> init = emptyStaticTensor(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError("aten.permute requires a static result shape");
  auto transposed = hip::TransposeOp::create(builder, op->getLoc(), context,
                                             op->getOperand(0), *init,
                                             builder.getI64ArrayAttr(*perm));
  op->getResult(0).replaceAllUsesWith(transposed->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerAtenBroadcastTo(Operation *op, OpBuilder &builder,
                                   Value context) {
  if (op->getNumOperands() != 2)
    return op->emitError("aten.broadcast_to expects 2 operands");
  FailureOr<RankedTensorType> resultType =
      floatResultType(op, "aten.broadcast_to");
  if (failed(resultType))
    return failure();
  FailureOr<Value> init = emptyStaticTensor(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError("aten.broadcast_to requires a static result shape");
  auto expanded =
      hip::ExpandOp::create(builder, op->getLoc(), context, op->getOperand(0),
                            op->getOperand(1), *init);
  op->getResult(0).replaceAllUsesWith(expanded->getResult(0));
  op->erase();
  return success();
}

LogicalResult lowerAtenMm(Operation *op, OpBuilder &builder, Value context) {
  if (op->getNumOperands() != 2)
    return op->emitError("aten.mm expects 2 operands");
  FailureOr<RankedTensorType> resultType = floatResultType(op, "aten.mm");
  if (failed(resultType))
    return failure();
  FailureOr<Value> init = emptyStaticTensor(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError("aten.mm requires a static result shape");
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

LogicalResult lowerAtenSum(Operation *op, OpBuilder &builder, Value context) {
  if (op->getNumOperands() != 4)
    return op->emitError("aten.sum.dim_IntList expects 4 operands");
  if (!isa<torch::TorchNoneType>(op->getOperand(3).getType()))
    return op->emitError("aten.sum.dim_IntList dtype must be none");
  FailureOr<RankedTensorType> resultType =
      floatResultType(op, "aten.sum.dim_IntList");
  if (failed(resultType))
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return op->emitError("aten.sum.dim_IntList requires a ranked input");
  FailureOr<bool> keepdim = readI1Scalar(op->getOperand(2));
  if (failed(keepdim))
    return op->emitError(
        "aten.sum.dim_IntList keepdim must be a constant bool");
  FailureOr<SmallVector<int64_t>> axes = readI64List(op->getOperand(1));
  if (failed(axes))
    return op->emitError(
        "aten.sum.dim_IntList dims must be a constant i64 list");
  // Before: sum dims (-1, -2) on a rank-4 tensor
  // After:  hip.reduce_sum axes dense<[3, 2]>
  for (int64_t &axis : *axes)
    if (failed(normalizeAxis(axis, inputType.getRank())))
      return op->emitError("aten.sum.dim_IntList dim is out of range");
  auto axesType = RankedTensorType::get({static_cast<int64_t>(axes->size())},
                                        builder.getI64Type());
  Value axesValue = arith::ConstantOp::create(
      builder, op->getLoc(), DenseIntElementsAttr::get(axesType, *axes));
  FailureOr<Value> init = emptyStaticTensor(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError("aten.sum.dim_IntList requires a static result shape");
  auto reduced = hip::ReduceSumOp::create(
      builder, op->getLoc(), context, op->getOperand(0), axesValue, *init,
      builder.getI64IntegerAttr(*keepdim ? 1 : 0),
      builder.getI64IntegerAttr(0));
  op->getResult(0).replaceAllUsesWith(reduced->getResult(0));
  op->erase();
  return success();
}

// aten padding is one value per spatial axis and applies to both sides.
// hip.pool / hip.conv store begin axes, then end axes.
LogicalResult expandSymmetricPads(ArrayRef<int64_t> padding, int64_t spatial,
                                  SmallVectorImpl<int64_t> &pads) {
  if (static_cast<int64_t>(padding.size()) != spatial)
    return failure();
  pads.assign(padding.begin(), padding.end());
  pads.append(padding.begin(), padding.end());
  return success();
}

LogicalResult lowerAtenMaxPool(Operation *op, OpBuilder &builder,
                               Value context) {
  if (op->getNumOperands() != 6)
    return op->emitError("aten.max_pool2d expects 6 operands");
  FailureOr<RankedTensorType> resultType =
      floatResultType(op, "aten.max_pool2d");
  if (failed(resultType))
    return failure();
  FailureOr<SmallVector<int64_t>> kernel = readI64List(op->getOperand(1));
  FailureOr<SmallVector<int64_t>> strides = readI64List(op->getOperand(2));
  FailureOr<SmallVector<int64_t>> padding = readI64List(op->getOperand(3));
  FailureOr<SmallVector<int64_t>> dilation = readI64List(op->getOperand(4));
  FailureOr<bool> ceilMode = readI1Scalar(op->getOperand(5));
  if (failed(kernel) || failed(strides) || failed(padding) ||
      failed(dilation) || failed(ceilMode)) {
    return op->emitError("aten.max_pool2d window arguments must be constant");
  }
  int64_t spatial = static_cast<int64_t>(kernel->size());
  SmallVector<int64_t> pads;
  if (spatial == 0 || static_cast<int64_t>(strides->size()) != spatial ||
      static_cast<int64_t>(dilation->size()) != spatial ||
      failed(expandSymmetricPads(*padding, spatial, pads))) {
    return op->emitError("aten.max_pool2d window ranks do not match");
  }
  FailureOr<Value> init = emptyStaticTensor(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError("aten.max_pool2d requires a static result shape");
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

LogicalResult lowerAtenConvolution(Operation *op, OpBuilder &builder,
                                   Value context) {
  if (op->getNumOperands() != 9)
    return op->emitError("aten.convolution expects 9 operands");
  FailureOr<RankedTensorType> resultType =
      floatResultType(op, "aten.convolution");
  if (failed(resultType))
    return failure();
  auto weightType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!weightType || !weightType.hasStaticShape() || weightType.getRank() < 3)
    return op->emitError(
        "aten.convolution weight must be a static ranked tensor");
  FailureOr<bool> transposed = readI1Scalar(op->getOperand(6));
  if (failed(transposed))
    return op->emitError("aten.convolution transposed must be a constant");
  FailureOr<int64_t> groups = readI64Scalar(op->getOperand(8));
  if (failed(groups) || *groups < 1)
    return op->emitError("aten.convolution groups must be a positive constant");
  FailureOr<SmallVector<int64_t>> strides = readI64List(op->getOperand(3));
  FailureOr<SmallVector<int64_t>> padding = readI64List(op->getOperand(4));
  FailureOr<SmallVector<int64_t>> dilation = readI64List(op->getOperand(5));
  if (failed(strides) || failed(padding) || failed(dilation))
    return op->emitError("aten.convolution window arguments must be constant");

  int64_t spatial = weightType.getRank() - 2;
  SmallVector<int64_t> kernel;
  for (int64_t axis : llvm::seq<int64_t>(spatial))
    kernel.push_back(weightType.getDimSize(2 + axis));
  SmallVector<int64_t> pads;
  // Before: aten padding [ph, pw]
  // After:  hip.conv pads [ph, pw, ph, pw]
  if (static_cast<int64_t>(strides->size()) != spatial ||
      static_cast<int64_t>(dilation->size()) != spatial ||
      failed(expandSymmetricPads(*padding, spatial, pads))) {
    return op->emitError(
        "aten.convolution window ranks do not match the weight");
  }

  FailureOr<Value> init = emptyStaticTensor(builder, op->getLoc(), *resultType);
  if (failed(init))
    return op->emitError("aten.convolution requires a static result shape");
  SmallVector<Value> operands = {context, op->getOperand(0), op->getOperand(1)};
  if (!isa<torch::TorchNoneType>(op->getOperand(2).getType()))
    operands.push_back(op->getOperand(2));
  operands.push_back(*init);
  SmallVector<NamedAttribute> attrs = {
      builder.getNamedAttr("kernel_shape", builder.getI64ArrayAttr(kernel)),
      builder.getNamedAttr("strides", builder.getI64ArrayAttr(*strides)),
      builder.getNamedAttr("pads", builder.getI64ArrayAttr(pads)),
      builder.getNamedAttr("dilations", builder.getI64ArrayAttr(*dilation)),
      builder.getNamedAttr("group", builder.getI64IntegerAttr(*groups))};
  Value result;
  if (*transposed) {
    // Torch and hip.conv_transpose share the [C_in, C_out / group, k...]
    // weight layout.
    FailureOr<SmallVector<int64_t>> outputPadding =
        readI64List(op->getOperand(7));
    if (failed(outputPadding) ||
        static_cast<int64_t>(outputPadding->size()) != spatial)
      return op->emitError(
          "aten.convolution output_padding must be a constant list");
    attrs.push_back(builder.getNamedAttr(
        "output_padding", builder.getI64ArrayAttr(*outputPadding)));
    result = hip::ConvTransposeOp::create(
                 builder, op->getLoc(), TypeRange{*resultType}, operands, attrs)
                 .getResult(0);
  } else {
    result = hip::ConvOp::create(builder, op->getLoc(), operands, attrs)
                 .getResult(0);
  }
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return success();
}

} // namespace

void mlir::torch::populateTorchToHipRewrites(
    SmallVectorImpl<TorchRewrite> &rewrites) {
  rewrites.push_back({"torch.aten.add.Tensor", lowerAtenAdd});
  rewrites.push_back({"torch.aten.add.Scalar", lowerAtenAdd});
  rewrites.push_back({"torch.aten.sub.Tensor", lowerAtenSub});
  rewrites.push_back({"torch.aten.sub.Scalar", lowerAtenSub});
  rewrites.push_back({"torch.aten.mul.Tensor", lowerAtenMulTensor});
  rewrites.push_back({"torch.aten.mul.Scalar", lowerAtenMulScalar});
  rewrites.push_back({"torch.aten.div.Scalar", lowerAtenDivScalar});
  rewrites.push_back({"torch.aten.div.Tensor", lowerAtenDivTensor});
  rewrites.push_back({"torch.aten.rsqrt", lowerAtenRsqrt});
  rewrites.push_back({"torch.aten.relu", lowerAtenRelu});
  rewrites.push_back({"torch.prim.ListConstruct", lowerListConstruct});
  rewrites.push_back({"torch.aten.unsqueeze", lowerAtenUnsqueeze});
  rewrites.push_back({"torch.aten.view", lowerAtenView});
  rewrites.push_back({"torch.aten.permute", lowerAtenPermute});
  rewrites.push_back({"torch.aten.broadcast_to", lowerAtenBroadcastTo});
  rewrites.push_back({"torch.aten.mm", lowerAtenMm});
  rewrites.push_back({"torch.aten.sum.dim_IntList", lowerAtenSum});
  rewrites.push_back({"torch.aten.max_pool2d", lowerAtenMaxPool});
  rewrites.push_back({"torch.aten.convolution", lowerAtenConvolution});
  populateTorchModelRewrites(rewrites);
}
