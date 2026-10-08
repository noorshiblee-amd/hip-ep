/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "PopulatePatterns.h"
#include "TorchFrontend.h"
#include "TorchToHipUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace {

LogicalResult lowerAtenAdd(Operation *op, OpBuilder &builder, Value context) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return op->emitError("aten.add expects 3 operands and 1 result");

  bool scalarOther = op->getName().getStringRef() == "torch.aten.add.Scalar";
  FailureOr<RankedTensorType> resultType =
      torch::toBuiltinTensorType(op->getResult(0).getType());
  if (failed(resultType)) {
    return op->emitError(
        "aten.add result element type must be f16, f32, or bf16");
  }

  Value self = op->getOperand(0);
  Value other = op->getOperand(1);
  Value alpha = op->getOperand(2);
  Location loc = op->getLoc();
  Type elementType = resultType->getElementType();

  auto selfType = dyn_cast<RankedTensorType>(self.getType());
  if (!selfType || !torch::isAcceptedFloatElementType(selfType.getElementType())) {
    return op->emitError(
        "aten.add self must be an f16, f32, or bf16 tensor");
  }

  if (scalarOther) {
    if (isa<ShapedType>(other.getType()))
      return op->emitError("aten.add.Scalar other must be a scalar");
    FailureOr<Value> otherTensor =
        hip::materializeScalarTensor(builder, loc, other, elementType);
    if (failed(otherTensor))
      return op->emitError("cannot materialize aten.add.Scalar other");
    other = *otherTensor;
  }

  auto otherType = dyn_cast<RankedTensorType>(other.getType());
  if (!otherType ||
      !torch::isAcceptedFloatElementType(otherType.getElementType())) {
    return op->emitError(
        "aten.add other must be an f16, f32, or bf16 tensor");
  }

  FailureOr<Value> castSelf =
      hip::castToElementType(builder, loc, context, self, elementType);
  FailureOr<Value> castOther =
      hip::castToElementType(builder, loc, context, other, elementType);
  if (failed(castSelf) || failed(castOther))
    return op->emitError("aten.add cannot cast operands to the result type");

  FailureOr<hip::PackedBinary> packed = hip::packBinaryBroadcast(
      builder, loc, *castSelf, *castOther, *resultType);
  if (failed(packed))
    return op->emitError("cannot pack aten.add into 4 dimensions");

  Value rhs = packed->rhs;
  if (!hip::isUnitScalar(alpha)) {
    FailureOr<Value> alphaTensor =
        hip::materializeScalarTensor(builder, loc, alpha, elementType);
    if (failed(alphaTensor))
      return op->emitError("cannot materialize aten.add alpha");
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    FailureOr<Value> scaled =
        hip::emitElementwise(builder, loc, context, rhs, *alphaTensor, rhsType,
                             hip::ElementwiseKind::Mul);
    if (failed(scaled))
      return op->emitError("aten.add alpha multiply failed");
    rhs = *scaled;
  }

  FailureOr<Value> sum = hip::emitElementwise(
      builder, loc, context, packed->lhs, rhs, packed->resultType,
      hip::ElementwiseKind::Add);
  if (failed(sum))
    return op->emitError("aten.add failed");

  Value result = *sum;
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

} // namespace

void mlir::torch::populateTorchToHipRewrites(
    SmallVectorImpl<TorchRewrite> &rewrites) {
  rewrites.push_back({"torch.aten.add.Tensor", lowerAtenAdd});
  rewrites.push_back({"torch.aten.add.Scalar", lowerAtenAdd});
}
