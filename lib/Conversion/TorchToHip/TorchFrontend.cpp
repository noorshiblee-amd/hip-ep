/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "TorchFrontend.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Torch/IR/TorchOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <limits>

using namespace mlir;
using namespace mlir::torch;

bool mlir::torch::isTorchDialectOp(Operation *op) {
  return op->getDialect() && op->getDialect()->getNamespace() == "torch";
}

bool mlir::torch::isTorchType(Type type) {
  return isa<IntType, TorchFloatType, TorchBoolType, ValueTensorType>(type);
}

bool mlir::torch::isAcceptedFloatElementType(Type elementType) {
  return elementType.isF16() || elementType.isF32() || elementType.isBF16();
}

FailureOr<RankedTensorType>
mlir::torch::toBuiltinTensorType(Type type) {
  auto valueTensor = dyn_cast<ValueTensorType>(type);
  if (!valueTensor || !isAcceptedFloatElementType(valueTensor.getDtype()))
    return failure();
  return RankedTensorType::get(valueTensor.getSizes(), valueTensor.getDtype());
}

FailureOr<Type> mlir::torch::toBuiltinType(Type type) {
  if (isa<ValueTensorType>(type)) {
    FailureOr<RankedTensorType> tensorType = toBuiltinTensorType(type);
    if (failed(tensorType))
      return failure();
    return Type(*tensorType);
  }
  if (isa<IntType>(type))
    return IntegerType::get(type.getContext(), 64);
  if (isa<TorchFloatType>(type))
    return Float64Type::get(type.getContext());
  if (isa<TorchBoolType>(type))
    return IntegerType::get(type.getContext(), 1);
  if (isTorchType(type))
    return failure();
  return type;
}

static LogicalResult literalTensorType(ValueTensorLiteralOp literal,
                                       RankedTensorType &resultType) {
  FailureOr<RankedTensorType> converted =
      toBuiltinTensorType(literal.getType());
  if (failed(converted)) {
    return literal.emitError(
        "torch.vtensor.literal element type must be f16, f32, or bf16");
  }
  auto dense = dyn_cast<DenseElementsAttr>(literal.getValue());
  if (!dense || isa<DenseResourceElementsAttr>(literal.getValue())) {
    return literal.emitError(
        "torch.vtensor.literal requires a dense elements attribute");
  }
  if (dense.getType() != *converted) {
    return literal.emitError(
        "torch.vtensor.literal payload type does not match its result type");
  }
  resultType = *converted;
  return success();
}

CarrierResult mlir::torch::lowerTorchCarrier(Operation *op,
                                             int64_t &serializationOrder) {
  OpBuilder builder(op);
  if (auto integer = dyn_cast<ConstantIntOp>(op)) {
    auto constant = arith::ConstantOp::create(
        builder, op->getLoc(), builder.getI64IntegerAttr(integer.getValue()));
    integer.getResult().replaceAllUsesWith(constant.getResult());
    integer.erase();
    return CarrierResult::Lowered;
  }
  if (auto floating = dyn_cast<ConstantFloatOp>(op)) {
    auto constant = arith::ConstantOp::create(
        builder, op->getLoc(),
        builder.getFloatAttr(builder.getF64Type(), floating.getValue()));
    floating.getResult().replaceAllUsesWith(constant.getResult());
    floating.erase();
    return CarrierResult::Lowered;
  }
  auto literal = dyn_cast<ValueTensorLiteralOp>(op);
  if (!literal)
    return CarrierResult::NotCarrier;

  RankedTensorType tensorType;
  if (failed(literalTensorType(literal, tensorType)))
    return CarrierResult::Failed;
  if (serializationOrder == std::numeric_limits<int64_t>::max()) {
    literal.emitError("hip.constant order overflows int64");
    return CarrierResult::Failed;
  }

  auto dense = cast<DenseElementsAttr>(literal.getValue());
  auto carrier = hip::ConstantOp::create(builder, literal.getLoc(), tensorType,
                                         dense);
  carrier.setSerializationOrderAttr(
      builder.getI64IntegerAttr(serializationOrder++));
  literal.replaceAllUsesWith(carrier.getResult());
  literal.erase();
  return CarrierResult::Lowered;
}
