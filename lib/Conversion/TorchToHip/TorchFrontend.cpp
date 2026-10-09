/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "TorchFrontend.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Torch/IR/TorchOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/APFloat.h"

#include <limits>

using namespace mlir;
using namespace mlir::torch;

bool mlir::torch::isTorchDialectOp(Operation *op) {
  return op->getDialect() && op->getDialect()->getNamespace() == "torch";
}

bool mlir::torch::isTorchType(Type type) {
  return isa<IntType, TorchFloatType, TorchBoolType, TorchNoneType, ListType,
             ValueTensorType>(type);
}

bool mlir::torch::isAcceptedFloatElementType(Type elementType) {
  return elementType.isF16() || elementType.isF32() || elementType.isBF16();
}

FailureOr<RankedTensorType> mlir::torch::toBuiltinTensorType(Type type) {
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

// `dense_resource` blobs are raw element bytes. The textual alignment prefix
// is stripped by the parser before `getData()`. A handle with no blob is a
// zero splat: some producers elide the payload and still name the shape.
static FailureOr<DenseElementsAttr>
materializeLiteral(ValueTensorLiteralOp literal, RankedTensorType resultType) {
  Attribute attr = literal.getValue();
  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    if (dense.getType() != resultType) {
      return literal.emitError(
          "torch.vtensor.literal payload type does not match its result type");
    }
    return dense;
  }
  auto resource = dyn_cast<DenseResourceElementsAttr>(attr);
  if (!resource) {
    return literal.emitError(
        "torch.vtensor.literal requires a dense elements attribute");
  }
  auto shaped = dyn_cast<ShapedType>(resource.getType());
  if (!shaped || shaped != resultType) {
    return literal.emitError(
        "torch.vtensor.literal payload type does not match its result type");
  }
  ArrayRef<char> data = resource.getData();
  if (data.empty() && resultType.getNumElements() != 0) {
    literal.emitWarning(
        "dense_resource has no payload; lowering a zero tensor");
    auto floatType = dyn_cast<FloatType>(resultType.getElementType());
    if (!floatType)
      return literal.emitError("zero literal requires a float element type");
    auto zero = FloatAttr::get(floatType,
                               APFloat::getZero(floatType.getFloatSemantics()));
    return DenseElementsAttr(SplatElementsAttr::get(resultType, zero));
  }
  bool splat = false;
  if (!DenseElementsAttr::isValidRawBuffer(resultType, data, splat)) {
    return literal.emitError(
        "torch.vtensor.literal dense_resource payload does not match its type");
  }
  return DenseElementsAttr::getFromRawBuffer(resultType, data);
}

static LogicalResult literalTensorType(ValueTensorLiteralOp literal,
                                       RankedTensorType &resultType) {
  FailureOr<RankedTensorType> converted =
      toBuiltinTensorType(literal.getType());
  if (failed(converted)) {
    return literal.emitError(
        "torch.vtensor.literal element type must be f16, f32, or bf16");
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
  if (auto flag = dyn_cast<ConstantBoolOp>(op)) {
    auto constant = arith::ConstantOp::create(
        builder, op->getLoc(), builder.getBoolAttr(flag.getValue()));
    flag.getResult().replaceAllUsesWith(constant.getResult());
    flag.erase();
    return CarrierResult::Lowered;
  }
  // A used None stays until the consumer drops the optional operand. An
  // unused None has no consumer and is erased here.
  if (isa<ConstantNoneOp>(op)) {
    if (!op->use_empty())
      return CarrierResult::NotCarrier;
    op->erase();
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

  FailureOr<DenseElementsAttr> dense = materializeLiteral(literal, tensorType);
  if (failed(dense))
    return CarrierResult::Failed;
  auto carrier =
      hip::ConstantOp::create(builder, literal.getLoc(), tensorType, *dense);
  carrier.setSerializationOrderAttr(
      builder.getI64IntegerAttr(serializationOrder++));
  literal.replaceAllUsesWith(carrier.getResult());
  literal.erase();
  return CarrierResult::Lowered;
}
