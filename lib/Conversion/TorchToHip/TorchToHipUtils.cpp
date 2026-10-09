/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "TorchToHipUtils.h"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

#include <functional>
#include <limits>

using namespace mlir;

namespace {

/// `tensor.empty` whose dynamic dimensions come from the operand that
/// actually contributes that axis. Same rule as the ONNX elementwise helper.
/// Kept in this file so it does not export the ONNX function's symbol.
FailureOr<Value> createBroadcastEmptyTensor(OpBuilder &builder, Location loc,
                                            RankedTensorType resultType,
                                            ValueRange operands) {
  int64_t resultRank = resultType.getRank();
  SmallVector<Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultRank)) {
    if (!resultType.isDynamicDim(dimIdx))
      continue;

    Value chosen;
    int64_t chosenDim = -1;
    Value fallback;
    int64_t fallbackDim = -1;
    for (Value operand : operands) {
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType)
        continue;
      int64_t offset = resultRank - operandType.getRank();
      if (dimIdx < offset)
        continue;
      int64_t operandDim = dimIdx - offset;
      if (!fallback) {
        fallback = operand;
        fallbackDim = operandDim;
      }
      if (!operandType.isDynamicDim(operandDim) &&
          operandType.getDimSize(operandDim) == 1)
        continue;
      chosen = operand;
      chosenDim = operandDim;
      break;
    }
    if (!chosen) {
      chosen = fallback;
      chosenDim = fallbackDim;
    }
    if (!chosen)
      return failure();
    dynSizes.push_back(tensor::DimOp::create(builder, loc, chosen, chosenDim));
  }
  return Value(tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes));
}

bool multiplyDim(int64_t &product, int64_t dim) {
  if (dim < 0)
    return false;
  if (dim != 0 && product > std::numeric_limits<int64_t>::max() / dim)
    return false;
  product *= dim;
  return true;
}

FailureOr<Value> rightAlignTensor(OpBuilder &builder, Location loc, Value value,
                                  int64_t targetRank) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() == 0 ||
      type.getRank() > targetRank)
    return failure();
  if (type.getRank() == targetRank)
    return value;

  int64_t leadingOnes = targetRank - type.getRank();
  SmallVector<int64_t> alignedShape(leadingOnes, 1);
  llvm::append_range(alignedShape, type.getShape());
  auto alignedType =
      RankedTensorType::get(alignedShape, type.getElementType());

  SmallVector<ReassociationIndices> reassociation;
  ReassociationIndices firstGroup;
  for (int64_t i = 0; i <= leadingOnes; ++i)
    firstGroup.push_back(i);
  reassociation.push_back(std::move(firstGroup));
  for (int64_t i = 1; i < type.getRank(); ++i)
    reassociation.push_back({leadingOnes + i});

  SmallVector<OpFoldResult> outputShape;
  for (int64_t dim : alignedShape)
    outputShape.push_back(builder.getIndexAttr(dim));
  return Value(tensor::ExpandShapeOp::create(builder, loc, alignedType, value,
                                             reassociation, outputShape));
}

FailureOr<Value> collapseTensor(OpBuilder &builder, Location loc, Value value,
                                ArrayRef<ReassociationIndices> reassociation) {
  auto type = cast<RankedTensorType>(value.getType());
  SmallVector<int64_t> packedShape;
  for (const ReassociationIndices &group : reassociation) {
    int64_t product = 1;
    for (int64_t dim : group)
      if (!multiplyDim(product, type.getDimSize(dim)))
        return failure();
    packedShape.push_back(product);
  }
  auto packedType = RankedTensorType::get(packedShape, type.getElementType());
  return Value(tensor::CollapseShapeOp::create(builder, loc, packedType, value,
                                               reassociation));
}

} // namespace

FailureOr<hip::PackedBinary>
mlir::hip::packBinaryBroadcast(OpBuilder &builder, Location loc, Value lhs,
                               Value rhs, RankedTensorType resultType) {
  PackedBinary unpacked{lhs, rhs, resultType, {}, false};
  if (resultType.getRank() <= 4)
    return unpacked;
  if (!resultType.hasStaticShape())
    return failure();

  int64_t rank = resultType.getRank();
  SmallVector<Value> operands = {lhs, rhs};
  SmallVector<SmallVector<int64_t>> operandShapes;
  for (Value operand : operands) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    if (!type || !type.hasStaticShape() || type.getRank() > rank)
      return failure();
    SmallVector<int64_t> shape(rank, 1);
    llvm::copy(type.getShape(), shape.begin() + rank - type.getRank());
    operandShapes.push_back(std::move(shape));
  }
  ArrayRef<int64_t> outShape = resultType.getShape();
  for (int64_t dim : llvm::seq<int64_t>(rank)) {
    bool someOperandCoversOut = false;
    for (const SmallVector<int64_t> &shape : operandShapes) {
      if (shape[dim] != 1 && shape[dim] != outShape[dim])
        return failure();
      if (shape[dim] == outShape[dim])
        someOperandCoversOut = true;
    }
    if (!someOperandCoversOut)
      return failure();
  }

  auto groupIsSafe = [&](int64_t begin, int64_t end) {
    int64_t outProduct = 1;
    for (int64_t axis = begin; axis < end; ++axis)
      if (!multiplyDim(outProduct, outShape[axis]))
        return false;
    for (const SmallVector<int64_t> &shape : operandShapes) {
      int64_t product = 1;
      for (int64_t axis = begin; axis < end; ++axis)
        if (!multiplyDim(product, shape[axis]))
          return false;
      if (product != 1 && product != outProduct)
        return false;
    }
    return true;
  };

  SmallVector<ReassociationIndices> chosen;
  for (int64_t wantedGroups = 4; wantedGroups >= 1 && chosen.empty();
       --wantedGroups) {
    SmallVector<ReassociationIndices> candidate;
    std::function<bool(int64_t, int64_t)> chooseGroups =
        [&](int64_t begin, int64_t groupsLeft) {
          if (groupsLeft == 1) {
            if (!groupIsSafe(begin, rank))
              return false;
            candidate.emplace_back();
            for (int64_t axis = begin; axis < rank; ++axis)
              candidate.back().push_back(axis);
            return true;
          }
          int64_t lastEnd = rank - (groupsLeft - 1);
          for (int64_t end = begin + 1; end <= lastEnd; ++end) {
            if (!groupIsSafe(begin, end))
              continue;
            candidate.emplace_back();
            for (int64_t axis = begin; axis < end; ++axis)
              candidate.back().push_back(axis);
            if (chooseGroups(end, groupsLeft - 1))
              return true;
            candidate.pop_back();
          }
          return false;
        };
    if (chooseGroups(0, wantedGroups))
      chosen = std::move(candidate);
  }
  if (chosen.empty())
    return failure();

  auto packOperand = [&](Value operand) -> FailureOr<Value> {
    auto operandType = cast<RankedTensorType>(operand.getType());
    if (operandType.getRank() == 0)
      return operand;
    FailureOr<Value> aligned = rightAlignTensor(builder, loc, operand, rank);
    if (failed(aligned))
      return failure();
    return collapseTensor(builder, loc, *aligned, chosen);
  };

  FailureOr<Value> packedLhs = packOperand(lhs);
  FailureOr<Value> packedRhs = packOperand(rhs);
  if (failed(packedLhs) || failed(packedRhs))
    return failure();

  SmallVector<int64_t> packedResultShape;
  for (const ReassociationIndices &group : chosen) {
    int64_t product = 1;
    for (int64_t axis : group)
      if (!multiplyDim(product, resultType.getDimSize(axis)))
        return failure();
    packedResultShape.push_back(product);
  }
  PackedBinary packed;
  packed.lhs = *packedLhs;
  packed.rhs = *packedRhs;
  packed.resultType = RankedTensorType::get(packedResultShape,
                                            resultType.getElementType());
  packed.reassociation = std::move(chosen);
  packed.needsExpand = true;
  return packed;
}

FailureOr<Value> mlir::hip::emitElementwise(OpBuilder &builder, Location loc,
                                            Value context, Value lhs, Value rhs,
                                            RankedTensorType resultType,
                                            ElementwiseKind kind) {
  FailureOr<Value> init =
      createBroadcastEmptyTensor(builder, loc, resultType, {lhs, rhs});
  if (failed(init))
    return failure();
  if (kind == ElementwiseKind::Add) {
    auto op =
        hip::AddOp::create(builder, loc, context, lhs, rhs, *init);
    return op->getResult(0);
  }
  auto op = hip::MulOp::create(builder, loc, context, lhs, rhs, *init);
  return op->getResult(0);
}

FailureOr<Value> mlir::hip::castToElementType(OpBuilder &builder, Location loc,
                                              Value context, Value input,
                                              Type elementType) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return failure();
  if (inputType.getElementType() == elementType)
    return input;

  int64_t onnxDataType = 0;
  if (elementType.isF32())
    onnxDataType = 1;
  else if (elementType.isF16())
    onnxDataType = 10;
  else if (elementType.isBF16())
    onnxDataType = 16;
  if (onnxDataType == 0)
    return failure();

  auto resultType =
      RankedTensorType::get(inputType.getShape(), elementType);
  FailureOr<Value> init =
      createBroadcastEmptyTensor(builder, loc, resultType, {input});
  if (failed(init))
    return failure();
  auto castOp = hip::CastOp::create(builder, loc, context, input, *init,
                                    builder.getI64IntegerAttr(onnxDataType));
  return castOp->getResult(0);
}

FailureOr<Value>
mlir::hip::materializeScalarTensor(OpBuilder &builder, Location loc,
                                   Value scalar, Type elementType) {
  if (!isa<FloatType>(elementType))
    return failure();
  Type sourceType = scalar.getType();
  Value converted = scalar;
  if (sourceType != elementType) {
    if (sourceType.isIntOrIndex()) {
      converted =
          arith::SIToFPOp::create(builder, loc, elementType, scalar);
    } else if (isa<FloatType>(sourceType)) {
      unsigned sourceBits = cast<FloatType>(sourceType).getWidth();
      unsigned destBits = cast<FloatType>(elementType).getWidth();
      if (sourceBits > destBits) {
        converted =
            arith::TruncFOp::create(builder, loc, elementType, scalar);
      } else if (sourceBits < destBits) {
        converted = arith::ExtFOp::create(builder, loc, elementType, scalar);
      } else {
        return failure();
      }
    } else {
      return failure();
    }
  }
  auto resultType = RankedTensorType::get({}, elementType);
  return Value(
      tensor::SplatOp::create(builder, loc, resultType, converted));
}

bool mlir::hip::isUnitScalar(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
    return integer.getValue().isOne();
  if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
    return floating.getValue().isExactlyValue(1.0);
  return false;
}
