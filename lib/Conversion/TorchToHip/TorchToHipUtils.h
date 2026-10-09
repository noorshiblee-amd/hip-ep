/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_TORCHTOHIP_UTILS_H
#define HIP_CONVERSION_TORCHTOHIP_UTILS_H

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hip {

/// HIP elementwise kernels take rank <= 4. Rank 4 and below is unchanged.
/// A static higher rank is packed with the same broadcast rule as the ONNX
/// elementwise frontend, then expanded back to the original result shape.
struct PackedBinary {
  Value lhs;
  Value rhs;
  RankedTensorType resultType;
  SmallVector<ReassociationIndices> reassociation;
  bool needsExpand = false;
};

FailureOr<PackedBinary> packBinaryBroadcast(OpBuilder &builder, Location loc,
                                            Value lhs, Value rhs,
                                            RankedTensorType resultType);

enum class ElementwiseKind { Add, Mul, Sub, Div, Max };

FailureOr<Value> emitElementwise(OpBuilder &builder, Location loc,
                                 Value context, Value lhs, Value rhs,
                                 RankedTensorType resultType,
                                 ElementwiseKind kind);

FailureOr<Value> castToElementType(OpBuilder &builder, Location loc,
                                   Value context, Value input,
                                   Type elementType);

/// Rank-0 tensor of `elementType` holding `scalar`. Integer scalars are
/// converted with sitofp; wider floats are truncated.
FailureOr<Value> materializeScalarTensor(OpBuilder &builder, Location loc,
                                         Value scalar, Type elementType);

/// True when `value` is an arith.constant integer 1 or float 1.0.
bool isUnitScalar(Value value);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_TORCHTOHIP_UTILS_H
