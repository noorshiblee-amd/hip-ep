/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_TORCHTOHIP_FRONTEND_H
#define HIP_CONVERSION_TORCHTOHIP_FRONTEND_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace torch {

/// These functions are the only conversion entry points that name the in-tree
/// Torch types. A later upstream Torch dialect replaces their bodies; rewrites
/// that already consume builtin tensors and scalars stay as they are.

bool isTorchDialectOp(Operation *op);
bool isTorchType(Type type);
bool isAcceptedFloatElementType(Type elementType);

/// Builtin type for a Torch scalar or value tensor. Non-Torch types pass
/// through. Integer and bool tensors fail.
FailureOr<Type> toBuiltinType(Type type);

/// Ranked builtin tensor for a Torch value tensor with an accepted element
/// type. Fails for every other type.
FailureOr<RankedTensorType> toBuiltinTensorType(Type type);

/// Lowers torch.constant.int, torch.constant.float, torch.constant.bool, and
/// torch.vtensor.literal. A used torch.constant.none is left for its consumer.
/// `NotCarrier` leaves the operation alone. `Failed` has already diagnosed.
enum class CarrierResult { NotCarrier, Lowered, Failed };

CarrierResult lowerTorchCarrier(Operation *op, int64_t &serializationOrder);

} // namespace torch
} // namespace mlir

#endif // HIP_CONVERSION_TORCHTOHIP_FRONTEND_H
