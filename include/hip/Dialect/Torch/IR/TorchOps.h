/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_TORCH_OPS_H
#define HIP_DIALECT_TORCH_OPS_H

#include "hip/Dialect/Torch/IR/TorchDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "hip/Dialect/Torch/IR/TorchOps.h.inc"

namespace mlir {
namespace torch {

/// Default assembly: operands, colon, operand types, arrow, result types.
ParseResult parseDefaultTorchOp(OpAsmParser &parser, OperationState &result,
                                 int numOperands, int numResults);
void printDefaultTorchOp(OpAsmPrinter &printer, Operation *op);

} // namespace torch
} // namespace mlir

#endif // HIP_DIALECT_TORCH_OPS_H
