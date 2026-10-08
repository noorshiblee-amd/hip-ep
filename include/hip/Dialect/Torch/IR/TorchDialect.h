/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_TORCH_DIALECT_H
#define HIP_DIALECT_TORCH_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

#include "hip/Dialect/Torch/IR/TorchDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hip/Dialect/Torch/IR/TorchTypes.h.inc"

#endif // HIP_DIALECT_TORCH_DIALECT_H
