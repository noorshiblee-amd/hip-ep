/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_TORCHTOHIP_POPULATE_H
#define HIP_CONVERSION_TORCHTOHIP_POPULATE_H

#include "mlir/IR/Builders.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace torch {

/// One accepted Torch operation. The rewrite sees builtin operand types.
/// On success it erases the operation, except a tensor `ListConstruct`,
/// which stays until cat or index has read it.
using TorchRewriteFn = LogicalResult (*)(Operation *op, OpBuilder &builder,
                                         Value context);

struct TorchRewrite {
  StringRef name;
  TorchRewriteFn rewrite;
};

void populateTorchToHipRewrites(SmallVectorImpl<TorchRewrite> &rewrites);

void populateTorchModelRewrites(SmallVectorImpl<TorchRewrite> &rewrites);

} // namespace torch
} // namespace mlir

#endif // HIP_CONVERSION_TORCHTOHIP_POPULATE_H
