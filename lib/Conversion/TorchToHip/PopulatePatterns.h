/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_TORCHTOHIP_POPULATE_H
#define HIP_CONVERSION_TORCHTOHIP_POPULATE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace torch {

/// One accepted Torch operation. The rewrite sees builtin operand types;
/// the result type may still be a Torch value tensor and is converted
/// through toBuiltinTensorType. On success the operation is erased.
using TorchRewriteFn = LogicalResult (*)(Operation *op, OpBuilder &builder,
                                         Value context);

struct TorchRewrite {
  StringRef name;
  TorchRewriteFn rewrite;
};

void populateTorchToHipRewrites(SmallVectorImpl<TorchRewrite> &rewrites);

void populateOnnxOperatorRewrites(SmallVectorImpl<TorchRewrite> &rewrites);

/// convert-onnx-to-hip reads metadata from @main_graph. A Torch module with
/// one function keeps its body and takes that name once it contains onnx ops.
void renameSoleOnnxEntry(ModuleOp module);

} // namespace torch
} // namespace mlir

#endif // HIP_CONVERSION_TORCHTOHIP_POPULATE_H
