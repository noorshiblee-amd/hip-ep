/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Torch/IR/TorchDialect.h"

#include "hip/Dialect/Torch/IR/TorchOps.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::torch;

#include "hip/Dialect/Torch/IR/TorchDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hip/Dialect/Torch/IR/TorchTypes.cpp.inc"

void TorchDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hip/Dialect/Torch/IR/TorchTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "hip/Dialect/Torch/IR/TorchOps.cpp.inc"
      >();
  // A pretty-printed operation this dialect does not spell fails at parse.
  // The generic quoted form still parses, so a test can show that an
  // accepted type with an unsupported operation fails in the pass.
  allowUnknownOperations();
}
