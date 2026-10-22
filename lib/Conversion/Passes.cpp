/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/Passes.h"
#include "hip/Conversion/TorchToHip/Passes.h"
#include "mlir/Pass/Pass.h"

namespace hip::compiler {

void registerConversionPasses() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createConvertOnnxToHipPass();
  });

#ifndef HIP_EP_TORCH_FROM_PLUGIN
  // torch_hip registers this pass itself. Registering it here as well would
  // claim the name convert-torch-to-hip twice.
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createConvertTorchToHipPass();
  });
#endif

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createOutlineOnnxToHipDNNPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createConvertHipToLLVMPass();
  });

  mlir::hipsr::registerHipsrConversionPasses();
}

} // namespace hip::compiler
