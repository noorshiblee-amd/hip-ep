##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(TorchToHip STATIC
  ElementwiseConversion.cpp
  OnnxOperatorConversion.cpp
  TorchFrontend.cpp
  TorchToHip.cpp
  TorchToHipUtils.cpp
)

add_dependencies(TorchToHip
  TorchDialectIncGen
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
)

target_link_libraries(TorchToHip PUBLIC
  TorchDialectIR
  HipDialectIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
)
