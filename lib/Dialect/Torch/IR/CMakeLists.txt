##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(TorchDialectIR STATIC
  TorchDialect.cpp
  TorchOps.cpp
  TorchTypes.cpp
)

add_dependencies(TorchDialectIR
  TorchDialectIncGen
)

target_link_libraries(TorchDialectIR PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
)
