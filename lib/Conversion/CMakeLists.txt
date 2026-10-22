##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Conversion Libraries")

add_subdirectory(OnnxToHip)
# torch_hip compiles these sources itself. Building them here as well defines
# createConvertTorchToHipPass twice when the plugin is linked.
if(NOT "torch_hip" IN_LIST HIPDNN_EP_COMPILER_PLUGINS)
  add_subdirectory(TorchToHip)
endif()
add_subdirectory(OnnxToHipsr)
add_subdirectory(OnnxToHipDNN)
add_subdirectory(HipToLLVM)
add_subdirectory(HipToTosa)

# Umbrella conversion library
add_library(HipConversion STATIC
  Passes.cpp
)

# Passes.cpp includes hip/Conversion/Passes.h -> Passes.h.inc.
add_dependencies(HipConversion HipsrConversionPassIncGen)

target_link_libraries(HipConversion PUBLIC
  OnnxToHip
  OnnxToHipsr
  OnnxToHipDNN
  HipToLLVM
  HipToTosa
  MLIRIR
  MLIRPass
)
# torch_hip defines createConvertTorchToHipPass. Linking TorchToHip as well
# is a duplicate symbol.
if(NOT "torch_hip" IN_LIST HIPDNN_EP_COMPILER_PLUGINS)
  target_link_libraries(HipConversion PUBLIC TorchToHip)
endif()

set_target_properties(HipConversion PROPERTIES FOLDER "mlir/Conversion")
