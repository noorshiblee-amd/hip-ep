// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @layouts
// CHECK-SAME: tensor<f32, #hip.layout<"">>
// CHECK-SAME: tensor<?xf32, #hip.layout<"?">>
// CHECK-SAME: tensor<1x2x3x4xf32, #hip.layout<"????">>
// CHECK-SAME: tensor<1x2x3x4x5x6xf32, #hip.layout<"??????">>
func.func @layouts(%scalar: tensor<f32, #hip.layout<"">>,
                   %dynamic: tensor<?xf32, #hip.layout<"?">>,
                   %four: tensor<1x2x3x4xf32, #hip.layout<"????">>,
                   %six: tensor<1x2x3x4x5x6xf32, #hip.layout<"??????">>) {
  return
}
