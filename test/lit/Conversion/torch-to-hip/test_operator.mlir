// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s | FileCheck %s

module {
  func.func @roundtrip_operator(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.none {
    // CHECK-LABEL: func.func @roundtrip_operator
    // CHECK: torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    // CHECK: torch.operator "onnx.QuantizeLinear"(%arg0, %{{.*}}) : (!torch.vtensor<[2, 3],f32>, !torch.vtensor<[],f32>) -> !torch.none
    %scale = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.0> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %q = torch.operator "onnx.QuantizeLinear"(%arg0, %scale) : (!torch.vtensor<[2,3],f32>, !torch.vtensor<[],f32>) -> !torch.none
    return %q : !torch.none
  }
}
