// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-torch-to-hip | FileCheck %s

module {
  func.func @test_literal(%input: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @test_literal
    // CHECK: hip.constant
    // CHECK-SAME: serialization_order = 0
    // CHECK-SAME: dense<1.000000e+00> : tensor<4xf32>
    // CHECK: hip.constant
    // CHECK-SAME: serialization_order = 1
    // CHECK-SAME: dense<2.000000e+00> : tensor<f32>
    // CHECK: hip.add
    %weight = torch.vtensor.literal(dense<1.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
    %bias = torch.vtensor.literal(dense<2.0> : tensor<f32>) : !torch.vtensor<[],f32>
    %alpha = torch.constant.int 1
    %sum = torch.aten.add.Tensor %input, %weight, %alpha : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4],f32>
    %output = torch.aten.add.Tensor %sum, %bias, %alpha : !torch.vtensor<[4],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[4],f32>
    return %output : !torch.vtensor<[4],f32>
  }
}
