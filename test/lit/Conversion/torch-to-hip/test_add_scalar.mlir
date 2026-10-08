// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-torch-to-hip | FileCheck %s

module {
  func.func @test_add_scalar(%input: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @test_add_scalar
    // CHECK: %[[THREE:.*]] = arith.constant 3 : i64
    // CHECK: %[[THREE_F:.*]] = arith.sitofp %[[THREE]] : i64 to f32
    // CHECK: %[[OTHER:.*]] = tensor.splat %[[THREE_F]] : tensor<f32>
    // CHECK: hip.add({{.*}}) ins({{.*}}, %[[OTHER]] : tensor<4xf32>, tensor<f32>)
    // CHECK-NOT: hip.mul
    %other = torch.constant.int 3
    %alpha = torch.constant.int 1
    %output = torch.aten.add.Scalar %input, %other, %alpha : !torch.vtensor<[4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4],f32>
    return %output : !torch.vtensor<[4],f32>
  }
}
