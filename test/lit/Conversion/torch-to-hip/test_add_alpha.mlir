// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-torch-to-hip | FileCheck %s

module {
  func.func @test_add_alpha_int(%lhs: !torch.vtensor<[4],f32>, %rhs: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @test_add_alpha_int
    // CHECK: %[[TWO:.*]] = arith.constant 2 : i64
    // CHECK: %[[TWO_F:.*]] = arith.sitofp %[[TWO]] : i64 to f32
    // CHECK: %[[ALPHA:.*]] = tensor.splat %[[TWO_F]] : tensor<f32>
    // CHECK: hip.mul({{.*}}) ins({{.*}}, %[[ALPHA]] : tensor<4xf32>, tensor<f32>)
    // CHECK: hip.add
    %alpha = torch.constant.int 2
    %output = torch.aten.add.Tensor %lhs, %rhs, %alpha : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4],f32>
    return %output : !torch.vtensor<[4],f32>
  }

  func.func @test_add_alpha_float(%lhs: !torch.vtensor<[4],f16>, %rhs: !torch.vtensor<[4],f16>) -> !torch.vtensor<[4],f16> {
    // CHECK-LABEL: func.func @test_add_alpha_float
    // CHECK: arith.constant 2.000000e+00 : f64
    // CHECK: arith.truncf {{.*}} : f64 to f16
    // CHECK: tensor.splat
    // CHECK: hip.mul
    // CHECK: hip.add
    %alpha = torch.constant.float 2.0
    %output = torch.aten.add.Tensor %lhs, %rhs, %alpha : !torch.vtensor<[4],f16>, !torch.vtensor<[4],f16>, !torch.float -> !torch.vtensor<[4],f16>
    return %output : !torch.vtensor<[4],f16>
  }

  func.func @test_add_runtime_alpha(%lhs: !torch.vtensor<[4],f32>, %rhs: !torch.vtensor<[4],f32>, %alpha: !torch.float) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @test_add_runtime_alpha
    // CHECK-SAME: tensor<4xf32>, tensor<4xf32>, f64
    // CHECK: arith.truncf
    // CHECK: hip.mul
    // CHECK: hip.add
    %output = torch.aten.add.Tensor %lhs, %rhs, %alpha : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],f32>
    return %output : !torch.vtensor<[4],f32>
  }
}
