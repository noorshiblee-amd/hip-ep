// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-torch-to-hip | FileCheck %s

module {
  func.func @test_add_same_rank(%lhs: !torch.vtensor<[1,128,32],f16>, %rhs: !torch.vtensor<[1,128,32],f16>) -> !torch.vtensor<[1,128,32],f16> {
    // CHECK-LABEL: func.func @test_add_same_rank
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[LHS:.*]]: tensor<1x128x32xf16>, %[[RHS:.*]]: tensor<1x128x32xf16>) -> tensor<1x128x32xf16>
    %alpha = torch.constant.int 1
    %output = torch.aten.add.Tensor %lhs, %rhs, %alpha : !torch.vtensor<[1,128,32],f16>, !torch.vtensor<[1,128,32],f16>, !torch.int -> !torch.vtensor<[1,128,32],f16>
    // CHECK: tensor.empty() : tensor<1x128x32xf16>
    // CHECK: hip.add(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<1x128x32xf16>, tensor<1x128x32xf16>) outs({{.*}} : tensor<1x128x32xf16>)
    // CHECK-NOT: hip.mul
    // CHECK-NOT: torch.
    return %output : !torch.vtensor<[1,128,32],f16>
  }

  func.func @test_add_broadcast(%input: !torch.vtensor<[1,128,32],f16>, %bias: !torch.vtensor<[32],f16>) -> !torch.vtensor<[1,128,32],f16> {
    // CHECK-LABEL: func.func @test_add_broadcast
    // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x128x32xf16>, %[[BIAS:.*]]: tensor<32xf16>) -> tensor<1x128x32xf16>
    %alpha = torch.constant.int 1
    %output = torch.aten.add.Tensor %input, %bias, %alpha : !torch.vtensor<[1,128,32],f16>, !torch.vtensor<[32],f16>, !torch.int -> !torch.vtensor<[1,128,32],f16>
    // CHECK: hip.add(%[[CTX2]]) ins(%[[INPUT]], %[[BIAS]] : tensor<1x128x32xf16>, tensor<32xf16>) outs({{.*}} : tensor<1x128x32xf16>)
    return %output : !torch.vtensor<[1,128,32],f16>
  }

  func.func @test_add_bf16(%lhs: !torch.vtensor<[4],bf16>, %rhs: !torch.vtensor<[4],bf16>) -> !torch.vtensor<[4],bf16> {
    // CHECK-LABEL: func.func @test_add_bf16
    // CHECK: hip.add({{.*}}) ins({{.*}}, {{.*}} : tensor<4xbf16>, tensor<4xbf16>) outs({{.*}} : tensor<4xbf16>)
    %alpha = torch.constant.float 1.0
    %output = torch.aten.add.Tensor %lhs, %rhs, %alpha : !torch.vtensor<[4],bf16>, !torch.vtensor<[4],bf16>, !torch.float -> !torch.vtensor<[4],bf16>
    return %output : !torch.vtensor<[4],bf16>
  }

  func.func @test_add_dynamic(%lhs: !torch.vtensor<[?,32],f32>, %rhs: !torch.vtensor<[?,32],f32>) -> !torch.vtensor<[?,32],f32> {
    // CHECK-LABEL: func.func @test_add_dynamic
    // CHECK: tensor.dim
    // CHECK: hip.add({{.*}}) ins({{.*}}, {{.*}} : tensor<?x32xf32>, tensor<?x32xf32>) outs({{.*}} : tensor<?x32xf32>)
    %alpha = torch.constant.int 1
    %output = torch.aten.add.Tensor %lhs, %rhs, %alpha : !torch.vtensor<[?,32],f32>, !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[?,32],f32>
    return %output : !torch.vtensor<[?,32],f32>
  }

  func.func @test_add_5d_scalar(%input: !torch.vtensor<[1,128,200,8,200],f32>, %bias: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,128,200,8,200],f32> {
    // CHECK-LABEL: func.func @test_add_5d_scalar
    // CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] : tensor<1x128x200x8x200xf32> into tensor<1x128x200x1600xf32>
    // CHECK: hip.add({{.*}}) ins({{.*}}, {{.*}} : tensor<1x128x200x1600xf32>, tensor<f32>) outs({{.*}} : tensor<1x128x200x1600xf32>)
    // CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] output_shape [1, 128, 200, 8, 200] : tensor<1x128x200x1600xf32> into tensor<1x128x200x8x200xf32>
    %alpha = torch.constant.int 1
    %output = torch.aten.add.Tensor %input, %bias, %alpha : !torch.vtensor<[1,128,200,8,200],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[1,128,200,8,200],f32>
    return %output : !torch.vtensor<[1,128,200,8,200],f32>
  }

  func.func @test_add_cast(%lhs: !torch.vtensor<[4],f32>, %rhs: !torch.vtensor<[4],f16>) -> !torch.vtensor<[4],f16> {
    // CHECK-LABEL: func.func @test_add_cast
    // CHECK: hip.cast({{.*}}) ins({{.*}} : tensor<4xf32>) outs({{.*}} : tensor<4xf16>) {to = 10 : i64}
    // CHECK: hip.add({{.*}}) ins({{.*}}, {{.*}} : tensor<4xf16>, tensor<4xf16>)
    %alpha = torch.constant.int 1
    %output = torch.aten.add.Tensor %lhs, %rhs, %alpha : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f16>, !torch.int -> !torch.vtensor<[4],f16>
    return %output : !torch.vtensor<[4],f16>
  }
}
