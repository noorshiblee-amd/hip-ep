// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --torch-to-hip-pipeline --verify-each | FileCheck %s

module {
  func.func @resnet_ops(%arg0: !torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1],f32> {
    // CHECK-LABEL: func.func @resnet_ops
    // CHECK-DAG: hip.alloc_output
    // CHECK-DAG: hip.conv
    // CHECK-DAG: kernel_shape = [1, 1]
    // CHECK-DAG: pads = [0, 0, 0, 0]
    // CHECK-DAG: hip.pool
    // CHECK-DAG: pool_mode = 1
    // CHECK-DAG: hip.reduce_sum
    // CHECK-DAG: hip.matmul
    // CHECK-DAG: hip.transpose
    // CHECK-DAG: hip.expand
    // CHECK: return
    // CHECK-NOT: torch.
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int4 = torch.constant.int 4
    %true = torch.constant.bool true
    %false = torch.constant.bool false
    %none = torch.constant.none
    %eps = torch.constant.float 1.000000e-05
    %weight = torch.vtensor.literal(dense<1.000000e+00> : tensor<1x1x1x1xf32>) : !torch.vtensor<[1,1,1,1],f32>
    %scale = torch.vtensor.literal(dense<1.000000e+00> : tensor<1xf32>) : !torch.vtensor<[1],f32>
    %fcWeight = torch.vtensor.literal(dense<1.000000e+00> : tensor<1x1xf32>) : !torch.vtensor<[1,1],f32>
    %fcBias = torch.vtensor.literal(dense<0.000000e+00> : tensor<1xf32>) : !torch.vtensor<[1],f32>
    %ones = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %zeros = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %conv = torch.aten.convolution %arg0, %weight, %none, %ones, %zeros, %ones, %false, %zeros, %int1 : !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,4,4],f32>
    %shifted = torch.aten.add.Scalar %scale, %eps, %int1 : !torch.vtensor<[1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1],f32>
    %inv = torch.aten.rsqrt %shifted : !torch.vtensor<[1],f32> -> !torch.vtensor<[1],f32>
    %unsq0 = torch.aten.unsqueeze %inv, %int0 : !torch.vtensor<[1],f32>, !torch.int -> !torch.vtensor<[1,1],f32>
    %unsq1 = torch.aten.unsqueeze %unsq0, %int0 : !torch.vtensor<[1,1],f32>, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %centered = torch.aten.sub.Tensor %conv, %unsq1, %int1 : !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int -> !torch.vtensor<[1,1,4,4],f32>
    %scaled = torch.aten.mul.Tensor %centered, %unsq1 : !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,4,4],f32>
    %activated = torch.aten.relu %scaled : !torch.vtensor<[1,1,4,4],f32> -> !torch.vtensor<[1,1,4,4],f32>
    %pooled = torch.aten.max_pool2d %activated, %ones, %ones, %zeros, %ones, %false : !torch.vtensor<[1,1,4,4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,4,4],f32>
    %int-1 = torch.constant.int -1
    %int-2 = torch.constant.int -2
    %axes = torch.prim.ListConstruct %int-1, %int-2 : (!torch.int, !torch.int) -> !torch.list<int>
    %summed = torch.aten.sum.dim_IntList %pooled, %axes, %true, %none : !torch.vtensor<[1,1,4,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1,1],f32>
    %averaged = torch.aten.div.Scalar %summed, %int4 : !torch.vtensor<[1,1,1,1],f32>, !torch.int -> !torch.vtensor<[1,1,1,1],f32>
    %flatShape = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %flat = torch.aten.view %averaged, %flatShape : !torch.vtensor<[1,1,1,1],f32>, !torch.list<int> -> !torch.vtensor<[1,1],f32>
    %perm = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %transposed = torch.aten.permute %fcWeight, %perm : !torch.vtensor<[1,1],f32>, !torch.list<int> -> !torch.vtensor<[1,1],f32>
    %logits = torch.aten.mm %flat, %transposed : !torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32> -> !torch.vtensor<[1,1],f32>
    %scaledLogits = torch.aten.mul.Scalar %logits, %int1 : !torch.vtensor<[1,1],f32>, !torch.int -> !torch.vtensor<[1,1],f32>
    %biasShape = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %bias = torch.aten.broadcast_to %fcBias, %biasShape : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[1,1],f32>
    %scaledBias = torch.aten.mul.Scalar %bias, %int1 : !torch.vtensor<[1,1],f32>, !torch.int -> !torch.vtensor<[1,1],f32>
    %result = torch.aten.add.Tensor %scaledLogits, %scaledBias, %int1 : !torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32>, !torch.int -> !torch.vtensor<[1,1],f32>
    return %result : !torch.vtensor<[1,1],f32>
  }
}
