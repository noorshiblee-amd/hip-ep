// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// convert-torch-to-hip records the graph's input/output shapes on the module,
// the same six attributes convert-onnx-to-hip stamps. Without them
// convert-hip-to-llvm skips building the (ctx, inputs) entry wrapper and
// generate-interface rejects @main_graph, so the C ABI cannot be produced from
// a Torch module.
//
// The shapes come from the !torch.vtensor types, so this has to happen before
// the rewrites replace the function signature.
// ============================================================================

// RUN: hip-mlir-opt %s --split-input-file --hip-add-context-arg --convert-torch-to-hip | FileCheck %s

// The !hip.context prepended by hip-add-context-arg is not a graph input, so
// input_count stays 1.
// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1 : i64
// CHECK-SAME: hipdnn.input_element_sizes = array<i64: 2>
// CHECK-SAME: hipdnn.input_shapes = [array<i64: 1, 128, 32>]
// CHECK-SAME: hipdnn.output_count = 2 : i64
// CHECK-SAME: hipdnn.output_element_sizes = array<i64: 2, 2>
// CHECK-SAME: hipdnn.output_shapes = [array<i64: 1, 128, 32>, array<i64: 32>]
module {
  func.func @main_graph(%arg0: !torch.vtensor<[1,128,32],f16>)
      -> (!torch.vtensor<[1,128,32],f16>, !torch.vtensor<[32],f16>) {
    %alpha = torch.constant.int 1
    %lit = torch.vtensor.literal(dense<1.000000e+00> : tensor<32xf16>) : !torch.vtensor<[32],f16>
    %sum = torch.aten.add.Tensor %arg0, %arg0, %alpha : !torch.vtensor<[1,128,32],f16>, !torch.vtensor<[1,128,32],f16>, !torch.int -> !torch.vtensor<[1,128,32],f16>
    return %sum, %lit : !torch.vtensor<[1,128,32],f16>, !torch.vtensor<[32],f16>
  }
}

// -----

// A dynamic dimension is recorded as the ? sentinel the shape carries; the
// runtime sizes it in-graph.
// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_shapes = [array<i64: -9223372036854775808, 16>]
module {
  func.func @main_graph(%arg0: !torch.vtensor<[?,16],f32>) -> !torch.vtensor<[?,16],f32> {
    return %arg0 : !torch.vtensor<[?,16],f32>
  }
}

// -----

// A module with no @main_graph is left alone: conversion-only tests lower a
// bare function and never reach the C ABI.
// CHECK-NOT: hipdnn.input_count
module {
  func.func @helper(%arg0: !torch.vtensor<[4],f16>) -> !torch.vtensor<[4],f16> {
    return %arg0 : !torch.vtensor<[4],f16>
  }
}
