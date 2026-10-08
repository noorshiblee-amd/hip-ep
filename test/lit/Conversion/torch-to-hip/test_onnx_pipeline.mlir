// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --torch-to-hip-pipeline --verify-each | FileCheck %s

module {
  func.func @matmul_graph(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.none {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: !hip.context
    // CHECK: hip.quantize_linear
    // CHECK: hip.matmul
    // CHECK: hip.alloc_output
    // CHECK: hip.add
    // CHECK-NOT: torch.
    // CHECK-NOT: "onnx.
    %scale = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.0> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %zp = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<ui8>} : () -> !torch.vtensor<[],ui8>
    %weight = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]> : tensor<3x4xui8>} : () -> !torch.vtensor<[3,4],ui8>
    %bias = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[[1.0, 1.0, 1.0, 1.0]]> : tensor<1x4xf32>} : () -> !torch.vtensor<[1,4],f32>
    %none = torch.constant.none
    %q = torch.operator "onnx.QuantizeLinear"(%arg0, %scale, %zp) : (!torch.vtensor<[2,3],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.none
    %act = torch.operator "onnx.DequantizeLinear"(%q, %scale, %zp) : (!torch.none, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.none
    %w = torch.operator "onnx.DequantizeLinear"(%weight, %scale, %zp) : (!torch.vtensor<[3,4],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.none
    %mm = torch.operator "onnx.MatMul"(%act, %w) : (!torch.none, !torch.none) -> !torch.none
    %sum = torch.operator "onnx.Add"(%mm, %bias) : (!torch.none, !torch.vtensor<[1,4],f32>) -> !torch.none
    return %sum : !torch.none
  }
}
