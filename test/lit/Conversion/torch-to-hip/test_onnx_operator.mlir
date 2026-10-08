// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-torch-to-hip | FileCheck %s

module {
  func.func @matmul_graph(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.none {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: !hip.context
    // CHECK-NOT: torch.
    // CHECK: "onnx.Constant"() {{.*}}value = dense<0> : tensor<ui8>
    // CHECK: "onnx.QuantizeLinear"
    // CHECK: "onnx.DequantizeLinear"
    // CHECK: "onnx.MatMul"
    // CHECK: "onnx.Add"
    // CHECK: tensor<2x4xf32>
    // CHECK-NOT: torch.
    // CHECK-NOT: dense_resource
    %scale = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.0> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %zp = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<zp> : tensor<ui8>} : () -> !torch.vtensor<[],ui8>
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

{-#
  dialect_resources: {
    builtin: {
      zp: "0x0800000000"
    }
  }
#-}
