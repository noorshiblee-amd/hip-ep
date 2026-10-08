// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A single quantized matmul from a torch-mlir ONNX import: dense_resource
// constants, an unused torch.constant.none, !torch.none results, and an
// explicit tensor return. The trailing quantize lets the shared fusion
// rewrite the dequantize/matmul/quantize chain.
// RUN: hip-mlir-opt %s --torch-to-hip-pipeline --verify-each | FileCheck %s

module {
  func.func @matmul_graph(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,4],f32> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: !hip.context
    // CHECK-SAME: memref<2x3xf32>
    // CHECK-SAME: memref<2x4xf32>
    // CHECK: hip.quantize_linear
    // CHECK: hip.qmatmul
    // CHECK: hip.alloc_output
    // CHECK: hip.dequantize_linear
    // CHECK-NOT: torch.
    // CHECK-NOT: "onnx.
    // CHECK-NOT: dense_resource
    %scale = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<scale> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %zp16 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<zp16> : tensor<ui16>} : () -> !torch.vtensor<[],ui16>
    %weight = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<weight> : tensor<3x4xui8>} : () -> !torch.vtensor<[3,4],ui8>
    %zp8 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<zp8> : tensor<ui8>} : () -> !torch.vtensor<[],ui8>
    %none = torch.constant.none
    %q = torch.operator "onnx.QuantizeLinear"(%arg0, %scale, %zp16) : (!torch.vtensor<[2,3],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui16>) -> !torch.none
    %act = torch.operator "onnx.DequantizeLinear"(%q, %scale, %zp16) : (!torch.none, !torch.vtensor<[],f32>, !torch.vtensor<[],ui16>) -> !torch.none
    %w = torch.operator "onnx.DequantizeLinear"(%weight, %scale, %zp8) : (!torch.vtensor<[3,4],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.none
    %mm = torch.operator "onnx.MatMul"(%act, %w) : (!torch.none, !torch.none) -> !torch.none
    %qm = torch.operator "onnx.QuantizeLinear"(%mm, %scale, %zp16) : (!torch.none, !torch.vtensor<[],f32>, !torch.vtensor<[],ui16>) -> !torch.none
    %out = torch.operator "onnx.DequantizeLinear"(%qm, %scale, %zp16) : (!torch.none, !torch.vtensor<[],f32>, !torch.vtensor<[],ui16>) -> !torch.vtensor<[2,4],f32>
    return %out : !torch.vtensor<[2,4],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      scale: "0x080000000000803F",
      zp16: "0x080000000000",
      zp8: "0x0800000000",
      weight: "0x08000000010101010101010101010101"
    }
  }
#-}
