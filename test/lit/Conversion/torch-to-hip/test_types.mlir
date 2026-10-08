// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s | FileCheck %s

module {
  func.func @roundtrip(%arg0: !torch.vtensor<[1,?,4],f16>, %arg1: !torch.int, %arg2: !torch.float, %arg3: !torch.bool, %arg4: !torch.vtensor<[],bf16>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @roundtrip
    // CHECK: !torch.vtensor<[1, ?, 4],f16>
    // CHECK: !torch.int
    // CHECK: !torch.float
    // CHECK: !torch.bool
    // CHECK: !torch.vtensor<[],bf16>
    // CHECK: -> !torch.vtensor<[4],f32>
    // CHECK: torch.constant.int 7
    // CHECK: torch.constant.float 1.000000e+00
    // CHECK: torch.vtensor.literal(dense<1.000000e+00> : tensor<4xf32>) : !torch.vtensor<[4],f32>
    // CHECK: torch.aten.add.Tensor {{.*}} : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4],f32>
    %int = torch.constant.int 7
    %float = torch.constant.float 1.0
    %lit = torch.vtensor.literal(dense<1.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
    %sum = torch.aten.add.Tensor %lit, %lit, %int : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4],f32>
    return %sum : !torch.vtensor<[4],f32>
  }

  // An operation this dialect does not spell still round-trips in generic form.
  func.func @generic_unknown(%arg0: !torch.vtensor<[4],f32>, %arg1: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @generic_unknown
    // CHECK: "torch.aten.sub.Tensor"
    %0 = "torch.aten.sub.Tensor"(%arg0, %arg1) : (!torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
    return %0 : !torch.vtensor<[4],f32>
  }
}
