// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --torch-to-hip-pipeline --verify-each | FileCheck %s

module {
  func.func @main_graph(%lhs: !torch.vtensor<[4],f32>, %rhs: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-SAME: !hip.context
    // CHECK: hip.add
    // CHECK: hip.alloc_output
    // CHECK-NOT: torch.
    %alpha = torch.constant.int 1
    %output = torch.aten.add.Tensor %lhs, %rhs, %alpha : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4],f32>
    return %output : !torch.vtensor<[4],f32>
  }
}
