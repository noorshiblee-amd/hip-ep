// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: not hip-mlir-opt %s --hip-add-context-arg --convert-torch-to-hip 2>&1 | FileCheck %s

module {
  func.func @generic_unknown(%lhs: !torch.vtensor<[4],f32>, %rhs: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK: no Torch-to-HIP lowering
    // CHECK: torch.aten.sub.Tensor
    %alpha = torch.constant.int 1
    %output = "torch.aten.sub.Tensor"(%lhs, %rhs, %alpha) : (!torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int) -> !torch.vtensor<[4],f32>
    return %output : !torch.vtensor<[4],f32>
  }
}
