// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: not hip-mlir-opt %s --convert-torch-to-hip 2>&1 | FileCheck %s --check-prefix=NOCTX
// RUN: not hip-mlir-opt %s --hip-add-context-arg --convert-torch-to-hip 2>&1 | FileCheck %s --check-prefix=DTYPE

module {
  // NOCTX: requires !hip.context
  func.func @missing_context(%lhs: !torch.vtensor<[4],f32>, %rhs: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    %alpha = torch.constant.int 1
    %output = torch.aten.add.Tensor %lhs, %rhs, %alpha : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4],f32>
    return %output : !torch.vtensor<[4],f32>
  }

  // i1 is not an integer width the elementwise kernels accept.
  // DTYPE: integer aten.add requires a static i16, i32, or i64 tensor
  func.func @bool_tensor(%lhs: !torch.vtensor<[4],i1>, %rhs: !torch.vtensor<[4],i1>) -> !torch.vtensor<[4],i1> {
    %alpha = torch.constant.int 1
    %output = torch.aten.add.Tensor %lhs, %rhs, %alpha : !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.int -> !torch.vtensor<[4],i1>
    return %output : !torch.vtensor<[4],i1>
  }
}
