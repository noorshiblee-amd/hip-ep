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

  // Integer and i1 arguments are accepted for embeddings / masks, but float
  // arithmetic still rejects them.
  // DTYPE: result element type must be f16, f32, or bf16
  func.func @integer_tensor(%lhs: !torch.vtensor<[4],si64>, %rhs: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],si64> {
    %alpha = torch.constant.int 1
    %output = torch.aten.add.Tensor %lhs, %rhs, %alpha : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    return %output : !torch.vtensor<[4],si64>
  }

  func.func @bool_tensor(%lhs: !torch.vtensor<[4],i1>, %rhs: !torch.vtensor<[4],i1>) -> !torch.vtensor<[4],i1> {
    %alpha = torch.constant.int 1
    %output = torch.aten.add.Tensor %lhs, %rhs, %alpha : !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1>, !torch.int -> !torch.vtensor<[4],i1>
    return %output : !torch.vtensor<[4],i1>
  }
}
