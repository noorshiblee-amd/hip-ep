// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: not hip-mlir-opt %s 2>&1 | FileCheck %s

module {
  // CHECK: aten.gelu
  func.func @pretty_unknown(%lhs: !torch.vtensor<[4],f32>, %rhs: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    %output = torch.aten.gelu %lhs : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    return %output : !torch.vtensor<[4],f32>
  }
}
