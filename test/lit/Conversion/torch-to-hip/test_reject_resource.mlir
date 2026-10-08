// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: not hip-mlir-opt %s 2>&1 | FileCheck %s

func.func @resource() -> !torch.vtensor<[4],f32> {
  // CHECK: dense elements attribute
  %0 = torch.vtensor.literal(dense_resource<blob> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}
