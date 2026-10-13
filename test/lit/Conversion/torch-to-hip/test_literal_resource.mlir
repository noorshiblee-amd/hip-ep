// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-torch-to-hip | FileCheck %s

module {
  func.func @resource() -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @resource
    // CHECK: hip.constant
    // CHECK-SAME: dense<1.000000e+00> : tensor<4xf32>
    // CHECK-NOT: dense_resource
    %0 = torch.vtensor.literal(dense_resource<blob> : tensor<4xf32>) : !torch.vtensor<[4],f32>
    return %0 : !torch.vtensor<[4],f32>
  }

  func.func @position_ids() -> !torch.vtensor<[2],si64> {
    // CHECK-LABEL: func.func @position_ids
    // CHECK: hip.constant
    // CHECK-SAME: dense<[0, 1]> : tensor<2xi64>
    %ids = torch.vtensor.literal(dense<[0, 1]> : tensor<2xsi64>) : !torch.vtensor<[2],si64>
    return %ids : !torch.vtensor<[2],si64>
  }

  func.func @missing_position_ids() -> !torch.vtensor<[2],si64> {
    // CHECK-LABEL: func.func @missing_position_ids
    // CHECK: hip.constant
    // CHECK-SAME: dense<0> : tensor<2xi64>
    %missing = torch.vtensor.literal(dense_resource<missing_ids> : tensor<2xsi64>) : !torch.vtensor<[2],si64>
    return %missing : !torch.vtensor<[2],si64>
  }
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000000000803F"
    }
  }
#-}
