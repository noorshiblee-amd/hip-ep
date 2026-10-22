// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s | FileCheck %s

module {
  func.func @roundtrip_none(%arg0: !torch.none) {
    // CHECK-LABEL: func.func @roundtrip_none
    // CHECK-SAME: !torch.none
    // CHECK: torch.constant.none
    %none = torch.constant.none
    return
  }
}
