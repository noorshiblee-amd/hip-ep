// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-torch-to-hip | FileCheck %s

module {
  func.func @sigmoid(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @sigmoid
    %cpu = torch.constant.device "cpu"
    %y = torch.aten.sigmoid %x : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    // CHECK: hip.sigmoid
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],f32>
  }

  func.func @cat(%a: !torch.vtensor<[2,4],f32>, %b: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[5,4],f32> {
    // CHECK-LABEL: func.func @cat
    %list = torch.prim.ListConstruct %a, %b : (!torch.vtensor<[2,4],f32>, !torch.vtensor<[3,4],f32>) -> !torch.list<vtensor>
    %dim = torch.constant.int 0
    %c = torch.aten.cat %list, %dim : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[5,4],f32>
    // CHECK: tensor.concat
    // CHECK-NOT: torch.
    return %c : !torch.vtensor<[5,4],f32>
  }

  func.func @floor_div(%a: !torch.vtensor<[4],f32>, %b: !torch.vtensor<[],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @floor_div
    %str = torch.constant.str "floor"
    %q = torch.aten.div.Tensor_mode %a, %b, %str : !torch.vtensor<[4],f32>, !torch.vtensor<[],f32>, !torch.str -> !torch.vtensor<[4],f32>
    // CHECK: hip.div
    // CHECK: hip.floor
    // CHECK-NOT: torch.
    return %q : !torch.vtensor<[4],f32>
  }
}
