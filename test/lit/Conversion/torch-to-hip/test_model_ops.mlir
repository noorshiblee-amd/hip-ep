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

  func.func @embedding(%w: !torch.vtensor<[8,4],f32>, %ids: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,4],f32> {
    // CHECK-LABEL: func.func @embedding
    %pad = torch.constant.int -1
    %false = torch.constant.bool false
    %y = torch.aten.embedding %w, %ids, %pad, %false, %false : !torch.vtensor<[8,4],f32>, !torch.vtensor<[2],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[2,4],f32>
    // CHECK: hip.gather
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[2,4],f32>
  }

  func.func @mean_and_square(%x: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,1],f32> {
    // CHECK-LABEL: func.func @mean_and_square
    %two = torch.constant.int 2
    %sq = torch.aten.pow.Tensor_Scalar %x, %two : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
    %dim = torch.constant.int -1
    %dims = torch.prim.ListConstruct %dim : (!torch.int) -> !torch.list<int>
    %keep = torch.constant.bool true
    %none = torch.constant.none
    %m = torch.aten.mean.dim %sq, %dims, %keep, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,1],f32>
    // CHECK: hip.mul
    // CHECK: hip.reduce_mean
    // CHECK-NOT: torch.
    return %m : !torch.vtensor<[2,1],f32>
  }

  func.func @gelu_exact(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @gelu_exact
    %none = torch.constant.str "none"
    %y = torch.aten.gelu %x, %none : !torch.vtensor<[4],f32>, !torch.str -> !torch.vtensor<[4],f32>
    // CHECK: hip.gelu
    // CHECK-NOT: approximate
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],f32>
  }

  func.func @gelu_tanh(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @gelu_tanh
    %tanh = torch.constant.str "tanh"
    %y = torch.aten.gelu %x, %tanh : !torch.vtensor<[4],f32>, !torch.str -> !torch.vtensor<[4],f32>
    // CHECK: hip.gelu{{.*}}approximate = "tanh"
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],f32>
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
