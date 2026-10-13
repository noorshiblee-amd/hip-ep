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

  func.func @tanh(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @tanh
    %y = torch.aten.tanh %x : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    // CHECK: hip.tanh
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

  func.func @sub_float_index(%x: !torch.vtensor<[4],f32>, %ids: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @sub_float_index
    %one = torch.constant.int 1
    %y = torch.aten.sub.Tensor %x, %ids, %one : !torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],f32>
    // CHECK: hip.cast
    // CHECK: hip.sub
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],f32>
  }

  func.func @clamp_min(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @clamp_min
    %zero = torch.constant.float 0.000000e+00
    %none = torch.constant.none
    %y = torch.aten.clamp %x, %zero, %none : !torch.vtensor<[4],f32>, !torch.float, !torch.none -> !torch.vtensor<[4],f32>
    // CHECK: hip.max
    // CHECK-NOT: hip.min
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],f32>
  }

  func.func @clamp_index(%x: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],si64> {
    // CHECK-LABEL: func.func @clamp_index
    %none = torch.constant.none
    %hi = torch.constant.int 26
    %y = torch.aten.clamp %x, %none, %hi : !torch.vtensor<[4],si64>, !torch.none, !torch.int -> !torch.vtensor<[4],si64>
    // CHECK: hip.min
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],si64>
  }

  func.func @pad_window(%x: !torch.vtensor<[1,2,2,4],f32>) -> !torch.vtensor<[1,4,4,4],f32> {
    // CHECK-LABEL: func.func @pad_window
    %c0 = torch.constant.int 0
    %c2 = torch.constant.int 2
    %pads = torch.prim.ListConstruct %c0, %c0, %c0, %c2, %c0, %c2 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %zero = torch.constant.float 0.000000e+00
    %y = torch.aten.constant_pad_nd %x, %pads, %zero : !torch.vtensor<[1,2,2,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,4,4,4],f32>
    // CHECK: hip.pad
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,4,4,4],f32>
  }

  func.func @index_full_slice(%x: !torch.vtensor<[1,4,8],f32>, %ids: !torch.vtensor<[8],si64>) -> !torch.vtensor<[1,4,8],f32> {
    // CHECK-LABEL: func.func @index_full_slice
    %none0 = torch.constant.none
    %none1 = torch.constant.none
    %list = torch.prim.ListConstruct %none0, %none1, %ids : (!torch.none, !torch.none, !torch.vtensor<[8],si64>) -> !torch.list<optional<vtensor>>
    %y = torch.aten.index.Tensor %x, %list : !torch.vtensor<[1,4,8],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[1,4,8],f32>
    // CHECK: hip.gather{{.*}}axis = 2
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,4,8],f32>
  }

  func.func @l2_norm(%x: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[1,1],f32> {
    // CHECK-LABEL: func.func @l2_norm
    %two = torch.constant.int 2
    %dim = torch.constant.int -1
    %dims = torch.prim.ListConstruct %dim : (!torch.int) -> !torch.list<int>
    %keep = torch.constant.bool true
    %none = torch.constant.none
    %y = torch.aten.linalg_vector_norm %x, %two, %dims, %keep, %none : !torch.vtensor<[1,4],f32>, !torch.int, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1],f32>
    // CHECK: hip.mul
    // CHECK: hip.reduce_sum
    // CHECK: hip.sqrt
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,1],f32>
  }

  func.func @squeeze_batch(%x: !torch.vtensor<[1,4,8],f32>) -> !torch.vtensor<[4,8],f32> {
    // CHECK-LABEL: func.func @squeeze_batch
    %dim = torch.constant.int 0
    %dims = torch.prim.ListConstruct %dim : (!torch.int) -> !torch.list<int>
    %y = torch.prims.squeeze %x, %dims : !torch.vtensor<[1,4,8],f32>, !torch.list<int> -> !torch.vtensor<[4,8],f32>
    // CHECK: tensor.reshape
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4,8],f32>
  }

  func.func @abs_sign(%x: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],si64> {
    // CHECK-LABEL: func.func @abs_sign
    %signed = torch.aten.sign %x : !torch.vtensor<[4],si64> -> !torch.vtensor<[4],si64>
    %y = torch.aten.abs %signed : !torch.vtensor<[4],si64> -> !torch.vtensor<[4],si64>
    // CHECK: hip.sign
    // CHECK: hip.abs
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],si64>
  }

  func.func @log_ceil_sqrt(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @log_ceil_sqrt
    %logged = torch.aten.log %x : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    %ceiled = torch.aten.ceil %logged : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    %y = torch.aten.sqrt %ceiled : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    // CHECK: hip.log
    // CHECK: hip.ceil
    // CHECK: hip.sqrt
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],f32>
  }

  func.func @bitwise_not(%x: !torch.vtensor<[4],i1>) -> !torch.vtensor<[4],i1> {
    // CHECK-LABEL: func.func @bitwise_not
    %y = torch.aten.bitwise_not %x : !torch.vtensor<[4],i1> -> !torch.vtensor<[4],i1>
    // CHECK: hip.not
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],i1>
  }

  func.func @order_scalar(%x: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],i1> {
    // CHECK-LABEL: func.func @order_scalar
    %one = torch.constant.int 1
    %gt = torch.aten.gt.Scalar %x, %one : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],i1>
    %lt = torch.aten.lt.Scalar %x, %one : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],i1>
    %le = torch.aten.le.Scalar %x, %one : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],i1>
    %both = torch.aten.bitwise_and.Tensor %gt, %lt : !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1> -> !torch.vtensor<[4],i1>
    %y = torch.aten.bitwise_and.Tensor %both, %le : !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1> -> !torch.vtensor<[4],i1>
    // CHECK: hip.less
    // CHECK: hip.not
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],i1>
  }

  func.func @gather_last(%x: !torch.vtensor<[2,4],f32>, %index: !torch.vtensor<[2,2],si64>) -> !torch.vtensor<[2,2],f32> {
    // CHECK-LABEL: func.func @gather_last
    %dim = torch.constant.int -1
    %sparse = torch.constant.bool false
    %y = torch.aten.gather %x, %dim, %index, %sparse : !torch.vtensor<[2,4],f32>, !torch.int, !torch.vtensor<[2,2],si64>, !torch.bool -> !torch.vtensor<[2,2],f32>
    // CHECK: hip.gather_elements
    // CHECK-SAME: axis = 1
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[2,2],f32>
  }

  func.func @clone_none(%x: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],si64> {
    // CHECK-LABEL: func.func @clone_none
    %none = torch.constant.none
    %y = torch.aten.clone %x, %none : !torch.vtensor<[4],si64>, !torch.none -> !torch.vtensor<[4],si64>
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],si64>
  }

  func.func @mul_i64(%a: !torch.vtensor<[1,4],si64>, %b: !torch.vtensor<[4,1],si64>) -> !torch.vtensor<[4,4],si64> {
    // CHECK-LABEL: func.func @mul_i64
    %y = torch.aten.mul.Tensor %a, %b : !torch.vtensor<[1,4],si64>, !torch.vtensor<[4,1],si64> -> !torch.vtensor<[4,4],si64>
    // CHECK: hip.mul
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4,4],si64>
  }
}
