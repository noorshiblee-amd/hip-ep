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

  func.func @cat_neg(%a: !torch.vtensor<[1,2,3],f32>, %b: !torch.vtensor<[1,4,3],f32>) -> !torch.vtensor<[1,6,3],f32> {
    // CHECK-LABEL: func.func @cat_neg
    %list = torch.prim.ListConstruct %a, %b : (!torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,4,3],f32>) -> !torch.list<vtensor>
    %dim = torch.constant.int -2
    %c = torch.aten.cat %list, %dim : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,6,3],f32>
    // CHECK: tensor.concat dim(1)
    // CHECK-NOT: torch.
    return %c : !torch.vtensor<[1,6,3],f32>
  }

  func.func @cat_empty(%empty: !torch.vtensor<[0],f32>, %x: !torch.vtensor<[1,2,3,4],f32>) -> !torch.vtensor<[1,2,3,4],f32> {
    // CHECK-LABEL: func.func @cat_empty
    // CHECK-SAME: %[[X:[a-z0-9]+]]: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
    %list = torch.prim.ListConstruct %empty, %x : (!torch.vtensor<[0],f32>, !torch.vtensor<[1,2,3,4],f32>) -> !torch.list<vtensor>
    %dim = torch.constant.int -2
    %c = torch.aten.cat %list, %dim : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
    // CHECK: return %[[X]]
    return %c : !torch.vtensor<[1,2,3,4],f32>
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

  func.func @pad_crop(%x: !torch.vtensor<[1,30,30],f32>) -> !torch.vtensor<[1,28,28],f32> {
    // CHECK-LABEL: func.func @pad_crop
    %m1 = torch.constant.int -1
    %pads = torch.prim.ListConstruct %m1, %m1, %m1, %m1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %zero = torch.constant.float 0.000000e+00
    %y = torch.aten.constant_pad_nd %x, %pads, %zero : !torch.vtensor<[1,30,30],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,28,28],f32>
    // CHECK: tensor.extract_slice
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,28,28],f32>
  }

  func.func @adaptive_max_pool(%x: !torch.vtensor<[1,2,8,8],f32>) -> !torch.vtensor<[1,2,2,1],f32> {
    // CHECK-LABEL: func.func @adaptive_max_pool
    %h = torch.constant.int 2
    %w = torch.constant.int 1
    %size = torch.prim.ListConstruct %h, %w : (!torch.int, !torch.int) -> !torch.list<int>
    %v, %i = torch.aten.adaptive_max_pool2d %x, %size : !torch.vtensor<[1,2,8,8],f32>, !torch.list<int> -> !torch.vtensor<[1,2,2,1],f32>, !torch.vtensor<[1,2,2,1],si64>
    // CHECK: hip.pool
    // CHECK-SAME: pool_mode = 1
    // CHECK-NOT: torch.
    return %v : !torch.vtensor<[1,2,2,1],f32>
  }

  func.func @unfold(%x: !torch.vtensor<[1,2,8],f32>) -> !torch.vtensor<[1,2,3,4],f32> {
    // CHECK-LABEL: func.func @unfold
    %dim = torch.constant.int 2
    %size = torch.constant.int 4
    %step = torch.constant.int 2
    %y = torch.aten.unfold %x, %dim, %size, %step : !torch.vtensor<[1,2,8],f32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
    // CHECK: tensor.extract_slice
    // CHECK: tensor.concat
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,2,3,4],f32>
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

  func.func @index_spatial(%x: !torch.vtensor<[1,2,4,4],f32>, %h: !torch.vtensor<[3,1],si64>, %w: !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,2,3,3],f32> {
    // CHECK-LABEL: func.func @index_spatial
    %none0 = torch.constant.none
    %none1 = torch.constant.none
    %list = torch.prim.ListConstruct %none0, %none1, %h, %w : (!torch.none, !torch.none, !torch.vtensor<[3,1],si64>, !torch.vtensor<[3],si64>) -> !torch.list<optional<vtensor>>
    %y = torch.aten.index.Tensor %x, %list : !torch.vtensor<[1,2,4,4],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[1,2,3,3],f32>
    // CHECK: hip.gather_nd
    // CHECK-SAME: batch_dims = 2
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,2,3,3],f32>
  }

  func.func @index_put(%x: !torch.vtensor<[1,2,4,4],f32>, %h: !torch.vtensor<[3,1],si64>, %w: !torch.vtensor<[3],si64>, %v: !torch.vtensor<[1,2,3,3],f32>) -> !torch.vtensor<[1,2,4,4],f32> {
    // CHECK-LABEL: func.func @index_put
    %none0 = torch.constant.none
    %none1 = torch.constant.none
    %list = torch.prim.ListConstruct %none0, %none1, %h, %w : (!torch.none, !torch.none, !torch.vtensor<[3,1],si64>, !torch.vtensor<[3],si64>) -> !torch.list<optional<vtensor>>
    %acc = torch.constant.bool true
    %y = torch.aten.index_put %x, %list, %v, %acc : !torch.vtensor<[1,2,4,4],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,2,3,3],f32>, !torch.bool -> !torch.vtensor<[1,2,4,4],f32>
    // CHECK: hip.scatter_nd
    // CHECK-SAME: reduction = "add"
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,2,4,4],f32>
  }

  func.func @num_to_tensor_float() -> !torch.vtensor<[],f32> {
    // CHECK-LABEL: func.func @num_to_tensor_float
    %one = torch.constant.float 1.000000e+00
    %scalar = torch.prim.NumToTensor.Scalar %one : !torch.float -> !torch.vtensor<[],f32>
    // CHECK: tensor.splat
    // CHECK-NOT: torch.
    return %scalar : !torch.vtensor<[],f32>
  }

  func.func @upsample_scales(%x: !torch.vtensor<[1,4,2,2],f32>) -> !torch.vtensor<[1,4,4,4],f32> {
    // CHECK-LABEL: func.func @upsample_scales
    %none = torch.constant.none
    %two = torch.constant.float 2.000000e+00
    %two_b = torch.constant.float 2.000000e+00
    %scales = torch.prim.ListConstruct %two, %two_b : (!torch.float, !torch.float) -> !torch.list<float>
    %y = torch.aten.upsample_nearest2d.vec %x, %none, %scales : !torch.vtensor<[1,4,2,2],f32>, !torch.none, !torch.list<float> -> !torch.vtensor<[1,4,4,4],f32>
    // CHECK: hip.resize
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,4,4,4],f32>
  }

  func.func @slice_scatter(%base: !torch.vtensor<[197,197],si64>, %src: !torch.vtensor<[196,197],si64>) -> !torch.vtensor<[197,197],si64> {
    // CHECK-LABEL: func.func @slice_scatter
    %dim = torch.constant.int 0
    %start = torch.constant.int 1
    %end = torch.constant.int 9223372036854775807
    %step = torch.constant.int 1
    %y = torch.aten.slice_scatter %base, %src, %dim, %start, %end, %step : !torch.vtensor<[197,197],si64>, !torch.vtensor<[196,197],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[197,197],si64>
    // CHECK: tensor.insert_slice
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[197,197],si64>
  }

  func.func @permute_i64(%x: !torch.vtensor<[2,3,4],si64>) -> !torch.vtensor<[3,4,2],si64> {
    // CHECK-LABEL: func.func @permute_i64
    %c1 = torch.constant.int 1
    %c2 = torch.constant.int 2
    %c0 = torch.constant.int 0
    %perm = torch.prim.ListConstruct %c1, %c2, %c0 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %y = torch.aten.permute %x, %perm : !torch.vtensor<[2,3,4],si64>, !torch.list<int> -> !torch.vtensor<[3,4,2],si64>
    // CHECK: hip.transpose
    // CHECK-SAME: perm = [1, 2, 0]
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[3,4,2],si64>
  }

  func.func @slice_empty_tail(%x: !torch.vtensor<[1,96],f32>) -> !torch.vtensor<[1,0],f32> {
    // CHECK-LABEL: func.func @slice_empty_tail
    %c1 = torch.constant.int 1
    %c96 = torch.constant.int 96
    %end = torch.constant.int 9223372036854775807
    %step = torch.constant.int 1
    %y = torch.aten.slice.Tensor %x, %c1, %c96, %end, %step : !torch.vtensor<[1,96],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,0],f32>
    // CHECK: tensor.extract_slice %{{.*}}[0, 0] [1, 0] [1, 1]
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,0],f32>
  }

  func.func @pow_tensor_tensor(%e: !torch.vtensor<[4],si32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @pow_tensor_tensor
    %base = torch.vtensor.literal(dense<0.5> : tensor<f32>) : !torch.vtensor<[],f32>
    %y = torch.aten.pow.Tensor_Tensor %base, %e : !torch.vtensor<[],f32>, !torch.vtensor<[4],si32> -> !torch.vtensor<[4],f32>
    // CHECK: hip.cast
    // CHECK: hip.log
    // CHECK: hip.mul
    // CHECK: hip.exp
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],f32>
  }

  func.func @mask_index_last(%x: !torch.vtensor<[1,4,8],f32>, %mask: !torch.vtensor<[1,4],i1>) -> !torch.vtensor<[1,8],f32> {
    // CHECK-LABEL: func.func @mask_index_last
    %c1 = torch.constant.int 1
    %c0 = torch.constant.int 0
    %cm1 = torch.constant.int -1
    %c8 = torch.constant.int 8
    %false = torch.constant.bool false
    %none = torch.constant.none
    %dims = torch.prim.ListConstruct %c1 : (!torch.int) -> !torch.list<int>
    // CHECK: hip.cast
    // CHECK: hip.reduce_sum
    %count = torch.aten.sum.dim_IntList %mask, %dims, %false, %none : !torch.vtensor<[1,4],i1>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1],si64>
    // CHECK: tensor.cast
    %u0, %u1, %u2 = torch.aten.unique_consecutive %count, %false, %false, %none : !torch.vtensor<[1],si64>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?],si64>, !torch.vtensor<[0],si64>, !torch.vtensor<[0],si64>
    %n = torch.aten.size.int %u0, %c0 : !torch.vtensor<[?],si64>, !torch.int -> !torch.int
    // CHECK: arith.cmpi eq
    %ok = torch.aten.eq.int %n, %c1 : !torch.int, !torch.int -> !torch.bool
    %oki = torch.aten.Int.bool %ok : !torch.bool -> !torch.int
    %msg = torch.constant.str "u0 == 1"
    torch.aten._assert_scalar %oki, %msg : !torch.int, !torch.str
    %idx = torch.prim.ListConstruct %mask : (!torch.vtensor<[1,4],i1>) -> !torch.list<optional<vtensor>>
    // CHECK: hip.nonzero
    // CHECK: hip.readback_dim
    // CHECK: hip.compress
    // CHECK-SAME: outs({{.*}} : tensor<?x8xf32>)
    %rows = torch.aten.index.Tensor %x, %idx : !torch.vtensor<[1,4,8],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[?,8],f32>
    %shape = torch.prim.ListConstruct %c1, %cm1, %c8 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    // CHECK: tensor.reshape {{.*}} -> tensor<1x?x8xf32>
    %v = torch.aten.view %rows, %shape : !torch.vtensor<[?,8],f32>, !torch.list<int> -> !torch.vtensor<[1,?,8],f32>
    // CHECK: tensor.dim
    // CHECK: arith.subi
    // CHECK: tensor.extract_slice
    %last = torch.aten.select.int %v, %c1, %cm1 : !torch.vtensor<[1,?,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,8],f32>
    // CHECK-NOT: torch.
    return %last : !torch.vtensor<[1,8],f32>
  }

  func.func @remainder_scalar(%x: !torch.vtensor<[1,16],si64>) -> !torch.vtensor<[1,16],si64> {
    // CHECK-LABEL: func.func @remainder_scalar
    %m = torch.constant.int 16384
    %y = torch.aten.remainder.Scalar %x, %m : !torch.vtensor<[1,16],si64>, !torch.int -> !torch.vtensor<[1,16],si64>
    // CHECK: hip.mod
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,16],si64>
  }

  func.func @log1p(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @log1p
    %y = torch.aten.log1p %x : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    // CHECK: hip.add
    // CHECK: hip.log
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],f32>
  }

  func.func @index_select(%t: !torch.vtensor<[10,4],f32>, %ids: !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,4],f32> {
    // CHECK-LABEL: func.func @index_select
    %c0 = torch.constant.int 0
    %y = torch.aten.index_select %t, %c0, %ids : !torch.vtensor<[10,4],f32>, !torch.int, !torch.vtensor<[3],si64> -> !torch.vtensor<[3,4],f32>
    // CHECK: hip.gather
    // CHECK-SAME: tensor<10x4xf32>, tensor<3xi64>
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[3,4],f32>
  }

  func.func @unsqueeze_rank0(%x: !torch.vtensor<[],si64>) -> !torch.vtensor<[1],si64> {
    // CHECK-LABEL: func.func @unsqueeze_rank0
    %c0 = torch.constant.int 0
    %y = torch.aten.unsqueeze %x, %c0 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    // CHECK: tensor.reshape
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1],si64>
  }

  func.func @sum_i64(%x: !torch.vtensor<[2,3,4],si64>) -> !torch.vtensor<[2,3],si64> {
    // CHECK-LABEL: func.func @sum_i64
    %cm1 = torch.constant.int -1
    %dims = torch.prim.ListConstruct %cm1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %y = torch.aten.sum.dim_IntList %x, %dims, %false, %none : !torch.vtensor<[2,3,4],si64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,3],si64>
    // CHECK: hip.reduce_sum
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[2,3],si64>
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

  func.func @l2_norm_dtype(%x: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[1,1],f32> {
    // CHECK-LABEL: func.func @l2_norm_dtype
    %two = torch.constant.int 2
    %dim = torch.constant.int -1
    %dims = torch.prim.ListConstruct %dim : (!torch.int) -> !torch.list<int>
    %keep = torch.constant.bool true
    %f32 = torch.constant.int 6
    %y = torch.aten.linalg_vector_norm %x, %two, %dims, %keep, %f32 : !torch.vtensor<[1,4],f32>, !torch.int, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1],f32>
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

  func.func @minimum(%a: !torch.vtensor<[16,16],si64>, %b: !torch.vtensor<[16,16],si64>) -> !torch.vtensor<[16,16],si64> {
    // CHECK-LABEL: func.func @minimum
    %y = torch.aten.minimum %a, %b : !torch.vtensor<[16,16],si64>, !torch.vtensor<[16,16],si64> -> !torch.vtensor<[16,16],si64>
    // CHECK: hip.min
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[16,16],si64>
  }

  func.func @log_ceil_sqrt(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @log_ceil_sqrt
    %logged = torch.aten.log %x : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    %ceiled = torch.aten.ceil %logged : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    %floored = torch.aten.floor %ceiled : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    %y = torch.aten.sqrt %floored : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    // CHECK: hip.log
    // CHECK: hip.ceil
    // CHECK: hip.floor
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

  func.func @add_tensor_i64(%a: !torch.vtensor<[16,16],si64>, %b: !torch.vtensor<[16,16],si64>) -> !torch.vtensor<[16,16],si64> {
    // CHECK-LABEL: func.func @add_tensor_i64
    %one = torch.constant.int 1
    %y = torch.aten.add.Tensor %a, %b, %one : !torch.vtensor<[16,16],si64>, !torch.vtensor<[16,16],si64>, !torch.int -> !torch.vtensor<[16,16],si64>
    // CHECK: hip.add
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[16,16],si64>
  }

  func.func @add_mul_i32(%x: !torch.vtensor<[1,4],si32>, %mask: !torch.vtensor<[1,4],si32>) -> !torch.vtensor<[1,4],si32> {
    // CHECK-LABEL: func.func @add_mul_i32
    %one = torch.constant.int 1
    %sum = torch.aten.add.Scalar %x, %one, %one : !torch.vtensor<[1,4],si32>, !torch.int, !torch.int -> !torch.vtensor<[1,4],si32>
    %y = torch.aten.mul.Tensor %sum, %mask : !torch.vtensor<[1,4],si32>, !torch.vtensor<[1,4],si32> -> !torch.vtensor<[1,4],si32>
    // CHECK: hip.add
    // CHECK: hip.mul
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,4],si32>
  }

  func.func @mul_i64(%a: !torch.vtensor<[1,4],si64>, %b: !torch.vtensor<[4,1],si64>) -> !torch.vtensor<[4,4],si64> {
    // CHECK-LABEL: func.func @mul_i64
    %y = torch.aten.mul.Tensor %a, %b : !torch.vtensor<[1,4],si64>, !torch.vtensor<[4,1],si64> -> !torch.vtensor<[4,4],si64>
    // CHECK: hip.mul
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4,4],si64>
  }

  func.func @mul_scalar_i64(%x: !torch.vtensor<[16,16],si64>) -> !torch.vtensor<[16,16],si64> {
    // CHECK-LABEL: func.func @mul_scalar_i64
    %sixteen = torch.constant.int 16
    %y = torch.aten.mul.Scalar %x, %sixteen : !torch.vtensor<[16,16],si64>, !torch.int -> !torch.vtensor<[16,16],si64>
    // CHECK: hip.mul
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[16,16],si64>
  }

  func.func @gt_tensor(%a: !torch.vtensor<[1,4],si64>, %b: !torch.vtensor<[4,1],si64>) -> !torch.vtensor<[4,4],i1> {
    // CHECK-LABEL: func.func @gt_tensor
    %y = torch.aten.gt.Tensor %a, %b : !torch.vtensor<[1,4],si64>, !torch.vtensor<[4,1],si64> -> !torch.vtensor<[4,4],i1>
    // CHECK: hip.less
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4,4],i1>
  }

  func.func @topk(%x: !torch.vtensor<[2,4],bf16>) -> !torch.vtensor<[2,1],si64> {
    // CHECK-LABEL: func.func @topk
    %k = torch.constant.int 1
    %dim = torch.constant.int -1
    %flag = torch.constant.bool true
    %values, %indices = torch.aten.topk %x, %k, %dim, %flag, %flag : !torch.vtensor<[2,4],bf16>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[2,1],bf16>, !torch.vtensor<[2,1],si64>
    // CHECK: hip.top_k
    // CHECK-NOT: torch.
    return %indices : !torch.vtensor<[2,1],si64>
  }

  func.func @max_dim(%x: !torch.vtensor<[2,4],bf16>) -> !torch.vtensor<[2],si64> {
    // CHECK-LABEL: func.func @max_dim
    %dim = torch.constant.int -1
    %keep = torch.constant.bool false
    %values, %indices = torch.aten.max.dim %x, %dim, %keep : !torch.vtensor<[2,4],bf16>, !torch.int, !torch.bool -> !torch.vtensor<[2],bf16>, !torch.vtensor<[2],si64>
    // CHECK: hip.top_k
    // CHECK: tensor.reshape
    // CHECK-NOT: torch.
    return %indices : !torch.vtensor<[2],si64>
  }

  func.func @argmax(%x: !torch.vtensor<[1,16],si32>) -> !torch.vtensor<[1],si64> {
    // CHECK-LABEL: func.func @argmax
    %dim = torch.constant.int -1
    %keep = torch.constant.bool false
    %indices = torch.aten.argmax %x, %dim, %keep : !torch.vtensor<[1,16],si32>, !torch.int, !torch.bool -> !torch.vtensor<[1],si64>
    // CHECK: hip.top_k
    // CHECK: tensor.reshape
    // CHECK-NOT: torch.
    return %indices : !torch.vtensor<[1],si64>
  }

  func.func @exp(%x: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
    // CHECK-LABEL: func.func @exp
    %y = torch.aten.exp %x : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    // CHECK: hip.exp
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[],f32>
  }

  func.func @split_sizes(%x: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,2],f32> {
    // CHECK-LABEL: func.func @split_sizes
    %two = torch.constant.int 2
    %sizes = torch.prim.ListConstruct %two, %two : (!torch.int, !torch.int) -> !torch.list<int>
    %dim = torch.constant.int -1
    %list = torch.aten.split_with_sizes %x, %sizes, %dim : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.int -> !torch.list<vtensor>
    %lo, %hi = torch.prim.ListUnpack %list : !torch.list<vtensor> -> !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>
    %alpha = torch.constant.int 1
    %y = torch.aten.add.Tensor %lo, %hi, %alpha : !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>, !torch.int -> !torch.vtensor<[2,2],f32>
    // CHECK: tensor.extract_slice
    // CHECK: tensor.extract_slice
    // CHECK: hip.add
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[2,2],f32>
  }

  func.func @copy(%self: !torch.vtensor<[1,2],f32>, %src: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,2],f32> {
    // CHECK-LABEL: func.func @copy
    // CHECK-SAME: %[[SRC:[a-z0-9]+]]: tensor<1x2xf32>) -> tensor<1x2xf32>
    %false = torch.constant.bool false
    %y = torch.aten.copy %self, %src, %false : !torch.vtensor<[1,2],f32>, !torch.vtensor<[1,2],f32>, !torch.bool -> !torch.vtensor<[1,2],f32>
    // CHECK: return %[[SRC]]
    return %y : !torch.vtensor<[1,2],f32>
  }

  func.func @cat_mixed(%a: !torch.vtensor<[1,1],si32>, %b: !torch.vtensor<[1,1],f32>) -> !torch.vtensor<[1,2],f32> {
    // CHECK-LABEL: func.func @cat_mixed
    %list = torch.prim.ListConstruct %a, %b : (!torch.vtensor<[1,1],si32>, !torch.vtensor<[1,1],f32>) -> !torch.list<vtensor>
    %dim = torch.constant.int 1
    %c = torch.aten.cat %list, %dim : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2],f32>
    // CHECK: hip.cast
    // CHECK: tensor.concat
    // CHECK-NOT: torch.
    return %c : !torch.vtensor<[1,2],f32>
  }

  func.func @repeat_rank(%x: !torch.vtensor<[5,4],f32>) -> !torch.vtensor<[1,2,5,4],f32> {
    // CHECK-LABEL: func.func @repeat_rank
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %reps = torch.prim.ListConstruct %int1, %int2, %int1, %int1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %y = torch.aten.repeat %x, %reps : !torch.vtensor<[5,4],f32>, !torch.list<int> -> !torch.vtensor<[1,2,5,4],f32>
    // CHECK: tensor.expand_shape {{.*}} into tensor<1x1x5x4xf32>
    // CHECK: hip.tile
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,2,5,4],f32>
  }

  func.func @flip(%x: !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[3,2],f32> {
    // CHECK-LABEL: func.func @flip
    %dim = torch.constant.int 0
    %dims = torch.prim.ListConstruct %dim : (!torch.int) -> !torch.list<int>
    %y = torch.aten.flip %x, %dims : !torch.vtensor<[3,2],f32>, !torch.list<int> -> !torch.vtensor<[3,2],f32>
    // CHECK: arith.constant dense<[2, 1, 0]> : tensor<3xi64>
    // CHECK: hip.gather
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[3,2],f32>
  }

  func.func @pow_float_cube(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @pow_float_cube
    %three = torch.constant.float 3.000000e+00
    %y = torch.aten.pow.Tensor_Scalar %x, %three : !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],f32>
    // CHECK: hip.mul
    // CHECK: hip.mul
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],f32>
  }

  func.func @sdpa_gqa(%q: !torch.vtensor<[1,4,2,8],f32>, %k: !torch.vtensor<[1,2,2,8],f32>, %v: !torch.vtensor<[1,2,2,8],f32>) -> !torch.vtensor<[1,4,2,8],f32> {
    // CHECK-LABEL: func.func @sdpa_gqa
    %none = torch.constant.none
    %none_scale = torch.constant.none
    %drop = torch.constant.float 0.000000e+00
    %true = torch.constant.bool true
    %y = torch.aten.scaled_dot_product_attention %q, %k, %v, %none, %drop, %true, %none_scale, %true : !torch.vtensor<[1,4,2,8],f32>, !torch.vtensor<[1,2,2,8],f32>, !torch.vtensor<[1,2,2,8],f32>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[1,4,2,8],f32>
    // CHECK: hip.gqa
    // CHECK-SAME: kv_num_heads = 2{{.*}}num_heads = 4
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,4,2,8],f32>
  }

  func.func @flash_attention(%q: !torch.vtensor<[1,2,2,8],f32>, %k: !torch.vtensor<[1,2,2,8],f32>, %v: !torch.vtensor<[1,2,2,8],f32>) -> !torch.vtensor<[1,2,2,8],f32> {
    // CHECK-LABEL: func.func @flash_attention
    %none = torch.constant.none
    %none_scale = torch.constant.none
    %drop = torch.constant.float 0.000000e+00
    %false = torch.constant.bool false
    %y:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%q, %k, %v, %drop, %false, %none, %none_scale) : (!torch.vtensor<[1,2,2,8],f32>, !torch.vtensor<[1,2,2,8],f32>, !torch.vtensor<[1,2,2,8],f32>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<[1,2,2,8],f32>, !torch.vtensor<[1,2,2],f32>)
    // CHECK: hip.gqa
    // CHECK-SAME: kv_num_heads = 2{{.*}}no_causal = true{{.*}}num_heads = 2
    // CHECK-NOT: torch.
    return %y#0 : !torch.vtensor<[1,2,2,8],f32>
  }

  func.func @qmoe(%x: !torch.vtensor<[1,2,8],f32>, %probs: !torch.vtensor<[2,2],f32>, %fc1w: !torch.vtensor<[2,8,4],ui8>, %fc1s: !torch.vtensor<[2,8,2],f16>, %fc1b: !torch.vtensor<[2,8],f32>, %fc2w: !torch.vtensor<[2,8,2],ui8>, %fc2s: !torch.vtensor<[2,8,1],f16>, %fc2b: !torch.vtensor<[2,8],f32>) -> !torch.vtensor<[1,2,8],f32> {
    // CHECK-LABEL: func.func @qmoe
    %k = torch.constant.int 1
    %alpha = torch.constant.float 1.702000e+00
    %beta = torch.constant.float 1.000000e+00
    %limit = torch.constant.float 7.000000e+00
    %y = torch.operator "torch.hipdnn.qmoe"(%x, %probs, %fc1w, %fc1s, %fc1b, %fc2w, %fc2s, %fc2b, %k, %alpha, %beta, %limit) : (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,8,4],ui8>, !torch.vtensor<[2,8,2],f16>, !torch.vtensor<[2,8],f32>, !torch.vtensor<[2,8,2],ui8>, !torch.vtensor<[2,8,1],f16>, !torch.vtensor<[2,8],f32>, !torch.int, !torch.float, !torch.float, !torch.float) -> !torch.vtensor<[1,2,8],f32>
    // CHECK: hip.qmoe
    // CHECK-SAME: activation_type = "swiglu"{{.*}}block_size = 4{{.*}}expert_weight_bits = 4{{.*}}k = 1{{.*}}swiglu_fusion = 1
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,2,8],f32>
  }

  func.func @reciprocal_f64(%x: !torch.vtensor<[64],f64>) -> !torch.vtensor<[64],f64> {
    // CHECK-LABEL: func.func @reciprocal_f64
    %y = torch.aten.reciprocal %x : !torch.vtensor<[64],f64> -> !torch.vtensor<[64],f64>
    // CHECK: hip.reciprocal
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[64],f64>
  }

  func.func @logical_and(%a: !torch.vtensor<[8,4],i1>, %b: !torch.vtensor<[8,4],i1>) -> !torch.vtensor<[8,4],i1> {
    // CHECK-LABEL: func.func @logical_and
    %y = torch.aten.logical_and %a, %b : !torch.vtensor<[8,4],i1>, !torch.vtensor<[8,4],i1> -> !torch.vtensor<[8,4],i1>
    // CHECK: hip.and
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[8,4],i1>
  }

  func.func @div_scalar_f64(%x: !torch.vtensor<[64],f64>) -> !torch.vtensor<[64],f64> {
    // CHECK-LABEL: func.func @div_scalar_f64
    %c = torch.constant.int 64
    %y = torch.aten.div.Scalar %x, %c : !torch.vtensor<[64],f64>, !torch.int -> !torch.vtensor<[64],f64>
    // CHECK: hip.div
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[64],f64>
  }

  func.func @mul_i1(%a: !torch.vtensor<[4],i1>, %b: !torch.vtensor<[4],i1>) -> !torch.vtensor<[4],i1> {
    // CHECK-LABEL: func.func @mul_i1
    %y = torch.aten.mul.Tensor %a, %b : !torch.vtensor<[4],i1>, !torch.vtensor<[4],i1> -> !torch.vtensor<[4],i1>
    // CHECK: hip.and
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],i1>
  }

  func.func @sub_i64_scalar(%x: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32> {
    // CHECK-LABEL: func.func @sub_i64_scalar
    %one = torch.constant.int 1
    %scalar = torch.prim.NumToTensor.Scalar %one : !torch.int -> !torch.vtensor<[],si64>
    %alpha = torch.constant.int 1
    %y = torch.aten.sub.Tensor %scalar, %x, %alpha : !torch.vtensor<[],si64>, !torch.vtensor<[2,2],f32>, !torch.int -> !torch.vtensor<[2,2],f32>
    // CHECK: hip.cast
    // CHECK: hip.sub
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[2,2],f32>
  }

  func.func @sub_i64_rank0(%x: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4],si64> {
    // CHECK-LABEL: func.func @sub_i64_rank0
    %n = torch.constant.int 63
    %scalar = torch.prim.NumToTensor.Scalar %n : !torch.int -> !torch.vtensor<[],si64>
    %one = torch.constant.int 1
    %y = torch.aten.sub.Tensor %scalar, %x, %one : !torch.vtensor<[],si64>, !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    // CHECK: hip.sub
    // CHECK-SAME: tensor<4xi64>
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],si64>
  }

  func.func @roll(%x: !torch.vtensor<[1,4,4],f32>) -> !torch.vtensor<[1,4,4],f32> {
    // CHECK-LABEL: func.func @roll
    %shift0 = torch.constant.int -1
    %shift1 = torch.constant.int 1
    %shifts = torch.prim.ListConstruct %shift0, %shift1 : (!torch.int, !torch.int) -> !torch.list<int>
    %dim0 = torch.constant.int 1
    %dim1 = torch.constant.int 2
    %dims = torch.prim.ListConstruct %dim0, %dim1 : (!torch.int, !torch.int) -> !torch.list<int>
    %y = torch.aten.roll %x, %shifts, %dims : !torch.vtensor<[1,4,4],f32>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[1,4,4],f32>
    // CHECK: tensor.extract_slice
    // CHECK: tensor.concat
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,4,4],f32>
  }

  func.func @one_hot(%ids: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,4],f32> {
    // CHECK-LABEL: func.func @one_hot
    %n = torch.constant.int 4
    %y = torch.aten.one_hot %ids, %n : !torch.vtensor<[2],si64>, !torch.int -> !torch.vtensor<[2,4],f32>
    // CHECK: hip.one_hot
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[2,4],f32>
  }

  func.func @rms_norm(%x: !torch.vtensor<[2,4],f32>, %w: !torch.vtensor<[4],f32>) -> !torch.vtensor<[2,4],f32> {
    // CHECK-LABEL: func.func @rms_norm
    %c4 = torch.constant.int 4
    %shape = torch.prim.ListConstruct %c4 : (!torch.int) -> !torch.list<int>
    %eps = torch.constant.float 1.000000e-05
    %y = torch.aten.rms_norm %x, %shape, %w, %eps : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[2,4],f32>
    // CHECK: hip.rms_norm
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[2,4],f32>
  }

  func.func @layer_norm(%x: !torch.vtensor<[2,4],f32>, %w: !torch.vtensor<[4],f32>) -> !torch.vtensor<[2,4],f32> {
    // CHECK-LABEL: func.func @layer_norm
    %c4 = torch.constant.int 4
    %shape = torch.prim.ListConstruct %c4 : (!torch.int) -> !torch.list<int>
    %none = torch.constant.none
    %eps = torch.constant.float 1.000000e-05
    %false = torch.constant.bool false
    %y = torch.aten.layer_norm %x, %shape, %w, %none, %eps, %false : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.vtensor<[4],f32>, !torch.none, !torch.float, !torch.bool -> !torch.vtensor<[2,4],f32>
    // CHECK: hip.layer_norm
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[2,4],f32>
  }

  func.func @eye() -> !torch.vtensor<[3,3],f32> {
    // CHECK-LABEL: func.func @eye
    %n = torch.constant.int 3
    %none = torch.constant.none
    %y = torch.aten.eye %n, %none, %none, %none, %none : !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3,3],f32>
    // CHECK: hip.equal
    // CHECK: hip.cast
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[3,3],f32>
  }

  func.func @isfinite(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],i1> {
    // CHECK-LABEL: func.func @isfinite
    %y = torch.aten.isfinite %x : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],i1>
    // CHECK: hip.abs
    // CHECK: hip.and
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],i1>
  }

  func.func @logaddexp(%a: !torch.vtensor<[4],f32>, %b: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @logaddexp
    %y = torch.aten.logaddexp %a, %b : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    // CHECK: hip.max
    // CHECK: hip.exp
    // CHECK: hip.log
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],f32>
  }

  func.func @isclose(%a: !torch.vtensor<[4],f32>, %b: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],i1> {
    // CHECK-LABEL: func.func @isclose
    %rtol = torch.constant.float 1.000000e-05
    %atol = torch.constant.float 1.000000e-08
    %false = torch.constant.bool false
    %y = torch.aten.isclose %a, %b, %rtol, %atol, %false : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[4],i1>
    // CHECK: hip.less
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],i1>
  }

  func.func @silu(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @silu
    %y = torch.aten.silu %x : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    // CHECK: hip.silu
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],f32>
  }

  func.func @matmul_vec(%a: !torch.vtensor<[3],f32>, %b: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[4],f32> {
    // CHECK-LABEL: func.func @matmul_vec
    %y = torch.aten.matmul %a, %b : !torch.vtensor<[3],f32>, !torch.vtensor<[3,4],f32> -> !torch.vtensor<[4],f32>
    // CHECK: hip.matmul
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[4],f32>
  }

  func.func @all_i1(%x: !torch.vtensor<[2,3],i1>) -> !torch.vtensor<[],i1> {
    // CHECK-LABEL: func.func @all_i1
    %y = torch.aten.all %x : !torch.vtensor<[2,3],i1> -> !torch.vtensor<[],i1>
    // CHECK: hip.reduce_min
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[],i1>
  }

  func.func @var_correction(%x: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,1],f32> {
    // CHECK-LABEL: func.func @var_correction
    %dim = torch.constant.int -1
    %dims = torch.prim.ListConstruct %dim : (!torch.int) -> !torch.list<int>
    %corr = torch.constant.int 1
    %keep = torch.constant.bool true
    %y = torch.aten.var.correction %x, %dims, %corr, %keep : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.int, !torch.bool -> !torch.vtensor<[2,1],f32>
    // CHECK: hip.reduce_mean
    // CHECK: hip.sub
    // CHECK: hip.mul
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[2,1],f32>
  }

  func.func @amax_amin(%x: !torch.vtensor<[2,4,3],f32>) -> (!torch.vtensor<[2,3],f32>, !torch.vtensor<[1,1,1],f32>) {
    // CHECK-LABEL: func.func @amax_amin
    %dim = torch.constant.int 1
    %dims = torch.prim.ListConstruct %dim : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %y = torch.aten.amax %x, %dims, %false : !torch.vtensor<[2,4,3],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,3],f32>
    // CHECK: hip.reduce_max
    %none = torch.prim.ListConstruct  : () -> !torch.list<int>
    %z = torch.aten.amin %x, %none, %true : !torch.vtensor<[2,4,3],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,1],f32>
    // CHECK: hip.reduce_min
    // CHECK-NOT: torch.
    return %y, %z : !torch.vtensor<[2,3],f32>, !torch.vtensor<[1,1,1],f32>
  }

  func.func @upsample_aa(%x: !torch.vtensor<[1,2,4,4],f32>) -> !torch.vtensor<[1,2,6,3],f32> {
    // CHECK-LABEL: func.func @upsample_aa
    %h = torch.constant.int 6
    %w = torch.constant.int 3
    %size = torch.prim.ListConstruct %h, %w : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %y = torch.operator "torch.aten._upsample_bicubic2d_aa"(%x, %size, %false, %none, %none) : (!torch.vtensor<[1,2,4,4],f32>, !torch.list<int>, !torch.bool, !torch.none, !torch.none) -> !torch.vtensor<[1,2,6,3],f32>
    // CHECK: arith.constant dense<{{.*}}> : tensor<4x3xf32>
    // CHECK: hip.matmul
    // CHECK: hip.transpose
    // CHECK: arith.constant dense<{{.*}}> : tensor<4x6xf32>
    // CHECK: hip.matmul
    // CHECK: hip.transpose
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[1,2,6,3],f32>
  }

  func.func @norm_scalar(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[],f32> {
    // CHECK-LABEL: func.func @norm_scalar
    %p = torch.constant.int 2
    %y = torch.aten.norm.Scalar %x, %p : !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[],f32>
    // CHECK: hip.sqrt
    // CHECK-NOT: torch.
    return %y : !torch.vtensor<[],f32>
  }
}
