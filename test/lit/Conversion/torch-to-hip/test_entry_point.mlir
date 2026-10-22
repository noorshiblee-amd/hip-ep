// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// torch-mlir's exporter names the graph entry @main, while the compiler /
// runtime contract spells it @main_graph. convert-torch-to-hip renames the
// entry so a stock Torch export reaches the C ABI without a hand edit:
// without the rename the module metadata is skipped and generate-interface
// then reports a missing @main_graph.
//
// Only a function carrying Torch IR is renamed, and only when the module has
// no @main_graph of its own.
// ============================================================================

// RUN: hip-mlir-opt %s --split-input-file --hip-add-context-arg --convert-torch-to-hip | FileCheck %s

// The renamed entry is the one metadata is stamped from, so input_count
// appearing at all proves the rename ran before generateModuleMetadata.
// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1 : i64
// CHECK-SAME: hipdnn.output_shapes = [array<i64: 1, 8>]
// CHECK: func.func @main_graph
module {
  func.func @main(%arg0: !torch.vtensor<[1,8],f32>) -> !torch.vtensor<[1,8],f32> {
    %alpha = torch.constant.int 1
    %sum = torch.aten.add.Tensor %arg0, %arg0, %alpha : !torch.vtensor<[1,8],f32>, !torch.vtensor<[1,8],f32>, !torch.int -> !torch.vtensor<[1,8],f32>
    return %sum : !torch.vtensor<[1,8],f32>
  }
}

// -----

// Next to an existing @main_graph, @main is an ordinary helper and keeps its
// name; the metadata still comes from @main_graph.
// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_shapes = [array<i64: 2, 2>]
// CHECK: func.func @main_graph
// CHECK: func.func @main(
module {
  func.func @main_graph(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32> {
    return %arg0 : !torch.vtensor<[2,2],f32>
  }
  func.func @main(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    return %arg0 : !torch.vtensor<[4],f32>
  }
}

// -----

// A @main with no Torch IR is not a Torch entry point and is left alone, so a
// non-Torch module passed through this pass is unchanged.
// CHECK-NOT: hipdnn.input_count
// CHECK: func.func @main(
module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}
