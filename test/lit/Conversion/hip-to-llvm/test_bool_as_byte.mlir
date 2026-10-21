// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// An i1 memref is one byte per element holding 0 or 1, so an op that only
// moves elements around, or compares them, is byte-identical on a bool and a
// ui8 buffer. Those ops pass HIPDNN_EP_DATATYPE_UINT8 (7) for an i1 operand
// rather than rejecting it.
//
// hip.cast is deliberately NOT in this set -- see test_cast.mlir, where
// narrowing to bool uses its own dtype because it is `x != 0`, not a
// truncation.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Broadcast of a bool mask: pure data movement.
  func.func @expand_bool(
      %ctx: !hip.context,
      %input: memref<1x1x4x8xi1, 1>,
      %shape: memref<4xi64, 1>,
      %output: memref<1x1x4x8xi1, 1>) {
    // CHECK-LABEL: llvm.func @expand_bool
    // CHECK-DAG: llvm.mlir.constant(7 : i64) : i64
    // CHECK: llvm.call @wrap_expand
    hip.expand(%ctx) ins(%input, %shape : memref<1x1x4x8xi1, 1>, memref<4xi64, 1>)
                     outs(%output : memref<1x1x4x8xi1, 1>)
    return
  }

  // Min over a bool axis is logical AND, which is how aten.all arrives here.
  func.func @reduce_min_bool(
      %ctx: !hip.context,
      %data: memref<1x4x8xi1, 1>,
      %axes: memref<1xi64, 1>,
      %output: memref<1x4x1xi1, 1>) {
    // CHECK-LABEL: llvm.func @reduce_min_bool
    // CHECK-DAG: llvm.mlir.constant(7 : i64) : i64
    // CHECK: llvm.call @wrap_reduce_min
    hip.reduce_min(%ctx) ins(%data, %axes : memref<1x4x8xi1, 1>, memref<1xi64, 1>)
                         outs(%output : memref<1x4x1xi1, 1>)
                         {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64}
    return
  }

  // Gathering rows of a bool mask: pure data movement.
  func.func @gather_nd_bool(
      %ctx: !hip.context,
      %data: memref<1x8xi1, 1>,
      %indices: memref<1x4x2xi64, 1>,
      %output: memref<1x4xi1, 1>) {
    // CHECK-LABEL: llvm.func @gather_nd_bool
    // CHECK-DAG: llvm.mlir.constant(7 : i64) : i64
    // CHECK: llvm.call @wrap_gather_nd
    hip.gather_nd(%ctx) ins(%data, %indices : memref<1x8xi1, 1>, memref<1x4x2xi64, 1>)
                        outs(%output : memref<1x4xi1, 1>) {batch_dims = 0 : i64}
    return
  }
}
