// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

// expected-error@+1 {{HIP layout length must match tensor rank 2}}
func.func @too_short(%arg: tensor<2x3xf32, #hip.layout<"?">>)

// -----

// expected-error@+1 {{HIP layout length must match tensor rank 1}}
func.func @too_long(%arg: tensor<2xf32, #hip.layout<"??">>)

// -----

// expected-error@+1 {{HIP layout length must match tensor rank 0}}
func.func @scalar(%arg: tensor<f32, #hip.layout<"?">>)
