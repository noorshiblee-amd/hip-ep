//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef TORCH_OPS
#define TORCH_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "hip/Dialect/Torch/IR/TorchDialect.td"
include "hip/Dialect/Torch/IR/TorchTypes.td"

// Every accepted operation uses a hand-written parser. The default form is
// `torch.<mnemonic> operands : operand-types -> result-types`, implemented by
// parseDefaultTorchOp. Constants and literals set their own printers because
// the producer does not use that form.
class Torch_Op<string mnemonic, list<Trait> traits = []> :
    Op<Torch_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}

def ConstantIntOp : Torch_Op<"constant.int", [Pure]> {
  let summary = "Torch integer scalar constant";
  let arguments = (ins I64Attr:$value);
  let results = (outs IntType:$result);
}

def ConstantFloatOp : Torch_Op<"constant.float", [Pure]> {
  let summary = "Torch float scalar constant";
  let arguments = (ins F64Attr:$value);
  let results = (outs TorchFloatType:$result);
}

def ConstantNoneOp : Torch_Op<"constant.none", [Pure]> {
  let summary = "Torch singleton None";
  let arguments = (ins);
  let results = (outs TorchNoneType:$result);
}

def ValueTensorLiteralOp : Torch_Op<"vtensor.literal", [Pure]> {
  let summary = "Dense value-tensor literal";
  let description = [{
    Carries a `dense` elements attribute. `dense_resource` is rejected: the
    HIP constant carrier needs a payload it can serialize.
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs ValueTensorType:$result);
}

def AtenAddTensorOp : Torch_Op<"aten.add.Tensor", [Pure]> {
  let summary = "self + alpha * other";
  let arguments = (ins
    ValueTensorType:$self,
    ValueTensorType:$other,
    AnyTypeOf<[IntType, TorchFloatType]>:$alpha
  );
  let results = (outs ValueTensorType:$result);
}

def AtenAddScalarOp : Torch_Op<"aten.add.Scalar", [Pure]> {
  let summary = "self + alpha * scalar";
  let arguments = (ins
    ValueTensorType:$self,
    AnyTypeOf<[IntType, TorchFloatType]>:$other,
    AnyTypeOf<[IntType, TorchFloatType]>:$alpha
  );
  let results = (outs ValueTensorType:$result);
}

#endif // TORCH_OPS
