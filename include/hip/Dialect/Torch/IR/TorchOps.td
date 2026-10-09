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

def ConstantBoolOp : Torch_Op<"constant.bool", [Pure]> {
  let summary = "Torch bool scalar constant";
  let arguments = (ins BoolAttr:$value);
  let results = (outs TorchBoolType:$result);
}

def ValueTensorLiteralOp : Torch_Op<"vtensor.literal", [Pure]> {
  let summary = "Dense value-tensor literal";
  let description = [{
    Carries a `dense` elements attribute, or a `dense_resource` whose blob is
    copied into one before `hip.constant` is built. A resource handle with no
    blob becomes a zero splat of the declared shape.
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs ValueTensorType:$result);
}

def PrimListConstructOp : Torch_Op<"prim.ListConstruct", [Pure]> {
  let summary = "Build a Torch list from scalar elements";
  let arguments = (ins Variadic<IntType>:$elements);
  let results = (outs ListType:$result);
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

// Assembly matches torch-mlir: `torch.operator "name"(operands) attrs : fn-type`.
// Regions are not accepted; the importer used here does not emit them.
def OperatorOp : Torch_Op<"operator"> {
  let summary = "Opaque torch operator";
  let arguments = (ins StrAttr:$name, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
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

def AtenSubTensorOp : Torch_Op<"aten.sub.Tensor", [Pure]> {
  let summary = "self - alpha * other";
  let arguments = (ins
    ValueTensorType:$self,
    ValueTensorType:$other,
    AnyTypeOf<[IntType, TorchFloatType]>:$alpha
  );
  let results = (outs ValueTensorType:$result);
}

def AtenMulTensorOp : Torch_Op<"aten.mul.Tensor", [Pure]> {
  let summary = "Elementwise product";
  let arguments = (ins ValueTensorType:$self, ValueTensorType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenMulScalarOp : Torch_Op<"aten.mul.Scalar", [Pure]> {
  let summary = "Elementwise product by a scalar";
  let arguments = (ins
    ValueTensorType:$self,
    AnyTypeOf<[IntType, TorchFloatType]>:$other
  );
  let results = (outs ValueTensorType:$result);
}

def AtenDivScalarOp : Torch_Op<"aten.div.Scalar", [Pure]> {
  let summary = "Elementwise division by a scalar";
  let arguments = (ins
    ValueTensorType:$self,
    AnyTypeOf<[IntType, TorchFloatType]>:$other
  );
  let results = (outs ValueTensorType:$result);
}

def AtenRsqrtOp : Torch_Op<"aten.rsqrt", [Pure]> {
  let summary = "Reciprocal square root";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenReluOp : Torch_Op<"aten.relu", [Pure]> {
  let summary = "max(self, 0)";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenUnsqueezeOp : Torch_Op<"aten.unsqueeze", [Pure]> {
  let summary = "Insert a size-1 dimension";
  let arguments = (ins ValueTensorType:$self, IntType:$dim);
  let results = (outs ValueTensorType:$result);
}

def AtenViewOp : Torch_Op<"aten.view", [Pure]> {
  let summary = "Reshape to a constant shape list";
  let arguments = (ins ValueTensorType:$self, ListType:$size);
  let results = (outs ValueTensorType:$result);
}

def AtenPermuteOp : Torch_Op<"aten.permute", [Pure]> {
  let summary = "Permute dimensions";
  let arguments = (ins ValueTensorType:$self, ListType:$dims);
  let results = (outs ValueTensorType:$result);
}

def AtenBroadcastToOp : Torch_Op<"aten.broadcast_to", [Pure]> {
  let summary = "Broadcast to a constant shape list";
  let arguments = (ins ValueTensorType:$self, ListType:$size);
  let results = (outs ValueTensorType:$result);
}

def AtenMmOp : Torch_Op<"aten.mm", [Pure]> {
  let summary = "Matrix product";
  let arguments = (ins ValueTensorType:$self, ValueTensorType:$mat2);
  let results = (outs ValueTensorType:$result);
}

def AtenSumDimIntListOp : Torch_Op<"aten.sum.dim_IntList", [Pure]> {
  let summary = "Sum over a list of dimensions";
  let arguments = (ins
    ValueTensorType:$self,
    ListType:$dim,
    TorchBoolType:$keepdim,
    AnyTypeOf<[TorchNoneType, IntType]>:$dtype
  );
  let results = (outs ValueTensorType:$result);
}

def AtenMaxPool2dOp : Torch_Op<"aten.max_pool2d", [Pure]> {
  let summary = "2-D max pool";
  let arguments = (ins
    ValueTensorType:$self,
    ListType:$kernel_size,
    ListType:$stride,
    ListType:$padding,
    ListType:$dilation,
    TorchBoolType:$ceil_mode
  );
  let results = (outs ValueTensorType:$result);
}

def AtenConvolutionOp : Torch_Op<"aten.convolution", [Pure]> {
  let summary = "Convolution";
  let arguments = (ins
    ValueTensorType:$input,
    ValueTensorType:$weight,
    AnyTypeOf<[ValueTensorType, TorchNoneType]>:$bias,
    ListType:$stride,
    ListType:$padding,
    ListType:$dilation,
    TorchBoolType:$transposed,
    ListType:$output_padding,
    IntType:$groups
  );
  let results = (outs ValueTensorType:$result);
}

#endif // TORCH_OPS
