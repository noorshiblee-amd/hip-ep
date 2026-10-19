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

def ConstantDeviceOp : Torch_Op<"constant.device", [Pure]> {
  let summary = "Torch device constant";
  let arguments = (ins StrAttr:$value);
  let results = (outs DeviceType:$result);
}

def ConstantStrOp : Torch_Op<"constant.str", [Pure]> {
  let summary = "Torch string constant";
  let arguments = (ins StrAttr:$value);
  let results = (outs TorchStringType:$result);
}

def PrimListConstructOp : Torch_Op<"prim.ListConstruct", [Pure]> {
  let summary = "Build a Torch list";
  let arguments = (ins Variadic<AnyType>:$elements);
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

def AtenHardtanhOp : Torch_Op<"aten.hardtanh", [Pure]> {
  let summary = "Clamp each element to a scalar range";
  let arguments = (ins ValueTensorType:$self, AnyType:$min_val, AnyType:$max_val);
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

def AtenFlipOp : Torch_Op<"aten.flip", [Pure]> {
  let summary = "Reverse a tensor along the given dimensions";
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

def AtenAvgPool2dOp : Torch_Op<"aten.avg_pool2d", [Pure]> {
  let summary = "2-D average pool";
  let arguments = (ins
    ValueTensorType:$self,
    ListType:$kernel_size,
    ListType:$stride,
    ListType:$padding,
    TorchBoolType:$ceil_mode,
    TorchBoolType:$count_include_pad,
    AnyType:$divisor_override
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

// Ops below are the pretty forms emitted by a full torch-mlir export. Optional
// trailing operands are AnyType because the producer mixes none, int, bool,
// and Device in the same slot.

def AtenExpandOp : Torch_Op<"aten.expand", [Pure]> {
  let summary = "Broadcast to a shape list";
  let arguments = (ins ValueTensorType:$self, ListType:$size, TorchBoolType:$implicit);
  let results = (outs ValueTensorType:$result);
}

def AtenBmmOp : Torch_Op<"aten.bmm", [Pure]> {
  let summary = "Batched matrix product";
  let arguments = (ins ValueTensorType:$self, ValueTensorType:$mat2);
  let results = (outs ValueTensorType:$result);
}

def AtenVarMeanCorrectionOp : Torch_Op<"aten.var_mean.correction", [Pure]> {
  let summary = "Variance and mean";
  let arguments = (ins
    ValueTensorType:$self,
    ListType:$dim,
    IntType:$correction,
    TorchBoolType:$keepdim
  );
  let results = (outs ValueTensorType:$var, ValueTensorType:$mean);
}

def AtenWhereSelfOp : Torch_Op<"aten.where.self", [Pure]> {
  let summary = "Select elements from two tensors";
  let arguments = (ins
    ValueTensorType:$condition,
    ValueTensorType:$self,
    ValueTensorType:$other
  );
  let results = (outs ValueTensorType:$result);
}

def AtenScalarTensorOp : Torch_Op<"aten.scalar_tensor", [Pure]> {
  let summary = "Rank-0 tensor holding one scalar";
  let arguments = (ins
    AnyType:$value,
    AnyType:$dtype,
    AnyType:$layout,
    AnyType:$device,
    AnyType:$pin_memory
  );
  let results = (outs ValueTensorType:$result);
}

def AtenFullLikeOp : Torch_Op<"aten.full_like", [Pure]> {
  let summary = "Tensor filled with a scalar, shaped like self";
  let arguments = (ins
    ValueTensorType:$self,
    AnyType:$fill_value,
    AnyType:$dtype,
    AnyType:$layout,
    AnyType:$device,
    TorchBoolType:$pin_memory,
    AnyType:$memory_format
  );
  let results = (outs ValueTensorType:$result);
}

def AtenCloneOp : Torch_Op<"aten.clone", [Pure]> {
  let summary = "Value-semantic copy";
  let arguments = (ins ValueTensorType:$self, AnyType:$memory_format);
  let results = (outs ValueTensorType:$result);
}

def AtenCopyOp : Torch_Op<"aten.copy", [Pure]> {
  let summary = "Value-semantic copy of src into the shape of self";
  let arguments = (ins ValueTensorType:$self, ValueTensorType:$src,
                       TorchBoolType:$non_blocking);
  let results = (outs ValueTensorType:$result);
}

def AtenScaledDotProductAttentionOp
    : Torch_Op<"aten.scaled_dot_product_attention", [Pure]> {
  let summary = "Fused scaled dot-product attention";
  let arguments = (ins
    ValueTensorType:$query,
    ValueTensorType:$key,
    ValueTensorType:$value,
    AnyType:$attn_mask,
    TorchFloatType:$dropout_p,
    TorchBoolType:$is_causal,
    AnyType:$scale,
    TorchBoolType:$enable_gqa
  );
  let results = (outs ValueTensorType:$result);
}

def AtenSoftmaxOp : Torch_Op<"aten._softmax", [Pure]> {
  let summary = "Softmax over one dimension";
  let arguments = (ins ValueTensorType:$self, IntType:$dim, TorchBoolType:$half_to_float);
  let results = (outs ValueTensorType:$result);
}

def AtenEqScalarOp : Torch_Op<"aten.eq.Scalar", [Pure]> {
  let summary = "Elementwise equality with a scalar";
  let arguments = (ins ValueTensorType:$self, AnyType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenAllDimOp : Torch_Op<"aten.all.dim", [Pure]> {
  let summary = "Logical and along one dimension";
  let arguments = (ins ValueTensorType:$self, IntType:$dim, TorchBoolType:$keepdim);
  let results = (outs ValueTensorType:$result);
}

def AtenAssertTensorMetadataOp : Torch_Op<"aten._assert_tensor_metadata"> {
  let summary = "Dropped tensor-metadata assertion";
  let arguments = (ins
    AnyType:$self,
    AnyType:$size,
    AnyType:$stride,
    AnyType:$dtype,
    AnyType:$device,
    AnyType:$layout
  );
}

def AtenArangeStartStepOp : Torch_Op<"aten.arange.start_step", [Pure]> {
  let summary = "Integer range";
  let arguments = (ins
    IntType:$start,
    IntType:$end,
    IntType:$step,
    AnyType:$dtype,
    AnyType:$layout,
    AnyType:$device,
    TorchBoolType:$pin_memory
  );
  let results = (outs ValueTensorType:$result);
}

def AtenSliceTensorOp : Torch_Op<"aten.slice.Tensor", [Pure]> {
  let summary = "Slice one dimension";
  let arguments = (ins
    ValueTensorType:$self,
    IntType:$dim,
    IntType:$start,
    IntType:$end,
    IntType:$step
  );
  let results = (outs ValueTensorType:$result);
}

def AtenSliceScatterOp : Torch_Op<"aten.slice_scatter", [Pure]> {
  let summary = "Write a tensor into one sliced dimension";
  let arguments = (ins
    ValueTensorType:$self,
    ValueTensorType:$src,
    IntType:$dim,
    IntType:$start,
    IntType:$end,
    IntType:$step
  );
  let results = (outs ValueTensorType:$result);
}

def AtenUpsampleNearest2dVecOp : Torch_Op<"aten.upsample_nearest2d.vec", [Pure]> {
  let summary = "Nearest-neighbor resize; size or scales may be none";
  let arguments = (ins ValueTensorType:$self, AnyType:$output_size, AnyType:$scale_factors);
  let results = (outs ValueTensorType:$result);
}

def AtenDivTensorOp : Torch_Op<"aten.div.Tensor", [Pure]> {
  let summary = "Elementwise division";
  let arguments = (ins ValueTensorType:$self, ValueTensorType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenBitwiseAndTensorOp : Torch_Op<"aten.bitwise_and.Tensor", [Pure]> {
  let summary = "Elementwise logical and";
  let arguments = (ins ValueTensorType:$self, ValueTensorType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenEmptyStridedOp : Torch_Op<"aten.empty_strided", [Pure]> {
  let summary = "Uninitialized tensor used as a shape template";
  let arguments = (ins
    ListType:$size,
    ListType:$stride,
    IntType:$dtype,
    AnyType:$layout,
    AnyType:$device,
    TorchBoolType:$pin_memory
  );
  let results = (outs ValueTensorType:$result);
}

def PrimsConvertElementTypeOp : Torch_Op<"prims.convert_element_type", [Pure]> {
  let summary = "Cast to a Torch scalar type code";
  let arguments = (ins ValueTensorType:$a, IntType:$dtype);
  let results = (outs ValueTensorType:$result);
}

def AtenCatOp : Torch_Op<"aten.cat", [Pure]> {
  let summary = "Concatenate a tensor list";
  let arguments = (ins ListType:$tensors, IntType:$dim);
  let results = (outs ValueTensorType:$result);
}

def AtenCumsumOp : Torch_Op<"aten.cumsum", [Pure]> {
  let summary = "Cumulative sum";
  let arguments = (ins ValueTensorType:$self, IntType:$dim, AnyType:$dtype);
  let results = (outs ValueTensorType:$result);
}

def AtenSinOp : Torch_Op<"aten.sin", [Pure]> {
  let summary = "Elementwise sine";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenCosOp : Torch_Op<"aten.cos", [Pure]> {
  let summary = "Elementwise cosine";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenAssertScalarOp : Torch_Op<"aten._assert_scalar"> {
  let summary = "Dropped runtime scalar assertion";
  let arguments = (ins AnyType:$self, AnyType:$message);
}

def AtenSizeIntOp : Torch_Op<"aten.size.int", [Pure]> {
  let summary = "Size of one tensor dimension";
  let arguments = (ins ValueTensorType:$self, IntType:$dim);
  let results = (outs IntType:$result);
}

def AtenGeIntOp : Torch_Op<"aten.ge.int", [Pure]> {
  let summary = "Integer scalar a >= b";
  let arguments = (ins IntType:$a, IntType:$b);
  let results = (outs TorchBoolType:$result);
}

def AtenLeIntOp : Torch_Op<"aten.le.int", [Pure]> {
  let summary = "Integer scalar a <= b";
  let arguments = (ins IntType:$a, IntType:$b);
  let results = (outs TorchBoolType:$result);
}

def AtenEqIntOp : Torch_Op<"aten.eq.int", [Pure]> {
  let summary = "Integer scalar a == b";
  let arguments = (ins IntType:$a, IntType:$b);
  let results = (outs TorchBoolType:$result);
}

def AtenUniqueConsecutiveOp : Torch_Op<"aten.unique_consecutive", [Pure]> {
  let summary = "Collapse runs of equal values";
  let arguments = (ins
    ValueTensorType:$self,
    TorchBoolType:$return_inverse,
    TorchBoolType:$return_counts,
    AnyType:$dim
  );
  let results = (outs
    ValueTensorType:$result,
    ValueTensorType:$inverse,
    ValueTensorType:$counts
  );
}

def AtenIntBoolOp : Torch_Op<"aten.Int.bool", [Pure]> {
  let summary = "Bool scalar to integer scalar";
  let arguments = (ins TorchBoolType:$a);
  let results = (outs IntType:$result);
}

def AtenRemainderScalarOp : Torch_Op<"aten.remainder.Scalar", [Pure]> {
  let summary = "Elementwise remainder whose sign follows the divisor";
  let arguments = (ins ValueTensorType:$self, AnyType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenGeScalarOp : Torch_Op<"aten.ge.Scalar", [Pure]> {
  let summary = "Elementwise greater-or-equal against a scalar";
  let arguments = (ins ValueTensorType:$self, AnyType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenIndexTensorOp : Torch_Op<"aten.index.Tensor", [Pure]> {
  let summary = "Index with a list of index tensors";
  let arguments = (ins ValueTensorType:$self, ListType:$indices);
  let results = (outs ValueTensorType:$result);
}

def AtenIndexSelectOp : Torch_Op<"aten.index_select", [Pure]> {
  let summary = "Select entries along one dimension with a 1-D index";
  let arguments = (ins ValueTensorType:$self, IntType:$dim, ValueTensorType:$index);
  let results = (outs ValueTensorType:$result);
}

def AtenMaxPool2dWithIndicesOp : Torch_Op<"aten.max_pool2d_with_indices", [Pure]> {
  let summary = "2-D max pool; the index result is dropped when unused";
  let arguments = (ins
    ValueTensorType:$self,
    ListType:$kernel_size,
    ListType:$stride,
    ListType:$padding,
    ListType:$dilation,
    TorchBoolType:$ceil_mode
  );
  let results = (outs ValueTensorType:$values, ValueTensorType:$indices);
}

def AtenSelectIntOp : Torch_Op<"aten.select.int", [Pure]> {
  let summary = "Select one index along a dimension";
  let arguments = (ins ValueTensorType:$self, IntType:$dim, IntType:$index);
  let results = (outs ValueTensorType:$result);
}

def PrimNumToTensorScalarOp : Torch_Op<"prim.NumToTensor.Scalar", [Pure]> {
  let summary = "Scalar integer or float to a rank-0 tensor";
  let arguments = (ins AnyType:$a);
  let results = (outs ValueTensorType:$result);
}

def AtenDivTensorModeOp : Torch_Op<"aten.div.Tensor_mode", [Pure]> {
  let summary = "Elementwise division with a rounding mode";
  let arguments = (ins ValueTensorType:$self, ValueTensorType:$other, TorchStringType:$rounding_mode);
  let results = (outs ValueTensorType:$result);
}

def AtenPowScalarOp : Torch_Op<"aten.pow.Scalar", [Pure]> {
  let summary = "Scalar base raised to a tensor exponent";
  let arguments = (ins AnyType:$self, ValueTensorType:$exponent);
  let results = (outs ValueTensorType:$result);
}

def AtenRepeatOp : Torch_Op<"aten.repeat", [Pure]> {
  let summary = "Tile by a repeat list";
  let arguments = (ins ValueTensorType:$self, ListType:$repeats);
  let results = (outs ValueTensorType:$result);
}

def AtenSigmoidOp : Torch_Op<"aten.sigmoid", [Pure]> {
  let summary = "Elementwise sigmoid";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenTanhOp : Torch_Op<"aten.tanh", [Pure]> {
  let summary = "Elementwise hyperbolic tangent";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenGeluOp : Torch_Op<"aten.gelu", [Pure]> {
  let summary = "GELU activation";
  let arguments = (ins ValueTensorType:$self, AnyType:$approximate);
  let results = (outs ValueTensorType:$result);
}

def AtenEmbeddingOp : Torch_Op<"aten.embedding", [Pure]> {
  let summary = "Gather embedding rows by index";
  let arguments = (ins
    ValueTensorType:$weight,
    ValueTensorType:$indices,
    IntType:$padding_idx,
    TorchBoolType:$scale_grad_by_freq,
    TorchBoolType:$sparse
  );
  let results = (outs ValueTensorType:$result);
}

def AtenAliasOp : Torch_Op<"aten.alias", [Pure]> {
  let summary = "Identity view of a tensor";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenEqTensorOp : Torch_Op<"aten.eq.Tensor", [Pure]> {
  let summary = "Elementwise equality of two tensors";
  let arguments = (ins ValueTensorType:$self, ValueTensorType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenLeTensorOp : Torch_Op<"aten.le.Tensor", [Pure]> {
  let summary = "Elementwise less-or-equal of two tensors";
  let arguments = (ins ValueTensorType:$self, ValueTensorType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenMeanDimOp : Torch_Op<"aten.mean.dim", [Pure]> {
  let summary = "Mean along a dimension list";
  let arguments = (ins
    ValueTensorType:$self,
    ListType:$dim,
    TorchBoolType:$keepdim,
    AnyType:$dtype
  );
  let results = (outs ValueTensorType:$result);
}

def AtenNeScalarOp : Torch_Op<"aten.ne.Scalar", [Pure]> {
  let summary = "Elementwise not-equal against a scalar";
  let arguments = (ins ValueTensorType:$self, AnyType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenNegOp : Torch_Op<"aten.neg", [Pure]> {
  let summary = "Elementwise negation";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenPowTensorScalarOp : Torch_Op<"aten.pow.Tensor_Scalar", [Pure]> {
  let summary = "Raise a tensor to a scalar power";
  let arguments = (ins ValueTensorType:$self, AnyType:$exponent);
  let results = (outs ValueTensorType:$result);
}

def AtenPowTensorTensorOp : Torch_Op<"aten.pow.Tensor_Tensor", [Pure]> {
  let summary = "Raise a tensor to an elementwise tensor power";
  let arguments = (ins ValueTensorType:$self, ValueTensorType:$exponent);
  let results = (outs ValueTensorType:$result);
}

def AtenSubScalarOp : Torch_Op<"aten.sub.Scalar", [Pure]> {
  let summary = "Subtract a scalar with alpha";
  let arguments = (ins ValueTensorType:$self, AnyType:$other, AnyType:$alpha);
  let results = (outs ValueTensorType:$result);
}

def AtenClampOp : Torch_Op<"aten.clamp", [Pure]> {
  let summary = "Clamp a tensor between optional scalar bounds";
  let arguments = (ins ValueTensorType:$self, AnyType:$min, AnyType:$max);
  let results = (outs ValueTensorType:$result);
}

def AtenConstantPadNdOp : Torch_Op<"aten.constant_pad_nd", [Pure]> {
  let summary = "Pad a tensor with a constant, pairs counted from the last axis";
  let arguments = (ins ValueTensorType:$self, ListType:$pad, AnyType:$value);
  let results = (outs ValueTensorType:$result);
}

def PrimsSqueezeOp : Torch_Op<"prims.squeeze", [Pure]> {
  let summary = "Drop size-1 dimensions";
  let arguments = (ins ValueTensorType:$self, ListType:$dimensions);
  let results = (outs ValueTensorType:$result);
}

def AtenLinalgVectorNormOp : Torch_Op<"aten.linalg_vector_norm", [Pure]> {
  let summary = "L2 vector norm along a dimension list";
  let arguments = (ins
    ValueTensorType:$self,
    AnyType:$ord,
    AnyType:$dim,
    TorchBoolType:$keepdim,
    AnyType:$dtype
  );
  let results = (outs ValueTensorType:$result);
}

def AtenAbsOp : Torch_Op<"aten.abs", [Pure]> {
  let summary = "Elementwise absolute value";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenSignOp : Torch_Op<"aten.sign", [Pure]> {
  let summary = "Elementwise sign";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenCeilOp : Torch_Op<"aten.ceil", [Pure]> {
  let summary = "Elementwise ceiling";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenFloorOp : Torch_Op<"aten.floor", [Pure]> {
  let summary = "Elementwise floor";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenLogOp : Torch_Op<"aten.log", [Pure]> {
  let summary = "Elementwise natural logarithm";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenLog1pOp : Torch_Op<"aten.log1p", [Pure]> {
  let summary = "Elementwise log(1 + x)";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenSqrtOp : Torch_Op<"aten.sqrt", [Pure]> {
  let summary = "Elementwise square root";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenBitwiseNotOp : Torch_Op<"aten.bitwise_not", [Pure]> {
  let summary = "Elementwise bitwise complement";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenGtScalarOp : Torch_Op<"aten.gt.Scalar", [Pure]> {
  let summary = "Elementwise greater-than against a scalar";
  let arguments = (ins ValueTensorType:$self, AnyType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenLtScalarOp : Torch_Op<"aten.lt.Scalar", [Pure]> {
  let summary = "Elementwise less-than against a scalar";
  let arguments = (ins ValueTensorType:$self, AnyType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenLeScalarOp : Torch_Op<"aten.le.Scalar", [Pure]> {
  let summary = "Elementwise less-or-equal against a scalar";
  let arguments = (ins ValueTensorType:$self, AnyType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenGatherOp : Torch_Op<"aten.gather", [Pure]> {
  let summary = "Gather elements along one axis";
  let arguments = (ins
    ValueTensorType:$self,
    IntType:$dim,
    ValueTensorType:$index,
    TorchBoolType:$sparse_grad
  );
  let results = (outs ValueTensorType:$result);
}

def AtenGtTensorOp : Torch_Op<"aten.gt.Tensor", [Pure]> {
  let summary = "Elementwise greater-than";
  let arguments = (ins ValueTensorType:$self, ValueTensorType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenMinimumOp : Torch_Op<"aten.minimum", [Pure]> {
  let summary = "Elementwise minimum of two tensors";
  let arguments = (ins ValueTensorType:$self, ValueTensorType:$other);
  let results = (outs ValueTensorType:$result);
}

def AtenMaxDimOp : Torch_Op<"aten.max.dim", [Pure]> {
  let summary = "Maximum value and its index along one axis";
  let arguments = (ins ValueTensorType:$self, IntType:$dim, TorchBoolType:$keepdim);
  let results = (outs ValueTensorType:$values, ValueTensorType:$indices);
}

def AtenArgmaxOp : Torch_Op<"aten.argmax", [Pure]> {
  let summary = "Index of the maximum along one axis";
  let arguments = (ins ValueTensorType:$self, AnyType:$dim, TorchBoolType:$keepdim);
  let results = (outs ValueTensorType:$result);
}

def AtenExpOp : Torch_Op<"aten.exp", [Pure]> {
  let summary = "Elementwise natural exponential";
  let arguments = (ins ValueTensorType:$self);
  let results = (outs ValueTensorType:$result);
}

def AtenTopkOp : Torch_Op<"aten.topk", [Pure]> {
  let summary = "Top-k values and indices along one axis";
  let arguments = (ins
    ValueTensorType:$self,
    IntType:$k,
    IntType:$dim,
    TorchBoolType:$largest,
    TorchBoolType:$sorted
  );
  let results = (outs ValueTensorType:$values, ValueTensorType:$indices);
}

def AtenSplitWithSizesOp : Torch_Op<"aten.split_with_sizes", [Pure]> {
  let summary = "Split a tensor into sized sections along one axis";
  let arguments = (ins ValueTensorType:$self, ListType:$split_sizes, IntType:$dim);
  let results = (outs ListType:$result);
}

def PrimListUnpackOp : Torch_Op<"prim.ListUnpack", [Pure]> {
  let summary = "Unpack a tensor list into its elements";
  let arguments = (ins AnyType:$list);
  let results = (outs Variadic<AnyType>:$tensors);
}

#endif // TORCH_OPS
