<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Torch to HIP

**Date:** 2026-10-06
**Document Type:** Design
**Status:** Draft
**Related:** [supported-operations.md](../supported-operations.md), [pipeline_pass_menu.md](../pipeline_pass_menu.md), [hip-shape-inference.md](hip-shape-inference.md), [unranked-tensor-handling.md](unranked-tensor-handling.md), [constant-handling-design.md](constant-handling-design.md), [output-allocator-design.md](output-allocator-design.md), [pool-allocs-memory-planning.md](pool-allocs-memory-planning.md), [compiler-runtime-contract.md](compiler-runtime-contract.md)

## Overview

`hip-mlir-opt` accepts a backend-contract Torch module produced outside this repository and lowers the accepted subset to tensor-mode HIP IR. The shared tail then bufferizes that IR, allocates graph outputs, and pools transients. The production compiler still ingests ONNX.

## Design

```text
Torch module
    |
    v
hip-add-context-arg
    |
    v
convert-torch-to-hip
    |
    v
shared tensor-HIP tail
    |
    v
hip-to-llvm-pipeline   (optional)
```

`torch-to-hip-pipeline` is `hip-add-context-arg`, then `convert-torch-to-hip`, then the tail of `onnx-to-hip-pipeline`. Pass names and tail order are in [pipeline_pass_menu.md](../pipeline_pass_menu.md). `hip-to-llvm-pipeline` is separate. It requires `@main_graph` when the C ABI is generated; conversion tests do not.

`convert-torch-to-hip` requires `!hip.context` as argument 0. `hip-add-context-arg` inserts it and does not rewrite other argument types. The pass then rewrites the function signature and the Torch operations in one step. Verifying between those two edits would reject the function, because a block argument cannot change type while a Torch operation still uses it.

### Entry contract

| Input | Accepted form |
|---|---|
| Tensor | Ranked `!torch.vtensor<[dims],dtype>`. `?` is a dynamic dimension. |
| Element type | `f16`, `f32`, `bf16` |
| Add | `torch.aten.add.Tensor`, `torch.aten.add.Scalar` |
| Sub / Mul / Div | `torch.aten.sub.Tensor`, `torch.aten.mul.Tensor`, `torch.aten.mul.Scalar`, `torch.aten.div.Scalar` |
| Alpha | `torch.constant.int`, `torch.constant.float`, or a `!torch.int` / `!torch.float` argument |
| Bool | `torch.constant.bool` |
| List | `!torch.list<int>` built by `torch.prim.ListConstruct` of constant integers |
| Tensor literal | `torch.vtensor.literal` with a `dense` attribute, or a `dense_resource` blob copied into one |
| Convolution / pool | `torch.aten.convolution` (transposed lowers to `hip.conv_transpose`), `torch.aten.max_pool2d`, `torch.aten.avg_pool2d` (average; `count_include_pad` selects the divisor and `divisor_override` must be none) |
| Normalization helpers | `torch.aten.rsqrt`, `torch.aten.relu`, `torch.aten.unsqueeze` |
| Reductions and shape | `torch.aten.sum.dim_IntList` (dtype `none`), `torch.aten.view`, `torch.aten.permute`, `torch.aten.broadcast_to`, `torch.aten.mm` |

Function arguments accept `f16`, `f32`, `bf16`, ranked `i1`, and ranked integer tensors (token ids). The float arithmetic ops (`aten.add`, `aten.sub`, `aten.mul`, `aten.div.Scalar`) still require float element types. Integer `aten.add.Scalar` / `aten.add.Tensor` / `aten.sub.Scalar` / `aten.sub.Tensor` / `aten.mul.Scalar` / `aten.mul.Tensor` lower when the result is `i16`, `i32`, or `i64`. A ranked `i1` or integer tensor produced inside the graph is accepted: comparisons, masks, and index vectors lower onto `hip.equal`, `hip.where`, `hip.and`, `hip.less`, `hip.not`, `hip.cast`, `hip.range`, `hip.gather`, and `hip.gather_nd`. Unranked values are rejected: [hip-shape-inference.md](hip-shape-inference.md) only narrows `?` on a ranked HIP result. A `dense_resource` literal is copied into a `DenseElementsAttr` before it becomes `hip.constant`. A handle whose blob is absent becomes a zero splat of the declared shape, including an integer tensor such as a position-id buffer. `torch.constant.none` is an omitted optional operand (convolution bias, sum dtype) and is dropped with that consumer. `torch.constant.device` and `torch.constant.str` are likewise dropped once their consumers have been lowered. `!torch.list<int>` becomes `tensor<Nxi64>`. `!torch.list<float>` is a scale-factor list and is dropped with its consumer. `!torch.list<vtensor>` and `!torch.list<optional<vtensor>>` stay as `prim.ListConstruct` until `aten.cat` or `aten.index.Tensor` reads the elements.

Full torch-mlir exports also lower `aten.expand`, `aten.bmm`, `aten.var_mean.correction` (population variance when correction is 0), `aten.where.self`, `aten.scalar_tensor`, `aten.full_like`, `aten.clone` (a `none` memory format is an identity), `aten.copy` (src forwarded when it matches the result type), `aten._softmax` (last dimension, or a transpose around it), `aten.eq.Scalar`, `aten.eq.Tensor`, `aten.ge.Scalar`, `aten.le.Tensor`, `aten.ne.Scalar`, `aten.all.dim`, `aten.mean.dim` (dtype `none`), `aten.arange.start_step`, `aten.slice.Tensor`, `aten.slice_scatter` (writes the source into one window of the base; a huge end is the open bound; an empty slice uses offset 0), `aten.select.int` (a negative index on a dynamic axis counts back from the runtime extent), `aten.index_select` (`hip.gather` on the selected axis), `aten.remainder.Scalar` (integer only, `hip.mod` with `fmod = 0`), `aten.pow.Tensor_Tensor` (`exp(exponent * log(base))`, so a negative base is undefined), `aten.log1p` (`hip.log` of `1 + x`), `aten.size.int`, `aten.ge.int` / `aten.le.int` / `aten.eq.int` (`arith.cmpi`), `aten._assert_scalar` (dropped), `aten.unique_consecutive` (only a 1-D input with at most one element, which is its own result), a single bool-mask `aten.index.Tensor` over the leading dims (`hip.nonzero` counts the kept rows, `hip.readback_dim` sizes the destination, and `hip.compress` writes them), `aten.upsample_nearest2d.vec` (the static result shape is the resize; the size list or the scale list may be none), `aten.bitwise_and.Tensor`, `aten.empty_strided`, `prims.convert_element_type`, `aten.cat` (mixed-dtype inputs are cast to the result element type; a negative dim counts from the result rank; a zero-element input is dropped), `aten.repeat` (extra repeat factors prepend size-1 dimensions), `aten.flip` (each axis is a `hip.gather` with reversed indices), `aten.scaled_dot_product_attention` and `aten._scaled_dot_product_flash_attention_for_cpu` (both `hip.gqa`; grouped-query uses the key head count), `torch.hipdnn.qmoe` (`hip.qmoe`; 4-bit packed expert weights, fused SwiGLU, block size taken from the scale groups), `aten.cumsum` (casts a bool input to the result element type), `aten.sin`, `aten.cos`, `aten.sigmoid`, `aten.tanh`, `aten.abs`, `aten.sign`, `aten.ceil`, `aten.floor` (`hip.floor`), `aten.log`, `aten.sqrt`, `aten.neg`, `aten.bitwise_not` (`i1` is `hip.not`), `aten.gt.Scalar` (`hip.less` with the scalar on the left), `aten.lt.Scalar` (`hip.less`), `aten.le.Scalar` (the complement of greater-than), `aten.gather` (`sparse_grad` false is `hip.gather_elements`), `aten.gt.Tensor` (`hip.less` with the operands swapped), `aten.minimum` (`hip.min`; both operands keep the result element type), `aten.max.dim` (top-1 `hip.top_k`; dropping keepdim reshapes the size-1 axis away), `aten.argmax` (the index half of that top-1 reduction; `dim` must be a constant), `aten.exp` (`hip.exp`), `aten.topk`, `aten.split_with_sizes` (static sections become `tensor.extract_slice` at `prim.ListUnpack`), `aten.Int.bool`, `aten.index.Tensor` (one tensor index is `hip.gather`, and `none` is a full slice; several tensor indices in one contiguous block are `hip.gather_nd`, and leading full slices are `batch_dims`), `aten.embedding` (`scale_grad_by_freq` and `sparse` false), `aten.alias`, `aten.max_pool2d_with_indices` (value only; a used index result is rejected), `prim.NumToTensor.Scalar` (an integer or float scalar becomes a rank-0 tensor), `aten.div.Tensor`, `aten.div.Tensor_mode` (`floor`), `aten.pow.Scalar`, `aten.pow.Tensor_Scalar` (exponent 2 is a multiply), `aten.repeat`, `aten.gelu` (`none` is exact erf, `tanh` is the hyperbolic approximation), `aten.clamp` (a missing bound is skipped; a present min is `hip.max` and a present max is `hip.min`), `aten.hardtanh` (both bounds present; `hip.max` then `hip.min`), `aten.constant_pad_nd` (padding pairs are counted from the last axis and stored as ONNX begin-then-end pads), `prims.squeeze` (a reshape that drops size-1 axes), `aten.linalg_vector_norm` (ord 2 is the square root of a sum of squares; a missing dim list reduces every axis; dtype `none` or the result element type), and `aten._assert_tensor_metadata` (erased). `aten.view` and `aten.unsqueeze` keep the input element type, so an `i1` or integer tensor can pass through them.

An operation this dialect does not spell fails at parse in pretty form. The generic quoted form parses, and `convert-torch-to-hip` then reports that it has no lowering.

### Add

`aten.add` is `self + alpha * other`. Alpha `1` or `1.0` is one `hip.add`. Any other alpha is `hip.mul` of `other` by a rank-0 tensor, then `hip.add`. A scalar `other` is materialized as a rank-0 tensor first. The result element type is authoritative: a mismatched operand is `hip.cast` before the add.

The result is destination-passing style: `tensor.empty` plus `hip.add`, the same tensor-mode form `convert-onnx-to-hip` emits. `hip.add` is the operation listed in [supported-operations.md](../supported-operations.md). Rank above 4 is packed to at most four dimensions inside this pass, then expanded back. The packer is the static broadcast rule already used for ONNX elementwise ops. A dynamic shape of rank 4 or less is left ranked and sized with `tensor.dim`. A dynamic shape of rank above 4 is rejected.

Tensor literals become `hip.constant` carriers with a deterministic `serialization_order`. The shared tail's `hip-externalize-constants` pass applies the storage policy. Scalar Torch constants, including bools, become `arith.constant` and are not carriers. A `!torch.list<int>` of constant integers becomes `tensor<Nxi64>` before any consumer reads it.

`aten.sub.Tensor` is `self - alpha * other`, with the same alpha rule as add. An integer tensor `other` is `hip.cast` to the float result type before the subtract. `aten.mul` and `aten.div.Scalar` are one `hip.mul` or `hip.div`; a scalar operand is a rank-0 tensor. `aten.rsqrt` is `hip.sqrt` then `hip.reciprocal`. `aten.relu` is `hip.max` against a rank-0 zero. `aten.unsqueeze` is `tensor.expand_shape` that groups the inserted size-1 axis with its neighbor. A rank-0 input is `tensor.reshape` to a length-1 vector. `aten.view` is `tensor.reshape`; one dynamic result dim is the input element count divided by the static result dims. `aten.permute` is `hip.transpose` (`perm[i]` is the input axis that moves to output axis `i`; the element type may be a float or an 8, 16, 32, or 64-bit integer). `aten.broadcast_to` is `hip.expand`. `aten.mm` is `hip.matmul`. `aten.sum.dim_IntList` is `hip.reduce_sum` after negative axes are rewritten against the input rank. The result element type may be f16, f32, bf16, i32, or i64; a bool or narrower input is `hip.cast` to it first. `aten.convolution`, `aten.max_pool2d`, and `aten.avg_pool2d` take one padding value per spatial axis and apply it to both sides. A convolution stride, padding, or dilation list of length one is repeated for every spatial axis; `hip.conv` and `hip.pool` receive that padding as begin axes followed by end axes. Convolution bias `none` is omitted. These rewrites require static shapes.

After the pass, a live `torch.*` operation or `!torch.*` type is an error. Graph outputs and transient buffers are unchanged from the ONNX tail: outputs are allocated with `hip.alloc_output`, and other transients are pooled. See [output-allocator-design.md](output-allocator-design.md) and [pool-allocs-memory-planning.md](pool-allocs-memory-planning.md).

### Adding another operation

An operation whose HIP counterpart already exists needs three additions:

1. A Torch operation declaration, using the default assembly (`operands : operand-types -> result-types`) unless the producer prints a different form.
2. A rewrite registered with the Torch-to-HIP population hook. The rewrite consumes builtin tensors and scalars. Type and constant conversion stay in the frontend adapter.
3. A conversion LIT test, plus a `torch-to-hip-pipeline` test when the operation changes bufferization or constants.

A HIP operation that does not exist yet has to be added before that rewrite. The rocmlir path is only for an operation that must enter `rocmlir-pipeline`. `torch-to-hip-pipeline` does not.

### Upstream Torch dialect

Type queries, scalar queries, and constant lowering go through one adapter. The Add rewrite does not name Torch tensor types. Replacing this parser with the upstream Torch dialect means reimplementing that adapter and the assembly parser. The Add rewrite stays.

## Related Documents

- [supported-operations.md](../supported-operations.md) — HIP Add, Mul, and Cast that this frontend targets.
- [pipeline_pass_menu.md](../pipeline_pass_menu.md) — `torch-to-hip-pipeline`, `convert-torch-to-hip`, and the shared tail.
- [hip-shape-inference.md](hip-shape-inference.md) — `?` narrowing happens after conversion, on ranked HIP results.
- [unranked-tensor-handling.md](unranked-tensor-handling.md) — why an unranked input is not given a rank here.
- [constant-handling-design.md](constant-handling-design.md) — `hip.constant` is a carrier; the externalizer chooses storage.
- [output-allocator-design.md](output-allocator-design.md) — returned tensors become `hip.alloc_output`.
- [pool-allocs-memory-planning.md](pool-allocs-memory-planning.md) — transient allocations are pooled.
- [compiler-runtime-contract.md](compiler-runtime-contract.md) — `@main_graph` and the C ABI, used only after the LLVM pipeline.
