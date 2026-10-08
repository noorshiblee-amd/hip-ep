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
| Alpha | `torch.constant.int`, `torch.constant.float`, or a `!torch.int` / `!torch.float` argument |
| Tensor literal | `torch.vtensor.literal` with a `dense` elements attribute |
| ONNX importer op | `torch.operator "onnx.<Op>"` for `Constant`, `QuantizeLinear`, `DequantizeLinear`, `MatMul`, and `Add` |
| Missing result type | `!torch.none`, inferred from the ONNX op before the op is emitted |
| Unused None | `torch.constant.none` with no uses is erased |

`aten.add` still rejects integer tensors and `i1` tensors. Bool `aten.add` is a logical or, and an integer tensor would become an arithmetic `hip.add`. `torch.operator "onnx.*"` accepts integer and unsigned tensors (`ui8`, `ui16`, `si32`, and the other builtin integer and float element types) because QuantizeLinear and DequantizeLinear use them. Unranked values are rejected: [hip-shape-inference.md](hip-shape-inference.md) only narrows `?` on a ranked HIP result. `torch.vtensor.literal` still rejects `dense_resource`. A `torch.onnx.value` `dense_resource` on `torch.operator "onnx.Constant"` is materialized to a `dense` payload, because [constant-handling-design.md](constant-handling-design.md) serializes inline dense data.

`!torch.none` is the importer's untyped result. QuantizeLinear takes the zero-point element type and the input shape. DequantizeLinear takes the scale element type and the input shape. MatMul and Add use the operand element type; MatMul builds the ONNX matrix shape and Add broadcasts. A module with a single function is renamed to `@main_graph` after those ops are emitted, because `convert-onnx-to-hip` reads that symbol.

An operation this dialect does not spell fails at parse in pretty form. The generic quoted form parses, and `convert-torch-to-hip` then reports that it has no lowering.

### Add

`aten.add` is `self + alpha * other`. Alpha `1` or `1.0` is one `hip.add`. Any other alpha is `hip.mul` of `other` by a rank-0 tensor, then `hip.add`. A scalar `other` is materialized as a rank-0 tensor first. The result element type is authoritative: a mismatched operand is `hip.cast` before the add.

The result is destination-passing style: `tensor.empty` plus `hip.add`, the same tensor-mode form `convert-onnx-to-hip` emits. `hip.add` is the operation listed in [supported-operations.md](../supported-operations.md). Rank above 4 is packed to at most four dimensions inside this pass, then expanded back. The packer is the static broadcast rule already used for ONNX elementwise ops. A dynamic shape of rank 4 or less is left ranked and sized with `tensor.dim`. A dynamic shape of rank above 4 is rejected.

Tensor literals become `hip.constant` carriers with a deterministic `serialization_order`. The shared tail's `hip-externalize-constants` pass applies the storage policy. Scalar Torch constants become `arith.constant` and are not carriers.

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
