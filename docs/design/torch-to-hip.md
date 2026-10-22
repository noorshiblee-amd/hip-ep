<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Torch to HIP

The Torch dialect and `convert-torch-to-hip` live in the `torch_hip` plugin. The design for the accepted subset, the entry contract, and how to add an operation is `torch_hip/docs/torch-to-hip.md` in `torch-hip-plugin`.

This tree still owns the pipeline around that pass. `CompilerDriver` selects `torch-to-hip-pipeline` when the parsed module carries Torch IR, matched on the `torch` dialect namespace. The pipeline is `hip-add-context-arg`, then `convert-torch-to-hip`, then the tail of `onnx-to-hip-pipeline`. Pass names are in [pipeline_pass_menu.md](../pipeline_pass_menu.md). A build that does not select the plugin has no Torch parser. `hip-compiler` reports that the plugin is required.

The lowered module uses the same HIP operations as the ONNX path. See [supported-operations.md](../supported-operations.md).
