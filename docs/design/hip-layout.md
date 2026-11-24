<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# HIP tensor layout encoding

`#hip.layout<"...">` is a ranked tensor encoding with one character per
tensor dimension. The encoding is invalid if the string length differs from
the tensor rank. `?` denotes an unknown dimension layout; for example,
`tensor<1x2x3x4xf32, #hip.layout<"????">>` has an unknown layout in each
dimension. A rank-zero tensor uses `#hip.layout<"">`.

The attribute does not annotate tensors automatically or change compilation
behavior. Annotation and enforcement of its presence are separate work.
