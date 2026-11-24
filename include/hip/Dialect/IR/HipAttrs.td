//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_ATTRS
#define HIP_ATTRS

include "hip/Dialect/IR/HipDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/TensorEncoding.td"

def Hip_LayoutAttr : AttrDef<Hip_Dialect, "Layout",
    [DeclareAttrInterfaceMethods<VerifiableTensorEncoding>]> {
  let mnemonic = "layout";
  let parameters = (ins "::mlir::StringAttr":$value);
  let assemblyFormat = "`<` $value `>`";
}

#endif // HIP_ATTRS
