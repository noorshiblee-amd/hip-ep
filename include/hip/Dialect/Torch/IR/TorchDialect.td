//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef TORCH_DIALECT
#define TORCH_DIALECT

include "mlir/IR/DialectBase.td"

def Torch_Dialect : Dialect {
  let name = "torch";
  let summary = "Accepted subset of backend-contract Torch IR";
  let description = [{
    Parses the Torch operations this compiler lowers to tensor-mode HIP IR.
    The producer is external. This dialect spells only the operations and
    types the conversion accepts, plus enough structure for an unknown
    operation to round-trip in generic form.

    `!torch.vtensor` is ranked and value-semantic. Integer and bool tensor
    element types parse so a later pass can reject them; the accepted compute
    element types are f16, f32, and bf16.
  }];
  let cppNamespace = "::mlir::torch";
  let useDefaultTypePrinterParser = 1;
}

#endif // TORCH_DIALECT
