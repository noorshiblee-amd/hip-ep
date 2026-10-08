//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef TORCH_TYPES
#define TORCH_TYPES

include "mlir/IR/AttrTypeBase.td"
include "hip/Dialect/Torch/IR/TorchDialect.td"

class Torch_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<Torch_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// The C++ class is the name parameter plus "Type".
def IntType : Torch_Type<"Int", "int"> {
  let summary = "Torch integer scalar";
}

// Class name avoids mlir::FloatType. The mnemonic is still `float`.
def TorchFloatType : Torch_Type<"TorchFloat", "float"> {
  let summary = "Torch float scalar";
}

def TorchBoolType : Torch_Type<"TorchBool", "bool"> {
  let summary = "Torch bool scalar";
}

// The C++ class is not mlir::NoneType. The mnemonic is still `none`.
def TorchNoneType : Torch_Type<"TorchNone", "none"> {
  let summary = "Torch singleton None";
}

def ValueTensorType : Torch_Type<"ValueTensor", "vtensor"> {
  let summary = "Ranked value-semantic Torch tensor";
  let description = [{
    Printed as `!torch.vtensor<[dims],element-type>`. A `?` dimension is
    dynamic. The element type is any builtin type the parser accepts; the
    conversion pass, not the parser, decides which element types lower.
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$sizes,
    "mlir::Type":$dtype
  );
  let hasCustomAssemblyFormat = 1;
}

#endif // TORCH_TYPES
