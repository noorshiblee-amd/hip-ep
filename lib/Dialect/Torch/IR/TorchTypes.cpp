/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Torch/IR/TorchDialect.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::torch;

Type ValueTensorType::parse(AsmParser &parser) {
  if (parser.parseLess() || parser.parseLSquare())
    return {};

  SmallVector<int64_t> sizes;
  // `[]` is rank 0. Otherwise each entry is a non-negative integer or `?`.
  if (failed(parser.parseOptionalRSquare())) {
    do {
      if (succeeded(parser.parseOptionalQuestion())) {
        sizes.push_back(ShapedType::kDynamic);
        continue;
      }
      int64_t dim = 0;
      if (parser.parseInteger(dim))
        return {};
      if (dim < 0) {
        parser.emitError(parser.getCurrentLocation())
            << "tensor dimension must be non-negative or '?'";
        return {};
      }
      sizes.push_back(dim);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare())
      return {};
  }

  Type dtype;
  if (parser.parseComma() || parser.parseType(dtype) || parser.parseGreater())
    return {};
  return ValueTensorType::get(parser.getContext(), sizes, dtype);
}

void ValueTensorType::print(AsmPrinter &printer) const {
  printer << '<';
  printer << '[';
  llvm::interleaveComma(getSizes(), printer, [&](int64_t dim) {
    if (ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
  });
  printer << "],";
  printer.printType(getDtype());
  printer << '>';
}
