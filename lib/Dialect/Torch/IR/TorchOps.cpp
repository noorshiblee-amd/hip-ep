/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Torch/IR/TorchOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::torch;

#define GET_OP_CLASSES
#include "hip/Dialect/Torch/IR/TorchOps.cpp.inc"

ParseResult torch::parseDefaultTorchOp(OpAsmParser &parser,
                                        OperationState &result,
                                        int numOperands, int numResults) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> operandTypes;
  SmallVector<Type> resultTypes;
  if (parser.parseOperandList(operands) ||
      parser.parseColonTypeList(operandTypes))
    return failure();
  if (numResults > 0 &&
      (parser.parseArrow() || parser.parseTypeList(resultTypes)))
    return failure();
  if (static_cast<int>(operands.size()) != numOperands ||
      static_cast<int>(operandTypes.size()) != numOperands ||
      static_cast<int>(resultTypes.size()) != numResults) {
    return parser.emitError(parser.getNameLoc())
           << "expected " << numOperands << " operands and " << numResults
           << " results";
  }
  if (parser.resolveOperands(operands, operandTypes, parser.getNameLoc(),
                             result.operands))
    return failure();
  result.addTypes(resultTypes);
  return success();
}

void torch::printDefaultTorchOp(OpAsmPrinter &printer, Operation *op) {
  printer << ' ';
  printer.printOperands(op->getOperands());
  printer << " : ";
  llvm::interleaveComma(op->getOperandTypes(), printer);
  if (op->getNumResults() != 0) {
    printer << " -> ";
    llvm::interleaveComma(op->getResultTypes(), printer);
  }
}

#define TORCH_DEFAULT_FORM(OpClass, NumOperands, NumResults)                   \
  ParseResult OpClass::parse(OpAsmParser &parser, OperationState &result) {   \
    return parseDefaultTorchOp(parser, result, NumOperands, NumResults);      \
  }                                                                            \
  void OpClass::print(OpAsmPrinter &printer) {                                 \
    printDefaultTorchOp(printer, *this);                                       \
  }

TORCH_DEFAULT_FORM(AtenAddTensorOp, 3, 1)
TORCH_DEFAULT_FORM(AtenAddScalarOp, 3, 1)

#undef TORCH_DEFAULT_FORM

ParseResult ConstantIntOp::parse(OpAsmParser &parser, OperationState &result) {
  int64_t value = 0;
  if (parser.parseInteger(value))
    return failure();
  result.addAttribute("value", parser.getBuilder().getI64IntegerAttr(value));
  result.addTypes(IntType::get(parser.getContext()));
  return success();
}

void ConstantIntOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getValue();
}

ParseResult ConstantFloatOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  double value = 0.0;
  if (parser.parseFloat(value))
    return failure();
  result.addAttribute("value", parser.getBuilder().getF64FloatAttr(value));
  result.addTypes(TorchFloatType::get(parser.getContext()));
  return success();
}

void ConstantFloatOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printFloat(getValue());
}

ParseResult ValueTensorLiteralOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  Attribute attr;
  Type type;
  if (parser.parseLParen() || parser.parseAttribute(attr) ||
      parser.parseRParen() || parser.parseColon() || parser.parseType(type))
    return failure();
  if (isa<DenseResourceElementsAttr>(attr) || !isa<DenseElementsAttr>(attr)) {
    return parser.emitError(parser.getNameLoc())
           << "torch.vtensor.literal requires a dense elements attribute";
  }
  result.addAttribute("value", attr);
  result.addTypes(type);
  return success();
}

void ValueTensorLiteralOp::print(OpAsmPrinter &printer) {
  printer << '(';
  printer.printAttribute(getValue());
  printer << ") : ";
  printer.printType(getResult().getType());
}
