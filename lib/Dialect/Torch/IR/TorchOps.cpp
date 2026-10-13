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
                                       OperationState &result, int numOperands,
                                       int numResults) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> operandTypes;
  SmallVector<Type> resultTypes;
  if (parser.parseOperandList(operands) ||
      parser.parseColonTypeList(operandTypes))
    return failure();
  if (numResults != 0 &&
      (parser.parseArrow() || parser.parseTypeList(resultTypes)))
    return failure();
  // A negative count accepts a variadic list. Zero results omit the arrow.
  bool resultCountOk =
      numResults < 0
          ? !resultTypes.empty()
          : static_cast<int>(resultTypes.size()) == numResults;
  if (operands.size() != operandTypes.size() ||
      (numOperands >= 0 && static_cast<int>(operands.size()) != numOperands) ||
      !resultCountOk) {
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
  ParseResult OpClass::parse(OpAsmParser &parser, OperationState &result) {    \
    return parseDefaultTorchOp(parser, result, NumOperands, NumResults);       \
  }                                                                            \
  void OpClass::print(OpAsmPrinter &printer) {                                 \
    printDefaultTorchOp(printer, *this);                                       \
  }

TORCH_DEFAULT_FORM(AtenAddTensorOp, 3, 1)
TORCH_DEFAULT_FORM(AtenAddScalarOp, 3, 1)
TORCH_DEFAULT_FORM(AtenSubTensorOp, 3, 1)
TORCH_DEFAULT_FORM(AtenMulTensorOp, 2, 1)
TORCH_DEFAULT_FORM(AtenMulScalarOp, 2, 1)
TORCH_DEFAULT_FORM(AtenDivScalarOp, 2, 1)
TORCH_DEFAULT_FORM(AtenRsqrtOp, 1, 1)
TORCH_DEFAULT_FORM(AtenReluOp, 1, 1)
TORCH_DEFAULT_FORM(AtenUnsqueezeOp, 2, 1)
TORCH_DEFAULT_FORM(AtenViewOp, 2, 1)
TORCH_DEFAULT_FORM(AtenPermuteOp, 2, 1)
TORCH_DEFAULT_FORM(AtenBroadcastToOp, 2, 1)
TORCH_DEFAULT_FORM(AtenMmOp, 2, 1)
TORCH_DEFAULT_FORM(AtenSumDimIntListOp, 4, 1)
TORCH_DEFAULT_FORM(AtenMaxPool2dOp, 6, 1)
TORCH_DEFAULT_FORM(AtenConvolutionOp, 9, 1)
TORCH_DEFAULT_FORM(AtenExpandOp, 3, 1)
TORCH_DEFAULT_FORM(AtenBmmOp, 2, 1)
TORCH_DEFAULT_FORM(AtenVarMeanCorrectionOp, 4, 2)
TORCH_DEFAULT_FORM(AtenWhereSelfOp, 3, 1)
TORCH_DEFAULT_FORM(AtenScalarTensorOp, 5, 1)
TORCH_DEFAULT_FORM(AtenFullLikeOp, 7, 1)
TORCH_DEFAULT_FORM(AtenCloneOp, 2, 1)
TORCH_DEFAULT_FORM(AtenSoftmaxOp, 3, 1)
TORCH_DEFAULT_FORM(AtenEqScalarOp, 2, 1)
TORCH_DEFAULT_FORM(AtenAllDimOp, 3, 1)
TORCH_DEFAULT_FORM(AtenAssertTensorMetadataOp, 6, 0)
TORCH_DEFAULT_FORM(AtenArangeStartStepOp, 7, 1)
TORCH_DEFAULT_FORM(AtenSliceTensorOp, 5, 1)
TORCH_DEFAULT_FORM(AtenUpsampleNearest2dVecOp, 3, 1)
TORCH_DEFAULT_FORM(AtenDivTensorOp, 2, 1)
TORCH_DEFAULT_FORM(AtenBitwiseAndTensorOp, 2, 1)
TORCH_DEFAULT_FORM(AtenEmptyStridedOp, 6, 1)
TORCH_DEFAULT_FORM(PrimsConvertElementTypeOp, 2, 1)
TORCH_DEFAULT_FORM(AtenCatOp, 2, 1)
TORCH_DEFAULT_FORM(AtenCumsumOp, 3, 1)
TORCH_DEFAULT_FORM(AtenSinOp, 1, 1)
TORCH_DEFAULT_FORM(AtenCosOp, 1, 1)
TORCH_DEFAULT_FORM(AtenIntBoolOp, 1, 1)
TORCH_DEFAULT_FORM(AtenGeScalarOp, 2, 1)
TORCH_DEFAULT_FORM(AtenIndexTensorOp, 2, 1)
TORCH_DEFAULT_FORM(AtenMaxPool2dWithIndicesOp, 6, 2)
TORCH_DEFAULT_FORM(AtenSelectIntOp, 3, 1)
TORCH_DEFAULT_FORM(PrimNumToTensorScalarOp, 1, 1)
TORCH_DEFAULT_FORM(AtenDivTensorModeOp, 3, 1)
TORCH_DEFAULT_FORM(AtenPowScalarOp, 2, 1)
TORCH_DEFAULT_FORM(AtenRepeatOp, 2, 1)
TORCH_DEFAULT_FORM(AtenSigmoidOp, 1, 1)
TORCH_DEFAULT_FORM(AtenTanhOp, 1, 1)
TORCH_DEFAULT_FORM(AtenGeluOp, 2, 1)
TORCH_DEFAULT_FORM(AtenEmbeddingOp, 5, 1)
TORCH_DEFAULT_FORM(AtenAliasOp, 1, 1)
TORCH_DEFAULT_FORM(AtenEqTensorOp, 2, 1)
TORCH_DEFAULT_FORM(AtenLeTensorOp, 2, 1)
TORCH_DEFAULT_FORM(AtenMeanDimOp, 4, 1)
TORCH_DEFAULT_FORM(AtenNeScalarOp, 2, 1)
TORCH_DEFAULT_FORM(AtenNegOp, 1, 1)
TORCH_DEFAULT_FORM(AtenPowTensorScalarOp, 2, 1)
TORCH_DEFAULT_FORM(AtenSubScalarOp, 3, 1)
TORCH_DEFAULT_FORM(AtenClampOp, 3, 1)
TORCH_DEFAULT_FORM(AtenConstantPadNdOp, 3, 1)
TORCH_DEFAULT_FORM(PrimsSqueezeOp, 2, 1)
TORCH_DEFAULT_FORM(AtenLinalgVectorNormOp, 5, 1)
TORCH_DEFAULT_FORM(AtenAbsOp, 1, 1)
TORCH_DEFAULT_FORM(AtenSignOp, 1, 1)
TORCH_DEFAULT_FORM(AtenCeilOp, 1, 1)
TORCH_DEFAULT_FORM(AtenLogOp, 1, 1)
TORCH_DEFAULT_FORM(AtenSqrtOp, 1, 1)
TORCH_DEFAULT_FORM(AtenBitwiseNotOp, 1, 1)
TORCH_DEFAULT_FORM(AtenGtScalarOp, 2, 1)
TORCH_DEFAULT_FORM(AtenLtScalarOp, 2, 1)
TORCH_DEFAULT_FORM(AtenLeScalarOp, 2, 1)
TORCH_DEFAULT_FORM(AtenGatherOp, 4, 1)
TORCH_DEFAULT_FORM(AtenGtTensorOp, 2, 1)
TORCH_DEFAULT_FORM(AtenMaxDimOp, 3, 2)
TORCH_DEFAULT_FORM(AtenTopkOp, 5, 2)
TORCH_DEFAULT_FORM(AtenSplitWithSizesOp, 3, 1)
TORCH_DEFAULT_FORM(PrimListUnpackOp, 1, -1)

#undef TORCH_DEFAULT_FORM

// The producer prints ` : (operand-types) -> result-types`. The default
// printer omits the parentheses. Both forms are accepted.
ParseResult PrimListConstructOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> operandTypes;
  SmallVector<Type> resultTypes;
  if (parser.parseOperandList(operands) || parser.parseColon())
    return failure();
  bool wrapped = succeeded(parser.parseOptionalLParen());
  // `()` is an empty list. parseTypeList rejects that form.
  if (!(wrapped && succeeded(parser.parseOptionalRParen()))) {
    if (parser.parseTypeList(operandTypes))
      return failure();
    if (wrapped && parser.parseRParen())
      return failure();
  }
  if (parser.parseArrow() || parser.parseTypeList(resultTypes))
    return failure();
  if (operandTypes.size() != operands.size() || resultTypes.size() != 1) {
    return parser.emitError(parser.getNameLoc()) << "expected one list result";
  }
  if (parser.resolveOperands(operands, operandTypes, parser.getNameLoc(),
                             result.operands))
    return failure();
  result.addTypes(resultTypes);
  return success();
}

void PrimListConstructOp::print(OpAsmPrinter &printer) {
  printDefaultTorchOp(printer, *this);
}

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

ParseResult ConstantNoneOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addTypes(TorchNoneType::get(parser.getContext()));
  return success();
}

void ConstantNoneOp::print(OpAsmPrinter &printer) {
  printer.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult ConstantBoolOp::parse(OpAsmParser &parser, OperationState &result) {
  bool value = false;
  if (succeeded(parser.parseOptionalKeyword("true")))
    value = true;
  else if (parser.parseKeyword("false"))
    return failure();
  result.addAttribute("value", BoolAttr::get(parser.getContext(), value));
  result.addTypes(TorchBoolType::get(parser.getContext()));
  return success();
}

void ConstantBoolOp::print(OpAsmPrinter &printer) {
  printer << ' ' << (getValue() ? "true" : "false");
}

static ParseResult parseStringConstant(OpAsmParser &parser, OperationState &result,
                                Type resultType) {
  Attribute attr;
  if (parser.parseAttribute(attr))
    return failure();
  auto value = dyn_cast<StringAttr>(attr);
  if (!value) {
    return parser.emitError(parser.getNameLoc())
           << "expected a string constant";
  }
  result.addAttribute("value", value);
  result.addTypes(resultType);
  return success();
}

ParseResult ConstantDeviceOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  return parseStringConstant(parser, result,
                             DeviceType::get(parser.getContext()));
}

void ConstantDeviceOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printAttribute(getValueAttr());
}

ParseResult ConstantStrOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseStringConstant(parser, result,
                             TorchStringType::get(parser.getContext()));
}

void ConstantStrOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printAttribute(getValueAttr());
}

ParseResult ValueTensorLiteralOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  Attribute attr;
  Type type;
  if (parser.parseLParen() || parser.parseAttribute(attr) ||
      parser.parseRParen() || parser.parseColon() || parser.parseType(type))
    return failure();
  if (!isa<DenseElementsAttr>(attr) && !isa<DenseResourceElementsAttr>(attr)) {
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

ParseResult OperatorOp::parse(OpAsmParser &parser, OperationState &result) {
  Attribute name;
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseAttribute(name, "name", result.attributes) ||
      parser.parseLParen() || parser.parseOperandList(operands) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon())
    return failure();
  if (!isa<StringAttr>(name))
    return parser.emitError(parser.getNameLoc())
           << "torch.operator name must be a string";

  FunctionType fnType;
  if (parser.parseType(fnType))
    return failure();
  if (parser.resolveOperands(operands, fnType.getInputs(), parser.getNameLoc(),
                             result.operands))
    return failure();
  result.addTypes(fnType.getResults());
  return success();
}

void OperatorOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printAttribute(getNameAttr());
  printer << '(';
  printer.printOperands(getOperands());
  printer << ')';
  printer.printOptionalAttrDict((*this)->getAttrs(), {"name"});
  printer << " : ";
  printer.printFunctionalType(getOperandTypes(), getResultTypes());
}
