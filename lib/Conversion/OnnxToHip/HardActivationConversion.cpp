/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HardActivationConversion.cpp - HardSigmoid / HardSwish ------------===//
//
// onnx.HardSigmoid: y = max(0, min(1, alpha * x + beta))
// onnx.HardSwish:   y = x * HardSigmoid(x) with alpha = 1/6 and beta = 1/2
//
// Emit hip.mul / hip.add / hip.min / hip.max directly. The convert pass runs
// with GreedyRewriteStrictness::ExistingOps, so a synthesized onnx.* op would
// survive and fail one-shot bufferize.
//
//   Before:
//     %y = "onnx.HardSigmoid"(%x) {alpha = 0.2 : f32, beta = 0.5 : f32}
//
//   After:
//     %s = hip.mul(%ctx) ins(%x, %alpha : ...)
//     %a = hip.add(%ctx) ins(%s, %beta : ...)
//     %c = hip.min(%ctx) ins(%a, %one : ...)
//     %y = hip.max(%ctx) ins(%c, %zero : ...)
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"

namespace mlir {
namespace hip {
namespace {

static mlir::Value floatScalar(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::FloatType elemType,
                               double value) {
  auto scalarType = mlir::RankedTensorType::get({}, elemType);
  llvm::APFloat literal(value);
  bool losesInfo = false;
  literal.convert(elemType.getFloatSemantics(),
                  llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  mlir::OperationState state(loc, "onnx.Constant");
  state.addTypes(scalarType);
  state.addAttribute("value",
                     mlir::DenseElementsAttr::get(scalarType, literal));
  return rewriter.create(state)->getResult(0);
}

static mlir::Value emitBinary(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::Value context,
                              mlir::RankedTensorType resultType,
                              mlir::Value lhs, mlir::Value rhs,
                              bool isMul) {
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, lhs);
  if (isMul) {
    return mlir::hip::MulOp::create(rewriter, loc, context, lhs, rhs, init)
        ->getResult(0);
  }
  return mlir::hip::AddOp::create(rewriter, loc, context, lhs, rhs, init)
      ->getResult(0);
}

/// y = max(0, min(1, alpha * x + beta)).
static mlir::Value emitHardSigmoid(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc, mlir::Value context,
                                   mlir::Value x,
                                   mlir::RankedTensorType resultType,
                                   double alpha, double beta) {
  auto elemType = mlir::cast<mlir::FloatType>(resultType.getElementType());
  mlir::Value alphaV = floatScalar(rewriter, loc, elemType, alpha);
  mlir::Value betaV = floatScalar(rewriter, loc, elemType, beta);
  mlir::Value oneV = floatScalar(rewriter, loc, elemType, 1.0);
  mlir::Value zeroV = floatScalar(rewriter, loc, elemType, 0.0);

  mlir::Value scaled =
      emitBinary(rewriter, loc, context, resultType, x, alphaV, /*isMul=*/true);
  mlir::Value shifted = emitBinary(rewriter, loc, context, resultType, scaled,
                                   betaV, /*isMul=*/false);
  mlir::Value initHi = createEmptyTensor(rewriter, loc, resultType, shifted);
  mlir::Value clippedHi =
      mlir::hip::MinOp::create(rewriter, loc, resultType, context, shifted,
                               oneV, initHi)
          ->getResult(0);
  mlir::Value initLo = createEmptyTensor(rewriter, loc, resultType, clippedHi);
  return mlir::hip::MaxOp::create(rewriter, loc, resultType, context, clippedHi,
                                 zeroV, initLo)
      ->getResult(0);
}

struct HardSigmoidToHip : public mlir::RewritePattern {
  HardSigmoidToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.HardSigmoid", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.HardSigmoid expects 1 operand and 1 result");
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !mlir::isa<mlir::FloatType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "onnx.HardSigmoid expects a ranked float result");

    double alpha = 0.2;
    double beta = 0.5;
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("alpha"))
      alpha = attr.getValueAsDouble();
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("beta"))
      beta = attr.getValueAsDouble();

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();

    mlir::Value y = emitHardSigmoid(rewriter, op->getLoc(), *ctxOrFailure,
                                    op->getOperand(0), resultType, alpha, beta);
    rewriter.replaceOp(op, y);
    return mlir::success();
  }
};

struct HardSwishToHip : public mlir::RewritePattern {
  HardSwishToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.HardSwish", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.HardSwish expects 1 operand and 1 result");
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !mlir::isa<mlir::FloatType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "onnx.HardSwish expects a ranked float result");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();

    mlir::Location loc = op->getLoc();
    mlir::Value x = op->getOperand(0);
    // ONNX HardSwish fixes alpha = 1/6 and beta = 1/2.
    mlir::Value sig = emitHardSigmoid(rewriter, loc, *ctxOrFailure, x,
                                      resultType, 1.0 / 6.0, 0.5);
    mlir::Value y =
        emitBinary(rewriter, loc, *ctxOrFailure, resultType, x, sig,
                   /*isMul=*/true);
    rewriter.replaceOp(op, y);
    return mlir::success();
  }
};

} // namespace

void populateHardActivationConversionPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx) {
  patterns.add<HardSigmoidToHip, HardSwishToHip>(ctx);
}

} // namespace hip
} // namespace mlir
