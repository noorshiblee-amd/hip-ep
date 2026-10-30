/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- EinsumConversion.cpp - selected onnx.Einsum -> hip.matmul ---------===//
//
// Two batched attention contractions used by convolutional vision
// transformers:
//
//   bhlk,bhtk->bhlt  ==  A[..., M, K] @ transpose(B[..., N, K])
//   bhlt,bhtv->bhlv  ==  A[..., M, K] @ B[..., K, N]
//
// Other equations are left unconverted.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

struct EinsumBatchMatmulToHip : public mlir::RewritePattern {
  EinsumBatchMatmulToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Einsum", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.Einsum lowering expects 2 operands and 1 result");
    auto equationAttr = op->getAttrOfType<mlir::StringAttr>("equation");
    if (!equationAttr)
      return rewriter.notifyMatchFailure(op, "onnx.Einsum has no equation");
    std::string equation;
    for (char c : equationAttr.getValue()) {
      if (c != ' ')
        equation.push_back(c);
    }
    int64_t transB = 0;
    if (equation == "bhlk,bhtk->bhlt")
      transB = 1;
    else if (equation == "bhlt,bhtv->bhlv")
      transB = 0;
    else
      return rewriter.notifyMatchFailure(
          op, "onnx.Einsum equation is not a supported batched matmul");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto aType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto bType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    if (!resultType || !aType || !bType || resultType.getRank() < 2)
      return rewriter.notifyMatchFailure(
          op, "onnx.Einsum expects ranked matmul tensors");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();

    mlir::Location loc = op->getLoc();
    mlir::Value a = op->getOperand(0);
    mlir::Value b = op->getOperand(1);
    llvm::SmallVector<mlir::Value> dynSizes;
    const int64_t rank = resultType.getRank();
    const int64_t bRank = bType.getRank();
    for (int64_t dimIdx : llvm::seq<int64_t>(rank)) {
      if (!resultType.isDynamicDim(dimIdx))
        continue;
      if (dimIdx == rank - 1) {
        int64_t bDim = transB ? bRank - 2 : bRank - 1;
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, b, bDim));
      } else if (dimIdx == rank - 2) {
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, a, aType.getRank() - 2));
      } else {
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, a, dimIdx));
      }
    }
    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);
    llvm::SmallVector<mlir::NamedAttribute> attrs = {
        rewriter.getNamedAttr("transA", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr("transB", rewriter.getI64IntegerAttr(transB)),
    };
    llvm::SmallVector<mlir::Value> operands = {*ctxOrFailure, a, b, init};
    auto hipOp = mlir::hip::MatmulOp::create(rewriter, loc, operands, attrs);
    rewriter.replaceOp(op, hipOp.getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateEinsumConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<EinsumBatchMatmulToHip>(ctx);
}

} // namespace hip
} // namespace mlir
