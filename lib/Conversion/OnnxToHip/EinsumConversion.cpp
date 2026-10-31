/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- EinsumConversion.cpp - selected onnx.Einsum -> hip.matmul ---------===//
//
// Batched attention contractions:
//
//   bhlk,bhtk->bhlt  ==  A[..., M, K] @ transpose(B[..., N, K])
//   bhlt,bhtv->bhlv  ==  A[..., M, K] @ B[..., K, N]
//
// SAM relative-position contractions. Both contract c. The shared spatial
// axis is a batch axis of the matmul, so the other operand is transposed
// and given a leading unit batch:
//
//   bhwc,hkc->bhwk  ==  A[b,h,w,c] @ transpose(B[h,k,c])
//   bhwc,wkc->bhwk  ==  transpose the w axis of A next to the batch, then
//                       the same matmul, then transpose w back
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

Value transposeStatic(mlir::PatternRewriter &rewriter, mlir::Location loc,
                     mlir::Value context, mlir::Value input,
                     llvm::ArrayRef<int64_t> perm) {
  auto inType = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t> outShape;
  for (int64_t axis : perm)
    outShape.push_back(inType.getDimSize(axis));
  auto outType =
      mlir::RankedTensorType::get(outShape, inType.getElementType());
  mlir::Value init = mlir::tensor::EmptyOp::create(
      rewriter, loc, outShape, inType.getElementType());
  auto transposed = mlir::hip::TransposeOp::create(
      rewriter, loc, context, input, init, rewriter.getI64ArrayAttr(perm));
  return transposed.getResult(0);
}

// [D0, D1, ...] -> [1, D0, D1, ...]. The unit axis shares the first
// input dimension's reassociation group.
mlir::Value unsqueezeLeading(mlir::PatternRewriter &rewriter,
                             mlir::Location loc, mlir::Value input) {
  auto inType = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t> outShape = {1};
  outShape.append(inType.getShape().begin(), inType.getShape().end());
  auto outType =
      mlir::RankedTensorType::get(outShape, inType.getElementType());
  llvm::SmallVector<mlir::ReassociationIndices> reassoc;
  reassoc.push_back({0, 1});
  for (int64_t axis = 2; axis < outType.getRank(); ++axis)
    reassoc.push_back({axis});
  llvm::SmallVector<mlir::OpFoldResult> shape;
  for (int64_t dim : outShape)
    shape.push_back(rewriter.getIndexAttr(dim));
  return mlir::tensor::ExpandShapeOp::create(rewriter, loc, outType, input,
                                             reassoc, shape)
      .getResult();
}

mlir::Value matmulStatic(mlir::PatternRewriter &rewriter, mlir::Location loc,
                         mlir::Value context, mlir::Value lhs, mlir::Value rhs,
                         mlir::RankedTensorType resultType) {
  mlir::Value init = mlir::tensor::EmptyOp::create(
      rewriter, loc, resultType.getShape(), resultType.getElementType());
  llvm::SmallVector<mlir::NamedAttribute> attrs = {
      rewriter.getNamedAttr("transA", rewriter.getI64IntegerAttr(0)),
      rewriter.getNamedAttr("transB", rewriter.getI64IntegerAttr(0)),
  };
  llvm::SmallVector<mlir::Value> operands = {context, lhs, rhs, init};
  auto hipOp = mlir::hip::MatmulOp::create(rewriter, loc, operands, attrs);
  return hipOp.getResult(0);
}

// bhwc,hkc->bhwk and bhwc,wkc->bhwk. SAM's relative-position einsums.
struct EinsumSpatialToHip : public mlir::RewritePattern {
  EinsumSpatialToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Einsum", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "einsum expects 2 operands");
    auto equationAttr = op->getAttrOfType<mlir::StringAttr>("equation");
    if (!equationAttr)
      return rewriter.notifyMatchFailure(op, "einsum has no equation");
    std::string equation;
    for (char c : equationAttr.getValue()) {
      if (c != ' ')
        equation.push_back(c);
    }
    bool contractH = equation == "bhwc,hkc->bhwk";
    bool contractW = equation == "bhwc,wkc->bhwk";
    if (!contractH && !contractW)
      return rewriter.notifyMatchFailure(
          op, "einsum equation is not a SAM spatial contraction");

    auto aType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto bType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!aType || !bType || !resultType || !aType.hasStaticShape() ||
        !bType.hasStaticShape() || !resultType.hasStaticShape() ||
        aType.getRank() != 4 || bType.getRank() != 3 ||
        resultType.getRank() != 4 ||
        !mlir::isa<mlir::FloatType>(aType.getElementType()) ||
        aType.getElementType() != bType.getElementType() ||
        aType.getElementType() != resultType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "SAM einsum expects static float tensors");
    }
    // A is [B, H, W, C]. B is [H, K, C] or [W, K, C]. Y is [B, H, W, K].
    int64_t shared = contractH ? aType.getDimSize(1) : aType.getDimSize(2);
    if (bType.getDimSize(0) != shared ||
        bType.getDimSize(2) != aType.getDimSize(3) ||
        resultType.getDimSize(0) != aType.getDimSize(0) ||
        resultType.getDimSize(1) != aType.getDimSize(1) ||
        resultType.getDimSize(2) != aType.getDimSize(2) ||
        resultType.getDimSize(3) != bType.getDimSize(1)) {
      return rewriter.notifyMatchFailure(op, "SAM einsum shapes do not match");
    }

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Location loc = op->getLoc();
    mlir::Value context = *ctxOrFailure;
    mlir::Value a = op->getOperand(0);
    mlir::Value b = transposeStatic(rewriter, loc, context, op->getOperand(1),
                                   {0, 2, 1});
    b = unsqueezeLeading(rewriter, loc, b);
    if (contractH) {
      rewriter.replaceOp(op, matmulStatic(rewriter, loc, context, a, b,
                                          resultType));
      return mlir::success();
    }
    a = transposeStatic(rewriter, loc, context, a, {0, 2, 1, 3});
    auto matType = mlir::RankedTensorType::get(
        {aType.getDimSize(0), aType.getDimSize(2), aType.getDimSize(1),
         bType.getDimSize(1)},
        aType.getElementType());
    mlir::Value product = matmulStatic(rewriter, loc, context, a, b, matType);
    rewriter.replaceOp(
        op, transposeStatic(rewriter, loc, context, product, {0, 2, 1, 3}));
    return mlir::success();
  }
};

} // namespace

void populateEinsumConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<EinsumSpatialToHip, EinsumBatchMatmulToHip>(ctx);
}

} // namespace hip
} // namespace mlir
