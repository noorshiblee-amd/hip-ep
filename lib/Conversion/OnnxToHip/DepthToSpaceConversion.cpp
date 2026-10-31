/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- DepthToSpaceConversion.cpp - onnx.DepthToSpace -> reshape/transpose ===//
//
// CRD (pixel shuffle), blocksize B, channels C = C_out * B * B:
//   [N, C, H, W]
//   -> [N, C_out, B, B, H, W]
//   -> transpose [0, 1, 4, 2, 5, 3]
//   -> [N, C_out, H * B, W * B]
//
// DCR uses [N, B, B, C_out, H, W] and perm [0, 3, 4, 1, 5, 2].
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

namespace mlir {
namespace hip {
namespace {

mlir::Value transposeStatic(mlir::PatternRewriter &rewriter,
                            mlir::Location loc, mlir::Value context,
                            mlir::Value input, llvm::ArrayRef<int64_t> perm) {
  auto inType = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t> outShape;
  for (int64_t axis : perm)
    outShape.push_back(inType.getDimSize(axis));
  mlir::Value init = mlir::tensor::EmptyOp::create(
      rewriter, loc, outShape, inType.getElementType());
  return mlir::hip::TransposeOp::create(rewriter, loc, context, input, init,
                                        rewriter.getI64ArrayAttr(perm))
      .getResult(0);
}

struct DepthToSpaceToHip : public mlir::RewritePattern {
  DepthToSpaceToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.DepthToSpace", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "DepthToSpace expects 1 tensor");
    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || inputType.getRank() != 4 ||
        resultType.getRank() != 4 ||
        inputType.getElementType() != resultType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "DepthToSpace expects static rank-4 tensors");
    }
    auto blockAttr = op->getAttrOfType<mlir::IntegerAttr>("blocksize");
    if (!blockAttr)
      return rewriter.notifyMatchFailure(op, "DepthToSpace needs blocksize");
    int64_t block = blockAttr.getInt();
    if (block < 1)
      return rewriter.notifyMatchFailure(op, "blocksize must be positive");
    std::string mode = "DCR";
    if (auto modeAttr = op->getAttrOfType<mlir::StringAttr>("mode"))
      mode = modeAttr.getValue().str();

    int64_t n = inputType.getDimSize(0);
    int64_t channels = inputType.getDimSize(1);
    int64_t height = inputType.getDimSize(2);
    int64_t width = inputType.getDimSize(3);
    int64_t blockArea = block * block;
    if (blockArea == 0 || channels % blockArea != 0)
      return rewriter.notifyMatchFailure(
          op, "blocksize does not divide the channel count");
    int64_t outChannels = channels / blockArea;
    if (resultType.getDimSize(0) != n ||
        resultType.getDimSize(1) != outChannels ||
        resultType.getDimSize(2) != height * block ||
        resultType.getDimSize(3) != width * block) {
      return rewriter.notifyMatchFailure(op, "DepthToSpace result shape");
    }

    llvm::SmallVector<int64_t> splitShape;
    llvm::SmallVector<int64_t> perm;
    if (mode == "CRD") {
      splitShape = {n, outChannels, block, block, height, width};
      perm = {0, 1, 4, 2, 5, 3};
    } else if (mode == "DCR") {
      splitShape = {n, block, block, outChannels, height, width};
      perm = {0, 3, 4, 1, 5, 2};
    } else {
      return rewriter.notifyMatchFailure(op, "DepthToSpace mode must be CRD or DCR");
    }

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Location loc = op->getLoc();
    auto splitType =
        mlir::RankedTensorType::get(splitShape, inputType.getElementType());
    llvm::SmallVector<mlir::ReassociationIndices> splitReassoc = {
        {0}, {1, 2, 3}, {4}, {5}};
    llvm::SmallVector<mlir::OpFoldResult> splitFold;
    for (int64_t dim : splitShape)
      splitFold.push_back(rewriter.getIndexAttr(dim));
    mlir::Value split = mlir::tensor::ExpandShapeOp::create(
                            rewriter, loc, splitType, op->getOperand(0),
                            splitReassoc, splitFold)
                            .getResult();
    mlir::Value moved =
        transposeStatic(rewriter, loc, *ctxOrFailure, split, perm);
    llvm::SmallVector<mlir::ReassociationIndices> mergeReassoc = {
        {0}, {1}, {2, 3}, {4, 5}};
    mlir::Value merged = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, resultType, moved, mergeReassoc);
    rewriter.replaceOp(op, merged);
    return mlir::success();
  }
};

} // namespace

void populateDepthToSpaceConversionPatterns(RewritePatternSet &patterns,
                                            MLIRContext *ctx) {
  patterns.add<DepthToSpaceToHip>(ctx);
}

} // namespace hip
} // namespace mlir
