/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "PopulatePatterns.h"
#include "TorchFrontend.h"

#include "hip/Conversion/TorchToHip/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Torch/IR/TorchOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

struct ConvertTorchToHipPass
    : public PassWrapper<ConvertTorchToHipPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTorchToHipPass)

  StringRef getArgument() const override { return "convert-torch-to-hip"; }

  StringRef getDescription() const override {
    return "Lower the accepted Torch subset to tensor-mode HIP";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hip::HipDialect, arith::ArithDialect, tensor::TensorDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override;
};

LogicalResult convertFunctionArguments(func::FuncOp func, bool &hasTorch) {
  hasTorch = false;
  for (BlockArgument arg : func.getArguments()) {
    if (torch::isTorchType(arg.getType()))
      hasTorch = true;
  }
  if (!hasTorch) {
    func.walk([&](Operation *op) {
      if (torch::isTorchDialectOp(op))
        hasTorch = true;
    });
  }
  if (!hasTorch)
    return success();
  if (func.isExternal())
    return func.emitError("Torch function has no body");
  if (func.getNumArguments() == 0 ||
      !isa<hip::ContextType>(func.getArgument(0).getType())) {
    return func.emitError(
        "convert-torch-to-hip requires !hip.context as argument 0; run "
        "hip-add-context-arg first");
  }

  SmallVector<Type> argTypes;
  argTypes.reserve(func.getNumArguments());
  for (auto [index, arg] : llvm::enumerate(func.getArguments())) {
    FailureOr<Type> converted = torch::toBuiltinType(arg.getType());
    if (failed(converted)) {
      return func.emitError()
             << "argument " << index << " has unsupported Torch type "
             << arg.getType()
             << "; tensor element type must be f16, f32, or bf16";
    }
    arg.setType(*converted);
    argTypes.push_back(*converted);
  }
  func.setFunctionType(
      FunctionType::get(func.getContext(), argTypes, func.getResultTypes()));
  return success();
}

LogicalResult lowerFunction(func::FuncOp func,
                            ArrayRef<torch::TorchRewrite> rewrites,
                            int64_t &serializationOrder) {
  bool hasTorch = false;
  if (failed(convertFunctionArguments(func, hasTorch)))
    return failure();
  if (!hasTorch)
    return success();

  SmallVector<Operation *> carriers;
  func.walk([&](Operation *op) {
    if (torch::isTorchDialectOp(op))
      carriers.push_back(op);
  });
  for (Operation *op : carriers) {
    torch::CarrierResult result =
        torch::lowerTorchCarrier(op, serializationOrder);
    if (result == torch::CarrierResult::Failed)
      return failure();
  }

  DenseMap<StringRef, torch::TorchRewriteFn> byName;
  for (const torch::TorchRewrite &rewrite : rewrites)
    byName[rewrite.name] = rewrite.rewrite;

  bool progress = true;
  while (progress) {
    progress = false;
    SmallVector<Operation *> pending;
    func.walk([&](Operation *op) {
      if (torch::isTorchDialectOp(op))
        pending.push_back(op);
    });
    for (Operation *op : pending) {
      auto it = byName.find(op->getName().getStringRef());
      if (it == byName.end())
        continue;
      bool ready = true;
      for (Value operand : op->getOperands()) {
        // None marks an omitted optional (convolution bias, sum dtype). The
        // consumer drops it; it must not keep that consumer unlowered.
        if (isa<torch::TorchNoneType>(operand.getType()))
          continue;
        if (torch::isTorchType(operand.getType()))
          ready = false;
      }
      if (!ready)
        continue;
      OpBuilder builder(op);
      Value context = func.getArgument(0);
      if (failed(it->second(op, builder, context)))
        return failure();
      progress = true;
    }
  }

  SmallVector<Operation *> leftoverNones;
  func.walk([&](torch::ConstantNoneOp op) { leftoverNones.push_back(op); });
  for (Operation *op : leftoverNones) {
    if (!op->use_empty()) {
      op->emitError("torch.constant.none still has uses");
      return failure();
    }
    op->erase();
  }

  LogicalResult leftover = success();
  func.walk([&](Operation *op) {
    if (!torch::isTorchDialectOp(op))
      return;
    op->emitError("no Torch-to-HIP lowering for ") << op->getName();
    leftover = failure();
  });
  if (failed(leftover))
    return failure();

  Operation *terminator = func.getBody().front().getTerminator();
  auto ret = dyn_cast<func::ReturnOp>(terminator);
  if (!ret)
    return func.emitError("expected func.return");
  for (Type type : ret.getOperandTypes()) {
    if (torch::isTorchType(type))
      return func.emitError("Torch type survived conversion: ") << type;
  }
  for (Type type : func.getArgumentTypes()) {
    if (torch::isTorchType(type))
      return func.emitError("Torch type survived conversion: ") << type;
  }
  func.setFunctionType(FunctionType::get(
      func.getContext(), func.getArgumentTypes(), ret.getOperandTypes()));
  return success();
}

void ConvertTorchToHipPass::runOnOperation() {
  SmallVector<torch::TorchRewrite> rewrites;
  torch::populateTorchToHipRewrites(rewrites);

  int64_t serializationOrder = 0;
  ModuleOp module = getOperation();
  auto result = module.walk([&](func::FuncOp func) {
    if (failed(lowerFunction(func, rewrites, serializationOrder)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::hip::createConvertTorchToHipPass() {
  return std::make_unique<ConvertTorchToHipPass>();
}
