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
             << "; tensor element type must be f16, f32, bf16, f64, i1, or "
                "an integer";
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
    int before = static_cast<int>(pending.size());
    for (Operation *op : pending) {
      auto it = byName.find(op->getName().getStringRef());
      if (it == byName.end())
        continue;
      bool ready = true;
      for (Value operand : op->getOperands()) {
        // None, device, and string operands are dropped by the consumer.
        // An integer list must already be an i64 tensor. A tensor list stays
        // a ListConstruct so cat and index can read its elements.
        Type type = operand.getType();
        if (isa<torch::TorchNoneType, torch::DeviceType,
                torch::TorchStringType>(type))
          continue;
        if (auto list = dyn_cast<torch::ListType>(type)) {
          if (!isa<torch::IntType>(list.getElementType()))
            continue;
        }
        if (torch::isTorchType(type))
          ready = false;
      }
      // split_with_sizes yields a tensor list. The unpack is ready only once
      // that split's tensor, size list, and axis have left the Torch type
      // system.
      if (ready && op->getName().getStringRef() == "torch.prim.ListUnpack") {
        Operation *split = op->getOperand(0).getDefiningOp();
        if (split &&
            split->getName().getStringRef() == "torch.aten.split_with_sizes" &&
            llvm::any_of(split->getOperands(), [](Value value) {
              return torch::isTorchType(value.getType());
            }))
          ready = false;
      }
      if (!ready)
        continue;
      OpBuilder builder(op);
      Value context = func.getArgument(0);
      if (failed(it->second(op, builder, context)))
        return failure();
      // The rewrite may have erased `op`. Do not touch it after this point.
    }
    int after = 0;
    func.walk([&](Operation *op) {
      if (torch::isTorchDialectOp(op))
        ++after;
    });
    // A rewrite that leaves its op in place (a tensor list waiting on cat or
    // index) must not count as progress, or this loop never finishes.
    if (after < before)
      progress = true;
  }

  SmallVector<Operation *> droppedLists;
  SmallVector<Operation *> droppedConstants;
  func.walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "torch.prim.ListConstruct")
      droppedLists.push_back(op);
    else if (name == "torch.constant.none" || name == "torch.constant.device" ||
             name == "torch.constant.str")
      droppedConstants.push_back(op);
  });
  // A use-empty list can still reference `torch.constant.none` (a full
  // slice inside aten.index.Tensor). Erase lists first so those constants
  // become dead.
  auto eraseIfDead = [](Operation *op) {
    if (!op->use_empty()) {
      op->emitError("still has uses after Torch-to-HIP conversion");
      return failure();
    }
    op->erase();
    return success();
  };
  for (Operation *op : droppedLists)
    if (failed(eraseIfDead(op)))
      return failure();
  for (Operation *op : droppedConstants)
    if (failed(eraseIfDead(op)))
      return failure();

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

/// Record the graph's input/output shapes on the module, the same six
/// attributes convert-onnx-to-hip stamps. GenerateInterface requires them, and
/// convert-hip-to-llvm silently skips building the (ctx, inputs) entry wrapper
/// without them, so the C ABI cannot be generated from a Torch module unless
/// they are present.
///
/// Must run before the rewrites replace the Torch function signature: the
/// shapes come from the `!torch.vtensor` types that the rewrites erase. A
/// module with no @main_graph is left alone -- conversion-only tests lower a
/// bare function and never reach the C ABI.
static LogicalResult generateModuleMetadata(ModuleOp module) {
  auto mainFunc = module.lookupSymbol<func::FuncOp>("main_graph");
  if (!mainFunc)
    return success();

  FunctionType funcType = mainFunc.getFunctionType();
  OpBuilder builder(module.getContext());

  auto collect = [&](TypeRange types, SmallVectorImpl<Attribute> &shapes,
                     SmallVectorImpl<int64_t> &elementSizes, int64_t &count,
                     StringRef what) -> LogicalResult {
    count = static_cast<int64_t>(types.size());
    for (Type type : types) {
      // hip-add-context-arg has already prepended !hip.context; it is not a
      // graph input.
      if (isa<hip::ContextType>(type)) {
        --count;
        continue;
      }
      // toBuiltinType, not toBuiltinTensorType: the latter accepts only float
      // element types, which would reject the integer token-id and mask
      // tensors the rewrites themselves accept as graph arguments.
      FailureOr<Type> converted = torch::toBuiltinType(type);
      auto tensorType = succeeded(converted)
                            ? dyn_cast<RankedTensorType>(*converted)
                            : nullptr;
      if (!tensorType)
        return mainFunc.emitError("unsupported ")
               << what << " type in @main_graph: " << type;
      Type elemType = tensorType.getElementType();
      if (!elemType.isIntOrFloat())
        return mainFunc.emitError("unsupported element type in @main_graph ")
               << what << ": " << elemType;
      shapes.push_back(builder.getDenseI64ArrayAttr(tensorType.getShape()));
      elementSizes.push_back(elemType.getIntOrFloatBitWidth() / 8);
    }
    return success();
  };

  int64_t inputCount = 0, outputCount = 0;
  SmallVector<Attribute> inputShapes, outputShapes;
  SmallVector<int64_t> inputElementSizes, outputElementSizes;

  if (failed(collect(funcType.getInputs(), inputShapes, inputElementSizes,
                     inputCount, "input")) ||
      failed(collect(funcType.getResults(), outputShapes, outputElementSizes,
                     outputCount, "output")))
    return failure();

  module->setAttr("hipdnn.input_count", builder.getI64IntegerAttr(inputCount));
  module->setAttr("hipdnn.input_shapes", builder.getArrayAttr(inputShapes));
  module->setAttr("hipdnn.input_element_sizes",
                  builder.getDenseI64ArrayAttr(inputElementSizes));
  module->setAttr("hipdnn.output_count",
                  builder.getI64IntegerAttr(outputCount));
  module->setAttr("hipdnn.output_shapes", builder.getArrayAttr(outputShapes));
  module->setAttr("hipdnn.output_element_sizes",
                  builder.getDenseI64ArrayAttr(outputElementSizes));
  return success();
}

void ConvertTorchToHipPass::runOnOperation() {
  SmallVector<torch::TorchRewrite> rewrites;
  torch::populateTorchToHipRewrites(rewrites);

  int64_t serializationOrder = 0;
  ModuleOp module = getOperation();
  if (failed(generateModuleMetadata(module))) {
    signalPassFailure();
    return;
  }
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
