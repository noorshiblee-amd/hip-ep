/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// torch.operator "onnx.*" is the torch-mlir ONNX importer's spelling. This
// rewrite turns that spelling into the generic onnx dialect ops that
// convert-onnx-to-hip already lowers. !torch.none results have no shape in
// the importer output; the result type is inferred from the ONNX op.
// dense_resource payloads are materialized to a DenseElementsAttr because
// hip.constant serializes inline dense data, not a resource handle.
//
// Before:
//   %z = torch.operator "onnx.Constant"()
//       {torch.onnx.value = dense_resource<z> : tensor<ui8>}
//       : () -> !torch.vtensor<[],ui8>
//   %q = torch.operator "onnx.QuantizeLinear"(%x, %s, %z)
//       : (!torch.vtensor<[2,3],f32>, !torch.vtensor<[],f32>,
//          !torch.vtensor<[],ui8>) -> !torch.none
// After:
//   %z = "onnx.Constant"() {value = dense<0> : tensor<ui8>} : () -> tensor<ui8>
//   %q = "onnx.QuantizeLinear"(%x, %s, %z)
//       : (tensor<2x3xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x3xui8>

#include "PopulatePatterns.h"
#include "TorchFrontend.h"

#include "hip/Dialect/Torch/IR/TorchOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

FailureOr<Attribute> materializeResource(DenseResourceElementsAttr resource) {
  auto shaped = dyn_cast<ShapedType>(resource.getType());
  if (!shaped)
    return failure();
  ArrayRef<char> data = resource.getData();
  if (data.empty() && shaped.getNumElements() != 0)
    return failure();
  bool splat = false;
  if (!DenseElementsAttr::isValidRawBuffer(shaped, data, splat))
    return failure();
  return DenseElementsAttr::getFromRawBuffer(shaped, data);
}

LogicalResult broadcastDims(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs,
                            SmallVectorImpl<int64_t> &out) {
  const int64_t rank = std::max(static_cast<int64_t>(lhs.size()),
                                static_cast<int64_t>(rhs.size()));
  out.resize(rank);
  for (int64_t index : llvm::seq<int64_t>(rank)) {
    const int64_t lhsDim = index < static_cast<int64_t>(lhs.size())
                               ? lhs[lhs.size() - 1 - index]
                               : 1;
    const int64_t rhsDim = index < static_cast<int64_t>(rhs.size())
                               ? rhs[rhs.size() - 1 - index]
                               : 1;
    int64_t dim = lhsDim;
    if (lhsDim == rhsDim || rhsDim == 1)
      dim = lhsDim;
    else if (lhsDim == 1)
      dim = rhsDim;
    else if (ShapedType::isDynamic(lhsDim) || ShapedType::isDynamic(rhsDim))
      dim = ShapedType::kDynamic;
    else
      return failure();
    out[rank - 1 - index] = dim;
  }
  return success();
}

FailureOr<RankedTensorType> rankedOperand(Operation *op, Value value,
                                          StringRef role) {
  auto ranked = dyn_cast<RankedTensorType>(value.getType());
  if (!ranked || !ranked.getElementType().isIntOrFloat()) {
    op->emitError() << "onnx " << role
                    << " must be a ranked integer or float tensor";
    return failure();
  }
  return ranked;
}

FailureOr<RankedTensorType> inferQuantizeLinear(Operation *op) {
  FailureOr<RankedTensorType> input =
      rankedOperand(op, op->getOperand(0), "input");
  if (failed(input))
    return failure();
  Type elementType =
      IntegerType::get(op->getContext(), 8, IntegerType::Unsigned);
  if (op->getNumOperands() >= 3) {
    FailureOr<RankedTensorType> zeroPoint =
        rankedOperand(op, op->getOperand(2), "zero point");
    if (failed(zeroPoint))
      return failure();
    if (!zeroPoint->getElementType().isInteger()) {
      op->emitError("onnx.QuantizeLinear zero point must be an integer tensor");
      return failure();
    }
    elementType = zeroPoint->getElementType();
  }
  return RankedTensorType::get(input->getShape(), elementType);
}

FailureOr<RankedTensorType> inferDequantizeLinear(Operation *op) {
  if (op->getNumOperands() < 2) {
    op->emitError("onnx.DequantizeLinear expects an input and a scale");
    return failure();
  }
  FailureOr<RankedTensorType> input =
      rankedOperand(op, op->getOperand(0), "input");
  FailureOr<RankedTensorType> scale =
      rankedOperand(op, op->getOperand(1), "scale");
  if (failed(input) || failed(scale))
    return failure();
  if (!scale->getElementType().isFloat()) {
    op->emitError("onnx.DequantizeLinear scale must be a float tensor");
    return failure();
  }
  return RankedTensorType::get(input->getShape(), scale->getElementType());
}

FailureOr<RankedTensorType> inferMatMul(Operation *op) {
  if (op->getNumOperands() != 2) {
    op->emitError("onnx.MatMul expects two operands");
    return failure();
  }
  FailureOr<RankedTensorType> lhs = rankedOperand(op, op->getOperand(0), "lhs");
  FailureOr<RankedTensorType> rhs = rankedOperand(op, op->getOperand(1), "rhs");
  if (failed(lhs) || failed(rhs))
    return failure();
  if (lhs->getElementType() != rhs->getElementType()) {
    op->emitError("onnx.MatMul operands have different element types");
    return failure();
  }
  if (lhs->getRank() == 0 || rhs->getRank() == 0) {
    op->emitError("onnx.MatMul operands must have rank at least 1");
    return failure();
  }

  const bool squeezeM = lhs->getRank() == 1;
  const bool squeezeN = rhs->getRank() == 1;
  const int64_t m = squeezeM ? 1 : lhs->getDimSize(lhs->getRank() - 2);
  const int64_t kLhs = lhs->getDimSize(lhs->getRank() - 1);
  const int64_t kRhs =
      squeezeN ? rhs->getDimSize(0) : rhs->getDimSize(rhs->getRank() - 2);
  const int64_t n = squeezeN ? 1 : rhs->getDimSize(rhs->getRank() - 1);
  if (!ShapedType::isDynamic(kLhs) && !ShapedType::isDynamic(kRhs) &&
      kLhs != kRhs) {
    op->emitError("onnx.MatMul inner dimensions differ");
    return failure();
  }

  ArrayRef<int64_t> lhsBatch =
      squeezeM ? ArrayRef<int64_t>{} : lhs->getShape().drop_back(2);
  ArrayRef<int64_t> rhsBatch =
      squeezeN ? ArrayRef<int64_t>{} : rhs->getShape().drop_back(2);
  SmallVector<int64_t> shape;
  if (failed(broadcastDims(lhsBatch, rhsBatch, shape))) {
    op->emitError("onnx.MatMul batch dimensions do not broadcast");
    return failure();
  }
  if (!squeezeM)
    shape.push_back(m);
  if (!squeezeN)
    shape.push_back(n);
  return RankedTensorType::get(shape, lhs->getElementType());
}

FailureOr<RankedTensorType> inferAdd(Operation *op) {
  if (op->getNumOperands() != 2) {
    op->emitError("onnx.Add expects two operands");
    return failure();
  }
  FailureOr<RankedTensorType> lhs = rankedOperand(op, op->getOperand(0), "lhs");
  FailureOr<RankedTensorType> rhs = rankedOperand(op, op->getOperand(1), "rhs");
  if (failed(lhs) || failed(rhs))
    return failure();
  if (lhs->getElementType() != rhs->getElementType()) {
    op->emitError("onnx.Add operands have different element types");
    return failure();
  }
  SmallVector<int64_t> shape;
  if (failed(broadcastDims(lhs->getShape(), rhs->getShape(), shape))) {
    op->emitError("onnx.Add shapes do not broadcast");
    return failure();
  }
  return RankedTensorType::get(shape, lhs->getElementType());
}

FailureOr<RankedTensorType> explicitTensorResult(Operation *op, Type type) {
  auto valueTensor = dyn_cast<torch::ValueTensorType>(type);
  if (!valueTensor) {
    op->emitError(
        "torch.operator result must be !torch.vtensor or !torch.none");
    return failure();
  }
  if (!valueTensor.getDtype().isIntOrFloat()) {
    op->emitError("torch.vtensor element type must be an integer or a float");
    return failure();
  }
  return RankedTensorType::get(valueTensor.getSizes(), valueTensor.getDtype());
}

FailureOr<RankedTensorType> inferNoneResult(Operation *op, StringRef onnxName) {
  if (onnxName == "onnx.QuantizeLinear")
    return inferQuantizeLinear(op);
  if (onnxName == "onnx.DequantizeLinear")
    return inferDequantizeLinear(op);
  if (onnxName == "onnx.MatMul")
    return inferMatMul(op);
  if (onnxName == "onnx.Add")
    return inferAdd(op);
  op->emitError("cannot infer !torch.none result of ") << onnxName;
  return failure();
}

LogicalResult lowerOnnxOperator(Operation *op, OpBuilder &builder, Value) {
  // hip-mlir-opt registers the stub (or the modeled dialect) but does not load
  // it until an onnx op is parsed. This rewrite is the first onnx op.
  if (!op->getContext()->getOrLoadDialect("onnx"))
    return op->emitError("onnx dialect is not registered");

  auto oper = cast<torch::OperatorOp>(op);
  StringRef onnxName = oper.getName();
  if (!onnxName.starts_with("onnx."))
    return op->emitError(
               "torch.operator lowering only accepts onnx.* names, got ")
           << onnxName;
  if (op->getNumResults() != 1)
    return op->emitError("torch.operator onnx lowering expects one result");

  Type torchResult = op->getResult(0).getType();
  FailureOr<RankedTensorType> resultType =
      isa<torch::TorchNoneType>(torchResult)
          ? inferNoneResult(op, onnxName)
          : explicitTensorResult(op, torchResult);
  if (failed(resultType))
    return failure();

  NamedAttrList onnxAttrs;
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef key = attr.getName().getValue();
    if (key == "name")
      continue;
    StringRef stripped = key;
    stripped.consume_front("torch.onnx.");
    Attribute value = attr.getValue();
    if (auto resource = dyn_cast<DenseResourceElementsAttr>(value)) {
      FailureOr<Attribute> dense = materializeResource(resource);
      if (failed(dense)) {
        return op->emitError("cannot read dense_resource attribute ")
               << stripped;
      }
      value = *dense;
    }
    onnxAttrs.set(StringAttr::get(op->getContext(), stripped), value);
  }

  if (onnxName == "onnx.Constant") {
    auto elements = dyn_cast_or_null<ElementsAttr>(onnxAttrs.get("value"));
    if (!elements) {
      return op->emitError(
          "onnx.Constant requires a torch.onnx.value elements attribute");
    }
    if (elements.getShapedType() != *resultType) {
      return op->emitError(
          "onnx.Constant value type does not match its result type");
    }
  }

  OperationState state(op->getLoc(), onnxName);
  state.addOperands(op->getOperands());
  state.addTypes(*resultType);
  state.addAttributes(onnxAttrs.getAttrs());
  Operation *lowered = builder.create(state);
  op->replaceAllUsesWith(lowered->getResults());
  op->erase();
  return success();
}

} // namespace

void mlir::torch::populateOnnxOperatorRewrites(
    SmallVectorImpl<TorchRewrite> &rewrites) {
  rewrites.push_back({"torch.operator", lowerOnnxOperator});
}

void mlir::torch::renameSoleOnnxEntry(ModuleOp module) {
  if (module.lookupSymbol<func::FuncOp>("main_graph"))
    return;
  SmallVector<func::FuncOp> funcs(module.getOps<func::FuncOp>());
  if (funcs.size() != 1)
    return;
  bool hasOnnx = false;
  funcs.front().walk([&](Operation *op) {
    if (op->getName().getStringRef().starts_with("onnx."))
      hasOnnx = true;
  });
  if (hasOnnx)
    funcs.front().setSymName("main_graph");
}
