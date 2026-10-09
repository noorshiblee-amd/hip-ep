##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

set(LLVM_TARGET_DEFINITIONS TorchDialect.td)
mlir_tablegen(TorchDialect.h.inc -gen-dialect-decls -dialect=torch)
mlir_tablegen(TorchDialect.cpp.inc -gen-dialect-defs -dialect=torch)

set(LLVM_TARGET_DEFINITIONS TorchTypes.td)
mlir_tablegen(TorchTypes.h.inc -gen-typedef-decls -typedefs-dialect=torch)
mlir_tablegen(TorchTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=torch)

set(LLVM_TARGET_DEFINITIONS TorchOps.td)
mlir_tablegen(TorchOps.h.inc -gen-op-decls)
mlir_tablegen(TorchOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(TorchDialectIncGen)
