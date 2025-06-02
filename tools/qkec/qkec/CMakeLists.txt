# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=qkec.mlir.")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Include MLIR Python utilities
include(AddMLIRPython)

# Set up include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${CUDAQ_INSTALL_DIR}/lib)
include_directories(${CUDAQ_INSTALL_DIR}/include)

# Declare the main Python source group
declare_mlir_python_sources(QuakeCPythonSources)

cudaq_declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QuakeCPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE ${CUDAQ_INSTALL_DIR}/include/cudaq/python/QuakeOps.td
  SOURCES
    dialects/quake.py
  DIALECT_NAME quake)

cudaq_declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QuakeCPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE ${CUDAQ_INSTALL_DIR}/include/cudaq/python/CCOps.td
  SOURCES
    dialects/cc.py
  DIALECT_NAME cc)

# Declare dialect-specific Python sources
declare_mlir_python_sources(QuakeCPythonSources.Dialects
  ADD_TO_PARENT QuakeCPythonSources
)

# Add core Python sources (if you have any custom Python files)
declare_mlir_python_sources(QuakeCPythonSources.Core
  ADD_TO_PARENT QuakeCPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
    visitor.py
)

declare_mlir_python_extension(QuakeCPythonSources.Extension
  MODULE_NAME _qkecExtension
  ADD_TO_PARENT QuakeCPythonSources
  SOURCES
    qkec_extension.cpp
)

# Create the CAPI library that links CUDA-Q dialects
add_mlir_python_common_capi_library(QuakeCPythonCAPI
  INSTALL_COMPONENT QuakeCPythonModules
  INSTALL_DESTINATION python/qkec/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/qkec/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    # Include core MLIR sources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    # Include our custom sources
    QuakeCPythonSources
)

# Create the Python modules
add_mlir_python_modules(QuakeCPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python/qkec/mlir"
  INSTALL_PREFIX "python/qkec/mlir"
  DECLARED_SOURCES
    QuakeCPythonSources
    # Include core MLIR Python sources for base functionality
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    QuakeCPythonCAPI
)

target_link_libraries(QuakeCPythonModules.extension._qkecExtension.dso 
  PRIVATE #cudaq-pybind-dialects
    -Wl,--whole-archive CCDialect -Wl,--no-whole-archive
    -Wl,--whole-archive QuakeDialect -Wl,--no-whole-archive
    CUDAQuantumMLIRCAPI)
