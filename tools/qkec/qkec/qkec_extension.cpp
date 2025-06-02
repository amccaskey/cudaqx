/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <pybind11/complex.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#if 1
#include "cudaq/python/cudaq_pybind_dialects.h"

namespace py = pybind11;

PYBIND11_MODULE(_qkecExtension, m) {
  // We do need the CUDA-Q Dialects here.
  cudaq::registerQuakeDialectAndTypes(m, [&]() { return; });
  cudaq::registerCCDialectAndTypes(m);
}
#else 
#include "cudaq/python/py_register_dialects.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
namespace py = pybind11;

PYBIND11_MODULE(_qkecExtension, m) {
  // We do need the CUDA-Q Dialects here.
  cudaq::registerQuakeDialectAndTypes(m, MAKE_MLIR_PYTHON_QUALNAME("ir"));
  cudaq::registerCCDialectAndTypes(m, MAKE_MLIR_PYTHON_QUALNAME("ir"));
}
#endif 