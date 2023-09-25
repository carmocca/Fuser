// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cuda.h>

// How to lazily load a driver API and invoke it? Just forget about lazy loading
// and write code as if you are using the driver API directly. Magic will
// happen. To understand how the magic works, please refer to the cpp file's doc
// "How does the magic work?"

namespace nvfuser {

#define DECLARE_DRIVER_API_WRAPPER(funcName) \
  extern decltype(::funcName)* funcName;

// List of driver APIs that you want the magic to happen.
DECLARE_DRIVER_API_WRAPPER(cuGetErrorName);
DECLARE_DRIVER_API_WRAPPER(cuGetErrorString);
DECLARE_DRIVER_API_WRAPPER(cuModuleLoadDataEx);
DECLARE_DRIVER_API_WRAPPER(cuModuleGetFunction);
DECLARE_DRIVER_API_WRAPPER(cuOccupancyMaxActiveBlocksPerMultiprocessor);
DECLARE_DRIVER_API_WRAPPER(cuFuncGetAttribute);
DECLARE_DRIVER_API_WRAPPER(cuLaunchKernel);
DECLARE_DRIVER_API_WRAPPER(cuLaunchCooperativeKernel);
DECLARE_DRIVER_API_WRAPPER(cuDeviceGetAttribute);
DECLARE_DRIVER_API_WRAPPER(cuDeviceGetName);

#if (CUDA_VERSION >= 12000)
DECLARE_DRIVER_API_WRAPPER(cuTensorMapEncodeTiled);
#endif

#undef DECLARE_DRIVER_API_WRAPPER

} // namespace nvfuser
