/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

__global__ void cast_i64_to_i32_kernel(
    const int64_t* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int32_t>(input[idx]);
  }
}

__global__ void cast_f32_to_f16_kernel(
    const float* __restrict__ input,
    __half* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __float2half(input[idx]);
  }
}

__global__ void cast_f16_to_f32_kernel(
    const __half* __restrict__ input,
    float* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __half2float(input[idx]);
  }
}

__global__ void cast_i64_to_f32_kernel(
    const int64_t* __restrict__ input,
    float* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

__global__ void cast_i32_to_f32_kernel(
    const int32_t* __restrict__ input,
    float* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

__global__ void cast_i64_to_f16_kernel(
    const int64_t* __restrict__ input,
    __half* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __float2half(static_cast<float>(input[idx]));
  }
}

__global__ void cast_i32_to_f16_kernel(
    const int32_t* __restrict__ input,
    __half* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __float2half(static_cast<float>(input[idx]));
  }
}

__global__ void cast_i32_to_i64_kernel(
    const int32_t* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int64_t>(input[idx]);
  }
}

__global__ void cast_f32_to_i64_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int64_t>(input[idx]);
  }
}

__global__ void cast_f32_to_i32_kernel(
    const float* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int32_t>(input[idx]);
  }
}

__global__ void cast_f16_to_i64_kernel(
    const __half* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int64_t>(__half2float(input[idx]));
  }
}

__global__ void cast_f16_to_i32_kernel(
    const __half* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int32_t>(__half2float(input[idx]));
  }
}

// Generic integer<->integer cast (used for the uint8 paths that the per-pair
// kernels above don't cover). static_cast handles narrowing/widening per C++
// rules; the kernels above stay as-is so existing FP paths keep their bespoke
// __float2half / __half2float intrinsics.
template <typename Src, typename Dst>
__global__ void cast_int_to_int_kernel(
    const Src* __restrict__ input,
    Dst* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<Dst>(input[idx]);
  }
}

__global__ void cast_ui8_to_f16_kernel(
    const uint8_t* __restrict__ input,
    __half* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __float2half(static_cast<float>(input[idx]));
  }
}

// bool is represented as uint8 in the EP, so ONNX Cast(bool -> float32)
// (e.g. a padding-mask `Not(...)` promoted to a float multiplier) arrives
// here as UINT8 -> FLOAT32. Without this path the dispatch fell through to
// the unsupported branch, left the output buffer untouched (pool zeros), and
// silently zeroed every masked value downstream.
__global__ void cast_ui8_to_f32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

__global__ void cast_f16_to_ui8_kernel(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<uint8_t>(__half2float(input[idx]));
  }
}

// Narrowing to bool is `x != 0`, not a width truncation: truncating 2.5 to a
// byte yields 2, but converting it to bool must yield 1. Every source type
// shares one kernel because the only operation is a comparison against zero.
__device__ __forceinline__ bool cast_is_nonzero(float v) { return v != 0.0f; }
__device__ __forceinline__ bool cast_is_nonzero(__half v) {
  return __half2float(v) != 0.0f;
}
template <typename SrcT>
__device__ __forceinline__ bool cast_is_nonzero(SrcT v) {
  return v != static_cast<SrcT>(0);
}

template <typename SrcT>
__global__ void cast_to_bool_kernel(
    const SrcT* __restrict__ input,
    uint8_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = cast_is_nonzero(input[idx]) ? uint8_t(1) : uint8_t(0);
  }
}

extern "C" int hip_cast(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int input_dtype,
    int output_dtype) {
  if (num_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  if (input_dtype == HIP_DTYPE_INT64 && output_dtype == HIP_DTYPE_INT32) {
    CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_cast: INT64->INT32, "
            "num_elements=%lld, grid=%d, block=%d\n",
            (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(cast_i64_to_i32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t*>(input),
                       static_cast<int32_t*>(output),
                       num_elements);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 && output_dtype == HIP_DTYPE_FLOAT16) {
    CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_cast: FLOAT32->FLOAT16, "
            "num_elements=%lld, grid=%d, block=%d\n",
            (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(cast_f32_to_f16_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float*>(input),
                       static_cast<__half*>(output),
                       num_elements);
  } else if (input_dtype == HIP_DTYPE_FLOAT16 && output_dtype == HIP_DTYPE_FLOAT32) {
    CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_cast: FLOAT16->FLOAT32, "
            "num_elements=%lld, grid=%d, block=%d\n",
            (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(cast_f16_to_f32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half*>(input),
                       static_cast<float*>(output),
                       num_elements);
  } else if (input_dtype == HIP_DTYPE_INT64 && output_dtype == HIP_DTYPE_FLOAT32) {
    CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_cast: INT64->FLOAT32, num_elements=%lld\n", (long long)num_elements);
    hipLaunchKernelGGL(cast_i64_to_f32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t*>(input),
                       static_cast<float*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_INT32 && output_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(cast_i32_to_f32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t*>(input),
                       static_cast<float*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_INT64 && output_dtype == HIP_DTYPE_FLOAT16) {
    hipLaunchKernelGGL(cast_i64_to_f16_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t*>(input),
                       static_cast<__half*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_INT32 && output_dtype == HIP_DTYPE_FLOAT16) {
    hipLaunchKernelGGL(cast_i32_to_f16_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t*>(input),
                       static_cast<__half*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_INT32 && output_dtype == HIP_DTYPE_INT64) {
    hipLaunchKernelGGL(cast_i32_to_i64_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t*>(input),
                       static_cast<int64_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 && output_dtype == HIP_DTYPE_INT64) {
    hipLaunchKernelGGL(cast_f32_to_i64_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float*>(input),
                       static_cast<int64_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 && output_dtype == HIP_DTYPE_INT32) {
    hipLaunchKernelGGL(cast_f32_to_i32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float*>(input),
                       static_cast<int32_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_FLOAT16 && output_dtype == HIP_DTYPE_INT64) {
    hipLaunchKernelGGL(cast_f16_to_i64_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half*>(input),
                       static_cast<int64_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_FLOAT16 && output_dtype == HIP_DTYPE_INT32) {
    hipLaunchKernelGGL(cast_f16_to_i32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half*>(input),
                       static_cast<int32_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_INT64 && output_dtype == HIP_DTYPE_UINT8) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_int_to_int_kernel<int64_t, uint8_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t*>(input),
                       static_cast<uint8_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_UINT8 && output_dtype == HIP_DTYPE_INT64) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_int_to_int_kernel<uint8_t, int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const uint8_t*>(input),
                       static_cast<int64_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_INT32 && output_dtype == HIP_DTYPE_UINT8) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_int_to_int_kernel<int32_t, uint8_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t*>(input),
                       static_cast<uint8_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_UINT8 && output_dtype == HIP_DTYPE_INT32) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_int_to_int_kernel<uint8_t, int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const uint8_t*>(input),
                       static_cast<int32_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_UINT8 && output_dtype == HIP_DTYPE_FLOAT16) {
    hipLaunchKernelGGL(cast_ui8_to_f16_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const uint8_t*>(input),
                       static_cast<__half*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_UINT8 && output_dtype == HIP_DTYPE_FLOAT32) {
    CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_cast: UINT8->FLOAT32, "
            "num_elements=%lld, grid=%d, block=%d\n",
            (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(cast_ui8_to_f32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const uint8_t*>(input),
                       static_cast<float*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_FLOAT16 && output_dtype == HIP_DTYPE_UINT8) {
    hipLaunchKernelGGL(cast_f16_to_ui8_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half*>(input),
                       static_cast<uint8_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_INT8 && output_dtype == HIP_DTYPE_INT16) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_int_to_int_kernel<int8_t, int16_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int8_t*>(input),
                       static_cast<int16_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_INT16 && output_dtype == HIP_DTYPE_UINT8) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_int_to_int_kernel<int16_t, uint8_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int16_t*>(input),
                       static_cast<uint8_t*>(output), num_elements);
  } else if (output_dtype == HIP_DTYPE_BOOL) {
    // One branch for every source type: the destination kernel is the same
    // comparison against zero regardless of what is being narrowed.
    switch (input_dtype) {
    case HIP_DTYPE_FLOAT32:
      hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_to_bool_kernel<float>),
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const float*>(input),
                         static_cast<uint8_t*>(output), num_elements);
      break;
    case HIP_DTYPE_FLOAT16:
      hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_to_bool_kernel<__half>),
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const __half*>(input),
                         static_cast<uint8_t*>(output), num_elements);
      break;
    case HIP_DTYPE_INT64:
      hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_to_bool_kernel<int64_t>),
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const int64_t*>(input),
                         static_cast<uint8_t*>(output), num_elements);
      break;
    case HIP_DTYPE_INT32:
      hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_to_bool_kernel<int32_t>),
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const int32_t*>(input),
                         static_cast<uint8_t*>(output), num_elements);
      break;
    case HIP_DTYPE_UINT8:
    case HIP_DTYPE_BOOL:
      hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_to_bool_kernel<uint8_t>),
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const uint8_t*>(input),
                         static_cast<uint8_t*>(output), num_elements);
      break;
    case HIP_DTYPE_INT8:
      hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_to_bool_kernel<int8_t>),
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const int8_t*>(input),
                         static_cast<uint8_t*>(output), num_elements);
      break;
    default:
      fprintf(stderr,
              "[custom_kernels] hip_cast: unsupported conversion "
              "input_dtype=%d -> BOOL\n",
              input_dtype);
      return -1;
    }
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_cast: unsupported conversion "
            "input_dtype=%d -> output_dtype=%d\n",
            input_dtype, output_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_cast launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
