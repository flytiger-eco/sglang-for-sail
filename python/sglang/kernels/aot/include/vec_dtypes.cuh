/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Self-contained vectorized load/store helper, replacing the dependency on
// flashinfer/vec_dtypes.cuh.
//
// API-compatible with the subset of `flashinfer::vec_t` that sgl-kernel
// actually uses: operator[], ptr(), fill(), load(), store(), cast_load(),
// cast_store(). The ROCm counterpart lives in include/hip/hip_vec_dtypes.h and
// keeps the same surface, so call sites stay platform-neutral.
//
// Conversions go through `float`, which every CUDA extended floating-point
// type (__half, __nv_bfloat16, __nv_fp8_e4m3, __nv_fp8_e5m2) provides a
// conversion operator for. That keeps one code path for every dtype pair
// instead of the per-dtype specializations flashinfer needs, at the cost of
// leaving the packed-vector conversion intrinsics to the compiler: loads and
// stores stay contiguous and unrolled, so the vectorized ld/st are still
// emitted, but no dtype-specific packed-conversion instruction is hand-picked.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifndef USE_ROCM

#include <hggc_bf16.h>
#include <hggc_fp16.h>
#include <hggc_fp8.h>
#include <hggc_runtime.h>

#define SGL_VEC_INLINE inline __attribute__((always_inline)) __device__

namespace sgl {

// Widest scalar type both source and destination can round-trip through.
template <typename T>
SGL_VEC_INLINE float to_float(const T& value) {
  return static_cast<float>(value);
}

template <typename T>
SGL_VEC_INLINE T from_float(float value) {
  return static_cast<T>(value);
}

template <typename dtype_t, size_t vec_size>
struct vec_t {
  static_assert(vec_size > 0, "vec_size must be positive");

  dtype_t data[vec_size];

  SGL_VEC_INLINE dtype_t& operator[](size_t i) {
    return data[i];
  }
  SGL_VEC_INLINE const dtype_t& operator[](size_t i) const {
    return data[i];
  }

  SGL_VEC_INLINE dtype_t* ptr() {
    return data;
  }
  SGL_VEC_INLINE const dtype_t* ptr() const {
    return data;
  }

  SGL_VEC_INLINE void fill(dtype_t value) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      data[i] = value;
    }
  }

  SGL_VEC_INLINE void load(const dtype_t* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      data[i] = src[i];
    }
  }

  SGL_VEC_INLINE void store(dtype_t* dst) const {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      dst[i] = data[i];
    }
  }

  // Convert from a vector of a different dtype.
  template <typename src_t>
  SGL_VEC_INLINE void cast_from(const vec_t<src_t, vec_size>& src) {
    if constexpr (std::is_same<src_t, dtype_t>::value) {
#pragma unroll
      for (size_t i = 0; i < vec_size; ++i) {
        data[i] = src[i];
      }
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size; ++i) {
        data[i] = from_float<dtype_t>(to_float(src[i]));
      }
    }
  }

  // Load `vec_size` elements of type src_t and convert them to dtype_t.
  template <typename src_t>
  SGL_VEC_INLINE void cast_load(const src_t* src) {
    if constexpr (std::is_same<src_t, dtype_t>::value) {
      load(src);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size; ++i) {
        data[i] = from_float<dtype_t>(to_float(src[i]));
      }
    }
  }

  // Convert to dst_t and store `vec_size` elements.
  template <typename dst_t>
  SGL_VEC_INLINE void cast_store(dst_t* dst) const {
    if constexpr (std::is_same<dst_t, dtype_t>::value) {
      store(dst);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size; ++i) {
        dst[i] = from_float<dst_t>(to_float(data[i]));
      }
    }
  }
};

}  // namespace sgl

#undef SGL_VEC_INLINE

#endif  // !USE_ROCM
