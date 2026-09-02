/* Copyright 2025 SGLang Team. All Rights Reserved.
   Copyright (c) 2024 by FlashInfer team.

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

// Self-contained segment bit-packing, replacing the dependency on
// flashinfer/quantization.cuh. Algorithm adapted from FlashInfer's
// quantization.cuh (Apache-2.0); only `SegmentPackBits` is kept, since
// `PackBits` has no call site in sgl-kernel.

#pragma once

#include <hggc_runtime.h>
#include <hggc_runtime_api.h>

#include <cstdint>
#include <cub/cub.cuh>

namespace sgl {
namespace quantization {

enum class BitOrder { kBig = 0U, kLittle = 1U };

#define SGL_DISPATCH_BITORDER(bitorder, BITORDER, ...) \
  if (bitorder == BitOrder::kBig) {                    \
    constexpr BitOrder BITORDER = BitOrder::kBig;      \
    __VA_ARGS__                                        \
  } else {                                             \
    constexpr BitOrder BITORDER = BitOrder::kLittle;   \
    __VA_ARGS__                                        \
  }

template <BitOrder BITORDER, typename IdType>
__global__ void SegmentPackBitsKernel(bool* input, uint8_t* output, IdType* input_indptr, IdType* output_indptr) {
  int64_t bx = blockIdx.x, tx = threadIdx.x;
  bool input_vec[8];
  typedef cub::BlockLoad<bool, 256, 8, cub::BLOCK_LOAD_VECTORIZE> BlockLoad;
  __shared__ typename BlockLoad::TempStorage temp_storage;
  int64_t num_elements = input_indptr[bx + 1] - input_indptr[bx];
  for (uint32_t start_offset = 0; start_offset < num_elements; start_offset += 8 * blockDim.x) {
    uint8_t ret = 0;
    BlockLoad(temp_storage)
        .Load(
            input + input_indptr[bx] + start_offset,
            input_vec,
            num_elements - start_offset,
            /*default=*/0);

    if constexpr (BITORDER == BitOrder::kBig) {
      ret = (input_vec[0] << 7) | (input_vec[1] << 6) | (input_vec[2] << 5) | (input_vec[3] << 4) |
            (input_vec[4] << 3) | (input_vec[5] << 2) | (input_vec[6] << 1) | input_vec[7];
    } else {
      ret = (input_vec[7] << 7) | (input_vec[6] << 6) | (input_vec[5] << 5) | (input_vec[4] << 4) |
            (input_vec[3] << 3) | (input_vec[2] << 2) | (input_vec[1] << 1) | input_vec[0];
    }
    if (start_offset + tx * 8 < num_elements) output[output_indptr[bx] + start_offset / 8 + tx] = ret;
  }
}

template <typename IdType>
hggcError_t SegmentPackBits(
    bool* input,
    uint8_t* output,
    IdType* input_indptr,
    IdType* output_indptr,
    uint32_t batch_size,
    BitOrder bitorder,
    hggcStream_t stream) {
  SGL_DISPATCH_BITORDER(bitorder, BITORDER, {
    auto kernel = SegmentPackBitsKernel<BITORDER, IdType>;
    const dim3 nthrs(256);
    const dim3 nblks(batch_size);
    void* args[] = {&input, &output, &input_indptr, &output_indptr};
    hggcError_t status = hggcLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream);
    if (status != hggcSuccess) {
      return status;
    }
  });
  return hggcSuccess;
}

}  // namespace quantization
}  // namespace sgl
