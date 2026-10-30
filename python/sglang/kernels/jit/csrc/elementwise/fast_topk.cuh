// Radix-select fast top-k with the AOT fast_topk_v2 semantics: for each row b,
// select the kTopK largest scores in [row_starts[b], row_starts[b] + lengths[b])
// and write their indices relative to row_starts[b]; order within a row is unspecified.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <type_traits>

namespace sglang {

namespace fast_topk_detail {

constexpr uint32_t kThreadsPerBlock = 1024;
// Candidate buffer for the refine passes: 4K entries per round (2 rounds = 8K
// entries = 32KB). A coarse threshold bin holding more candidates than that
// takes the exact rescan fallback in radix_select_topk instead of being clipped.
constexpr size_t kSmemBytes = 8 * 1024 * sizeof(uint32_t);

template <typename DType>
struct FastTopKParams {
  const DType* __restrict__ input;
  const int32_t* __restrict__ row_starts;
  int32_t* __restrict__ indices;
  const int32_t* __restrict__ lengths;
  int64_t input_stride;
};

template <typename DType>
SGL_DEVICE auto load_score(const DType* __restrict__ score, int64_t idx) -> float {
  return static_cast<float>(score[idx]);
}

SGL_DEVICE auto convert_to_uint8(float x) -> uint8_t {
  const __half h = __float2half_rn(x);
  const uint16_t bits = __half_as_ushort(h);
  const uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

SGL_DEVICE auto convert_to_uint32(float x) -> uint32_t {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <int kTopK>
SGL_DEVICE void naive_topk(int32_t* __restrict__ indices, int32_t length) {
  for (int i = threadIdx.x; i < kTopK; i += kThreadsPerBlock) {
    indices[i] = i < length ? i : -1;
  }
}

template <int kTopK>
SGL_DEVICE void naive_topk_transform(int32_t length, int32_t* __restrict__ dst, const int32_t* __restrict__ src) {
  for (int i = threadIdx.x; i < kTopK; i += kThreadsPerBlock) {
    dst[i] = i < length ? src[i] : -1;
  }
}

template <int kTopK>
SGL_DEVICE void naive_topk_transform_ragged(int32_t length, int32_t* __restrict__ dst, int32_t offset) {
  for (int i = threadIdx.x; i < kTopK; i += kThreadsPerBlock) {
    dst[i] = i < length ? i + offset : -1;
  }
}

template <int kTopK, typename DType>
SGL_DEVICE void
radix_select_topk(const DType* __restrict__ input, int32_t* __restrict__ indices, int32_t row_start, int32_t length) {
  int topk = kTopK;
  constexpr int kRadix = 256;
  constexpr int kSmemInputSize = kSmemBytes / (2 * sizeof(int));

  alignas(128) __shared__ int histogram_buffers[2][kRadix + 128];
  alignas(128) __shared__ int counter;
  alignas(128) __shared__ int threshold_bin_id;
  alignas(128) __shared__ int num_input[2];

  auto& histogram = histogram_buffers[0];
  extern __shared__ int input_indices[][kSmemInputSize];

  const int tid = threadIdx.x;
  if (tid < kRadix + 1) histogram[tid] = 0;
  __syncthreads();

  for (int idx = tid; idx < length; idx += kThreadsPerBlock) {
    const auto bin = convert_to_uint8(load_score(input, idx + row_start));
    ::atomicAdd(&histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == kRadix);
      if (tid < kRadix) {
        const int distance = 1 << i;
        const int buffer = i & 1;
        auto value = histogram_buffers[buffer][tid];
        if (tid < kRadix - distance) {
          value += histogram_buffers[buffer][tid + distance];
        }
        histogram_buffers[buffer ^ 1][tid] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tid < kRadix && histogram[tid] > topk && histogram[tid + 1] <= topk) {
    threshold_bin_id = tid;
    num_input[0] = 0;
    counter = 0;
  }
  __syncthreads();

  auto threshold_bin = threshold_bin_id;
  topk -= histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tid; idx < length; idx += kThreadsPerBlock) {
      const auto bin = static_cast<int>(convert_to_uint8(load_score(input, idx + row_start)));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&counter, 1);
        indices[pos] = idx;
      }
    }
    __syncthreads();
    return;
  }

  __syncthreads();
  if (tid < kRadix + 1) histogram[tid] = 0;
  __syncthreads();

  for (int idx = tid; idx < length; idx += kThreadsPerBlock) {
    const auto value = load_score(input, idx + row_start);
    const auto bin = static_cast<int>(convert_to_uint8(value));
    if (bin > threshold_bin) {
      const auto pos = ::atomicAdd(&counter, 1);
      indices[pos] = idx;
    } else if (bin == threshold_bin) {
      const auto pos = ::atomicAdd(&num_input[0], 1);
      if (pos < kSmemInputSize) {
        input_indices[0][pos] = idx;
        const auto sub_bin = (convert_to_uint32(value) >> 24) & 0xFF;
        ::atomicAdd(&histogram[sub_bin], 1);
      }
    }
  }
  __syncthreads();

  // Exact fallback for a coarse threshold bin that overflowed the candidate
  // buffer: the buffer (and its fused sub-histogram) then holds an arbitrary
  // subset of the bin, so refining it could drop true winners. Instead refine
  // the whole bin with histogram-only rescans of the row, 8 bits of the fp32
  // key per round, then emit once. The fast path only pays this branch.
  if (num_input[0] > kSmemInputSize) [[unlikely]] {
    __shared__ int overflow_remain;
    uint32_t prefix = 0;       // decided high bits of the fp32 threshold key
    uint32_t prefix_mask = 0;  // which bits of `prefix` are decided
#pragma unroll 1
    for (int round = 0; round < 4 && topk > 0; ++round) {
      const auto offset = 24 - round * 8;
      if (tid < kRadix + 1) histogram[tid] = 0;
      __syncthreads();
      for (int idx = tid; idx < length; idx += kThreadsPerBlock) {
        const auto value = load_score(input, idx + row_start);
        if (static_cast<int>(convert_to_uint8(value)) != threshold_bin) continue;
        const auto key = convert_to_uint32(value);
        if ((key & prefix_mask) == prefix) ::atomicAdd(&histogram[(key >> offset) & 0xFF], 1);
      }
      __syncthreads();
      run_cumsum();
      if (tid < kRadix && histogram[tid] > topk && histogram[tid + 1] <= topk) {
        threshold_bin_id = tid;
      }
      __syncthreads();
      const auto sub_bin = threshold_bin_id;
      topk -= histogram[sub_bin + 1];
      prefix |= static_cast<uint32_t>(sub_bin) << offset;
      prefix_mask |= 0xFFu << offset;
      // every thread must read the histogram before the next round clears it
      __syncthreads();
    }

    // Keys above the decided prefix are selected outright (appended after the
    // bins above the coarse threshold); `topk` of the keys equal to it fill the
    // trailing slots. Once all 32 bits are decided those are identical scores.
    if (tid == 0) overflow_remain = topk;
    __syncthreads();
    for (int idx = tid; idx < length; idx += kThreadsPerBlock) {
      const auto value = load_score(input, idx + row_start);
      if (static_cast<int>(convert_to_uint8(value)) != threshold_bin) continue;
      const auto key = convert_to_uint32(value) & prefix_mask;
      if (key > prefix) {
        const auto pos = ::atomicAdd(&counter, 1);
        indices[pos] = idx;
      } else if (key == prefix) {
        const auto pos = ::atomicAdd(&overflow_remain, -1);
        if (pos > 0) indices[kTopK - pos] = idx;
      }
    }
    __syncthreads();
    return;
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int last_remain;
    const int input_buffer = round & 1;
    const int raw_num_input = num_input[input_buffer];
    const int current_num_input = raw_num_input < kSmemInputSize ? raw_num_input : kSmemInputSize;

    run_cumsum();
    if (tid < kRadix && histogram[tid] > topk && histogram[tid + 1] <= topk) {
      threshold_bin_id = tid;
      num_input[input_buffer ^ 1] = 0;
      last_remain = topk - histogram[tid + 1];
    }
    __syncthreads();

    threshold_bin = threshold_bin_id;
    topk -= histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tid; i < current_num_input; i += kThreadsPerBlock) {
        const int idx = input_indices[input_buffer][i];
        const int shift = 24 - round * 8;
        const auto bin = (convert_to_uint32(load_score(input, idx + row_start)) >> shift) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&counter, 1);
          indices[pos] = idx;
        }
      }
      __syncthreads();
      break;
    }

    __syncthreads();
    if (tid < kRadix + 1) histogram[tid] = 0;
    __syncthreads();
    for (int i = tid; i < current_num_input; i += kThreadsPerBlock) {
      const int idx = input_indices[input_buffer][i];
      const auto value = load_score(input, idx + row_start);
      const int shift = 24 - round * 8;
      const auto bin = (convert_to_uint32(value) >> shift) & 0xFF;
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&counter, 1);
        indices[pos] = idx;
      } else if (bin == threshold_bin) {
        if (round == 3) {
          const auto pos = ::atomicAdd(&last_remain, -1);
          if (pos > 0) indices[kTopK - pos] = idx;
        } else {
          const auto pos = ::atomicAdd(&num_input[input_buffer ^ 1], 1);
          if (pos < kSmemInputSize) {
            input_indices[input_buffer ^ 1][pos] = idx;
            const auto sub_bin = (convert_to_uint32(value) >> (shift - 8)) & 0xFF;
            ::atomicAdd(&histogram[sub_bin], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

template <int kTopK, bool kUsePDL, typename DType>
__global__
__launch_bounds__(kThreadsPerBlock) void fast_topk_kernel(const FastTopKParams<DType> __grid_constant__ params) {
  device::PDLWaitPrimary<kUsePDL>();
  const auto row = static_cast<uint64_t>(blockIdx.x);
  const int32_t row_start = params.row_starts == nullptr ? 0 : params.row_starts[row];
  const int32_t length = params.lengths[row];
  auto* indices = params.indices + row * kTopK;
  const auto* score = params.input + row * params.input_stride;
  if (length <= kTopK) {
    naive_topk<kTopK>(indices, length);
  } else {
    radix_select_topk<kTopK>(score, indices, row_start, length);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <int kTopK, bool kUsePDL, typename DType>
__global__ __launch_bounds__(kThreadsPerBlock) void topk_transform_decode_kernel(
    const FastTopKParams<DType> __grid_constant__ params,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table,
    int64_t src_stride) {
  device::PDLWaitPrimary<kUsePDL>();
  const auto row = static_cast<uint64_t>(blockIdx.x);
  const int32_t length = params.lengths[row];
  const auto* score = params.input + row * params.input_stride;
  const auto* src = src_page_table + row * src_stride;
  auto* dst = dst_page_table + row * kTopK;
  if (length <= kTopK) {
    naive_topk_transform<kTopK>(length, dst, src);
  } else {
    __shared__ int32_t indices[kTopK];
    radix_select_topk<kTopK>(score, indices, 0, length);
    for (int i = threadIdx.x; i < kTopK; i += kThreadsPerBlock) {
      dst[i] = src[indices[i]];
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <int kTopK, bool kUsePDL, typename DType>
__global__ __launch_bounds__(kThreadsPerBlock) void topk_transform_prefill_kernel(
    const FastTopKParams<DType> __grid_constant__ params,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table,
    int64_t src_stride,
    const int32_t* __restrict__ cu_seqlens_q,
    int64_t prefill_batch) {
  device::PDLWaitPrimary<kUsePDL>();
  const auto row = static_cast<uint64_t>(blockIdx.x);
  const int32_t length = params.lengths[row];
  const int32_t row_start = params.row_starts == nullptr ? 0 : params.row_starts[row];
  const auto* score = params.input + row * params.input_stride;
  auto* dst = dst_page_table + row * kTopK;

  __shared__ const int32_t* src;
  if (prefill_batch <= kThreadsPerBlock) {
    if (threadIdx.x < prefill_batch && row >= cu_seqlens_q[threadIdx.x] && row < cu_seqlens_q[threadIdx.x + 1]) {
      src = src_page_table + threadIdx.x * src_stride;
    }
  } else {
    for (int64_t i = threadIdx.x; i < prefill_batch; i += kThreadsPerBlock) {
      if (row >= cu_seqlens_q[i] && row < cu_seqlens_q[i + 1]) src = src_page_table + i * src_stride;
    }
  }
  __syncthreads();

  if (length <= kTopK) {
    naive_topk_transform<kTopK>(length, dst, src);
  } else {
    __shared__ int32_t indices[kTopK];
    radix_select_topk<kTopK>(score, indices, row_start, length);
    for (int i = threadIdx.x; i < kTopK; i += kThreadsPerBlock) {
      dst[i] = src[indices[i]];
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <int kTopK, bool kUsePDL, typename DType>
__global__ __launch_bounds__(kThreadsPerBlock) void topk_transform_ragged_kernel(
    const FastTopKParams<DType> __grid_constant__ params,
    int32_t* __restrict__ dst_indices,
    const int32_t* __restrict__ offsets) {
  device::PDLWaitPrimary<kUsePDL>();
  const auto row = static_cast<uint64_t>(blockIdx.x);
  const int32_t length = params.lengths[row];
  const int32_t row_start = params.row_starts == nullptr ? 0 : params.row_starts[row];
  const auto* score = params.input + row * params.input_stride;
  auto* dst = dst_indices + row * kTopK;
  const int32_t offset = offsets[row];
  if (length <= kTopK) {
    naive_topk_transform_ragged<kTopK>(length, dst, offset);
  } else {
    __shared__ int32_t indices[kTopK];
    radix_select_topk<kTopK>(score, indices, row_start, length);
    for (int i = threadIdx.x; i < kTopK; i += kThreadsPerBlock) {
      dst[i] = indices[i] + offset;
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <typename DType>
auto make_params(
    const tvm::ffi::TensorView score,
    const tvm::ffi::TensorView lengths,
    const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts,
    int32_t* indices,
    int64_t input_stride) -> FastTopKParams<DType> {
  return FastTopKParams<DType>{
      .input = static_cast<const DType*>(score.data_ptr()),
      .row_starts = row_starts.has_value() ? static_cast<const int32_t*>(row_starts.value().data_ptr()) : nullptr,
      .indices = indices,
      .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
      .input_stride = input_stride,
  };
}

}  // namespace fast_topk_detail

template <int kTopK, bool kUsePDL, typename DType>
struct FastTopKKernel {
  static_assert(kTopK == 512 || kTopK == 2048, "fast_topk supports topk=512 or 2048");
  static_assert(std::is_same_v<DType, fp32_t> || std::is_same_v<DType, bf16_t>, "unsupported score dtype");

  static void
  run(const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts) {
    using namespace host;
    auto batch = SymbolicSize{"batch"};
    auto stride = SymbolicSize{"score_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({batch, -1}).with_strides({stride, 1}).with_dtype<DType>().with_device(device).verify(score);
    TensorMatcher({batch, kTopK}).with_dtype<int32_t>().with_device(device).verify(indices);
    TensorMatcher({batch}).with_dtype<int32_t>().with_device(device).verify(lengths);
    if (row_starts.has_value()) {
      TensorMatcher({batch}).with_dtype<int32_t>().with_device(device).verify(row_starts.value());
    }

    const auto params = fast_topk_detail::make_params<DType>(
        score, lengths, row_starts, static_cast<int32_t*>(indices.data_ptr()), stride.unwrap());
    const auto kernel = fast_topk_detail::fast_topk_kernel<kTopK, kUsePDL, DType>;
    LaunchKernel(
        static_cast<uint32_t>(batch.unwrap()),
        fast_topk_detail::kThreadsPerBlock,
        device.unwrap(),
        fast_topk_detail::kSmemBytes)
        .enable_pdl(kUsePDL)(kernel, params);
  }

  static void transform(
      const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView dst_page_table,
      const tvm::ffi::TensorView src_page_table,
      const tvm::ffi::TensorView cu_seqlens_q,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts) {
    using namespace host;
    auto batch = SymbolicSize{"batch"};
    auto score_stride = SymbolicSize{"score_stride"};
    auto prefill_batch = SymbolicSize{"prefill_batch"};
    auto src_stride = SymbolicSize{"page_table_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({batch, -1}).with_strides({score_stride, 1}).with_dtype<DType>().with_device(device).verify(score);
    TensorMatcher({batch}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({batch, kTopK}).with_dtype<int32_t>().with_device(device).verify(dst_page_table);
    TensorMatcher({prefill_batch, -1})
        .with_strides({src_stride, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(src_page_table);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(cu_seqlens_q);
    RuntimeCheck(
        cu_seqlens_q.size(0) == prefill_batch.unwrap() + 1, "cu_seqlens_q length must equal page-table rows + 1");
    // Empty query segments may leave more request table rows than score rows.
    if (row_starts.has_value()) {
      TensorMatcher({batch}).with_dtype<int32_t>().with_device(device).verify(row_starts.value());
    }

    const auto params =
        fast_topk_detail::make_params<DType>(score, lengths, row_starts, nullptr, score_stride.unwrap());
    const bool is_decode = !row_starts.has_value() && prefill_batch.unwrap() == batch.unwrap();
    if (is_decode) {
      const auto kernel = fast_topk_detail::topk_transform_decode_kernel<kTopK, kUsePDL, DType>;
      LaunchKernel(
          static_cast<uint32_t>(batch.unwrap()),
          fast_topk_detail::kThreadsPerBlock,
          device.unwrap(),
          fast_topk_detail::kSmemBytes)
          .enable_pdl(kUsePDL)(
              kernel,
              params,
              static_cast<int32_t*>(dst_page_table.data_ptr()),
              static_cast<const int32_t*>(src_page_table.data_ptr()),
              src_stride.unwrap());
    } else {
      const auto kernel = fast_topk_detail::topk_transform_prefill_kernel<kTopK, kUsePDL, DType>;
      LaunchKernel(
          static_cast<uint32_t>(batch.unwrap()),
          fast_topk_detail::kThreadsPerBlock,
          device.unwrap(),
          fast_topk_detail::kSmemBytes)
          .enable_pdl(kUsePDL)(
              kernel,
              params,
              static_cast<int32_t*>(dst_page_table.data_ptr()),
              static_cast<const int32_t*>(src_page_table.data_ptr()),
              src_stride.unwrap(),
              static_cast<const int32_t*>(cu_seqlens_q.data_ptr()),
              prefill_batch.unwrap());
    }
  }

  static void transform_ragged(
      const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView dst_indices,
      const tvm::ffi::TensorView offsets,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts) {
    using namespace host;
    auto batch = SymbolicSize{"batch"};
    auto score_stride = SymbolicSize{"score_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({batch, -1}).with_strides({score_stride, 1}).with_dtype<DType>().with_device(device).verify(score);
    TensorMatcher({batch}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({batch, kTopK}).with_dtype<int32_t>().with_device(device).verify(dst_indices);
    TensorMatcher({batch}).with_dtype<int32_t>().with_device(device).verify(offsets);
    if (row_starts.has_value()) {
      TensorMatcher({batch}).with_dtype<int32_t>().with_device(device).verify(row_starts.value());
    }

    const auto params =
        fast_topk_detail::make_params<DType>(score, lengths, row_starts, nullptr, score_stride.unwrap());
    const auto kernel = fast_topk_detail::topk_transform_ragged_kernel<kTopK, kUsePDL, DType>;
    LaunchKernel(
        static_cast<uint32_t>(batch.unwrap()),
        fast_topk_detail::kThreadsPerBlock,
        device.unwrap(),
        fast_topk_detail::kSmemBytes)
        .enable_pdl(kUsePDL)(
            kernel,
            params,
            static_cast<int32_t*>(dst_indices.data_ptr()),
            static_cast<const int32_t*>(offsets.data_ptr()));
  }
};

}  // namespace sglang
