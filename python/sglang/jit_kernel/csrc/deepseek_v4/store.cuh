#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <cuda_fp8.h>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

// Quantize one fp32 value to an E2M1 4-bit nibble stored in a uint8.
// Bucket boundaries [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] mirror torch's
// `bucketize(..., right=False)`; sign bit is set only for non-zero codes.
__device__ __forceinline__ uint8_t e2m1_nibble(float x) {
  float a = fminf(fabsf(x), 6.0f);
  uint8_t code;
  if (a <= 0.25f)
    code = 0;
  else if (a <= 0.75f)
    code = 1;
  else if (a <= 1.25f)
    code = 2;
  else if (a <= 1.75f)
    code = 3;
  else if (a <= 2.5f)
    code = 4;
  else if (a <= 3.5f)
    code = 5;
  else if (a <= 5.0f)
    code = 6;
  else
    code = 7;
  uint8_t sign = (x < 0.0f && code != 0) ? 1u : 0u;
  return code | static_cast<uint8_t>(sign << 3);
}

struct FusedStoreCacheParam {
  const void* __restrict__ input;
  void* __restrict__ cache;
  const void* __restrict__ indices;
  uint32_t num_tokens;
};

template <typename Float, typename IndicesT, uint32_t kPageBits, bool kUsePDL>
__global__ void fused_store_flashmla_cache(const __grid_constant__ FusedStoreCacheParam param) {
  using namespace device;

  /// NOTE: 584 = 576 + 8
  constexpr int64_t kPageBytes = host::div_ceil(584 << kPageBits, 576) * 576;

  // each warp handles 64 elements, 8 warps, each block handles 1 row
  const auto& [input, cache, indices, num_tokens] = param;
  const uint32_t bid = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t wid = tid / 32;

  PDLWaitPrimary<kUsePDL>();

  // prefetch the index
  const auto index = static_cast<const IndicesT*>(indices)[bid];
  // always load the value from input (don't store if invalid)
  using Float2 = packed_t<Float>;
  const auto elems = static_cast<const Float2*>(input)[tid + bid * 256];
  if (wid != 7) {
    const auto [x, y] = cast<fp32x2_t>(elems);
    const auto abs_max = warp::reduce_max(fmaxf(fabs(x), fabs(y)));
    const auto scale_raw = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
    const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    const auto result = pack_fp8(x * inv_scale, y * inv_scale);
    const int32_t page = index >> kPageBits;
    const int32_t offset = index & ((1 << kPageBits) - 1);
    const auto page_ptr = pointer::offset(cache, page * kPageBytes);
    const auto value_ptr = pointer::offset(page_ptr, offset * 576);
    const auto scale_ptr = pointer::offset(page_ptr, 576 << kPageBits, offset * 8);
    static_cast<fp8x2_e4m3_t*>(value_ptr)[tid] = result;
    static_cast<uint8_t*>(scale_ptr)[wid] = scale_ue8m0;
  } else {
    const auto result = cast<bf16x2_t>(elems);
    const int32_t page = index >> kPageBits;
    const int32_t offset = index & ((1 << kPageBits) - 1);
    const auto page_ptr = pointer::offset(cache, page * kPageBytes);
    const auto value_ptr = pointer::offset(page_ptr, offset * 576, 448);
    static_cast<bf16x2_t*>(value_ptr)[tid - 7 * 32] = result;
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename Float, typename IndicesT, uint32_t kPageBits, bool kUsePDL>
__global__ void fused_store_indexer_cache(const __grid_constant__ FusedStoreCacheParam param) {
  using namespace device;

  /// NOTE: 132 = 128 + 4
  constexpr int64_t kPageBytes = 132 << kPageBits;

  // each warp handles 128 elements, 1 warp, each block handles multiple rows
  const auto& [input, cache, indices, num_tokens] = param;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;

  if (global_wid >= num_tokens) return;

  PDLWaitPrimary<kUsePDL>();

  // prefetch the index
  const auto index = static_cast<const IndicesT*>(indices)[global_wid];
  // always load the value from input (don't store if invalid)
  using Float2 = packed_t<Float>;
  using InStorage = AlignedVector<Float2, 2>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;
  const auto elems = static_cast<const InStorage*>(input)[global_tid];
  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);
  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);
  const auto local_max = fmaxf(fmaxf(fabs(x0), fabs(x1)), fmaxf(fabs(y0), fabs(y1)));
  const auto abs_max = warp::reduce_max(local_max);
  // use normal fp32 scale
  const auto scale = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
  const auto inv_scale = 1.0f / scale;
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  const auto page_ptr = pointer::offset(cache, page * kPageBytes);
  const auto value_ptr = pointer::offset(page_ptr, offset * 128);
  const auto scale_ptr = pointer::offset(page_ptr, 128 << kPageBits, offset * 4);
  OutStorage result;
  result[0] = pack_fp8(x0 * inv_scale, x1 * inv_scale);
  result[1] = pack_fp8(y0 * inv_scale, y1 * inv_scale);
  static_cast<OutStorage*>(value_ptr)[lane_id] = result;
  static_cast<float*>(scale_ptr)[0] = scale;

  PDLTriggerSecondary<kUsePDL>();
}

template <typename Float, typename IndicesT, uint32_t kPageBits, bool kUsePDL>
__global__ void fused_store_indexer_mxfp4_cache(const __grid_constant__ FusedStoreCacheParam param) {
  using namespace device;

  // MXFP4: head_dim=128, block=32 elements (4 blocks/token), 2 nibbles/byte.
  // Per token: 64 packed FP4 bytes + 4 ue8m0 scale bytes = 68 bytes.
  constexpr int kHeadDim = 128;
  constexpr int kMxfp4Block = 32;
  constexpr int kNumScaleBlocks = kHeadDim / kMxfp4Block;      // 4
  constexpr int kPackedBytesPerToken = kHeadDim / 2;           // 64
  constexpr int kScaleBytesPerToken = kNumScaleBlocks;         // 4
  constexpr int kElemsPerLane = 4;                             // 4 fp32 per lane
  constexpr int kLanesPerBlock = kMxfp4Block / kElemsPerLane;  // 8
  constexpr int64_t kPageBytes = static_cast<int64_t>((kPackedBytesPerToken + kScaleBytesPerToken) << kPageBits);

  const auto& [input, cache, indices, num_tokens] = param;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;
  // Each warp handles one token; lane k holds elems [4k, 4k+3].
  // Block id = lane_id / 8; lanes 0..7 → block 0, 8..15 → block 1, etc.
  const auto block_idx_in_token = lane_id / kLanesPerBlock;

  if (global_wid >= num_tokens) return;

  PDLWaitPrimary<kUsePDL>();

  const auto index = static_cast<const IndicesT*>(indices)[global_wid];

  using Float2 = packed_t<Float>;
  using InStorage = AlignedVector<Float2, 2>;
  const auto elems = static_cast<const InStorage*>(input)[global_tid];
  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);
  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);

  // Per-block (8-lane group) absmax → ue8m0 = ceil(log2(amax / 6.0)) + 127.
  const float local_max = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(y0), fabsf(y1)));
  float amax = warp::reduce_max<kLanesPerBlock>(local_max);
  amax = fmaxf(amax, 1e-4f);
  // ceil(log2(amax / 6.0)) using the same fp32 bit trick as cast_to_ue8m0
  // but on `amax / 6.0` rather than the raw scale.
  const float ratio = amax / 6.0f;
  uint32_t ru = __float_as_uint(ratio);
  int32_t exp_biased = static_cast<int32_t>((ru >> 23) & 0xFF);
  uint32_t mant = ru & 0x7FFFFF;
  exp_biased = exp_biased + (mant != 0 ? 1 : 0);
  exp_biased = exp_biased < 0 ? 0 : (exp_biased > 254 ? 254 : exp_biased);
  const uint8_t ue8m0 = static_cast<uint8_t>(exp_biased);
  // inv_scale = 2^-(exp_biased - 127) = inv_scale_ue8m0 of (exp_biased)
  const float inv_scale = inv_scale_ue8m0(exp_biased);

  const uint8_t n0 = e2m1_nibble(x0 * inv_scale);
  const uint8_t n1 = e2m1_nibble(x1 * inv_scale);
  const uint8_t n2 = e2m1_nibble(y0 * inv_scale);
  const uint8_t n3 = e2m1_nibble(y1 * inv_scale);
  const uint8_t byte0 = n0 | static_cast<uint8_t>(n1 << 4);
  const uint8_t byte1 = n2 | static_cast<uint8_t>(n3 << 4);
  const uint16_t packed = static_cast<uint16_t>(byte0) | (static_cast<uint16_t>(byte1) << 8);

  // Page layout (kPageSize tokens):
  //   [0, kPageSize * 64):           packed FP4 (64 bytes/token)
  //   [kPageSize * 64, +kPageSize*4): ue8m0 scales (4 bytes/token)
  // DeepGEMM treats each token's 4 ue8m0 bytes as a single int32 (kv_sf).
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  const auto page_ptr = pointer::offset(cache, page * kPageBytes);
  const auto value_ptr = pointer::offset(page_ptr, offset * kPackedBytesPerToken);
  const auto scale_ptr = pointer::offset(page_ptr, kPackedBytesPerToken << kPageBits, offset * kScaleBytesPerToken);

  // Each lane writes 2 packed bytes (one uint16) at byte offset 2*lane_id.
  static_cast<uint16_t*>(value_ptr)[lane_id] = packed;

  // The first lane in each 8-lane group writes its block's ue8m0 byte.
  if ((lane_id % kLanesPerBlock) == 0) {
    static_cast<uint8_t*>(scale_ptr)[block_idx_in_token] = ue8m0;
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename Float, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct FusedStoreCacheFlashMLAKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = host::div_ceil(584 * kPageSize, 576) * 576;
  static constexpr auto kernel = fused_store_flashmla_cache<Float, IndicesT, kLogSize, kUsePDL>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);

  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 512})  // input
        .with_dtype<Float>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({N})  // indices
        .with_dtype<IndicesT>()
        .with_device(device_)
        .verify(indices);
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = FusedStoreCacheParam{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = num_tokens,
    };
    const auto kBlockSize = 256;
    const auto num_blocks = num_tokens;
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

template <typename Float, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct FusedStoreCacheIndexerMXFP4Kernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  // 64 packed FP4 bytes + 4 ue8m0 bytes per token = 68 bytes.
  static constexpr int64_t kPageBytes = 68 * kPageSize;
  static constexpr auto kernel = fused_store_indexer_mxfp4_cache<Float, IndicesT, kLogSize, kUsePDL>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);

  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 128})  // input
        .with_dtype<Float>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({N})  // indices
        .with_dtype<IndicesT>()
        .with_device(device_)
        .verify(indices);
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = FusedStoreCacheParam{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = num_tokens,
    };
    const auto kBlockSize = 128;
    const auto num_blocks = div_ceil(num_tokens * 32, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

template <typename Float, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct FusedStoreCacheIndexerKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = 132 * kPageSize;
  static constexpr auto kernel = fused_store_indexer_cache<Float, IndicesT, kLogSize, kUsePDL>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);

  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 128})  // input
        .with_dtype<Float>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({N})  // indices
        .with_dtype<IndicesT>()
        .with_device(device_)
        .verify(indices);
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = FusedStoreCacheParam{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = num_tokens,
    };
    const auto kBlockSize = 128;
    const auto num_blocks = div_ceil(num_tokens * 32, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
