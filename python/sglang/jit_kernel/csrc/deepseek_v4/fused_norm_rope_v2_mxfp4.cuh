// Fused norm + rope + Hadamard + MXFP4 quant + paged store for the C4 indexer.
//
// Mirrors `fused_norm_rope_indexer` in fused_norm_rope_v2.cuh for parts 1-3
// (norm, rope, hadamard), and swaps the per-warp FP8 quant+store in part 4 for
// the per-sub-block (8-lane) MXFP4 quant+store from
// `fused_store_indexer_mxfp4_cache` in store.cuh.
//
// Cache layout (matches `fused_store_indexer_mxfp4_cache`):
//   per token: 64 packed FP4 bytes + 4 ue8m0 scale bytes = 68 bytes
//   per page : [0, kPageSize*64)        packed FP4 (64 B/token)
//              [kPageSize*64, +ks*4)    ue8m0 scales (4 B/token, treated as int32)

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/compress_v2.cuh>
#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

using PlanC = device::compress::CompressPlan;
using PlanD = device::compress::DecodePlan;
using deepseek_v4::fp8::inv_scale_ue8m0;

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;

struct FusedNormRopeStoreMxfp4Params {
  void* __restrict__ input;
  const void* __restrict__ handle;  // PlanC* (extend) or PlanD* (decode)
  const void* __restrict__ weight;
  const float* __restrict__ freqs_cis;
  const int32_t* __restrict__ out_loc;
  uint8_t* __restrict__ kvcache;
  float eps;
  uint32_t compress_ratio;
  uint32_t num_tokens;
};

enum class ForwardMode : bool {
  CompressExtend = 0,
  CompressDecode = 1,
};

// Quantize one fp32 value to an E2M1 4-bit nibble. Bucket boundaries
// [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] mirror torch's
// `bucketize(..., right=False)`; sign bit is set only for non-zero codes.
// (Copy of store.cuh:28-49 to keep this header self-contained.)
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

#define INDEXER_KERNEL __global__ __launch_bounds__(kBlockSize, 8)

// Indexer variant: kHeadDim = 128, 1 token per warp (8 tokens per block).
// Each warp's 32 lanes cover the full 128-elem head_dim (kVecSize = 4 each).
// FP4 sub-blocks of 32 elems → 4 sub-blocks per token, 8 lanes per sub-block.
template <typename DType, ForwardMode kMode, int32_t kPageBits, bool kUsePDL>
INDEXER_KERNEL void fused_norm_rope_indexer_mxfp4(const __grid_constant__ FusedNormRopeStoreMxfp4Params params) {
  using namespace device;
  using enum ForwardMode;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  // FP4 layout constants (must match fused_store_indexer_mxfp4_cache).
  constexpr int kMxfp4Block = 32;
  constexpr int kLanesPerBlock = kMxfp4Block / kVecSize;   // 8
  constexpr int kNumScaleBlocks = kHeadDim / kMxfp4Block;  // 4
  constexpr int kPackedBytesPerToken = kHeadDim / 2;       // 64
  constexpr int kScaleBytesPerToken = kNumScaleBlocks;     // 4
  constexpr int64_t kPageBytes = static_cast<int64_t>(kPackedBytesPerToken + kScaleBytesPerToken) << kPageBits;
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kNumWarps + warp_id;
  // Lanes whose 4-elem pack lies in the rope tail (= last `kRopeSize` packs).
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;
  // Sub-block index within the token: lanes 0..7 → 0, 8..15 → 1, etc.
  const int block_idx_in_token = lane_id / kLanesPerBlock;

  if (work_id >= params.num_tokens) return;

  const auto input = static_cast<DType*>(params.input) + work_id * kHeadDim;
  int32_t position;
  int32_t out_loc;
  if constexpr (kMode == CompressExtend) {
    const auto plan = static_cast<const PlanC*>(params.handle)[work_id];
    if (plan.is_invalid()) return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[plan.ragged_id];
  } else if constexpr (kMode == CompressDecode) {
    const auto plan = static_cast<const PlanD*>(params.handle)[work_id];
    if (plan.seq_len % params.compress_ratio != 0) return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[work_id];
  } else {
    static_assert(host::dependent_false_v<DType>, "Unsupported Mode");
  }
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq;

  // part 1: norm
  {
    Storage input_vec, weight_vec;
    input_vec.load(input, lane_id);
    weight_vec.load(params.weight, lane_id);
    if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeSize));

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      sum_of_squares += fp32_input * fp32_input;
    }

    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      const auto fp32_weight = cast<float>(weight_vec[i]);
      data[i] = fp32_input * norm_factor * fp32_weight;
    }
  }

  // part 2: rope (rope-lane only, 4 elems per lane = 2 (real, imag) pairs)
  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto freq_x_real = freq[0];
    const auto freq_x_imag = freq[1];
    const auto freq_y_real = freq[2];
    const auto freq_y_imag = freq[3];
    data[0] = x_real * freq_x_real - x_imag * freq_x_imag;
    data[1] = x_real * freq_x_imag + x_imag * freq_x_real;
    data[2] = y_real * freq_y_real - y_imag * freq_y_imag;
    data[3] = y_real * freq_y_imag + y_imag * freq_y_real;
  }

  // part 3: hadamard transform
  {
    // Stage 1: butterfly (data[0], data[1]) and (data[2], data[3]).
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
      data[2] = a2 + a3;
      data[3] = a2 - a3;
    }
    // Stage 2: butterfly (data[0], data[2]) and (data[1], data[3]).
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a2;
      data[1] = a1 + a3;
      data[2] = a0 - a2;
      data[3] = a1 - a3;
    }
    // Stages 3..7: cross-lane butterflies. Lower-lane (mask bit clear) keeps
    // the sum, upper-lane (mask bit set) keeps the difference.
#pragma unroll
    for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
        const float other = __shfl_xor_sync(0xFFFFFFFFu, data[i], mask, kWarpThreads);
        data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
      }
    }
    const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
  }

  // part 4: per-sub-block MXFP4 quant + store (8 lanes per scale group).
  {
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = math::max(local_max, math::abs(data[i]));
    }
    // Reduce across 8 lanes per sub-block, not the whole warp.
    float amax = warp::reduce_max<kLanesPerBlock>(local_max);
    amax = fmaxf(amax, 1e-4f);
    // ceil(log2(amax / 6.0)) using the fp32 bit trick from
    // fused_store_indexer_mxfp4_cache (store.cuh:188-196).
    const float ratio = amax / 6.0f;
    uint32_t ru = __float_as_uint(ratio);
    int32_t exp_biased = static_cast<int32_t>((ru >> 23) & 0xFF);
    uint32_t mant = ru & 0x7FFFFF;
    exp_biased = exp_biased + (mant != 0 ? 1 : 0);
    exp_biased = exp_biased < 0 ? 0 : (exp_biased > 254 ? 254 : exp_biased);
    const uint8_t ue8m0 = static_cast<uint8_t>(exp_biased);
    const float inv_scale = inv_scale_ue8m0(exp_biased);

    const uint8_t n0 = e2m1_nibble(data[0] * inv_scale);
    const uint8_t n1 = e2m1_nibble(data[1] * inv_scale);
    const uint8_t n2 = e2m1_nibble(data[2] * inv_scale);
    const uint8_t n3 = e2m1_nibble(data[3] * inv_scale);
    const uint8_t byte0 = n0 | static_cast<uint8_t>(n1 << 4);
    const uint8_t byte1 = n2 | static_cast<uint8_t>(n3 << 4);
    const uint16_t packed = static_cast<uint16_t>(byte0) | (static_cast<uint16_t>(byte1) << 8);

    const int32_t page = out_loc >> kPageBits;
    const int32_t offset = out_loc & ((1 << kPageBits) - 1);
    const auto page_ptr = params.kvcache + page * kPageBytes;
    const auto value_ptr = page_ptr + offset * kPackedBytesPerToken;
    const auto scale_ptr = page_ptr + (kPackedBytesPerToken << kPageBits) + offset * kScaleBytesPerToken;

    PDLTriggerSecondary<kUsePDL>();
    // Each lane writes one uint16 (= 2 packed FP4 bytes) at lane_id*2.
    reinterpret_cast<uint16_t*>(value_ptr)[lane_id] = packed;
    // The first lane in each 8-lane sub-block publishes its ue8m0 byte.
    if ((lane_id % kLanesPerBlock) == 0) {
      static_cast<uint8_t*>(scale_ptr)[block_idx_in_token] = ue8m0;
    }
  }
}

template <typename DType, int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageSize, bool kUsePDL>
struct FusedNormRopeMxfp4Kernel {
  static constexpr int32_t kLogPageSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = 68 * kPageSize;

  static_assert(kHeadDim == 128, "MXFP4 indexer kernel only supports head_dim=128");
  static_assert(kRopeDim == 64);
  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");

  template <ForwardMode kMode>
  static constexpr auto select_kernel() {
    return fused_norm_rope_indexer_mxfp4<DType, kMode, kLogPageSize, kUsePDL>;
  }

  static void forward(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView plan,
      const tvm::ffi::TensorView weight,
      const float eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      const bool is_decode,
      const uint32_t compress_ratio) {
    using namespace host;
    using enum ForwardMode;

    const auto mode = static_cast<ForwardMode>(is_decode);

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, kHeadDim})  // input
        .with_dtype<DType>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({kHeadDim})  // weight
        .with_dtype<DType>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({-1, kRopeDim})  // freqs_cis
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    TensorMatcher({-1})  // out_loc
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(out_loc);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(kvcache);

    switch (mode) {
      case CompressExtend:
        compress::verify_plan_c(plan, N, device_);
        RuntimeCheck(out_loc.size(0) >= N.unwrap());
        break;
      case CompressDecode:
        compress::verify_plan_d(plan, N, device_);
        RuntimeCheck(out_loc.size(0) == N.unwrap());
        break;
    }

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    const auto params = FusedNormRopeStoreMxfp4Params{
        .input = input.data_ptr(),
        .handle = plan.data_ptr(),
        .weight = weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .out_loc = static_cast<const int32_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .eps = eps,
        .compress_ratio = compress_ratio,
        .num_tokens = num_tokens,
    };
    // Same warp-major packing as FP8 indexer variant: kNumWarps tokens per block.
    const uint32_t num_blocks = div_ceil(num_tokens, kNumWarps);
    const auto device = device_.unwrap();
    const auto kernel = mode == CompressExtend ? select_kernel<CompressExtend>() : select_kernel<CompressDecode>();
    LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
