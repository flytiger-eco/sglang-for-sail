// Fused RoPE + 128-pt Hadamard + MXFP4 act-quant for the C4 indexer Q path.
//
// Mirrors `fused_q_indexer_rope_hadamard_quant` in main_norm_rope.cuh for
// parts 1-3 (load, rope, hadamard), and swaps the per-warp FP8 quant in
// part 4 for the per-sub-block (8-lane) MXFP4 quant from the K-side
// `fused_norm_rope_indexer_mxfp4`. Output is two plain tensors --
// (q_packed, q_sf) -- plus a weights tensor `weight * weight_scale` (fp32).
// q_scale is NOT folded into weights (MXFP4 has 4 per-token scales, so the
// fold trick used by the FP8 Q kernel does not apply); the scales travel
// alongside q_packed as q_sf, matching the contract DeepGEMM's
// `fp8_fp4_mqa_logits` expects.
//
// q_sf shape is (batch, next_n=1, num_heads) int32 to match DeepGEMM's
// `fp8_fp4_paged_mqa_logits` contract directly (no Python-side unsqueeze
// needed). The packed-scale byte order inside each int32 matches the Triton
// `_downcast_to_mxfp4_opt` output (sub-block i ue8m0 byte goes to byte i of
// the int32, little-endian).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>

namespace {

using deepseek_v4::fp8::inv_scale_ue8m0;

// 1 warp per block -- finer scheduling granularity, higher SM occupancy ceiling,
// and simpler work_id = blockIdx.x (no warp_id arithmetic).
constexpr uint32_t kFusedQMxfp4BlockSize = device::kWarpThreads;  // 32
constexpr uint32_t kFusedQMxfp4NumWarps = kFusedQMxfp4BlockSize / device::kWarpThreads;  // 1

#define Q_MXFP4_KERNEL __global__ __launch_bounds__(kFusedQMxfp4BlockSize, 64)

struct FusedQIndexerRopeHadamardMxfp4Params {
  const void* __restrict__ q_input;  // (T, H, 128)   DType
  void* __restrict__ q_packed;       // (T, H, 64)    uint8 (two e2m1 / byte)
  void* __restrict__ q_sf;           // (T, next_n=1, H) int32 (4 ue8m0 packed)
  const void* __restrict__ weight;   // (T, H)        DType
  float* __restrict__ weights_out;   // (T, H, 1)     fp32 = weight * weight_scale
  float weight_scale;
  const float* __restrict__ freqs_cis;  // (max_pos, 64) fp32
  const void* __restrict__ positions;   // (T,) PosT
  uint32_t batch_size;
  uint32_t num_heads;
};

// Quantize one fp32 value to an E2M1 4-bit nibble. Bucket boundaries
// [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] mirror torch's
// `bucketize(..., right=False)`; sign bit is set only for non-zero codes.
// (Copy of fused_norm_rope_v2_mxfp4.cuh:59-80 to keep this header self-contained.)
__device__ __forceinline__ uint8_t e2m1_nibble_q(float x) {
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

// Pack two fp32 values into one e2m1x2 byte via PTX hardware instruction.
// lo nibble = e2m1(lo_val), hi nibble = e2m1(hi_val).
__device__ __forceinline__ uint8_t e2m1x2_pack_ptx(float hi_val, float lo_val) {
  uint8_t tmp;
  asm(
      "cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;"
      : "=r"(tmp)
      : "f"(hi_val), "f"(lo_val)
  );
  return tmp;
}

template <typename DType, typename PosT, bool kUsePDL>
Q_MXFP4_KERNEL void
fused_q_indexer_rope_hadamard_mxfp4(const __grid_constant__ FusedQIndexerRopeHadamardMxfp4Params params) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;  // = 16
  // MXFP4 layout (matches fused_norm_rope_indexer_mxfp4 / Triton downcast).
  constexpr int kMxfp4Block = 32;
  constexpr int kLanesPerBlock = kMxfp4Block / kVecSize;   // 8
  constexpr int kNumScaleBlocks = kHeadDim / kMxfp4Block;  // 4
  constexpr int kPackedBytesPerToken = kHeadDim / 2;       // 64
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);

  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;

  const auto lane_id = threadIdx.x;  // 1 warp/block: lane_id = threadIdx.x
  const auto work_id = blockIdx.x;   // 1 work item per block
  // Last `kRopeSize` lanes own the rope tail; their 4-elem packs cover the
  // trailing kRopeDim elements.
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;
  // Sub-block index within the token-head: lanes 0..7 -> 0, 8..15 -> 1, etc.
  const int block_idx_in_token = lane_id / kLanesPerBlock;

  const uint32_t total_works = params.batch_size * params.num_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_heads;
  const auto input_ptr = static_cast<const DType*>(params.q_input) + work_id * kHeadDim;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[batch_id]);
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq;
  const auto weight_val = cast<float>(static_cast<const DType*>(params.weight)[work_id]);

  // part 1: load (no norm). Each lane owns a 4-elem pack.
  {
    Storage input_vec;
    input_vec.load(input_ptr, lane_id);
    if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeSize));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = cast<float>(input_vec[i]);
    }
  }

  // part 2: rope on rope lanes only (4 elems / lane = 2 (real, imag) pairs).
  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto fxr = freq[0];
    const auto fxi = freq[1];
    const auto fyr = freq[2];
    const auto fyi = freq[3];
    data[0] = x_real * fxr - x_imag * fxi;
    data[1] = x_real * fxi + x_imag * fxr;
    data[2] = y_real * fyr - y_imag * fyi;
    data[3] = y_real * fyi + y_imag * fyr;
  }

  PDLTriggerSecondary<kUsePDL>();

  // part 3: 128-point Hadamard (2 local stages + 5 cross-lane shfl_xor stages).
  // Same recipe as `fused_q_indexer_rope_hadamard_quant` / K-side mxfp4 kernel,
  // so the (token, head) basis matches what the K-side cache write produced.
  {
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
      data[2] = a2 + a3;
      data[3] = a2 - a3;
    }
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a2;
      data[1] = a1 + a3;
      data[2] = a0 - a2;
      data[3] = a1 - a3;
    }
    // ILP: batch all 4 shuffles first, then compute — breaks shuffle→compute dep chain.
#pragma unroll
    for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
      const float o0 = __shfl_xor_sync(0xFFFFFFFFu, data[0], mask, kWarpThreads);
      const float o1 = __shfl_xor_sync(0xFFFFFFFFu, data[1], mask, kWarpThreads);
      const float o2 = __shfl_xor_sync(0xFFFFFFFFu, data[2], mask, kWarpThreads);
      const float o3 = __shfl_xor_sync(0xFFFFFFFFu, data[3], mask, kWarpThreads);
      const bool flip = (lane_id & mask) != 0;
      data[0] = flip ? (o0 - data[0]) : (data[0] + o0);
      data[1] = flip ? (o1 - data[1]) : (data[1] + o1);
      data[2] = flip ? (o2 - data[2]) : (data[2] + o2);
      data[3] = flip ? (o3 - data[3]) : (data[3] + o3);
    }
    const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
  }

  // part 4: per-sub-block MXFP4 quant + store, plus per-(token, head) weight scale.
  {
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = math::max(local_max, math::abs(data[i]));
    }
    // Reduce across the 8 lanes that share a sub-block, not the whole warp.
    float amax = warp::reduce_max<kLanesPerBlock>(local_max);
    amax = fmaxf(amax, 1e-4f);
    // ceil(log2(amax / 6.0)) using fp32 bit trick:
    //   (float_to_bits(ratio) + 0x007FFFFF) & 0x7F800000
    const float ratio = amax / 6.0f;
    uint32_t dequant_scale_exp = (__float_as_uint(ratio) + 0x007FFFFFu) & 0x7F800000u;
    int32_t exp_biased = static_cast<int32_t>(dequant_scale_exp >> 23);
    exp_biased = exp_biased > 254 ? 254 : exp_biased;
    const uint8_t ue8m0 = static_cast<uint8_t>(exp_biased);
    const float inv_scale = __uint_as_float(static_cast<uint32_t>(254 - exp_biased) << 23);

#if __CUDA_ARCH__ >= 890
    // ── PTX cvt.rn.satfinite.e2m1x2.f32 (requires sm_89+ / PPU >= 150) ──
    const float s0 = data[0] * inv_scale;
    const float s1 = data[1] * inv_scale;
    const float s2 = data[2] * inv_scale;
    const float s3 = data[3] * inv_scale;
    const uint8_t byte0 = e2m1x2_pack_ptx(s1, s0);
    const uint8_t byte1 = e2m1x2_pack_ptx(s3, s2);
#else
    // ── Fallback: manual bucket quantization ──
    const uint8_t n0 = e2m1_nibble_q(data[0] * inv_scale);
    const uint8_t n1 = e2m1_nibble_q(data[1] * inv_scale);
    const uint8_t n2 = e2m1_nibble_q(data[2] * inv_scale);
    const uint8_t n3 = e2m1_nibble_q(data[3] * inv_scale);
    const uint8_t byte0 = n0 | static_cast<uint8_t>(n1 << 4);
    const uint8_t byte1 = n2 | static_cast<uint8_t>(n3 << 4);
#endif
    const uint16_t packed = static_cast<uint16_t>(byte0) | (static_cast<uint16_t>(byte1) << 8);

    // q_packed row: 64 bytes per (token, head).
    // Streaming store: bypass L1/L2 cache for large sequential writes.
    {
      auto value_ptr = static_cast<uint8_t*>(params.q_packed) + work_id * kPackedBytesPerToken;
      const uint16_t* store_addr = reinterpret_cast<const uint16_t*>(value_ptr) + lane_id;
      asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(store_addr), "h"(packed) : "memory");
    }

    // q_sf row: int32 per (token, head) holding 4 ue8m0 bytes; write the
    // sub-block ue8m0 byte into byte offset `block_idx_in_token` of that
    // int32 -- little-endian packing matches Triton's `packed_scale`.
    if ((lane_id % kLanesPerBlock) == 0) {
      auto scale_ptr = static_cast<uint8_t*>(params.q_sf) + work_id * kNumScaleBlocks;
      const uint8_t* scale_addr = scale_ptr + block_idx_in_token;
      asm volatile("st.global.cs.u8 [%0], %1;" :: "l"(scale_addr), "r"((uint32_t)ue8m0) : "memory");
    }

    if (lane_id == 0) {
      float* wout_ptr = &params.weights_out[work_id];
      float wout_val = weight_val * params.weight_scale;
      asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(wout_ptr), "f"(wout_val) : "memory");
    }
  }
}

template <typename DType, bool kUsePDL>
struct FusedQIndexerRopeHadamardMxfp4Kernel {
  template <typename PosT>
  static constexpr auto kernel = fused_q_indexer_rope_hadamard_mxfp4<DType, PosT, kUsePDL>;

  static void forward(
      const tvm::ffi::TensorView q_input,
      const tvm::ffi::TensorView q_packed,
      const tvm::ffi::TensorView q_sf,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView weights_out,
      double weight_scale,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions) {
    using namespace host;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kRopeDim = 64;
    constexpr int64_t kPackedBytesPerToken = kHeadDim / 2;  // 64
    constexpr int64_t kNumScaleBlocks = kHeadDim / 32;      // 4

    auto B = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"num_heads"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q_input);
    TensorMatcher({B, H, kPackedBytesPerToken})  //
        .with_strides({-1, kPackedBytesPerToken, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(q_packed);
    // q_sf shape matches DeepGEMM's `fp8_fp4_paged_mqa_logits` contract:
    // (batch, next_n=1, num_heads) int32 with 4 ue8m0 bytes packed per int32.
    TensorMatcher({B, 1, H})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(q_sf);
    TensorMatcher({B, H})  //
        .with_dtype<DType>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({B, H, 1})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(weights_out);
    TensorMatcher({-1, kRopeDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto num_heads = static_cast<uint32_t>(H.unwrap());
    if (batch_size == 0) return;

    // Row pointer math assumes (batch, head, elem) contiguous order.
    const int64_t expected_q_input_batch = static_cast<int64_t>(num_heads) * kHeadDim;
    const int64_t expected_q_packed_batch = static_cast<int64_t>(num_heads) * kPackedBytesPerToken;
    const int64_t expected_q_sf_batch = static_cast<int64_t>(num_heads) * kNumScaleBlocks;
    RuntimeCheck(
        q_input.stride(0) == expected_q_input_batch,
        "q_input must be contiguous (B, H, kHeadDim); got stride[0]=",
        q_input.stride(0));
    RuntimeCheck(
        q_packed.stride(0) == expected_q_packed_batch,
        "q_packed must be contiguous (B, H, kPackedBytesPerToken); got stride[0]=",
        q_packed.stride(0));
    // q_sf is allocated as int32 [B, 1, H]; its byte stride between rows is
    // num_heads * 4 (= num_heads * kNumScaleBlocks). The kernel writes 4
    // uint8 bytes per (token, head) at base + work_id * 4 -- byte layout
    // identical to a flat [B*H] int32 array, so the (T, 1, H) reshape is
    // a no-op metadata view.
    RuntimeCheck(
        q_sf.stride(0) == static_cast<int64_t>(num_heads),
        "q_sf must be contiguous (B, 1, H) int32; got stride[0]=",
        q_sf.stride(0));

    const auto params = FusedQIndexerRopeHadamardMxfp4Params{
        .q_input = q_input.data_ptr(),
        .q_packed = q_packed.data_ptr(),
        .q_sf = q_sf.data_ptr(),
        .weight = weight.data_ptr(),
        .weights_out = static_cast<float*>(weights_out.data_ptr()),
        .weight_scale = static_cast<float>(weight_scale),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .batch_size = batch_size,
        .num_heads = num_heads,
    };
    const auto total_works = batch_size * num_heads;
    const auto num_blocks = total_works;  // 1 warp/block: 1 block per work item
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(num_blocks, kFusedQMxfp4BlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

}  // namespace
