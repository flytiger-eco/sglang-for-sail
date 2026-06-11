#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

// ============================================================
// DSv4 paged FP8 K-cache dequantize + gather kernel.
//
// Mirrors `_dequantize_and_gather_k_kernel` in
// `python/sglang/srt/layers/attention/dsv4/sparse_prefill.py`.
//
// Per-token layout (576 bytes data + 8 bytes scales = 584 B/token):
//   [  448 B FP8 nope  | 128 B BF16 rope ] data
//   [  7 UE8M0 scales  | 1 B pad        ] scales
// Per-page layout (padded to 576-byte multiples; passed in as
// `block_stride` to handle the padding correctly):
//   page_size * 576 bytes data, then page_size * 8 bytes scales.
//
// Output dim per token = 448 dequantized BF16 (FP8 -> BF16 with
// per-quant-block UE8M0 scale) + 64 directly-copied BF16 = 512.
// ============================================================

namespace dequant_gather {

constexpr uint32_t kBlockSize = 128;       // threads per CTA (matches Triton NUM_WORKERS)
constexpr uint32_t kFP8Dim = 448;          // FP8 nope width
constexpr uint32_t kBF16Dim = 64;          // BF16 rope width
constexpr uint32_t kScaleDim = 8;          // 7 scales + 1 pad per token
constexpr uint32_t kQuantBlock = 64;       // FP8 quant block length
constexpr uint32_t kTokenDataSize = 576;   // 448 + 128
constexpr uint32_t kOutputDim = 512;       // kFP8Dim + kBF16Dim
constexpr uint32_t kNumQuantBlocks = 7;    // kFP8Dim / kQuantBlock
constexpr uint32_t kMaxCTAsPerReq = 64;    // CTA 数上限

// Two-regime seqlen-adaptive dispatch:
//   短序列 (< kSeqLenThreshold): TPT=1, 每线程处理 1 token, 最大化 CTA 数 / SM 利用率
//   长序列 (>= kSeqLenThreshold): TPT=4, 每线程处理 4 tokens, CTA 数已足够
// 分界点 = kBlockSize * kMaxCTAsPerReq * kTokensPerThreadLong = 32768,
// 即 TPT=4 恰好 cap 到 kMaxCTAsPerReq CTAs 的序列长度. 低于此值时 TPT=1
// 始终能提供更多 CTA, 避免 CTA 数断崖 (原 8192 阈值在 seq_k=8500 时
// CTA 从 64 骤降到 17, 导致 ~1.9x 性能下降).
constexpr uint32_t kTokensPerThreadShort = 1;
constexpr uint32_t kTokensPerThreadLong = 4;
constexpr uint32_t kSeqLenThreshold = kBlockSize * kMaxCTAsPerReq * kTokensPerThreadLong;  // 32768

}  // namespace dequant_gather

struct DequantGatherParams {
  const uint8_t* __restrict__ k_cache;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ block_table;
  __nv_bfloat16* __restrict__ out;
  const int32_t* __restrict__ gather_lens;  // nullable
  int64_t out_stride0;                      // batch stride in bf16 elements
  int64_t out_stride1;                      // token stride in bf16 elements
  int64_t block_stride;                     // bytes per physical page
  int32_t offset;
  int32_t max_blocks_per_seq;
  int32_t cache_block_size;
  uint32_t num_ctas_per_req;  // 每个 request 使用的 CTA 数
  int32_t use_fp8_native;     // 1: __nv_cvt_fp8_to_halfraw 路径；0: IEEE754 位构造融合路径
};

// FP8 E4M3 (FN, no Inf) -> float32 via the standard CUDA intrinsic.
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t bits) {
  const __nv_fp8_storage_t storage = static_cast<__nv_fp8_storage_t>(bits);
  const __half h = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
  return __half2float(h);
}

// UE8M0: encoded scale `s` represents `2^(s - 127)`.
__device__ __forceinline__ float ue8m0_to_scale(uint8_t encoded) {
  return exp2f(static_cast<float>(static_cast<int32_t>(encoded) - 127));
}

// Fused FP8 E4M3 + UE8M0 -> float32 via IEEE754 bit construction.
// Used on hardware without native FP8 support (e.g. SM80/A100). Mirrors
// `_fused_fp8_ue8m0_dequant` in sparse_prefill.py and produces bit-identical
// results to `fp8e4m3_to_float(x) * ue8m0_to_scale(s)` for all FP8 normal
// values. Subnormals (exp == 0) are flushed to zero (FTZ).
__device__ __forceinline__ float fused_fp8_ue8m0_to_float(uint8_t fp8_bits, uint8_t ue8m0) {
  const int32_t x = static_cast<int32_t>(fp8_bits);
  const int32_t sign = x >> 7;
  const int32_t exp = (x >> 3) & 0xF;
  const int32_t mant = x & 0x7;
  const int32_t ue = static_cast<int32_t>(ue8m0);
  // Construct float32: sign | (fp8_exp + ue8m0 - 7) << 23 | mant << 20
  const int32_t f32_bits = (sign << 31) | ((exp + ue - 7) << 23) | (mant << 20);
  // FTZ for FP8 subnormals (exp == 0) — matches the Triton fused path.
  if (exp == 0) return 0.0f;
  return __int_as_float(f32_bits);
}

template <uint32_t TOKENS_PER_THREAD>
__global__ __launch_bounds__(dequant_gather::kBlockSize, 1) void dequantize_gather_k_kernel(
    const DequantGatherParams params) {
  using namespace dequant_gather;

  const uint32_t batch_idx = blockIdx.x / params.num_ctas_per_req;
  const uint32_t cta_idx = blockIdx.x % params.num_ctas_per_req;
  const uint32_t global_worker_id = cta_idx * kBlockSize + threadIdx.x;
  const uint32_t total_workers = params.num_ctas_per_req * kBlockSize;

  const int32_t seq_len = params.seq_lens[batch_idx];
  const int32_t gather_len = (params.gather_lens != nullptr) ? params.gather_lens[batch_idx] : seq_len;
  const int32_t start_pos = seq_len - gather_len;

  for (int32_t i = static_cast<int32_t>(global_worker_id); i < gather_len; i += static_cast<int32_t>(total_workers)) {
    const int32_t pos = start_pos + i;
    const int32_t block_in_seq = pos / params.cache_block_size;
    const int32_t pos_in_block = pos - block_in_seq * params.cache_block_size;

    const int32_t physical_block_idx =
        params.block_table[static_cast<int64_t>(batch_idx) * params.max_blocks_per_seq + block_in_seq];

    // Token data + scale pointers within the physical page.
    const uint8_t* cache_block_ptr =
        params.k_cache + static_cast<int64_t>(physical_block_idx) * params.block_stride;
    const uint8_t* token_data_ptr = cache_block_ptr + static_cast<int64_t>(pos_in_block) * kTokenDataSize;
    const uint8_t* token_scale_ptr =
        cache_block_ptr + static_cast<int64_t>(params.cache_block_size) * kTokenDataSize +
        static_cast<int64_t>(pos_in_block) * kScaleDim;

    // Output row pointer (bf16 elements).
    __nv_bfloat16* out_row = params.out + static_cast<int64_t>(batch_idx) * params.out_stride0 +
                             static_cast<int64_t>(params.offset + i) * params.out_stride1;

    // ------------------------------------------------------------
    // Stage 1: FP8 dequantization (7 blocks * 64 = 448 elements).
    //
    // Each quant block of 64 fp8 bytes shares one UE8M0 scale.
    // Process the block as 4 chunks of 16 fp8 bytes (vector loaded
    // with a single uint4) and emit 16 bf16 outputs per chunk via
    // two uint4 stores.
    // ------------------------------------------------------------
    // `params.use_fp8_native` is uniform across all threads in the grid, so
    // the branch below is uniform — no warp divergence.
    const bool use_fp8_native = params.use_fp8_native != 0;
#pragma unroll
    for (uint32_t qb = 0; qb < kNumQuantBlocks; ++qb) {
      const uint8_t raw_scale = token_scale_ptr[qb];
      // Native path uses a precomputed fp32 scale; fused path consumes the
      // raw UE8M0 byte directly inside `fused_fp8_ue8m0_to_float`.
      const float scale = use_fp8_native ? ue8m0_to_scale(raw_scale) : 0.0f;
      const uint32_t base = qb * kQuantBlock;

#pragma unroll
      for (uint32_t chunk = 0; chunk < kQuantBlock / 16u; ++chunk) {
        const uint32_t elem_offset = base + chunk * 16u;
        const uint4 raw = *reinterpret_cast<const uint4*>(token_data_ptr + elem_offset);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);

        __nv_bfloat16 results[16];
#pragma unroll
        for (uint32_t k = 0; k < 16u; ++k) {
          float f;
          if (use_fp8_native) {
            f = fp8e4m3_to_float(bytes[k]) * scale;
          } else {
            f = fused_fp8_ue8m0_to_float(bytes[k], raw_scale);
          }
          results[k] = __float2bfloat16(f);
        }

        // 16 bf16 = 32 bytes = 2 * uint4. out_row is bf16-typed so
        // (out_row + elem_offset) advances by elem_offset bf16 elements
        // (= 2 * elem_offset bytes); elem_offset is a multiple of 16
        // so each store is 16-byte aligned.
        *reinterpret_cast<uint4*>(out_row + elem_offset) = *reinterpret_cast<const uint4*>(&results[0]);
        *reinterpret_cast<uint4*>(out_row + elem_offset + 8u) = *reinterpret_cast<const uint4*>(&results[8]);
      }
    }

    // ------------------------------------------------------------
    // Stage 2: BF16 direct copy of the rope tail (64 bf16 = 128 B).
    // ------------------------------------------------------------
    const __nv_bfloat16* bf16_src = reinterpret_cast<const __nv_bfloat16*>(token_data_ptr + kFP8Dim);
    __nv_bfloat16* bf16_dst = out_row + kFP8Dim;
#pragma unroll
    for (uint32_t j = 0; j < kBF16Dim / 8u; ++j) {
      *reinterpret_cast<uint4*>(bf16_dst + j * 8u) = *reinterpret_cast<const uint4*>(bf16_src + j * 8u);
    }
  }
}

struct DequantizeGatherKernel {
  static void run(
      const tvm::ffi::TensorView out,                           // [B, T, 512] bf16, last-dim contiguous
      const tvm::ffi::TensorView k_cache,                       // [num_pages, page_bytes] uint8
      const tvm::ffi::TensorView seq_lens,                      // [B] int32
      const tvm::ffi::TensorView block_table,                   // [B, P] int32
      const int64_t offset,
      const tvm::ffi::Optional<tvm::ffi::TensorView> gather_lens,  // [B] int32, optional
      const int64_t block_size,
      const int64_t use_fp8_native) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto OS0 = SymbolicSize{"out_stride0"};
    auto OS1 = SymbolicSize{"out_stride1"};
    auto P = SymbolicSize{"max_blocks_per_seq"};
    auto BS = SymbolicSize{"block_stride"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    // out: [B, T, 512] bf16, inner-dim contiguous, free strides on B/T.
    TensorMatcher({B, -1, static_cast<int64_t>(dequant_gather::kOutputDim)})
        .with_strides({OS0, OS1, 1})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(out);

    // k_cache: [num_pages, page_bytes] uint8. block_stride = stride(0).
    TensorMatcher({-1, -1}).with_strides({BS, 1}).with_dtype<uint8_t>().with_device(device_).verify(k_cache);

    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({B, P}).with_dtype<int32_t>().with_device(device_).verify(block_table);

    const int32_t* gather_lens_ptr = nullptr;
    if (gather_lens.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(gather_lens.value());
      gather_lens_ptr = static_cast<const int32_t*>(gather_lens.value().data_ptr());
    }

    RuntimeCheck(block_size > 0, "block_size must be positive");

    const auto batch_size = static_cast<uint32_t>(B.unwrap());

    // Seqlen-adaptive dispatch: pick TOKENS_PER_THREAD based on gather length.
    const int64_t max_gather_len = out.shape()[1] - offset;

    auto launch = [&](uint32_t tokens_per_thread, auto kernel_fn) {
      const uint32_t desired_ctas = static_cast<uint32_t>(
          (max_gather_len + dequant_gather::kBlockSize * tokens_per_thread - 1) /
          (dequant_gather::kBlockSize * tokens_per_thread));
      const uint32_t num_ctas_per_req =
          std::max(1u, std::min(desired_ctas, dequant_gather::kMaxCTAsPerReq));

      const DequantGatherParams params = {
          .k_cache = static_cast<const uint8_t*>(k_cache.data_ptr()),
          .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
          .block_table = static_cast<const int32_t*>(block_table.data_ptr()),
          .out = static_cast<__nv_bfloat16*>(out.data_ptr()),
          .gather_lens = gather_lens_ptr,
          .out_stride0 = OS0.unwrap(),
          .out_stride1 = OS1.unwrap(),
          .block_stride = BS.unwrap(),
          .offset = static_cast<int32_t>(offset),
          .max_blocks_per_seq = static_cast<int32_t>(P.unwrap()),
          .cache_block_size = static_cast<int32_t>(block_size),
          .num_ctas_per_req = num_ctas_per_req,
          .use_fp8_native = static_cast<int32_t>(use_fp8_native),
      };

      const uint32_t grid_size = batch_size * num_ctas_per_req;
      LaunchKernel(grid_size, dequant_gather::kBlockSize, device_.unwrap(), 0)(kernel_fn, params);
    };

    if (max_gather_len <= static_cast<int64_t>(dequant_gather::kSeqLenThreshold)) {
      // 短序列: TPT=1, 最大化 SM 利用率
      launch(dequant_gather::kTokensPerThreadShort,
             dequantize_gather_k_kernel<dequant_gather::kTokensPerThreadShort>);
    } else {
      // 长序列: TPT=4, CTA 数已充足
      launch(dequant_gather::kTokensPerThreadLong,
             dequantize_gather_k_kernel<dequant_gather::kTokensPerThreadLong>);
    }
  }
};

}  // namespace
