#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cfloat>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <math_constants.h>

#ifndef SGL_TOPK
#define SGL_TOPK 512
#endif

namespace sglang {
namespace {

// ============================================================
// SM80 BF16 prefill streaming top-k kernel.
//
// Long rows use direct vectorized loads, half2 bin conversion, a 13-bit
// histogram, and warp-level work stealing for scatter. Short rows retain
// the register-resident 12-bit selection path.
// ============================================================

namespace prefill_bf16 {

constexpr uint32_t kTopK = SGL_TOPK;
constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kNumWarps = kBlockSize / 32;

constexpr uint32_t kHistBits = 12;
constexpr uint32_t kHistBins = 1u << kHistBits;          // 4096
constexpr uint32_t kHistItems = kHistBins / kBlockSize;  // 4
constexpr uint32_t kStreamHistBits = 13;
constexpr uint32_t kStreamHistBins = 1u << kStreamHistBits;
constexpr uint32_t kStreamHistItems = kStreamHistBins / kBlockSize;

constexpr uint32_t kElemPerStage = 8;                                      // bf16 elems per thread per stage
constexpr uint32_t kSizePerStage = kElemPerStage * kBlockSize;             // 8192 bf16 / stage
constexpr uint32_t kCpAsyncBytes = kElemPerStage * sizeof(__nv_bfloat16);  // 16B

// The threshold bin must retain at least kTopK candidates. Otherwise a
// tie-heavy row can leave part of s_topk_indices unwritten. This was latent
// at topk=512 and became memory-unsafe at topk=2048 when the output loop
// started consuming all 2048 slots.
constexpr uint32_t kMaxTies = kTopK > 1024u ? kTopK : 1024u;
constexpr uint32_t kMax1PassLength = 16384;                // single-pass register cap
constexpr uint32_t kMax2PassLength = 2 * kMax1PassLength;  // 32768
constexpr uint32_t kVecsPerThread = 4;
// Max slack the aligned-pointer shift can add (kElemPerStage - 1 bf16
// elements). Host-side dispatch reserves this so that per-row
// `length_padded = length + misalign` never exceeds kMax2PassLength.
constexpr uint32_t kMaxMisalign = kElemPerStage - 1;  // 7

}  // namespace prefill_bf16

// ============================================================
// Public params (mirrors TopKPrefillParams but with bf16 scores).
// ============================================================
struct TopKPrefillBF16Params {
  const __nv_bfloat16* __restrict__ scores;
  const int32_t* __restrict__ row_starts;
  const int32_t* __restrict__ row_ends;
  const int32_t* __restrict__ page_table;
  // Optional [num_rows] map giving the physical page_table row for each query
  // row; nullptr means page_table is already laid out per query row (row_id).
  const int32_t* __restrict__ row_to_page;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;  // optional, may be nullptr
  const int64_t score_stride;
  const int64_t page_table_stride;
  uint32_t page_bits;
};

// ============================================================
// Lightweight POD helpers (kept local to this TU to avoid
// colliding with topk_bf16.cuh when both share the same JIT cache).
// ============================================================
struct alignas(16) MatchBinV3 {
  uint32_t bin;
  uint32_t above_count;
  uint32_t equal_count;
};

struct alignas(8) TieV3 {
  uint32_t idx;
  float score;
};

// ============================================================
// Streaming shared memory layout (about 32.5 KB at topk=2048).
//   counter_gt  : 128B (alignas)
//   counter_eq  : 128B
//   match       : 128B
//   warp_sum    : 128B (32 warps * 4B = 128B)
//   union { histogram[8192]  -> 32KB
//           tie_buffer[2048] -> 16KB }
// s_topk_indices (kTopK * int32) = 8KB lives at the tail of dynamic smem
// (allocated separately by the caller).
// ============================================================
struct alignas(128) PrefillBF16Smem {
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  alignas(128) MatchBinV3 match;
  alignas(128) uint32_t warp_sum[prefill_bf16::kNumWarps];
  union {
    uint32_t histogram[prefill_bf16::kStreamHistBins];
    TieV3 tie_buffer[prefill_bf16::kMaxTies];
  };
};

// ============================================================
// Register-fast-path shared memory layout (~80 KB).
//
// Mirrors `RegisterTopKSmem` from topk_bf16.cuh but parameterized
// against the `prefill_bf16` namespace so we can share dynamic
// smem with the streaming layout via overlay.
//   counter_gt / counter_eq : 256B
//   match + warp_sum        : 32B + 128B
//   union { histogram[4096] -> 16KB
//           histogram_vec   -> same storage as histogram
//           tie_buffer[1024]-> 8KB }
//   bf16_stage[16384]       -> 32KB
// ============================================================
using Vec4PF = device::AlignedVector<float, 4>;

struct RegisterTopKSmemPF {
  using HistVec = device::AlignedVector<uint32_t, prefill_bf16::kHistBins / prefill_bf16::kBlockSize>;
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  MatchBinV3 match;
  uint32_t warp_sum[prefill_bf16::kNumWarps];
  union {
    uint32_t histogram[prefill_bf16::kHistBins];
    HistVec histogram_vec[prefill_bf16::kBlockSize];
    TieV3 tie_buffer[prefill_bf16::kMaxTies];
  };
  alignas(128) __nv_bfloat16 bf16_stage[prefill_bf16::kMax1PassLength];
};

// ============================================================
// Helpers
// ============================================================

SGL_DEVICE uint32_t canonical_score_bits_pf(float x) {
  const auto bits = __float_as_uint(x);
  const auto magnitude = bits & 0x7FFFFFFFu;
  if (magnitude == 0) return 0;                     // +0 and -0 are numerical ties.
  if (magnitude > 0x7F800000u) return 0x7FC00000u;  // NaNs sort above +inf.
  return bits;
}

SGL_DEVICE uint32_t exact_score_key_pf(float x) {
  uint32_t bits = canonical_score_bits_pf(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

SGL_DEVICE uint64_t make_topk_key_pf(float score, uint32_t index) {
  return (static_cast<uint64_t>(exact_score_key_pf(score)) << 32) | ~index;
}

template <uint32_t kBits>
SGL_DEVICE uint32_t extract_coarse_bin_pf(float x) {
  // Order-preserving fp32->fp16->uint16 mapping, then top-kBits as bin idx.
  __half h = __float2half_rn(__uint_as_float(canonical_score_bits_pf(x)));
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return key >> (16 - kBits);
}

template <uint32_t kBits>
SGL_DEVICE uint32_t extract_coarse_bins_bf16x2_pf(__nv_bfloat162 input, float2& values) {
  values = __bfloat1622float2(input);
  values.x = __uint_as_float(canonical_score_bits_pf(values.x));
  values.y = __uint_as_float(canonical_score_bits_pf(values.y));
  const __half2 halves = __float22half2_rn(values);
  const uint32_t bits = reinterpret_cast<const uint32_t&>(halves);
  // Map both packed FP16 signs into monotonic unsigned keys together.  For a
  // positive lane xor the sign bit; for a negative lane xor every bit.  This
  // is bit-identical to the two scalar select chains but exposes packed SWAR
  // operations to the ppu0015 vector scheduler.
  const uint32_t sign01 = (bits >> 15) & 0x00010001u;
  const uint32_t negative_masks = sign01 * 0xffffu;
  const uint32_t flips = 0x80008000u | (negative_masks & 0x7fff7fffu);
  const uint32_t keys = bits ^ flips;
  constexpr uint32_t kShift = 16 - kBits;
  constexpr uint32_t kLaneMask = (1u << kBits) - 1u;
  return (keys >> kShift) & (kLaneMask | (kLaneMask << 16));
}

// Reuse the dynamic scratch after either coarse selector reports overflow.
// Every candidate participates in all exact score/index radix passes; neither
// FP16 conversion nor a capped candidate buffer can drop a distinct BF16 value.
struct ExactTopKSmemPF {
  uint32_t histogram[2][288];
  uint32_t threshold;
  uint32_t counter;
  uint64_t prefix;
};
static_assert(sizeof(ExactTopKSmemPF) <= sizeof(RegisterTopKSmemPF));
static_assert(sizeof(ExactTopKSmemPF) <= sizeof(PrefillBF16Smem));

SGL_DEVICE static void exact_topk_bf16_pf(
    const __nv_bfloat16* __restrict__ scores,
    int32_t* __restrict__ indices,
    const uint32_t length,
    const uint32_t misalign,
    void* scratch) {
  using namespace prefill_bf16;
  constexpr uint32_t RADIX = 256;
  auto* smem = static_cast<ExactTopKSmemPF*>(scratch);
  const auto tx = threadIdx.x;
  auto& histogram = smem->histogram[0];
  uint32_t remaining = kTopK;
  if (tx == 0) smem->prefix = 0;
  __syncthreads();

#pragma unroll 8
  for (uint32_t round = 0; round < 8; ++round) {
    if (tx < RADIX + 1) histogram[tx] = 0;
    __syncthreads();
    const auto prefix = smem->prefix;
    const auto offset = 56 - round * 8;
    for (uint32_t idx = misalign + tx; idx < length; idx += kBlockSize) {
      const auto key = make_topk_key_pf(static_cast<float>(scores[idx]), idx);
      if (round == 0 || (key >> (64 - round * 8)) == prefix) atomicAdd(&histogram[(key >> offset) & 255], 1u);
    }
    __syncthreads();
#pragma unroll 8
    for (uint32_t pass = 0; pass < 8; ++pass) {
      if (tx < RADIX) {
        const auto shift = 1u << pass;
        auto value = smem->histogram[pass & 1][tx];
        if (tx + shift < RADIX) value += smem->histogram[pass & 1][tx + shift];
        smem->histogram[(pass & 1) ^ 1][tx] = value;
      }
      __syncthreads();
    }
    if (tx < RADIX && histogram[tx] > remaining && histogram[tx + 1] <= remaining) smem->threshold = tx;
    __syncthreads();
    const auto bin = smem->threshold;
    remaining -= histogram[bin + 1];
    if (tx == 0) smem->prefix = (prefix << 8) | bin;
    __syncthreads();
    if (remaining == 0) {
      const auto selected = smem->prefix;
      const auto bits = (round + 1) * 8;
      if (tx == 0) smem->counter = 0;
      __syncthreads();
      for (uint32_t idx = misalign + tx; idx < length; idx += kBlockSize) {
        if ((make_topk_key_pf(static_cast<float>(scores[idx]), idx) >> (64 - bits)) > selected)
          indices[atomicAdd(&smem->counter, 1u)] = idx;
      }
      __syncthreads();
      return;
    }
  }
  const auto threshold = smem->prefix;
  if (tx == 0) smem->counter = 0;
  __syncthreads();
  for (uint32_t idx = misalign + tx; idx < length; idx += kBlockSize) {
    if (make_topk_key_pf(static_cast<float>(scores[idx]), idx) >= threshold)
      indices[atomicAdd(&smem->counter, 1u)] = idx;
  }
  __syncthreads();
}

SGL_DEVICE int32_t page_to_indices_pf(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

SGL_DEVICE uint32_t cdiv_pf(uint32_t a, uint32_t b) {
  return (a + b - 1u) / b;
}

// The scores tensor may itself be a contiguous storage-offset view. Align
// the actual row address rather than assuming scores.data_ptr() is 16B-aligned.
SGL_DEVICE uint32_t bf16_misalign_elems_pf(const __nv_bfloat16* p) {
  return static_cast<uint32_t>((reinterpret_cast<uintptr_t>(p) >> 1) & (prefill_bf16::kElemPerStage - 1u));
}

SGL_DEVICE uint32_t warp_inclusive_sum_pf(uint32_t lane_id, uint32_t val) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

SGL_DEVICE uint32_t warp_reduce_sum_stream_pf(uint32_t val) {
#if !defined(USE_ROCM) && defined(__HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
  // ppu0015 lowers this intrinsic to one v.red.add instead of the generic
  // shuffle tree used by device::warp::reduce_sum.
  return __reduce_add_sync(0xffffffffu, val);
#else
  return device::warp::reduce_sum(val);
#endif
}

// A direct vectorized load is faster for the histogram pass on ppu0015:
// every score is consumed once, so staging through shared memory only adds
// pipeline waits and CTA barriers without creating data reuse.
SGL_DEVICE static void histogram_pass_bf16_direct(
    const __nv_bfloat16* __restrict__ scores_bf16,
    const uint32_t length,
    const uint32_t misalign,
    PrefillBF16Smem* smem) {
  using namespace prefill_bf16;
  const uint32_t tx = threadIdx.x;
  constexpr uint32_t kBlockTileElems = kBlockSize * kElemPerStage;

  const auto accumulate_vector = [&](uint4 data, uint32_t global_base) {
    const auto* values = reinterpret_cast<const __nv_bfloat16*>(&data);

    // Materialize all eight independent bin addresses before issuing the
    // asynchronous no-return TSM atomics. Keeping one packed-bin variable
    // inside the unrolled loop forces a tsmcnt wait before the compiler can
    // safely recycle that address register for the next pair.
    float2 ignored_values;
    const uint32_t packed_01 = extract_coarse_bins_bf16x2_pf<kStreamHistBits>(
        *reinterpret_cast<const __nv_bfloat162*>(&values[0]), ignored_values);
    const uint32_t packed_23 = extract_coarse_bins_bf16x2_pf<kStreamHistBits>(
        *reinterpret_cast<const __nv_bfloat162*>(&values[2]), ignored_values);
    const uint32_t packed_45 = extract_coarse_bins_bf16x2_pf<kStreamHistBits>(
        *reinterpret_cast<const __nv_bfloat162*>(&values[4]), ignored_values);
    const uint32_t packed_67 = extract_coarse_bins_bf16x2_pf<kStreamHistBits>(
        *reinterpret_cast<const __nv_bfloat162*>(&values[6]), ignored_values);
    const uint32_t bin_0 = packed_01 & 0xffffu;
    const uint32_t bin_1 = packed_01 >> 16;
    const uint32_t bin_2 = packed_23 & 0xffffu;
    const uint32_t bin_3 = packed_23 >> 16;
    const uint32_t bin_4 = packed_45 & 0xffffu;
    const uint32_t bin_5 = packed_45 >> 16;
    const uint32_t bin_6 = packed_67 & 0xffffu;
    const uint32_t bin_7 = packed_67 >> 16;

    if (global_base >= misalign) {
      atomicAdd(&smem->histogram[bin_0], 1u);
      atomicAdd(&smem->histogram[bin_1], 1u);
      atomicAdd(&smem->histogram[bin_2], 1u);
      atomicAdd(&smem->histogram[bin_3], 1u);
      atomicAdd(&smem->histogram[bin_4], 1u);
      atomicAdd(&smem->histogram[bin_5], 1u);
      atomicAdd(&smem->histogram[bin_6], 1u);
      atomicAdd(&smem->histogram[bin_7], 1u);
    } else {
      // Only thread zero's first aligned vector can contain the dead prefix.
      const uint32_t bins[kElemPerStage] = {bin_0, bin_1, bin_2, bin_3, bin_4, bin_5, bin_6, bin_7};
#pragma unroll
      for (uint32_t e = 0; e < kElemPerStage; ++e) {
        if (global_base + e >= misalign) {
          atomicAdd(&smem->histogram[bins[e]], 1u);
        }
      }
    }
  };

  // Pipeline block-uniform full rounds: issue the next 16-byte vector load
  // before consuming the current vector so its latency can overlap bin
  // conversion and the eight no-return histogram atomics.
  const uint32_t full_rounds = length / kBlockTileElems;
  if (full_rounds != 0u) {
    uint32_t global_base = tx * kElemPerStage;
    uint4 current = *reinterpret_cast<const uint4*>(scores_bf16 + global_base);
    for (uint32_t round = 1u; round < full_rounds; ++round) {
      const uint32_t next_base = round * kBlockTileElems + tx * kElemPerStage;
      const uint4 next = *reinterpret_cast<const uint4*>(scores_bf16 + next_base);
      accumulate_vector(current, global_base);
      current = next;
      global_base = next_base;
    }
    accumulate_vector(current, global_base);
  }

  const uint32_t tail_base = full_rounds * kBlockTileElems + tx * kElemPerStage;
  if (tail_base + kElemPerStage <= length) {
    const uint4 data = *reinterpret_cast<const uint4*>(scores_bf16 + tail_base);
    accumulate_vector(data, tail_base);
  } else if (tail_base < length) {
    for (uint32_t e = 0; e < kElemPerStage; ++e) {
      const uint32_t global_idx = tail_base + e;
      if (global_idx >= length) break;
      if (global_idx >= misalign) {
        const float value = __bfloat162float(scores_bf16[global_idx]);
        const uint32_t bin = extract_coarse_bin_pf<kStreamHistBits>(value);
        atomicAdd(&smem->histogram[bin], 1u);
      }
    }
  }
  __syncthreads();
}

// ============================================================
// scatter_pass_bf16_warp_steal
//
// Work-stealing scatter pass for ultra-long sequences. Each warp
// independently claims tiles from a shared atomic counter and
// loads data directly from HBM via vectorized loads (uint4).
// No block-level __syncthreads() in the main loop -- warps that
// miss topk hits immediately steal the next unscanned tile.
// ============================================================
SGL_DEVICE static void scatter_pass_bf16_warp_steal(
    const __nv_bfloat16* __restrict__ scores_bf16,
    const uint32_t length,
    const uint32_t misalign,
    const uint32_t thr_bin,
    int32_t* __restrict__ s_topk_indices,
    PrefillBF16Smem* smem) {
  // The pointer is aligned down; positions before misalign are a dead prefix.
  using namespace prefill_bf16;
  const auto tx = threadIdx.x;
  const auto lane_id = tx & 31u;
  const auto warp_id = tx >> 5u;

  // find_threshold_bf16 guarantees match.above_count < kTopK, and this pass
  // uses the identical `bin > thr_bin` predicate over the identical range.
  // Therefore every counter_gt return value is a valid top-k slot.

  // Warp-tile granularity: 32 lanes * 8 elements = 256 BF16 values.
  constexpr uint32_t kWarpTileElems = 32u * kElemPerStage;  // 256
  const uint32_t num_warp_tiles = cdiv_pf(length, kWarpTileElems);

  // Reuse warp_sum[0] as shared work-steal counter.
  // Pre-assign first kNumWarps tiles statically to avoid initial stampede.
  if (tx == 0) {
    smem->warp_sum[0] = kNumWarps;
  }
  __syncthreads();  // single init barrier

  // Each warp starts with its statically-assigned tile.
  uint32_t my_tile = warp_id;

  while (my_tile < num_warp_tiles) {
    const uint32_t base = my_tile * kWarpTileElems;
    const uint32_t global_base = base + lane_id * kElemPerStage;

    if (global_base + kElemPerStage <= length) {
      // Fast path: vectorized 16-byte load (8 bf16 = uint4).
      const uint4 data = *reinterpret_cast<const uint4*>(scores_bf16 + global_base);
      const __nv_bfloat16* vals = reinterpret_cast<const __nv_bfloat16*>(&data);

      if (global_base < misalign) {
        // At most the first aligned vector overlaps the dead prefix. Keep its
        // checks out of the common path taken by every other vector.
#pragma unroll
        for (uint32_t e = 0; e < kElemPerStage; ++e) {
          const uint32_t idx = global_base + e;
          if (idx >= misalign) {
            const float fval = __bfloat162float(vals[e]);
            const uint32_t bin = extract_coarse_bin_pf<kStreamHistBits>(fval);
            if (bin > thr_bin) {
              const auto pos = atomicAdd(&smem->counter_gt, 1u);
              s_topk_indices[pos] = static_cast<int32_t>(idx);
            } else if (bin == thr_bin) {
              const auto pos = atomicAdd(&smem->counter_eq, 1u);
              if (pos < kMaxTies) {
                smem->tie_buffer[pos] = TieV3{.idx = idx, .score = fval};
              }
            }
          }
        }
      } else {
        float2 ignored_values;
        const uint32_t bins_01 = extract_coarse_bins_bf16x2_pf<kStreamHistBits>(
            *reinterpret_cast<const __nv_bfloat162*>(&vals[0]), ignored_values);
        const uint32_t bins_23 = extract_coarse_bins_bf16x2_pf<kStreamHistBits>(
            *reinterpret_cast<const __nv_bfloat162*>(&vals[2]), ignored_values);
        const uint32_t bins_45 = extract_coarse_bins_bf16x2_pf<kStreamHistBits>(
            *reinterpret_cast<const __nv_bfloat162*>(&vals[4]), ignored_values);
        const uint32_t bins_67 = extract_coarse_bins_bf16x2_pf<kStreamHistBits>(
            *reinterpret_cast<const __nv_bfloat162*>(&vals[6]), ignored_values);

        const uint32_t max_01 = (bins_01 & 0xffffu) > (bins_01 >> 16) ? bins_01 & 0xffffu : bins_01 >> 16;
        const uint32_t max_23 = (bins_23 & 0xffffu) > (bins_23 >> 16) ? bins_23 & 0xffffu : bins_23 >> 16;
        const uint32_t max_45 = (bins_45 & 0xffffu) > (bins_45 >> 16) ? bins_45 & 0xffffu : bins_45 >> 16;
        const uint32_t max_67 = (bins_67 & 0xffffu) > (bins_67 >> 16) ? bins_67 & 0xffffu : bins_67 >> 16;
        const uint32_t max_0123 = max_01 > max_23 ? max_01 : max_23;
        const uint32_t max_4567 = max_45 > max_67 ? max_45 : max_67;
        const uint32_t vector_max = max_0123 > max_4567 ? max_0123 : max_4567;

#if !defined(USE_ROCM) && defined(__HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
        // Hoist once immediately before the gate so both the gate and its
        // unrolled GT/EQ comparisons reuse one cross-domain threshold move.
        const uint32_t scatter_thr_bin = __ppu_read_firstlane(thr_bin);
#else
        const uint32_t scatter_thr_bin = thr_bin;
#endif
        if (vector_max >= scatter_thr_bin) {
          const auto scatter_pair = [&](uint32_t bins, uint32_t pair_offset) {
#pragma unroll
            for (uint32_t p = 0; p < 2; ++p) {
              const uint32_t idx = global_base + pair_offset + p;
              const uint32_t bin = (bins >> (p * 16)) & 0xffffu;
              if (bin > scatter_thr_bin) {
                const auto pos = atomicAdd(&smem->counter_gt, 1u);
                s_topk_indices[pos] = static_cast<int32_t>(idx);
              } else if (bin == scatter_thr_bin) {
                const auto pos = atomicAdd(&smem->counter_eq, 1u);
                if (pos < kMaxTies) {
                  smem->tie_buffer[pos] = TieV3{
                      .idx = idx,
                      .score = __bfloat162float(vals[pair_offset + p]),
                  };
                }
              }
            }
          };
          if (max_01 >= scatter_thr_bin) scatter_pair(bins_01, 0u);
          if (max_23 >= scatter_thr_bin) scatter_pair(bins_23, 2u);
          if (max_45 >= scatter_thr_bin) scatter_pair(bins_45, 4u);
          if (max_67 >= scatter_thr_bin) scatter_pair(bins_67, 6u);
        }
      }
    } else if (global_base < length) {
      // Tail path: element-by-element for partial tile.
      for (uint32_t e = 0; e < kElemPerStage; ++e) {
        const uint32_t global_idx = global_base + e;
        if (global_idx >= length) break;
        if (global_idx >= misalign) {
          const float fval = __bfloat162float(scores_bf16[global_idx]);
          const uint32_t bin = extract_coarse_bin_pf<kStreamHistBits>(fval);

          if (bin > thr_bin) {
            const auto pos = atomicAdd(&smem->counter_gt, 1u);
            s_topk_indices[pos] = static_cast<int32_t>(global_idx);
          } else if (bin == thr_bin) {
            const auto pos = atomicAdd(&smem->counter_eq, 1u);
            if (pos < kMaxTies) {
              smem->tie_buffer[pos] = TieV3{.idx = global_idx, .score = fval};
            }
          }
        }
      }
    }
    // else: entire lane range OOB, skip.

    // Warp leader atomically claims the next tile; broadcast to all lanes.
    if (lane_id == 0) {
      my_tile = atomicAdd(&smem->warp_sum[0], 1u);
    }
    my_tile = __shfl_sync(0xFFFFFFFF, my_tile, 0);
  }

  // Warps leave the work-stealing loop independently. The following tie
  // handling consumes CTA-wide counters and buffers, so all publishers must
  // finish before any warp starts ranking or translating candidates.
  __syncthreads();
}

// ============================================================
// find_threshold_bf16
//
// 8192 bins / 1024 threads = 8 bins/thread. Builds a warp-then-block
// inclusive prefix scan of the histogram and locates the bin whose
// inclusive prefix straddles the K-th element.
// ============================================================
SGL_DEVICE static void find_threshold_bf16(uint32_t length, PrefillBF16Smem* smem) {
  using namespace prefill_bf16;
  const auto tx = threadIdx.x;
  const auto lane_id = tx % 32;
  const auto warp_id = tx / 32;

  uint32_t orig[kStreamHistItems];
  uint32_t local_sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < kStreamHistItems; ++i) {
    orig[i] = smem->histogram[tx * kStreamHistItems + i];
    local_sum += orig[i];
  }

  const auto warp_inc = warp_inclusive_sum_pf(lane_id, local_sum);
  const auto warp_exc = warp_inc - local_sum;
  if (lane_id == 31) smem->warp_sum[warp_id] = warp_inc;
  __syncthreads();

  // Inter-warp inclusive prefix; kNumWarps == 32 fits in one lane vector.
  const uint32_t tmp = smem->warp_sum[lane_id];
  uint32_t prefix_sum = warp_reduce_sum_stream_pf(lane_id < warp_id ? tmp : 0u);
  prefix_sum += warp_exc;

#pragma unroll
  for (uint32_t i = 0; i < kStreamHistItems; ++i) {
    prefix_sum += orig[i];
    const auto above = length - prefix_sum;  // strict-above count
    if (above < kTopK && above + orig[i] >= kTopK) {
      smem->match = MatchBinV3{
          .bin = tx * kStreamHistItems + i,
          .above_count = above,
          .equal_count = orig[i],
      };
    }
  }
  __syncthreads();
}

// ============================================================
// tie_handle_and_transform_bf16
//
// Resolves the equality bin and translates the final raw indices
// through the page table. When num_above + num_equal == kTopK we
// just append all ties; otherwise rank ties by (score desc, idx asc).
// ============================================================
SGL_DEVICE static void tie_handle_and_transform_bf16(
    const TopKPrefillBF16Params& params,
    int32_t* __restrict__ s_topk_indices,
    int32_t* __restrict__ page_indices_ptr,
    int32_t* __restrict__ raw_indices_ptr,
    const int32_t* __restrict__ page_ptr,
    const uint32_t misalign,
    const uint32_t length_real,
    PrefillBF16Smem* smem) {
  using namespace prefill_bf16;
  const auto tx = threadIdx.x;
  const auto lane_id = tx % 32;
  const auto warp_id = tx / 32;

  const auto num_above = smem->match.above_count;
  const auto num_equal = smem->counter_eq;
  const uint32_t num_ties = (num_equal < kMaxTies) ? num_equal : kMaxTies;
  const uint32_t topk_remain = (kTopK > num_above) ? (kTopK - num_above) : 0u;
  const bool need_tiebreak = (num_above + num_equal > kTopK);

  if (topk_remain == 0u) {
    // Already filled by the strictly-greater bucket; nothing to do.
  } else if (!need_tiebreak) {
    // Fast path: every tie element fits.
    for (uint32_t i = tx; i < num_ties; i += kBlockSize) {
      const auto pos = num_above + i;
      if (pos < kTopK) {
        s_topk_indices[pos] = static_cast<int32_t>(smem->tie_buffer[i].idx);
      }
    }
  } else {
    // Tie-break: rank each tie within the bin and keep top `topk_remain`.
    const auto is_greater = [](const TieV3& a, const TieV3& b) {
      return make_topk_key_pf(a.score, a.idx) > make_topk_key_pf(b.score, b.idx);
    };

    if (num_ties <= 32u) {
      // Single-warp ballot fast path.
      if (lane_id < num_ties && warp_id < num_ties) {
        const uint32_t mask = (num_ties == 32u) ? 0xFFFFFFFFu : ((1u << num_ties) - 1u);
        const auto tie = smem->tie_buffer[lane_id];
        const auto target = smem->tie_buffer[warp_id];
        const bool pred = is_greater(tie, target);
        const auto rank = static_cast<uint32_t>(__popc(__ballot_sync(mask, pred)));
        if (lane_id == 0 && rank < topk_remain) {
          s_topk_indices[num_above + rank] = static_cast<int32_t>(target.idx);
        }
      }
    } else if (num_ties <= 64u) {
      // Two-batch ballot: each warp ranks one target via two halves.
      const auto tie_0 = smem->tie_buffer[lane_id];
      uint32_t tie_1_idx = 0xFFFFFFFFu;
      float tie_1_score = -CUDART_INF_F;
      if (lane_id + 32u < num_ties) {
        tie_1_idx = smem->tie_buffer[lane_id + 32u].idx;
        tie_1_score = smem->tie_buffer[lane_id + 32u].score;
      }

      auto rank_target = [&](const TieV3& target) {
        const bool p0 = is_greater(tie_0, target);
        const bool p1 = make_topk_key_pf(tie_1_score, tie_1_idx) > make_topk_key_pf(target.score, target.idx);
        const auto r0 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFFu, p0)));
        const auto r1 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFFu, p1)));
        return r0 + r1;
      };

      if (warp_id < num_ties) {
        const auto target = smem->tie_buffer[warp_id];
        const auto rank = rank_target(target);
        if (lane_id == 0 && rank < topk_remain) {
          s_topk_indices[num_above + rank] = static_cast<int32_t>(target.idx);
        }
      }
      if (warp_id + 32u < num_ties) {
        const auto target = smem->tie_buffer[warp_id + 32u];
        const auto rank = rank_target(target);
        if (lane_id == 0 && rank < topk_remain) {
          s_topk_indices[num_above + rank] = static_cast<int32_t>(target.idx);
        }
      }
    } else {
      // Generic path: each warp independently ranks one target at a time.
      for (uint32_t i = warp_id; i < num_ties; i += kNumWarps) {
        const auto target = smem->tie_buffer[i];
        uint32_t local_rank = 0;
        for (uint32_t j = lane_id; j < num_ties; j += 32) {
          const auto t = smem->tie_buffer[j];
          if (is_greater(t, target)) ++local_rank;
        }
        const auto rank = warp_reduce_sum_stream_pf(local_rank);
        if (lane_id == 0 && rank < topk_remain) {
          s_topk_indices[num_above + rank] = static_cast<int32_t>(target.idx);
        }
      }
    }
  }
  __syncthreads();

  // Page-translate the final raw indices and emit. Sentinels from
  // initialization stay as -1. Indices from the topk pipeline live in
  // the aligned-pointer (shifted) space; subtract misalign here to get
  // caller-visible row-local indices in [0, length_real).
  for (uint32_t i = tx; i < kTopK; i += kBlockSize) {
    const auto raw = s_topk_indices[i];
    const bool raw_in_range =
        raw >= static_cast<int32_t>(misalign) && static_cast<uint32_t>(raw) < length_real + misalign;
    if (raw_in_range) {
      const auto row_local = static_cast<uint32_t>(raw) - misalign;
      page_indices_ptr[i] = page_to_indices_pf(page_ptr, row_local, params.page_bits);
      if (raw_indices_ptr != nullptr) raw_indices_ptr[i] = static_cast<int32_t>(row_local);
    } else {
      page_indices_ptr[i] = -1;
      if (raw_indices_ptr != nullptr) raw_indices_ptr[i] = -1;
    }
  }
}

// ============================================================
// naive_transform_bf16
//
// Length <= kTopK: every position is selected; just translate.
// ============================================================
SGL_DEVICE static void naive_transform_bf16(
    const int32_t* __restrict__ page_ptr,
    int32_t* __restrict__ page_indices_ptr,
    int32_t* __restrict__ raw_indices_ptr,
    const uint32_t length,
    const uint32_t page_bits) {
  using namespace prefill_bf16;
  for (uint32_t i = threadIdx.x; i < kTopK; i += kBlockSize) {
    if (i < length) {
      page_indices_ptr[i] = page_to_indices_pf(page_ptr, i, page_bits);
      if (raw_indices_ptr != nullptr) raw_indices_ptr[i] = static_cast<int32_t>(i);
    } else {
      page_indices_ptr[i] = -1;
      if (raw_indices_ptr != nullptr) raw_indices_ptr[i] = -1;
    }
  }
}

// ============================================================
// register_topk_bf16_pf
//
// Adapted from topk_bf16.cuh's `register_topk_bf16<kIs2Pass>`.
// Loads the entire row (<= 32K bf16 scores) into per-thread
// registers (and a 16K-float smem staging buffer for the second
// pass), then runs a coarse 12-bit histogram + tie-breaking pipeline.
// Avoids the streaming path entirely for short rows.
//
// `_smem` is reinterpret_cast<RegisterTopKSmemPF*>; the caller is
// responsible for sizing the dynamic smem accordingly.
// `indices` (== s_topk_indices) MUST be initialized to -1 by the
// caller before this is invoked, since num_above + num_equal == kTopK
// is only guaranteed when length > kTopK.
// ============================================================
template <bool kIs2Pass>
[[maybe_unused]]
SGL_DEVICE bool register_topk_bf16_pf(
    const __nv_bfloat16* __restrict__ scores,
    int32_t* __restrict__ indices,
    const uint32_t length,
    const uint32_t misalign,
    void* _smem) {
  // The input pointer was aligned down, so ignore its dead prefix:
  // `scores` is pre-shifted to a 16B-aligned address, `length` includes
  // the misalign padding, positions [0, misalign) are dead prefix, and
  // s_topk_indices values are stored in this shifted space.
  auto* smem = static_cast<RegisterTopKSmemPF*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % 32;
  const auto warp_id = tx / 32;

  // Initialize shared memory histogram
  {
    typename RegisterTopKSmemPF::HistVec hist_vec;
    hist_vec.fill(0);
    smem->histogram_vec[tx] = hist_vec;
    if (tx == 0) {
      smem->counter_gt = smem->counter_eq = 0;
    }
    __syncthreads();
  }

  // Load scores into registers (bf16 -> float cast per element)
  Vec4PF local[prefill_bf16::kVecsPerThread];
#pragma unroll
  for (uint32_t v = 0; v < prefill_bf16::kVecsPerThread; ++v) {
    const uint32_t base = (tx + v * prefill_bf16::kBlockSize) * 4;
    if (base >= length) break;
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const uint32_t idx = base + e;
      local[v][e] = (idx < length) ? static_cast<float>(scores[idx]) : 0.0f;
    }
  }

  // Async copy bf16 data from global to shared memory staging buffer, then cast to float
  if constexpr (kIs2Pass) {
    const uint32_t extra_length = length - prefill_bf16::kMax1PassLength;
    const __nv_bfloat16* src = scores + prefill_bf16::kMax1PassLength;

    // Issue async copies: each thread copies kElemPerStage (8) bf16 elements per round.
    // extra_length is at most kMax1PassLength (16384) and kSizePerStage = 8192, so at most 2 rounds.
#pragma unroll 2
    for (uint32_t r = 0; r < 2; ++r) {
      const uint32_t global_base = r * prefill_bf16::kSizePerStage + tx * prefill_bf16::kElemPerStage;
      if (global_base + prefill_bf16::kElemPerStage <= extra_length) {
        __pipeline_memcpy_async(&smem->bf16_stage[global_base], &src[global_base], prefill_bf16::kCpAsyncBytes);
      } else if (global_base < extra_length) {
#pragma unroll
        for (uint32_t e = 0; e < prefill_bf16::kElemPerStage; ++e) {
          if (global_base + e < extra_length) {
            smem->bf16_stage[global_base + e] = src[global_base + e];
          }
        }
      }
    }
    __pipeline_commit();
  }

  // Accumulate histogram via shared-memory atomics.
  // Dead-prefix positions [0, misalign) from the aligned-pointer shift
  // must be skipped so histogram sum equals the real (unpadded) length.
  // Skip via an `if` wrapper because `continue` inside an unrolled loop is
  // unreliable with this toolchain.
#pragma unroll
  for (uint32_t v = 0; v < prefill_bf16::kVecsPerThread; ++v) {
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const uint32_t idx = (tx + v * prefill_bf16::kBlockSize) * 4 + e;
      if constexpr (!kIs2Pass) {
        if (idx >= length) goto LABEL_ACC_FINISH;
      }
      if (idx >= misalign) {
        atomicAdd(&smem->histogram[extract_coarse_bin_pf<prefill_bf16::kHistBits>(local[v][e])], 1);
      }
    }
  }
  if constexpr (kIs2Pass) {
    const uint32_t extra_length = length - prefill_bf16::kMax1PassLength;
    __pipeline_wait_prior(0);
    __syncthreads();

    // Read 32-bit (2 bf16) from staging buffer and accumulate histogram directly
    const __nv_bfloat162* bf16_pairs = reinterpret_cast<const __nv_bfloat162*>(smem->bf16_stage);
    const uint32_t pair_len = extra_length / 2;
    for (uint32_t i = tx; i < pair_len; i += prefill_bf16::kBlockSize) {
      const __nv_bfloat162 pair = bf16_pairs[i];
      const float val0 = __bfloat162float(__low2bfloat16(pair));
      const float val1 = __bfloat162float(__high2bfloat16(pair));
      atomicAdd(&smem->histogram[extract_coarse_bin_pf<prefill_bf16::kHistBits>(val0)], 1);
      atomicAdd(&smem->histogram[extract_coarse_bin_pf<prefill_bf16::kHistBits>(val1)], 1);
    }
    // Handle trailing odd element
    if ((extra_length & 1) && tx == 0) {
      const float val = __bfloat162float(smem->bf16_stage[extra_length - 1]);
      atomicAdd(&smem->histogram[extract_coarse_bin_pf<prefill_bf16::kHistBits>(val)], 1);
    }
  }
[[maybe_unused]] LABEL_ACC_FINISH:
  __syncthreads();

  // Phase 2: Exclusive prefix scan -> find threshold bin
  {
    constexpr uint32_t kItems = prefill_bf16::kHistBins / prefill_bf16::kBlockSize;
    uint32_t orig[kItems];
    const auto hist_vec = smem->histogram_vec[tx];
    uint32_t tmp_local_sum = 0;

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      orig[i] = hist_vec[i];
      tmp_local_sum += orig[i];
    }

    const auto warp_inc = warp_inclusive_sum_pf(lane_id, tmp_local_sum);
    const auto warp_exc = warp_inc - tmp_local_sum;
    if (lane_id == 31) {
      smem->warp_sum[warp_id] = warp_inc;
    }

    __syncthreads();

    const auto tmp = smem->warp_sum[lane_id];
    uint32_t prefix_sum = device::warp::reduce_sum(lane_id < warp_id ? tmp : 0u);
    prefix_sum += warp_exc;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      prefix_sum += orig[i];
      // `length` is the padded iteration bound; the real element count
      // (matching the histogram sum) is `length - misalign`.
      const auto above = (length - misalign) - prefix_sum;
      if (above < prefill_bf16::kTopK && above + orig[i] >= prefill_bf16::kTopK) {
        smem->match = {
            .bin = tx * kItems + i,
            .above_count = above,
            .equal_count = orig[i],
        };
      }
    }
    __syncthreads();
  }

  const auto [thr_bin, num_above, num_equal] = smem->match;
  if (num_equal > prefill_bf16::kMaxTies) return false;

  // Phase 3: Scatter
  constexpr uint32_t kMaxTolerance = 0;
  const bool need_tiebreak = (num_equal + num_above > prefill_bf16::kTopK + kMaxTolerance);
  const auto topk_indices = indices;
  const auto tie_buffer = smem->tie_buffer;

#pragma unroll
  for (uint32_t v = 0; v < prefill_bf16::kVecsPerThread; ++v) {
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const uint32_t idx = (tx + v * prefill_bf16::kBlockSize) * 4 + e;
      if constexpr (!kIs2Pass) {
        if (idx >= length) goto LABEL_SCATTER_DONE;
      }
      // Skip dead-prefix positions (chunk 0 only; chunk 1 below has
      // idx >= kMax1PassLength > misalign). The `if` wrapper avoids
      // `continue` inside this unrolled loop on this toolchain.
      if (idx >= misalign) {
        const uint32_t bin = extract_coarse_bin_pf<prefill_bf16::kHistBits>(local[v][e]);
        if (bin > thr_bin) {
          topk_indices[atomicAdd(&smem->counter_gt, 1)] = idx;
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1);
          if (need_tiebreak) {
            if (pos < prefill_bf16::kMaxTies) {
              tie_buffer[pos] = {.idx = idx, .score = local[v][e]};
            }
          } else {
            if (const auto which = pos + num_above; which < prefill_bf16::kTopK) {
              topk_indices[which] = idx;
            }
          }
        }
      }
    }
    // prefetch the next scores from bf16_stage into registers (for 2-pass)
    if constexpr (kIs2Pass) {
      const uint32_t pf_base = (tx + v * prefill_bf16::kBlockSize) * 4;
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        local[v][e] = __bfloat162float(smem->bf16_stage[pf_base + e]);
      }
    }
  }

  // 16K ~ 32K: process second chunk from registers (now loaded from bf16_stage)
  if constexpr (kIs2Pass) {
#pragma unroll
    for (uint32_t v = 0; v < prefill_bf16::kVecsPerThread; ++v) {
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const uint32_t idx = (tx + v * prefill_bf16::kBlockSize) * 4 + e + prefill_bf16::kMax1PassLength;
        if (idx >= length) goto LABEL_SCATTER_DONE;
        const uint32_t bin = extract_coarse_bin_pf<prefill_bf16::kHistBits>(local[v][e]);
        if (bin > thr_bin) {
          topk_indices[atomicAdd(&smem->counter_gt, 1)] = idx;
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1);
          if (need_tiebreak) {
            if (pos < prefill_bf16::kMaxTies) {
              tie_buffer[pos] = {.idx = idx, .score = local[v][e]};
            }
          } else {
            if (const auto which = pos + num_above; which < prefill_bf16::kTopK) {
              topk_indices[which] = idx;
            }
          }
        }
      }
    }
  }

[[maybe_unused]] LABEL_SCATTER_DONE:
  if (!need_tiebreak) return true;

  // Phase 4: Tie-breaking within the threshold bin
  __syncthreads();

  const uint32_t num_ties = min(num_equal, prefill_bf16::kMaxTies);
  const uint32_t topk_remain = prefill_bf16::kTopK - num_above;

  const auto is_greater = [](const TieV3& a, const TieV3& b) {
    return make_topk_key_pf(a.score, a.idx) > make_topk_key_pf(b.score, b.idx);
  };

  if (num_ties <= 32u) {
    if (lane_id >= num_ties || warp_id >= num_ties) return true;
    const uint32_t mask = (1ull << num_ties) - 1u;
    const auto tie = tie_buffer[lane_id];
    const auto target_tie = tie_buffer[warp_id];
    const bool pred = is_greater(tie, target_tie);
    const auto rank = static_cast<uint32_t>(__popc(__ballot_sync(mask, pred)));
    if (lane_id == 0 && rank < topk_remain) {
      topk_indices[num_above + rank] = target_tie.idx;
    }
  } else if (num_ties <= 64u) {
    const auto lane_id_1 = lane_id + 32;
    const auto warp_id_1 = warp_id + 32;
    const auto invalid = TieV3{.idx = 0xFFFFFFFF, .score = -CUDART_INF_F};
    const auto tie_0 = tie_buffer[lane_id];
    const auto tie_1 = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;
    if (true) {
      const auto target = tie_buffer[warp_id];
      const bool pred_0 = is_greater(tie_0, target);
      const bool pred_1 = is_greater(tie_1, target);
      const auto rank_0 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_0)));
      const auto rank_1 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_1)));
      const auto rank = rank_0 + rank_1;
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target.idx;
      }
    }
    if (warp_id_1 < num_ties) {
      const auto target = tie_buffer[warp_id_1];
      const bool pred_0 = is_greater(tie_0, target);
      const bool pred_1 = is_greater(tie_1, target);
      const auto rank_0 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_0)));
      const auto rank_1 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_1)));
      const auto rank = rank_0 + rank_1;
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target.idx;
      }
    }
  } else {
    [[unlikely]];
    for (auto i = warp_id; i < num_ties; i += prefill_bf16::kNumWarps) {
      const auto target_tie = tie_buffer[i];
      uint32_t local_rank = 0;
      for (auto j = lane_id; j < num_ties; j += 32) {
        const auto tie = tie_buffer[j];
        if (is_greater(tie, target_tie)) local_rank++;
      }
      const auto rank = device::warp::reduce_sum(local_rank);
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target_tie.idx;
      }
    }
  }
  return true;
}

// ============================================================
// register_transform_bf16
//
// Page-translates the indices written by `register_topk_bf16_pf`
// (raw absolute offsets within the row) and emits to the global
// outputs. Mirrors the tail of `tie_handle_and_transform_bf16`.
// ============================================================
SGL_DEVICE static void register_transform_bf16(
    const TopKPrefillBF16Params& params,
    const int32_t* __restrict__ s_topk_indices,
    int32_t* __restrict__ page_indices_ptr,
    int32_t* __restrict__ raw_indices_ptr,
    const int32_t* __restrict__ page_ptr,
    const uint32_t misalign,
    const uint32_t length_real) {
  using namespace prefill_bf16;
  for (uint32_t i = threadIdx.x; i < kTopK; i += kBlockSize) {
    const auto raw = s_topk_indices[i];
    const bool raw_in_range =
        raw >= static_cast<int32_t>(misalign) && static_cast<uint32_t>(raw) < length_real + misalign;
    if (raw_in_range) {
      // Recover the caller-visible row-local index by removing the
      // aligned-pointer shift baked into s_topk_indices.
      const auto row_local = static_cast<uint32_t>(raw) - misalign;
      page_indices_ptr[i] = page_to_indices_pf(page_ptr, row_local, params.page_bits);
      if (raw_indices_ptr != nullptr) raw_indices_ptr[i] = static_cast<int32_t>(row_local);
    } else {
      page_indices_ptr[i] = -1;
      if (raw_indices_ptr != nullptr) raw_indices_ptr[i] = -1;
    }
  }
}

// ============================================================
// streaming_topk_bf16
//
// Drives the full streaming flow over an arbitrary length:
//   A. histogram pass  (direct vectorized global loads)
//   B. find threshold
//   C. scatter pass    (warp-level dynamic work stealing)
//
// The caller is responsible for calling tie_handle_and_transform_bf16
// afterwards.
// ============================================================
SGL_DEVICE static bool streaming_topk_bf16(
    const __nv_bfloat16* __restrict__ scores_bf16,
    const uint32_t length_padded,
    const uint32_t misalign,
    const uint32_t length_real,
    int32_t* __restrict__ s_topk_indices,
    PrefillBF16Smem* smem) {
  using namespace prefill_bf16;
  const auto tx = threadIdx.x;

  // Init shared state: histogram, counters, sentinel topk slots.
  {
    // Two warp-contiguous 16-byte stores cover all 8192 bins.  The previous
    // thread-major [tx][8] initialization made adjacent lanes stride by 32 B,
    // which maps poorly onto ppu0015's four TSM groups.
    auto* histogram_vec = reinterpret_cast<uint4*>(smem->histogram);
    constexpr uint32_t kHistogramVecs = kStreamHistBins / 4;
    constexpr uint4 kZero4 = {0u, 0u, 0u, 0u};
    histogram_vec[tx] = kZero4;
    histogram_vec[tx + kBlockSize] = kZero4;
    static_assert(kHistogramVecs == 2 * kBlockSize);
    if (tx == 0) {
      smem->counter_gt = 0u;
      smem->counter_eq = 0u;
    }
    for (uint32_t i = tx; i < kTopK; i += kBlockSize) {
      s_topk_indices[i] = -1;
    }
    __syncthreads();
  }

  // Phase A: build the 13-bit histogram with vectorized direct loads.
  // Iterates over the padded length so that dead-prefix positions are
  // covered by aligned loads, but skipped from atomicAdd — so
  // the histogram sum equals length_real.
  histogram_pass_bf16_direct(scores_bf16, length_padded, misalign, smem);

  // Phase B: locate threshold bin. Uses the real (non-padded) length so
  // that `prefix_sum` / `above = length - prefix_sum` is consistent with
  // the histogram sum computed above.
  find_threshold_bf16(length_real, smem);
  if (smem->match.equal_count > kMaxTies) return false;

  // Reset counters before scatter; tie_buffer aliases histogram so
  // we must not touch histogram between this point and the scatter.
  if (tx == 0) {
    smem->counter_gt = 0u;
    smem->counter_eq = 0u;
  }
  __syncthreads();

  // Phase C: scatter pass.
  const auto thr_bin = smem->match.bin;
  scatter_pass_bf16_warp_steal(scores_bf16, length_padded, misalign, thr_bin, s_topk_indices, smem);
  return true;
}

// ============================================================
// Register kernel (short/medium sequences <= 32K)
//
// Handles naive passthrough (<=kTopK), 1-pass register (<=16K),
// and 2-pass register (<=32K) paths. Only allocates
// RegisterTopKSmemPF + s_topk_indices smem.
// ============================================================
__global__ __launch_bounds__(prefill_bf16::kBlockSize, 1) void topk_prefill_bf16_register_kernel(
    const TopKPrefillBF16Params params) {
  using namespace prefill_bf16;

  extern __shared__ char smem_raw[];
  auto* s_topk_indices = reinterpret_cast<int32_t*>(smem_raw + sizeof(RegisterTopKSmemPF));

  const auto row_id = blockIdx.x;
  const auto row_start = params.row_starts[row_id];
  const auto row_end = params.row_ends[row_id];
  const auto length = row_end > row_start ? static_cast<uint32_t>(row_end - row_start) : 0u;

  // Align the row base pointer down to a 16-byte boundary (8 bf16
  // elements). All vectorized loads in this kernel require the source
  // to be 16B-aligned; `row_start` from NSA is arbitrary in production
  // so we shift back by `misalign` and iterate over `length + misalign`
  // positions. Use the actual pointer to support tensor storage-offset views.
  const int64_t row_offset = static_cast<int64_t>(row_id) * params.score_stride + row_start;
  const auto scores_raw = params.scores + row_offset;
  const uint32_t misalign = bf16_misalign_elems_pf(scores_raw);
  const uint32_t length_padded = length + misalign;

  const auto scores_ptr = scores_raw - misalign;
  const auto page_row = params.row_to_page == nullptr ? row_id : static_cast<uint32_t>(params.row_to_page[row_id]);
  const auto page_ptr = params.page_table + static_cast<int64_t>(page_row) * params.page_table_stride;
  auto page_indices_ptr = params.page_indices + static_cast<int64_t>(row_id) * kTopK;
  auto raw_indices_ptr =
      params.raw_indices == nullptr ? nullptr : params.raw_indices + static_cast<int64_t>(row_id) * kTopK;

  // register_topk_bf16_pf may under-fill on an unexpected input or a future
  // algorithm regression. Initialize every slot so the translation stage can
  // contain that failure instead of interpreting shared-memory garbage as a
  // page-table index. The loop form is required when kTopK > kBlockSize.
  if (length > kTopK) {
    for (uint32_t i = threadIdx.x; i < kTopK; i += kBlockSize) {
      s_topk_indices[i] = -1;
    }
    __syncthreads();
  }

  if (length <= kTopK) {
    // Trivial case: just transform indices. No bf16 loads → misalign
    // does not apply here.
    naive_transform_bf16(page_ptr, page_indices_ptr, raw_indices_ptr, length, params.page_bits);
  } else if (length_padded <= kMax1PassLength) {
    // Dispatch on length_padded, not length: with misalign > 0 a row
    // whose real length just fits kMax1PassLength would overflow the
    // 1-pass register capacity. Host-side dispatch guarantees
    // length_padded <= kMax2PassLength so 2-pass always suffices.
    const bool exact = register_topk_bf16_pf<false>(scores_ptr, s_topk_indices, length_padded, misalign, smem_raw);
    __syncthreads();
    if (!exact) exact_topk_bf16_pf(scores_ptr, s_topk_indices, length_padded, misalign, smem_raw);
    __syncthreads();
    register_transform_bf16(params, s_topk_indices, page_indices_ptr, raw_indices_ptr, page_ptr, misalign, length);
  } else if (length_padded <= kMax2PassLength) {
    [[likely]];
    const bool exact = register_topk_bf16_pf<true>(scores_ptr, s_topk_indices, length_padded, misalign, smem_raw);
    __syncthreads();
    if (!exact) exact_topk_bf16_pf(scores_ptr, s_topk_indices, length_padded, misalign, smem_raw);
    __syncthreads();
    register_transform_bf16(params, s_topk_indices, page_indices_ptr, raw_indices_ptr, page_ptr, misalign, length);
  }
}

// ============================================================
// Streaming kernel (long sequences > 32K)
//
// Handles the direct-load streaming top-k path.
// Allocates PrefillBF16Smem + s_topk_indices smem.
// For batch rows whose actual length happens to be short,
// falls back to naive passthrough or streaming (since smem is
// already sized for streaming).
// ============================================================
__global__ __launch_bounds__(prefill_bf16::kBlockSize, 1) void topk_prefill_bf16_stream_kernel(
    const TopKPrefillBF16Params params) {
  using namespace prefill_bf16;

  extern __shared__ char smem_raw[];
  auto* smem = reinterpret_cast<PrefillBF16Smem*>(smem_raw);
  auto* s_topk_indices = reinterpret_cast<int32_t*>(smem_raw + sizeof(PrefillBF16Smem));

  const auto row_id = blockIdx.x;
  const auto row_start = params.row_starts[row_id];
  const auto row_end = params.row_ends[row_id];
  const auto length = row_end > row_start ? static_cast<uint32_t>(row_end - row_start) : 0u;

  // Align-down the per-row base pointer for the uint4 fast path.
  // See the register-kernel launcher for the aligned-pointer contract.
  const int64_t row_offset = static_cast<int64_t>(row_id) * params.score_stride + row_start;
  const auto scores_raw = params.scores + row_offset;
  const uint32_t misalign = bf16_misalign_elems_pf(scores_raw);
  const uint32_t length_padded = length + misalign;

  const auto scores_ptr = scores_raw - misalign;
  const auto page_row = params.row_to_page == nullptr ? row_id : static_cast<uint32_t>(params.row_to_page[row_id]);
  const auto page_ptr = params.page_table + static_cast<int64_t>(page_row) * params.page_table_stride;
  auto page_indices_ptr = params.page_indices + static_cast<int64_t>(row_id) * kTopK;
  auto raw_indices_ptr =
      params.raw_indices == nullptr ? nullptr : params.raw_indices + static_cast<int64_t>(row_id) * kTopK;

  if (length <= kTopK) {
    naive_transform_bf16(page_ptr, page_indices_ptr, raw_indices_ptr, length, params.page_bits);
    return;
  }

  // All rows with length > kTopK use the streaming path in this kernel,
  // since smem is already sized for PrefillBF16Smem.
  const bool exact =
      streaming_topk_bf16(scores_ptr, length_padded, misalign, /*length_real=*/length, s_topk_indices, smem);
  __syncthreads();
  if (exact) {
    tie_handle_and_transform_bf16(
        params, s_topk_indices, page_indices_ptr, raw_indices_ptr, page_ptr, misalign, length, smem);
  } else {
    exact_topk_bf16_pf(scores_ptr, s_topk_indices, length_padded, misalign, smem_raw);
    register_transform_bf16(params, s_topk_indices, page_indices_ptr, raw_indices_ptr, page_ptr, misalign, length);
  }
}

// ============================================================
// Host-side launcher
// ============================================================
template <auto* f, size_t kMaxDynamicSMEM>
void setup_kernel_smem_once_pf(host::DebugInfo where = {}) {
  [[maybe_unused]]
  static const auto result = [] {
    const auto fptr = std::bit_cast<const void*>(f);
    return ::cudaFuncSetAttribute(fptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
  }();
  host::RuntimeDeviceCheck(result, where);
}

struct TopKPrefillBF16Kernel {
  static void transform(
      const tvm::ffi::TensorView scores,        // [batch, max_len] bf16
      const tvm::ffi::TensorView row_starts,    // [batch] int32
      const tvm::ffi::TensorView row_ends,      // [batch] int32
      const tvm::ffi::TensorView page_table,    // [batch, num_pages] int32
      const tvm::ffi::TensorView page_indices,  // [batch, kTopK] int32
      const uint32_t page_size,
      const tvm::ffi::Optional<tvm::ffi::TensorView> raw_indices,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_to_page) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, -1}).with_strides({S, 1}).with_dtype<bf16_t>().with_device(device_).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(row_starts);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(row_ends);
    // page_table rows are addressed by row_id unless an indirection map is
    // supplied, which decouples the physical row count from batch B and lets
    // callers skip materializing a per-query [B, P] page table.
    int32_t* row_to_page_ptr = nullptr;
    if (row_to_page.has_value()) {
      TensorMatcher({-1, -1}).with_strides({P, 1}).with_dtype<int32_t>().with_device(device_).verify(page_table);
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(row_to_page.value());
      row_to_page_ptr = static_cast<int32_t*>(row_to_page.value().data_ptr());
    } else {
      TensorMatcher({B, -1}).with_strides({P, 1}).with_dtype<int32_t>().with_device(device_).verify(page_table);
    }
    TensorMatcher({B, prefill_bf16::kTopK}).with_dtype<int32_t>().with_device(device_).verify(page_indices);

    int32_t* raw_indices_ptr = nullptr;
    if (raw_indices.has_value()) {
      TensorMatcher({B, prefill_bf16::kTopK}).with_dtype<int32_t>().with_device(device_).verify(raw_indices.value());
      raw_indices_ptr = static_cast<int32_t*>(raw_indices.value().data_ptr());
    }

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto params = TopKPrefillBF16Params{
        .scores = static_cast<const __nv_bfloat16*>(scores.data_ptr()),
        .row_starts = static_cast<int32_t*>(row_starts.data_ptr()),
        .row_ends = static_cast<int32_t*>(row_ends.data_ptr()),
        .page_table = static_cast<int32_t*>(page_table.data_ptr()),
        .row_to_page = row_to_page_ptr,
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .raw_indices = raw_indices_ptr,
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .page_bits = page_bits,
    };

    // Host-side smem routing based on max possible per-row length.
    // The score tensor's second dim (score_stride S) is an upper bound on
    // (row_ends[i] - row_starts[i]), so we can safely pick the kernel with
    // the minimal smem footprint. We reserve kMaxMisalign so that the
    // aligned-pointer shift inside the register kernel never causes
    // `length_padded = length + misalign` to overflow kMax2PassLength.
    const auto max_seq_len = static_cast<uint32_t>(S.unwrap());

    if (max_seq_len + prefill_bf16::kMaxMisalign <= prefill_bf16::kMax2PassLength) {
      // Short/medium sequences: register kernel only needs RegisterTopKSmemPF.
      constexpr auto kernel = topk_prefill_bf16_register_kernel;
      constexpr auto kSMEM_ = sizeof(RegisterTopKSmemPF) + prefill_bf16::kTopK * sizeof(int32_t);
      setup_kernel_smem_once_pf<kernel, kSMEM_>();
      LaunchKernel(batch_size, prefill_bf16::kBlockSize, device_.unwrap(), kSMEM_)(kernel, params);
    } else {
      // Long sequences: streaming kernel needs PrefillBF16Smem.
      constexpr auto kernel = topk_prefill_bf16_stream_kernel;
      constexpr auto kSMEM_ = sizeof(PrefillBF16Smem) + prefill_bf16::kTopK * sizeof(int32_t);
      setup_kernel_smem_once_pf<kernel, kSMEM_>();
      LaunchKernel(batch_size, prefill_bf16::kBlockSize, device_.unwrap(), kSMEM_)(kernel, params);
    }
  }
};

}  // namespace
}  // namespace sglang
