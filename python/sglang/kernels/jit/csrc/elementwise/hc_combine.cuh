#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

struct HcCombineParams {
  const void* block_output;     // [M, H]
  const void* residual;         // [M, HC * H]
  const void* normed_residual;  // [M, HC * H]
  const void* inject_weight;    // [HC, HC * H]
  void* output;                 // [M, HC * H]
};

/**
 * \brief Fused HyperConnection (gated residual) combine:
 *
 *   a[m, c] = 2 * sigmoid(dot(normed_residual[m, :], inject_weight[c, :]) / kHcCount)
 *   output[m, c*H + i] = residual[m, c*H + i] + a[m, c] * block_output[m, i]
 *
 * One CTA handles one token row. Phase 1 computes the kHcCount gate values with a
 * block reduction over the full HC*H row (fp32 accumulation). Phase 2 streams the
 * HC*H output elements with vectorized 16B accesses; the block_output row is only
 * H elements, so its re-read per branch stays in L2.
 *
 * \tparam kHcCount    Number of hyper-connection branches (4 in production).
 * \tparam kHiddenSize Per-branch hidden size H. HC*H must be a multiple of
 *                     kNumThreads * kVecLen so the row maps exactly onto the CTA.
 * \tparam kUsePDL     Whether to emit the PDL wait/trigger pair.
 * \tparam Float       Element type: bf16_t | fp16_t.
 */
template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(256) void hc_combine_kernel(const HcCombineParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;  // 8 elements, 16 bytes
  constexpr uint32_t kVecLen = 8;
  constexpr uint32_t kNumThreads = 256;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;            // 1280 for 4x2560
  constexpr uint32_t kVecsPerThread = kVecsPerRow / kNumThreads;  // 5 for 4x2560
  constexpr uint32_t kVecsPerBranch = kHiddenSize / kVecLen;      // 320 for 2560
  constexpr uint32_t kNumWarps = kNumThreads / kWarpThreads;

  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  const uint32_t m = blockIdx.x;

  const auto y_ptr = pointer::offset<Float>(params.block_output, static_cast<int64_t>(m) * kHiddenSize);
  const auto r_ptr = pointer::offset<Float>(params.residual, static_cast<int64_t>(m) * kRowSize);
  const auto n_ptr = pointer::offset<Float>(params.normed_residual, static_cast<int64_t>(m) * kRowSize);
  const auto w_ptr = static_cast<const Float*>(params.inject_weight);
  const auto out_ptr = pointer::offset<Float>(params.output, static_cast<int64_t>(m) * kRowSize);

  PDLWaitPrimary<kUsePDL>();

  // Phase 1: gate values a_c = 2 * sigmoid(dot(N[m, :], W[c, :]) / HC), fp32.
  // Each thread accumulates its kVecsPerThread strided vectors against all
  // kHcCount weight rows, then the partials are reduced across the CTA.
  Storage n_vec[kVecsPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    n_vec[j] = gmem.load(n_ptr, j);
  }

  float acc[kHcCount];
#pragma unroll
  for (int c = 0; c < kHcCount; ++c) {
    const auto wc_ptr = w_ptr + static_cast<int64_t>(c) * kRowSize;
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      const Storage w_vec = gmem.load(wc_ptr, j);
#pragma unroll
      for (uint32_t i = 0; i < kVecLen / 2; ++i) {
        const auto [nx, ny] = cast<fp32x2_t>(n_vec[j][i]);
        const auto [wx, wy] = cast<fp32x2_t>(w_vec[i]);
        sum += nx * wx + ny * wy;
      }
    }
    acc[c] = warp::reduce_sum(sum);
  }

  __shared__ float smem[kHcCount][kNumWarps];
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < kHcCount; ++c) {
      smem[c][warp_id] = acc[c];
    }
  }
  __syncthreads();
  __shared__ float a_shared[kHcCount];
  if (threadIdx.x < kHcCount) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kNumWarps; ++w) {
      total += smem[threadIdx.x][w];
    }
    a_shared[threadIdx.x] = 2.0f / (1.0f + math::exp(-total / kHcCount));
  }
  __syncthreads();

  // Phase 2: stream the output row. Vector `vec_idx` lies entirely inside
  // branch `vec_idx / kVecsPerBranch` (H is a multiple of kVecLen).
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t vec_idx = threadIdx.x + j * kNumThreads;
    const uint32_t branch = vec_idx / kVecsPerBranch;
    const uint32_t col_in_branch = (vec_idx % kVecsPerBranch) * kVecLen;
    const float a = a_shared[branch];

    const Storage r_vec = gmem.load(r_ptr, j);
    Storage y_vec;
    y_vec.load(y_ptr, col_in_branch / kVecLen);
    Storage out_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen / 2; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r_vec[i]);
      const auto [yx, yy] = cast<fp32x2_t>(y_vec[i]);
      out_vec[i] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});
    }
    gmem.store(out_ptr, out_vec, j);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename DType>
struct HcCombineKernel {
  static_assert(sizeof(DType) == 2, "HcCombine only supports 2-byte dtypes");
  static_assert(kHcCount > 0, "kHcCount must be positive");
  static_assert(kHiddenSize > 0 && kHiddenSize % 8 == 0, "kHiddenSize must be a multiple of 8");
  static_assert((kHcCount * kHiddenSize) % (256 * 8) == 0, "kHcCount * kHiddenSize must be a multiple of 2048");
  static constexpr auto kernel = hc_combine_kernel<kHcCount, kHiddenSize, kUsePDL, DType>;
  static constexpr uint32_t kBlockSize = 256;

  /**
   * \brief Validate tensors and launch one CTA per token row.
   * \param block_output    [M, H] contiguous
   * \param residual        [M, HC * H] contiguous
   * \param normed_residual [M, HC * H] contiguous, same dtype/device as residual
   * \param inject_weight   [HC, HC * H] contiguous, same dtype/device as residual
   * \param output          [M, HC * H] contiguous, same shape/dtype/device as residual
   */
  static void
  run(const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView normed_residual,
      const tvm::ffi::TensorView inject_weight,
      const tvm::ffi::TensorView output) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHiddenSize})  // block_output
        .with_dtype<DType>()
        .with_device(device)
        .verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize})  // residual, normed_residual, output
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual)
        .verify(normed_residual)
        .verify(output);
    TensorMatcher({kHcCount, kHcCount * kHiddenSize})  // inject_weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(inject_weight);

    const auto params = HcCombineParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .normed_residual = normed_residual.data_ptr(),
        .inject_weight = inject_weight.data_ptr(),
        .output = output.data_ptr(),
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    LaunchKernel(num_tokens, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

struct HcCombineSplitParams {
  const void* block_output;
  const void* residual;
  const void* normed_residual;
  const void* inject_weight;
  void* output;
  float* partials;
};

namespace hc_combine_split_detail {

// One CTA per warp of the original kernel: CTA s replays exactly the work of
// threads [32s, 32s+32) of the 256-thread reference, so every float is
// accumulated in the reference order and results stay bit-identical.
constexpr uint32_t kSplit = 8;
constexpr uint32_t kRefThreads = 256;
constexpr uint32_t kGateThreads = 32;
constexpr uint32_t kApplyThreads = 160;
constexpr uint32_t kVecLen = 8;

}  // namespace hc_combine_split_detail

/**
 * \brief Stage 1 of the split combine: partial gate dots over a K slice.
 *
 * Grid is (rows, kSplit) so the [HC, HC*H] inject weight is read once across
 * the whole grid instead of once per row, and the row's traffic is spread over
 * kSplit CTAs. Each CTA writes its own partials slot, so no atomics and no
 * buffer clearing are needed.
 */
template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(hc_combine_split_detail::kGateThreads) void hc_combine_gate_kernel(
    const HcCombineSplitParams __grid_constant__ params) {
  using namespace device;
  using namespace hc_combine_split_detail;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;
  constexpr uint32_t kVecLen = 8;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;
  constexpr uint32_t kVecsPerThread = kVecsPerRow / kRefThreads;
  static_assert(kVecsPerRow % kRefThreads == 0);
  static_assert(kRefThreads / kGateThreads == kSplit);

  const uint32_t m = blockIdx.x;
  const uint32_t split = blockIdx.y / kHcCount;
  const uint32_t c = blockIdx.y % kHcCount;
  const uint32_t ref_tid = split * kGateThreads + threadIdx.x;

  const auto n_ptr = pointer::offset<Float>(params.normed_residual, static_cast<int64_t>(m) * kRowSize);
  const auto w_ptr = static_cast<const Float*>(params.inject_weight);

  PDLWaitPrimary<kUsePDL>();

  Storage n_vec[kVecsPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    n_vec[j].load(n_ptr, ref_tid + j * kRefThreads);
  }

  {
    const auto wc_ptr = w_ptr + static_cast<int64_t>(c) * kRowSize;
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      Storage w_vec;
      w_vec.load(wc_ptr, ref_tid + j * kRefThreads);
#pragma unroll
      for (uint32_t i = 0; i < kVecLen / 2; ++i) {
        const auto [nx, ny] = cast<fp32x2_t>(n_vec[j][i]);
        const auto [wx, wy] = cast<fp32x2_t>(w_vec[i]);
        sum += nx * wx + ny * wy;
      }
    }
    sum = warp::reduce_sum(sum);
    if (threadIdx.x == 0) {
      params.partials[(static_cast<int64_t>(m) * kSplit + split) * kHcCount + c] = sum;
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Stage 2: reduce the partial dots and stream the combined row.
 *
 * Each CTA owns one contiguous vector slice of the row. kSplit divides
 * kHiddenSize, so a slice never straddles two branches and the gate is a
 * per-CTA scalar.
 */
template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(hc_combine_split_detail::kApplyThreads) void hc_combine_apply_kernel(
    const HcCombineSplitParams __grid_constant__ params) {
  using namespace device;
  using namespace hc_combine_split_detail;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;
  constexpr uint32_t kVecsPerSplit = kVecsPerRow / kSplit;
  constexpr uint32_t kVecsPerThread = kVecsPerSplit / kApplyThreads;
  constexpr uint32_t kVecsPerBranch = kHiddenSize / kVecLen;
  static_assert(kVecsPerBranch % kVecsPerSplit == 0);

  const uint32_t m = blockIdx.x;
  const uint32_t split = blockIdx.y;
  const uint32_t vec_base = split * kVecsPerSplit;
  const uint32_t branch = vec_base / kVecsPerBranch;

  const auto y_ptr = pointer::offset<Float>(params.block_output, static_cast<int64_t>(m) * kHiddenSize);
  const auto r_ptr = pointer::offset<Float>(params.residual, static_cast<int64_t>(m) * kRowSize);
  const auto out_ptr = pointer::offset<Float>(params.output, static_cast<int64_t>(m) * kRowSize);

  PDLWaitPrimary<kUsePDL>();

  float total = 0.0f;
#pragma unroll
  for (uint32_t s = 0; s < kSplit; ++s) {
    total += params.partials[(static_cast<int64_t>(m) * kSplit + s) * kHcCount + branch];
  }
  const float a = 2.0f / (1.0f + math::exp(-total / kHcCount));

#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t vec_idx = vec_base + threadIdx.x + j * kApplyThreads;
    const uint32_t col_in_branch = (vec_idx % kVecsPerBranch);
    Storage r_vec;
    r_vec.load(r_ptr, vec_idx);
    Storage y_vec;
    y_vec.load(y_ptr, col_in_branch);
    Storage out_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen / 2; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r_vec[i]);
      const auto [yx, yy] = cast<fp32x2_t>(y_vec[i]);
      out_vec[i] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});
    }
    out_vec.store(out_ptr, vec_idx);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename DType>
struct HcCombineSplitKernel {
  static_assert(sizeof(DType) == 2, "HcCombine only supports 2-byte dtypes");
  static constexpr auto gate_kernel = hc_combine_gate_kernel<kHcCount, kHiddenSize, kUsePDL, DType>;
  static constexpr auto apply_kernel = hc_combine_apply_kernel<kHcCount, kHiddenSize, kUsePDL, DType>;

  static void
  run(const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView normed_residual,
      const tvm::ffi::TensorView inject_weight,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView partials) {
    using namespace host;
    using namespace hc_combine_split_detail;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHiddenSize}).with_dtype<DType>().with_device(device).verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize})
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual)
        .verify(normed_residual)
        .verify(output);
    TensorMatcher({kHcCount, kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(inject_weight);
    auto part_rows = SymbolicSize{"partial_rows"};
    TensorMatcher({part_rows, kSplit, kHcCount}).with_dtype<fp32_t>().with_device(device).verify(partials);

    const auto params = HcCombineSplitParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .normed_residual = normed_residual.data_ptr(),
        .inject_weight = inject_weight.data_ptr(),
        .output = output.data_ptr(),
        .partials = static_cast<float*>(partials.data_ptr()),
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    LaunchKernel(dim3(num_tokens, kSplit * kHcCount, 1), kGateThreads, device.unwrap())
        .enable_pdl(kUsePDL)(gate_kernel, params);
    LaunchKernel(dim3(num_tokens, kSplit, 1), kApplyThreads, device.unwrap()).enable_pdl(kUsePDL)(apply_kernel, params);
  }
};

struct HcCombineApplyParams {
  const void* block_output;  // [M, H]
  const void* residual;      // [M, HC * H]
  const void* gates;         // [M, HC] fp32, precomputed a[m,c]
  void* output;              // [M, HC * H]
};

/**
 * \brief Apply-only combine: output = residual + gates * block_output.
 *
 * The gate values a[m, c] = 2 * sigmoid(dot(normed[m], W[c]) / HC) are computed
 * upstream (hoisted into the mix-time down projection) and passed in as a dense
 * [M, HC] fp32 tensor, so this kernel never reads normed_residual or
 * inject_weight. That drops one full HC*H (~640MB at prefill) read versus the
 * fused `hc_combine`, leaving a clean high-occupancy streaming kernel: read
 * residual + block, write output. One CTA per token row.
 */
template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float>
__global__
__launch_bounds__(256) void hc_combine_apply_only_kernel(const HcCombineApplyParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;  // 8 elements, 16 bytes
  constexpr uint32_t kVecLen = 8;
  constexpr uint32_t kNumThreads = 256;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerBranch = kHiddenSize / kVecLen;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;
  constexpr uint32_t kVecsPerThread = kVecsPerRow / kNumThreads;

  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  const uint32_t m = blockIdx.x;
  const auto y_ptr = pointer::offset<Float>(params.block_output, static_cast<int64_t>(m) * kHiddenSize);
  const auto r_ptr = pointer::offset<Float>(params.residual, static_cast<int64_t>(m) * kRowSize);
  const auto out_ptr = pointer::offset<Float>(params.output, static_cast<int64_t>(m) * kRowSize);
  const auto g_ptr = static_cast<const float*>(params.gates) + static_cast<int64_t>(m) * kHcCount;

  PDLWaitPrimary<kUsePDL>();

  // The HC gate scalars are tiny; every thread loads all of them into registers.
  float a_reg[kHcCount];
#pragma unroll
  for (int c = 0; c < kHcCount; ++c) {
    a_reg[c] = g_ptr[c];
  }

#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t vec_idx = threadIdx.x + j * kNumThreads;
    const uint32_t branch = vec_idx / kVecsPerBranch;
    const uint32_t col_in_branch = (vec_idx % kVecsPerBranch) * kVecLen;
    const float a = a_reg[branch];

    const Storage r_vec = gmem.load(r_ptr, j);
    Storage y_vec;
    y_vec.load(y_ptr, col_in_branch / kVecLen);
    Storage out_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen / 2; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r_vec[i]);
      const auto [yx, yy] = cast<fp32x2_t>(y_vec[i]);
      out_vec[i] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});
    }
    gmem.store(out_ptr, out_vec, j);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename DType>
struct HcCombineApplyKernel {
  static_assert(sizeof(DType) == 2, "HcCombine only supports 2-byte dtypes");
  static_assert(kHiddenSize > 0 && kHiddenSize % 8 == 0, "kHiddenSize must be a multiple of 8");
  static_assert((kHcCount * kHiddenSize) % (256 * 8) == 0, "kHcCount * kHiddenSize must be a multiple of 2048");
  static constexpr auto kernel = hc_combine_apply_only_kernel<kHcCount, kHiddenSize, kUsePDL, DType>;
  static constexpr uint32_t kBlockSize = 256;

  static void
  run(const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView gates,
      const tvm::ffi::TensorView output) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHiddenSize}).with_dtype<DType>().with_device(device).verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(residual).verify(output);
    TensorMatcher({M, kHcCount}).with_dtype<fp32_t>().with_device(device).verify(gates);

    const auto params = HcCombineApplyParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .gates = gates.data_ptr(),
        .output = output.data_ptr(),
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    LaunchKernel(num_tokens, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

// ─── Fused combine-apply + grouped-Gemma-RMSNorm ───────────────────────────
//
// Eliminates the 160 MB DRAM round-trip of the intermediate updated residual
// between hc_combine_apply and grouped_gemma_rmsnorm.  One CTA per token row,
// three phases:
//   Phase 1 — combine (same vectorised pattern as apply-only) + accumulate
//             per-branch sum-of-squares in fp32.
//   Phase 2 — CTA-level reduce of 4 variance values, compute rsqrt.
//   Phase 3 — re-read updated from L1/L2 (the CTA just wrote 20 KB to the
//             *same* addresses), load norm weight, write normed output.
//

struct HcCombineNormParams {
  const void* block_output;              // [M, H]
  const void* residual;                  // [M, HC * H]
  const void* gates;                     // [M, HC] fp32
  const void* __restrict__ norm_weight;  // [HC * H]
  void* updated;                         // [M, HC * H]
  void* normed;                          // [M, HC * H]
  float eps;
};

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(256) void hc_combine_norm_kernel(const HcCombineNormParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;  // 8 elements, 16 bytes
  constexpr uint32_t kVecLen = 8;
  constexpr uint32_t kNumThreads = 256;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerBranch = kHiddenSize / kVecLen;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;
  constexpr uint32_t kVecsPerThread = kVecsPerRow / kNumThreads;
  constexpr uint32_t kNumWarps = kNumThreads / kWarpThreads;

  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  const uint32_t m = blockIdx.x;
  const auto y_ptr = pointer::offset<Float>(params.block_output, static_cast<int64_t>(m) * kHiddenSize);
  const auto r_ptr = pointer::offset<Float>(params.residual, static_cast<int64_t>(m) * kRowSize);
  const auto upd_ptr = pointer::offset<Float>(params.updated, static_cast<int64_t>(m) * kRowSize);
  const auto nrm_ptr = pointer::offset<Float>(params.normed, static_cast<int64_t>(m) * kRowSize);
  const auto g_ptr = static_cast<const float*>(params.gates) + static_cast<int64_t>(m) * kHcCount;
  const auto w_ptr = static_cast<const Float*>(params.norm_weight);

  PDLWaitPrimary<kUsePDL>();

  // Load gate scalars into registers.
  float a_reg[kHcCount];
#pragma unroll
  for (int c = 0; c < kHcCount; ++c) {
    a_reg[c] = g_ptr[c];
  }

  // ── Phase 1: combine + per-branch variance accumulation ──────────────
  float var_partial[kHcCount] = {};

#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t vec_idx = threadIdx.x + j * kNumThreads;
    const uint32_t branch = vec_idx / kVecsPerBranch;
    const uint32_t col_in_branch = (vec_idx % kVecsPerBranch) * kVecLen;
    const float a = a_reg[branch];

    const Storage r_vec = gmem.load(r_ptr, j);
    Storage y_vec;
    y_vec.load(y_ptr, col_in_branch / kVecLen);
    Storage upd_vec;
    float local_sq = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen / 2; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r_vec[i]);
      const auto [yx, yy] = cast<fp32x2_t>(y_vec[i]);
      const float ux = rx + a * yx;
      const float uy = ry + a * yy;
      upd_vec[i] = cast<Float2>(fp32x2_t{ux, uy});
      // Match the sequential contract: RMSNorm consumes the rounded 2-byte
      // combine output, not the pre-round fp32 accumulator.
      const auto [urx, ury] = cast<fp32x2_t>(upd_vec[i]);
      local_sq += urx * urx + ury * ury;
    }
    gmem.store(upd_ptr, upd_vec, j);
    var_partial[branch] += local_sq;
  }

  // ── Phase 2: CTA-level variance reduce + rsqrt ───────────────────────
  __shared__ float var_smem[kHcCount][kNumWarps];
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;

#pragma unroll
  for (int c = 0; c < kHcCount; ++c) {
    var_partial[c] = warp::reduce_sum(var_partial[c]);
  }
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < kHcCount; ++c) {
      var_smem[c][warp_id] = var_partial[c];
    }
  }
  __syncthreads();

  // Warp 0 finalises the per-branch rsqrt values.
  __shared__ float rsqrt_shared[kHcCount];
  if (warp_id == 0) {
#pragma unroll
    for (int c = 0; c < kHcCount; ++c) {
      float total = (lane < kNumWarps) ? var_smem[c][lane] : 0.0f;
      total = warp::reduce_sum(total);
      if (lane == 0) {
        rsqrt_shared[c] = math::rsqrt(total / kHiddenSize + params.eps);
      }
    }
  }
  __syncthreads();

  float rsq_reg[kHcCount];
#pragma unroll
  for (int c = 0; c < kHcCount; ++c) {
    rsq_reg[c] = rsqrt_shared[c];
  }

  // ── Phase 3: apply Gemma RMSNorm  (re-read updated from L1/L2) ───────
  const auto gmem_upd = tile::Memory<Storage>::cta(kNumThreads);
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t vec_idx = threadIdx.x + j * kNumThreads;
    const uint32_t branch = vec_idx / kVecsPerBranch;
    const float rsq = rsq_reg[branch];

    const Storage u_vec = gmem_upd.load(upd_ptr, j);
    Storage w_vec;
    w_vec.load(w_ptr, vec_idx);
    Storage nrm_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen / 2; ++i) {
      const auto [ux, uy] = cast<fp32x2_t>(u_vec[i]);
      const auto [wx, wy] = cast<fp32x2_t>(w_vec[i]);
      nrm_vec[i] = cast<Float2>(fp32x2_t{ux * rsq * (1.0f + wx), uy * rsq * (1.0f + wy)});
    }
    gmem.store(nrm_ptr, nrm_vec, j);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename DType>
struct HcCombineNormKernel {
  static_assert(sizeof(DType) == 2, "HcCombineNorm only supports 2-byte dtypes");
  static_assert(kHiddenSize > 0 && kHiddenSize % 8 == 0, "kHiddenSize must be a multiple of 8");
  static_assert((kHcCount * kHiddenSize) % (256 * 8) == 0, "kHcCount * kHiddenSize must be a multiple of 2048");
  static constexpr auto kernel = hc_combine_norm_kernel<kHcCount, kHiddenSize, kUsePDL, DType>;
  static constexpr uint32_t kBlockSize = 256;

  static void
  run(const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView gates,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView updated,
      const tvm::ffi::TensorView normed,
      float eps) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHiddenSize}).with_dtype<DType>().with_device(device).verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize})
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual)
        .verify(updated)
        .verify(normed);
    TensorMatcher({M, kHcCount}).with_dtype<fp32_t>().with_device(device).verify(gates);
    TensorMatcher({kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(norm_weight);

    const auto params = HcCombineNormParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .gates = gates.data_ptr(),
        .norm_weight = norm_weight.data_ptr(),
        .updated = updated.data_ptr(),
        .normed = normed.data_ptr(),
        .eps = eps,
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    LaunchKernel(num_tokens, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
