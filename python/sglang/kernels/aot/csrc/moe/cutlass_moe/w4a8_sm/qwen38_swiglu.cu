#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <torch/all.h>

#include <type_traits>

#include "cutlass/cutlass.h"
#include "qwen38_common.h"
#include "sm_int4b_t.cuh"
#include "w4a8_sm_grouped_mm_c3x_swiglu.cuh"

using namespace cute;

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Tile dispatch for the FUSED gate_up GEMM, sign-magnitude int4 flavour.
//
// Mainloop is the SM fast-convert one (QuantType = cutlass::sm_int4b_t), same as
// w4a8_sm_grouped_mm_c3x.cu; the epilogue is the contracting SwiGLU+quant fork.

//
// Only gate_up shapes are listed; down_proj never reaches this entry point.
// ─────────────────────────────────────────────────────────────────────────────

template <int M, int N, int K, int A, int B, int C>
struct SM90W4A8SmSwiGLUConfig {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>;
  using Cutlass3xW4A8Gemm =
      cutlass_3x_w4a8_group_gemm_swiglu<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule, cutlass::sm_int4b_t>;
};

template <int M, int N, int K, int A, int B, int C>
using SM90_CO = SM90W4A8SmSwiGLUConfig<M, N, K, A, B, C>;

// Small decode tiles use one math warpgroup with a 64-channel tile.
template <int M, int N, int K, int A, int B, int C>
struct SM90W4A8SmSwiGLUPPConfig {
  static_assert(M == 64, "fused ping-pong configurations require 64 output channels");
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>;
  using Cutlass3xW4A8Gemm =
      cutlass_3x_w4a8_group_gemm_swiglu<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule, cutlass::sm_int4b_t>;
};

template <int M, int N, int K, int A, int B, int C>
using SM90_PP = SM90W4A8SmSwiGLUPPConfig<M, N, K, A, B, C>;

template <typename Config>
inline void invoke_gemm(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& out_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t chunk_size) {
  cutlass_w4a8_group_gemm_swiglu_caller<typename Config::Cutlass3xW4A8Gemm>(
      d_tensors,
      a_tensors,
      b_tensors,
      a_scales,
      b_scales,
      out_scales,
      expert_offsets,
      problem_sizes,
      a_strides,
      b_strides,
      d_strides,
      s_strides,
      chunk_size);
}

#define INVOKE_GEMM_WITH_CONFIG_HELPER(...) \
  invoke_gemm<__VA_ARGS__>(                 \
      d_tensors,                            \
      a_tensors,                            \
      b_tensors,                            \
      a_scales,                             \
      b_scales,                             \
      out_scales,                           \
      expert_offsets,                       \
      problem_sizes,                        \
      a_strides,                            \
      b_strides,                            \
      d_strides,                            \
      s_strides,                            \
      chunk_size)
#define INVOKE_GEMM_WITH_CONFIG(Config) INVOKE_GEMM_WITH_CONFIG_HELPER Config

}  // namespace

void qwen38_w4a8_sm_swiglu(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& out_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t config_id) {
  c10::cuda::CUDAGuard guard(a_tensors.device());
  TORCH_CHECK(config_id >= 0 && config_id < 7, "invalid Qwen38 W4A8 config id");
  check_qwen38_w4a8(
      d_tensors,
      a_tensors,
      b_tensors,
      a_scales,
      b_scales,
      expert_offsets,
      problem_sizes,
      a_strides,
      b_strides,
      d_strides,
      s_strides,
      true,
      config_id == 6 ? 4 : 2);
  TORCH_CHECK(
      out_scales.device() == a_tensors.device() && out_scales.scalar_type() == torch::kFloat32 &&
          out_scales.numel() == 1 && out_scales.is_contiguous(),
      "output scale must be one FP32 value on the input device");
  if (a_tensors.numel() == 0) return;
  int64_t chunk_size = 128;
  switch (config_id) {
    case 0:
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 32, 256, 1, 1, 1>));
      break;
    case 1:
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 80, 256, 2, 1, 1>));
      break;
    case 2:
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 112, 256, 1, 1, 1>));
      break;
    case 3:
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 128, 256, 1, 1, 1>));
      break;
    case 4:
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 144, 256, 1, 1, 1>));
      break;
    case 5:
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 160, 256, 1, 1, 1>));
      break;
    case 6:
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 1, 1, 1>));
      break;
  }
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "qwen38_w4a8_sm_swiglu(Tensor(a!) d, Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, Tensor out_scales, "
      "Tensor expert_offsets, Tensor problem_sizes, Tensor a_strides, Tensor b_strides, Tensor d_strides, Tensor "
      "s_strides, int config_id) -> ()");
  m.impl("qwen38_w4a8_sm_swiglu", torch::kCUDA, &qwen38_w4a8_sm_swiglu);
}
