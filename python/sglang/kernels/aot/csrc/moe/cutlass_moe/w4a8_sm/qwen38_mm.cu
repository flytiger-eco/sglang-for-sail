#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <torch/all.h>

#include <type_traits>

#include "cutlass/cutlass.h"
#include "qwen38_common.h"
#include "w4a8_sm_grouped_mm_c3x.cuh"

using namespace cute;

namespace {

enum class Sched { PP, CO };

template <int M, int N, int K, int A, int B, int C, Sched S>
struct SM90W4A8SmConfig {
  using KernelSchedule = std::conditional_t<
      S == Sched::PP,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;

  using EpilogueSchedule = std::conditional_t<
      S == Sched::PP,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>;

  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>;
  using Cutlass3xW4A8SmGemm = cutlass_3x_w4a8_sm_group_gemm<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <int M, int N, int K, int A, int B, int C>
using SM90_PP = SM90W4A8SmConfig<M, N, K, A, B, C, Sched::PP>;

template <int M, int N, int K, int A, int B, int C>
using SM90_CO = SM90W4A8SmConfig<M, N, K, A, B, C, Sched::CO>;

template <typename Config>
inline void invoke_gemm(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t chunk_size) {
  using GemmT = typename Config::Cutlass3xW4A8SmGemm;
  cutlass_w4a8_sm_group_gemm_caller<GemmT>(
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
      chunk_size);
}

#define INVOKE_GEMM_WITH_CONFIG_HELPER(...) \
  invoke_gemm<__VA_ARGS__>(                 \
      d_tensors,                            \
      a_tensors,                            \
      b_tensors,                            \
      a_scales,                             \
      b_scales,                             \
      expert_offsets,                       \
      problem_sizes,                        \
      a_strides,                            \
      b_strides,                            \
      d_strides,                            \
      s_strides,                            \
      chunk_size)
#define INVOKE_GEMM_WITH_CONFIG(Config) INVOKE_GEMM_WITH_CONFIG_HELPER Config

}  // namespace

void qwen38_w4a8_sm_mm(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t config_id) {
  c10::cuda::CUDAGuard guard(a_tensors.device());
  TORCH_CHECK(config_id >= 0 && config_id < 5, "invalid Qwen38 W4A8 config id");
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
      false,
      1);
  if (a_tensors.numel() == 0) return;
  int64_t chunk_size = 128;
  switch (config_id) {
    case 0:
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 32, 128, 1, 1, 1>));
      break;
    case 1:
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<256, 80, 128, 1, 1, 1>));
      break;
    case 2:
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 128, 128, 1, 1, 1>));
      break;
    case 3:
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 160, 128, 1, 1, 1>));
      break;
    case 4:
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 128, 1, 1, 1>));
      break;
  }
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "qwen38_w4a8_sm_mm(Tensor(a!) d, Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, Tensor expert_offsets, "
      "Tensor problem_sizes, Tensor a_strides, Tensor b_strides, Tensor d_strides, Tensor s_strides, int config_id) -> "
      "()");
  m.impl("qwen38_w4a8_sm_mm", torch::kCUDA, &qwen38_w4a8_sm_mm);
}
