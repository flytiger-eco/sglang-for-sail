#pragma once

/**
 * @file w4a8_sm_grouped_mm_c3x.cuh
 * @brief Sign-magnitude variant of W4A8 grouped GEMM.
 *
 * Identical to w4a8_grouped_mm_c3x.cuh except:
 *   - QuantType = sm_int4b_t (sign-magnitude INT4)
 *   - NumericArrayConverter uses bitwise ALU ops instead of LUT-based PRMT
 *   - Weight scales must be pre-multiplied by 512 to compensate for FP8 subnormal mapping
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "../w4a8/w4a8_get_group_starts.cuh"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass_extensions/w4a8_sm/collective/collective_builder_mixed_input.hpp"
#include "sm_int4b_t.cuh"

using namespace cute;

namespace {

using MmaType = cutlass::float_e4m3_t;
using QuantType = cutlass::sm_int4b_t;     // sign-magnitude INT4
using ElementAccumulator = float;          // FP8 x FP8 ->FP32 accumulate
using ElementScale = cutlass::bfloat16_t;  // weight group scale BF16, scale[e,n,k_group]
using ElementC = cutlass::bfloat16_t;      //(D=alpha AB+ beta C)
using ElementD = ElementC;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  //[expert, 3]

using ArchTag = cutlass::arch::Sm90;                   // builder Hopper TMA、WGMMA warp-specialized kernel
using OperatorClass = cutlass::arch::OpClassTensorOp;  // cuda core

using LayoutA = cutlass::layout::RowMajor;     // activation [M,K]
using LayoutB = cutlass::layout::ColumnMajor;  // weight [N,K] colmajor；[N,K] RowMajor & [K,N] ColumnMajor
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;

using LayoutA_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutA>::type;  // LayoutA = RowMajor，LayoutA_Transpose = ColumnMajor
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using LayoutC_Transpose = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

// 128-bit访存对齐；
static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;    // AlignmentA= 128/8=16
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;  // AlignmentB = 32
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// tileshape：CTA GEMM tile(Tile M * Tile N * Tile K)
// ClusterShape：ClusterM × ClusterN × ClusterK,
// KernelSchedule：PingPong /Cooperative
template <typename TileShape, typename ClusterShape, typename KernelSchedule, typename EpilogueSchedule>
struct cutlass_3x_w4a8_sm_group_gemm {
  static constexpr int GroupSize = 128;
  static constexpr int PackedScalesNum = get<2>(TileShape{}) / GroupSize;    // Tile K/128
  using ElementScalePacked = cutlass::Array<ElementScale, PackedScalesNum>;  // 把一个逻辑 scale
                                                                             // 元素定义成固定长度数组，

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutC_Transpose*,
      AlignmentC,
      ElementD,
      LayoutD_Transpose*,
      AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilderMixedInput<
      ArchTag,
      OperatorClass,
      cute::tuple<QuantType, ElementScalePacked>,
      LayoutB_Transpose*,
      AlignmentB,
      MmaType,
      LayoutA_Transpose*,
      AlignmentA,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernelScaleOnly =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopScaleOnly, CollectiveEpilogue>;

  using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;

  using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
  using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
  using StrideC = typename GemmKernelScaleOnly::InternalStrideC;
  using StrideD = typename GemmKernelScaleOnly::InternalStrideD;
  using StrideS = typename CollectiveMainloopScaleOnly::StrideScale;
};

template <typename Gemm>
void cutlass_w4a8_sm_group_gemm_caller(
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
  using Args = typename Gemm::GemmScaleOnly::Arguments;

  int num_experts = static_cast<int>(expert_offsets.size(0));
  bool per_act_token = a_scales.numel() != 1;
  bool per_out_ch = b_scales.numel() != num_experts;

  TORCH_CHECK(a_tensors.dim() == 2 or a_tensors.dim() == 3, "A tensor must be 2D/3D");
  TORCH_CHECK(b_tensors.dim() == 3, "B tensor must be 3D [E, N, K/2]");
  TORCH_CHECK(b_scales.dim() == 3, "Scale tensor must be 3D [E, K//512, N*4]");
  TORCH_CHECK(a_scales.dim() == 1, "A Scale tensor must be 1D [1]");
  TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be a 1D tensor");
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(0) == num_experts, "problem_sizes must have num_experts rows");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have 3 columns (N, M, K)");
  TORCH_CHECK(b_tensors.size(0) == num_experts, "B tensor first dimension must match number of groups");
  TORCH_CHECK(b_scales.size(0) == num_experts, "Scale tensor first dimension must match number of groups");
  TORCH_CHECK(
      b_tensors.size(2) * 2 == a_tensors.size(1) or b_tensors.size(2) * 2 == a_tensors.size(2),
      "B tensor K/2 dimension must match A tensor K dimension");
  TORCH_CHECK(a_tensors.scalar_type() == torch::kFloat8_e4m3fn, "A tensor must be fp8 (float_e4m3_t) type");
  TORCH_CHECK(b_tensors.scalar_type() == torch::kInt8, "B tensor must contain packed int4 values (stored as int8)");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32, "Expert offsets must be int32 type");
  TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32, "Problem sizes must be int32 type");

  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(a_tensors.device());

  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);  // activation address
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a_tensors.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  Args arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 0;
  fusion_args.beta = 0;
  fusion_args.alpha_ptr = a_scales.data_ptr<float>();
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<ProblemShape::UnderlyingProblemShape*>(problem_sizes.data_ptr());

  run_int4_fp8_get_group_gemm_starts(
      expert_offsets,
      a_ptrs,
      b_ptrs,
      out_ptrs,
      a_scales_ptrs,
      b_scales_ptrs,
      a_tensors,
      b_tensors,
      d_tensors,
      a_scales,
      b_scales);

  arguments = Args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      {static_cast<const QuantType**>(b_ptrs.data_ptr()),  // weight
       static_cast<typename Gemm::StrideB*>(b_strides.data_ptr()),
       static_cast<const MmaType**>(a_ptrs.data_ptr()),  // activation
       static_cast<typename Gemm::StrideA*>(a_strides.data_ptr()),
       static_cast<const typename Gemm::ElementScalePacked**>(b_scales_ptrs.data_ptr()),
       static_cast<typename Gemm::StrideS*>(s_strides.data_ptr()),
       static_cast<int>(chunk_size)},
      {fusion_args,
       nullptr,
       nullptr,                                       // C
       static_cast<ElementD**>(out_ptrs.data_ptr()),  // D
       static_cast<typename Gemm::StrideD*>(d_strides.data_ptr())},
      hw_info};

  typename Gemm::GemmScaleOnly gemm;
  size_t workspace_size = Gemm::GemmScaleOnly::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a_tensors.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  // Host-only sticky-error peek (~5ns, no stream op, safe inside CUDA graph capture).
  cudaError_t pre_err = cudaPeekAtLastError();

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    TORCH_CHECK(
        false,
        "w4a8 SM GEMM can_implement failed: ",
        cutlass::cutlassGetStatusString(status),
        " | pre_cuda_err=",
        cudaGetErrorString(pre_err),
        " | M_total=",
        a_tensors.size(0),
        " N=",
        d_tensors.size(1),
        " K=",
        a_tensors.size(1),
        " num_experts=",
        num_experts,
        " chunk_size=",
        chunk_size);
  }

  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess, "w4a8 SM GEMM initialize failed: ", cutlass::cutlassGetStatusString(status));

  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    cudaError_t run_err = cudaGetLastError();
    TORCH_CHECK(
        false,
        "w4a8 SM GEMM run failed: ",
        cutlass::cutlassGetStatusString(status),
        " | run_cuda_err=",
        cudaGetErrorString(run_err));
  }
}

}  // namespace
