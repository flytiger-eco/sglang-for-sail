#pragma once

/**
 * @file w4a8_sm_grouped_mm_c3x_swiglu.cuh
 * @brief int4 x fp8 grouped GEMM with a FUSED SwiGLU + fp8-quant epilogue.
 *
 * This is the gate_up GEMM of the W4A8 MoE with the downstream
 * `silu(gate) * up -> fp8` kernel folded into the epilogue. The GEMM emits
 * [T, I] fp8 directly instead of [T, 2I] bf16, so one full round trip through
 * HBM and one kernel launch disappear.
 *
 * Relative to w4a8_grouped_mm_c3x.cuh, exactly three things change:
 *   - ElementC = void, ElementD = float_e4m3_t
 *   - the epilogue is Sm90SwiGLUContractEpilogueArray, rebuilt from the stock
 *     CollectiveBuilder result, so every template parameter the builder computes
 *     is still the stock one
 *   - the fusion callback is Sm90FusedSwiGLUQuant instead of LinearCombination
 * The mainloop is byte-identical to the unfused one.
 *
 * The weight rows must be pre-permuted so that CUTLASS M reads 8 gate channels
 * then 8 up channels, repeating every 16 -- that is the only pairing a Hopper
 * WGMMA accumulator can do register-locally. See SWIGLU_FUSION.md.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass_extensions/epilogue/sm90_epilogue_array_tma_ws_swiglu_contract.hpp"
#include "cutlass_extensions/epilogue/sm90_swiglu_quant_fusion.hpp"
#include "cutlass_extensions/w4a8_sm/collective/collective_builder_mixed_input.hpp"

using namespace cute;

// ─────────────────────────────────────────────────────────────────────────────
// Per-expert base pointers.
//
// This cannot reuse run_int4_fp8_get_group_gemm_starts: that helper assumes D
// and B share the same N. Under the contraction they do not -- D advances by
// n_out = I per token while B and the scales still advance by n_w = 2I.
//
// These live at file scope, not in the anonymous namespace below, and spell
// their types out: nvcc cannot generate launch stubs for a __global__ that is
// itself in an anonymous namespace or whose parameters name types from one.
// w4a8_get_group_starts.cuh is laid out the same way for the same reason.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void w4a8_swiglu_group_gemm_starts(
    int32_t const* expert_offsets,
    cutlass::float_e4m3_t** a_offsets,
    cutlass::int8_t** b_offsets,
    cutlass::float_e4m3_t** out_offsets,
    float** a_scales_offsets,
    cutlass::bfloat16_t** b_scales_offsets,
    cutlass::float_e4m3_t* a_base,
    cutlass::int8_t* b_base,
    cutlass::float_e4m3_t* out_base,
    float* a_scales_base,
    cutlass::bfloat16_t* b_scales_base,
    int64_t n_out,
    int64_t n_w,
    int64_t k,
    bool per_act_token,
    bool per_out_ch) {
  int expert_id = threadIdx.x;
  int32_t expert_offset = expert_offsets[expert_id];

  a_offsets[expert_id] = a_base + expert_offset * k;
  b_offsets[expert_id] = b_base + expert_id * k * n_w / 2;
  out_offsets[expert_id] = out_base + expert_offset * n_out;
  a_scales_offsets[expert_id] = a_scales_base + (per_act_token ? expert_offset : 0);
  b_scales_offsets[expert_id] = b_scales_base + (per_out_ch ? expert_id * n_w * k / 128 : expert_id);
}

__global__ void w4a8_swiglu_group_gemm_starts_3d(
    cutlass::float_e4m3_t** a_offsets,
    cutlass::int8_t** b_offsets,
    cutlass::float_e4m3_t** out_offsets,
    float** a_scales_offsets,
    cutlass::bfloat16_t** b_scales_offsets,
    cutlass::float_e4m3_t* a_base,
    cutlass::int8_t* b_base,
    cutlass::float_e4m3_t* out_base,
    float* a_scales_base,
    cutlass::bfloat16_t* b_scales_base,
    int64_t n_out,
    int64_t n_w,
    int64_t m,
    int64_t k,
    bool per_out_ch,
    int num_experts) {
  int expert_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert_id >= num_experts) return;

  a_offsets[expert_id] = a_base + expert_id * m * k;
  b_offsets[expert_id] = b_base + expert_id * k * n_w / 2;
  out_offsets[expert_id] = out_base + expert_id * m * n_out;
  a_scales_offsets[expert_id] = a_scales_base;
  b_scales_offsets[expert_id] = b_scales_base + (per_out_ch ? expert_id * n_w * 4 * k / 512 : expert_id);
}

namespace {

// orin： D = alpha × Accumulator + beta × C
// fuse： Accumulator = A × B， D = FP8Quant(SiLU(gate) × up)
using MmaType = cutlass::float_e4m3_t;
using ElementAccumulator = float;
using ElementScale = cutlass::bfloat16_t;
using ElementC = void;                   // the fused epilogue has no C source
using ElementD = cutlass::float_e4m3_t;  // contracted, quantized output
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;

using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using LayoutC_Transpose = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
static constexpr int AlignmentC = AlignmentD;  // unused, ElementC = void

// QuantType_ selects the int4 encoding of B: cutlass::int4b_t (two's complement,
// LUT/PRMT convert) or cutlass::sm_int4b_t (sign-magnitude, ALU-only convert).
// It reaches only the mainloop -- the fused epilogue is identical either way.
template <
    typename TileShape,
    typename ClusterShape,
    typename KernelSchedule,
    typename EpilogueSchedule,
    typename QuantType_ = cutlass::int4b_t>
struct cutlass_3x_w4a8_group_gemm_swiglu {
  using QuantType = QuantType_;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;

  static constexpr int GroupSize = 128;
  static constexpr int PackedScalesNum = get<2>(TileShape{}) / GroupSize;  // Tile K / 128
  using ElementScalePacked = cutlass::Array<ElementScale, PackedScalesNum>;

  // Stock builder picks the epilogue tile (EpiTileN = gcd(min(32, TILE_N), TILE_N)).
  // sm90_builder.inl relu, EpiTileM = min(_128, TILE_M)、EpiTileN = gcd(min(_32, TILE_N), TILE_N)
  using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;

  // Stock builder first -- it owns stage count, epilogue tile, copy atoms, dispatch policy.
  using StockEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,  // sm90
      OperatorClass,
      TileShape,     // CTA tile(M,N,K)
      ClusterShape,  // cluster
      EpilogueTile,  // EpilogueTileAuto
      ElementAccumulator,
      ElementAccumulator,
      ElementC,  // void
      LayoutC_Transpose*,
      AlignmentC,
      ElementD,  // output D:fp8_e4m
      LayoutD_Transpose*,
      AlignmentD,
      EpilogueSchedule,  // PtrArrayTmaWarpSpecializedCooperative(WS schedule)
      sglang::epilogue::fusion::Sm90FusedSwiGLUQuant<float>>::CollectiveOp;  // fusion
                                                                             // operation;builder->FusionCallbacks
  // before fusion operation: LinearCombination(builder default) , D = alpha·acc + beta·C
  // Sm90FusedSwiGLUQuant->sm90_swiglu_quant_fusion.hpp

  // ...then re-instantiate those same parameters on the contracting fork.
  // FP32 accumulator -> gate/up pairing -> SwiGLU -> output scale -> FP8 conversion -> contracted TMA store
  using CollectiveEpilogue = sglang::epilogue::collective::MakeSwiGLUContractEpilogue<StockEpilogue>;

  // TAM load -> int4 weight unquant FP8 -> WGMMA -> FP32 accumulator
  using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilderMixedInput<
      ArchTag,
      OperatorClass,
      cute::tuple<QuantType, ElementScalePacked>,  // QuantType: sm_int4b_t;
      LayoutB_Transpose*,                          // cutlass internal A, b_tensors weight
      AlignmentB,
      MmaType,             // cutlass::float_e4m3_t
      LayoutA_Transpose*,  // cutlass internal B,  a_tensors activation
      AlignmentA,
      ElementAccumulator,  // WGMMA->FP32 accumulator
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<
          int>(  // smem stage;  main_loop smem = CTA_total - sizeofsizeof(CollectiveEpilogue::SharedStorage)
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernelScaleOnly =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopScaleOnly, CollectiveEpilogue>;
  // host warrp,ex  using Args = typename Gemm::GemmScaleOnly::Arguments
  using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;

  using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
  using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
  using StrideD = typename GemmKernelScaleOnly::InternalStrideD;
  using StrideS = typename CollectiveMainloopScaleOnly::StrideScale;
};

// ─────────────────────────────────────────────────────────────────────────────
/**
 * @brief Fused gate_up GEMM: D = quant(silu(gate) * up), [T, I] fp8.
 *
 * @param d_tensors  output, fp8_e4m3, [T, I] (2D) or [E, M, I] (3D). I is HALF
 *                   the weight's output-channel count.
 * @param out_scales scalar float tensor; D is divided by it before the e4m3
 *                   convert. This is the a2_scale the standalone quant kernel
 *                   used to apply.
 * All other arguments match cutlass_w4a8_group_gemm_caller. problem_sizes still
 * carries M = 2I: the mainloop and the tile scheduler see the stock geometry and
 * only the D tensor is halved.
 */
template <typename Gemm>
void cutlass_w4a8_group_gemm_swiglu_caller(
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
  using Args = typename Gemm::GemmScaleOnly::Arguments;

  int num_experts = static_cast<int>(expert_offsets.size(0));
  bool per_act_token = a_scales.numel() != 1;
  bool per_out_ch = b_scales.numel() != num_experts;

  TORCH_CHECK(a_tensors.dim() == 2 or a_tensors.dim() == 3, "A tensor must be 2D/3D");
  TORCH_CHECK(b_tensors.dim() == 3, "B tensor must be 3D [E, 2I, K/2]");
  TORCH_CHECK(b_scales.dim() == 3, "Scale tensor must be 3D [E, K//512, 2I*4]");
  TORCH_CHECK(a_scales.dim() == 1, "A Scale tensor must be 1D [1]");
  TORCH_CHECK(d_tensors.dim() == a_tensors.dim(), "D and A must have the same rank");

  TORCH_CHECK(a_tensors.scalar_type() == torch::kFloat8_e4m3fn, "A tensor must be fp8 e4m3");
  TORCH_CHECK(b_tensors.scalar_type() == torch::kInt8, "B tensor must be packed int4 in int8");
  TORCH_CHECK(
      d_tensors.scalar_type() == torch::kFloat8_e4m3fn,
      "fused SwiGLU output must be fp8 e4m3, got ",
      d_tensors.scalar_type());
  TORCH_CHECK(out_scales.scalar_type() == torch::kFloat32, "out_scales must be float32");
  TORCH_CHECK(out_scales.numel() == 1, "out_scales must be a scalar (per-tensor quant)");
  TORCH_CHECK(a_scales.scalar_type() == torch::kFloat32, "a_scales must be float32");
  TORCH_CHECK(b_scales.scalar_type() == torch::kBFloat16, "b_scales must be bfloat16");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32, "Expert offsets must be int32");
  TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32, "Problem sizes must be int32");
  TORCH_CHECK(problem_sizes.size(0) == num_experts, "problem_sizes must have num_experts rows");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have 3 columns");
  TORCH_CHECK(b_tensors.size(0) == num_experts, "B first dim must be num_experts");
  TORCH_CHECK(b_scales.size(0) == num_experts, "b_scales first dim must be num_experts");

  int64_t const n_out = d_tensors.size(d_tensors.dim() - 1);
  int64_t const n_w = b_tensors.size(1);
  TORCH_CHECK(
      n_w == 2 * n_out,
      "fused SwiGLU contracts 2:1 along the weight output channels: expected B N=",
      2 * n_out,
      " for D N=",
      n_out,
      ", got ",
      n_w);
  // Two independent constraints, and AlignmentD is the binding one:
  //   - 8  : the gate/up pairing block (a WGMMA accumulator thread owns rows m, m+8)
  //   - 16 : AlignmentD, the 128-bit vectorised store of D, whose contiguous dim is n_out
  // A multiple of 8 that is not a multiple of 16 does NOT fault -- it silently
  // stores NaN (measured: n_out=24 -> 2.9% bit-exact, maxabs=nan). Hence a hard
  // check rather than trusting can_implement.
  static_assert(AlignmentD % 8 == 0, "AlignmentD must still cover the gate/up block");
  TORCH_CHECK(
      n_out % AlignmentD == 0,
      "contracted N must be a multiple of ",
      AlignmentD,
      " (128-bit store alignment for D; the gate/up block alone would only need 8), got ",
      n_out);

  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());
  auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(a_tensors.device());

  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);

  if (a_tensors.dim() == 3) {
    w4a8_swiglu_group_gemm_starts_3d<<<1, num_experts, 0, stream>>>(
        static_cast<MmaType**>(a_ptrs.data_ptr()),
        static_cast<cutlass::int8_t**>(b_ptrs.data_ptr()),
        static_cast<ElementD**>(out_ptrs.data_ptr()),
        static_cast<float**>(a_scales_ptrs.data_ptr()),
        static_cast<ElementScale**>(b_scales_ptrs.data_ptr()),
        static_cast<MmaType*>(a_tensors.data_ptr()),
        static_cast<cutlass::int8_t*>(b_tensors.data_ptr()),
        static_cast<ElementD*>(d_tensors.data_ptr()),
        static_cast<float*>(a_scales.data_ptr()),
        static_cast<ElementScale*>(b_scales.data_ptr()),
        n_out,
        n_w,
        a_tensors.size(1),
        a_tensors.size(2),
        per_out_ch,
        num_experts);
  } else {
    w4a8_swiglu_group_gemm_starts<<<1, num_experts, 0, stream>>>(
        static_cast<int32_t const*>(expert_offsets.data_ptr()),
        static_cast<MmaType**>(a_ptrs.data_ptr()),
        static_cast<cutlass::int8_t**>(b_ptrs.data_ptr()),
        static_cast<ElementD**>(out_ptrs.data_ptr()),
        static_cast<float**>(a_scales_ptrs.data_ptr()),
        static_cast<ElementScale**>(b_scales_ptrs.data_ptr()),
        static_cast<MmaType*>(a_tensors.data_ptr()),
        static_cast<cutlass::int8_t*>(b_tensors.data_ptr()),
        static_cast<ElementD*>(d_tensors.data_ptr()),
        static_cast<float*>(a_scales.data_ptr()),
        static_cast<ElementScale*>(b_scales.data_ptr()),
        n_out,
        n_w,
        a_tensors.size(1),
        per_act_token,
        per_out_ch);
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a_tensors.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  // a1_scale rides the same channel the unfused path put on alpha, but here it is
  // applied to the accumulator BEFORE silu, which is where it belongs.
  decltype(std::declval<Args>().epilogue.thread) fusion_args;
  fusion_args.acc_scale_ptr = a_scales.data_ptr<float>();
  fusion_args.out_scale_ptr = out_scales.data_ptr<float>();

  ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<ProblemShape::UnderlyingProblemShape*>(problem_sizes.data_ptr());

  Args arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      {static_cast<const typename Gemm::QuantType**>(b_ptrs.data_ptr()),
       static_cast<typename Gemm::StrideB*>(b_strides.data_ptr()),
       static_cast<const MmaType**>(a_ptrs.data_ptr()),
       static_cast<typename Gemm::StrideA*>(a_strides.data_ptr()),
       static_cast<const typename Gemm::ElementScalePacked**>(b_scales_ptrs.data_ptr()),
       static_cast<typename Gemm::StrideS*>(s_strides.data_ptr()),
       static_cast<int>(chunk_size)},
      {fusion_args,
       nullptr,
       nullptr,
       static_cast<ElementD**>(out_ptrs.data_ptr()),
       static_cast<typename Gemm::StrideD*>(d_strides.data_ptr())},
      hw_info};

  typename Gemm::GemmScaleOnly gemm;
  size_t workspace_size = Gemm::GemmScaleOnly::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a_tensors.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  cudaError_t pre_err = cudaPeekAtLastError();

  cutlass::Status status = gemm.can_implement(arguments);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "w4a8 fused-SwiGLU GEMM can_implement failed: ",
      cutlass::cutlassGetStatusString(status),
      " | pre_cuda_err=",
      cudaGetErrorString(pre_err),
      " | N_out=",
      n_out,
      " N_w=",
      n_w,
      " K=",
      a_tensors.size(a_tensors.dim() - 1),
      " num_experts=",
      num_experts,
      " chunk_size=",
      chunk_size);

  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "w4a8 fused-SwiGLU GEMM initialize failed: ",
      cutlass::cutlassGetStatusString(status));

  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    cudaError_t run_err = cudaGetLastError();
    TORCH_CHECK(
        false,
        "w4a8 fused-SwiGLU GEMM run failed: ",
        cutlass::cutlassGetStatusString(status),
        " | run_cuda_err=",
        cudaGetErrorString(run_err));
  }
}

}  // namespace
