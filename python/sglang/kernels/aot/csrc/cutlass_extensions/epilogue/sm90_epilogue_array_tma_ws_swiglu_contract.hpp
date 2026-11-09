/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief SwiGLU-contracting fork of CUTLASS's Sm90 ptr-array TMA warp-specialized epilogue.

  ── What this is ────────────────────────────────────────────────────────────────
  A byte-for-byte copy of
      cutlass/epilogue/collective/sm90_epilogue_array_tma_warpspecialized.hpp
  (cutlass 57e3cfb, the GIT_TAG sgl-kernel pins) with the D path modified to contract
  the M extent 2:1. Diff it against the stock file to review; every change is marked
  with a `SWIGLU:` comment.

  It is deliberately NOT a specialization of cutlass::epilogue::collective::CollectiveEpilogue
  and it lives in its own namespace. Nothing in the stock CUTLASS tree is touched, so no
  other operator that instantiates the Sm90 WS epilogue (fp8 GEMM, blockwise MoE, ...) can
  be affected. GemmUniversal takes its CollectiveEpilogue_ by duck typing, so a plain class
  template is enough. Use MakeSwiGLUContractEpilogue<StockCollectiveOp> to build it from a
  stock CollectiveBuilder result.

  ── Why the D path has to change ────────────────────────────────────────────────
  The w4a8 MoE gate_up GEMM uses the transpose trick: CUTLASS M = 2I (weight output
  channels), N = tokens. With the weights offline-permuted so that gate_i and up_i land on
  accumulator rows (m, m+8), a fusion EVT node can compute silu(g)*u entirely in registers
  (proven by w4a8_epi_frag_probe.cu / w4a8_fused_harness_evt.cu). But an EVT node cannot
  CONTRACT: visit() must return an Array of the same FragmentSize, so D would stay [T, 2I].

  Contraction is thread-local -- after dropping the up slots each thread owns exactly ONE
  output row -- but it destroys the (m, m+8) register pair that CopyAtomC/STSM is built
  around. So the D path gets its own TiledCopy, whose TV layout was derived and verified
  against ground truth on all three shipping tile shapes by w4a8_contract_tv_probe.cu.

  ── Restrictions relative to the stock epilogue ─────────────────────────────────
  This fork drops what w4a8 does not use, so that the surviving code is auditable:
    * no C source / bias (ElementC must be void)  -> no smem reuse, no G2S/S2R path
    * no R2R shuffle, no im2col
    * EPI_TILE_M must equal CTA_M and be even; EPI_TILE_N must be a multiple of 8
    * the contracted TV assumes the 256-thread cooperative TiledMma (asserted in store())
  The grouped-GEMM tensormap machinery is kept verbatim.
*/

#pragma once

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
// SWIGLU: for detail::sm90_get_epilogue_smem_swizzle_layout_atom, re-run on the
// contracted epilogue tile (see SmemLayoutAtomDContract below).
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/detail/collective.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace sglang {
namespace epilogue {
namespace collective {

using namespace cute;
using namespace cutlass;
using cutlass::epilogue::Sm90PtrArrayTmaWarpSpecialized;

// SWIGLU: the stock file lives in cutlass::epilogue::collective, where a bare `detail::`
// or `fusion::` resolved against the enclosing cutlass namespaces. Here the enclosing
// namespaces are sglang's, and both using-directives above inject cute::detail and
// cutlass::detail at global scope, so a bare `detail::` would be ambiguous. Pin both
// names to what the stock file meant.
namespace detail = cutlass::epilogue::collective::detail;
namespace fusion = cutlass::epilogue::fusion;
using cutlass::sizeof_bits;
using cutlass::sizeof_bits_v;

/////////////////////////////////////////////////////////////////////////////////////////////////

// SWIGLU: a plain class template, NOT a partial specialization of
// cutlass::epilogue::collective::CollectiveEpilogue. Same 17 parameters and same public
// interface, so GemmUniversal accepts it in the CollectiveEpilogue_ slot.
template <
    class DispatchPolicy_,
    class CtaTileMNK_,    //     (CTA_M,CTA_N,CTA_K)
    class EpilogueTile_,  // (EPI_TILE_M,EPI_TILE_N)
    class ElementC_,
    class StrideC_,
    class ElementD_,
    class StrideD_,
    class FusionCallbacks_,
    class CopyOpG2S_,
    class SmemLayoutAtomC_,
    class CopyOpS2R_,
    class CopyOpS2G_,
    class SmemLayoutAtomD_,
    class CopyOpR2S_,
    class CopyAtomC_,
    class CopyOpR2R_>
class Sm90SwiGLUContractEpilogueArray {
  static_assert(
      cutlass::detail::dependent_false<DispatchPolicy_>,
      "Sm90SwiGLUContractEpilogueArray: only Sm90PtrArrayTmaWarpSpecialized is supported");
};

template <
    int StagesC_,
    int StagesD_,
    int FragmentSize_,
    bool ReuseSmemC_,
    bool DelayTmaStore_,
    int NumEpilogueWarpGroups_,
    class CtaTileMNK_,    //     (CTA_M,CTA_N,CTA_K)
    class EpilogueTile_,  // (EPI_TILE_M,EPI_TILE_N)
    class ElementC_,
    class StrideC_,
    class ElementD_,
    class StrideD_,
    class FusionCallbacks_,
    class CopyOpG2S_,
    class SmemLayoutAtomC_,
    class CopyOpS2R_,
    class CopyOpS2G_,
    class SmemLayoutAtomD_,
    class CopyOpR2S_,
    class CopyAtomC_,
    class CopyOpR2R_>
class Sm90SwiGLUContractEpilogueArray<
    Sm90PtrArrayTmaWarpSpecialized<
        StagesC_,
        StagesD_,
        FragmentSize_,
        ReuseSmemC_,
        DelayTmaStore_,
        NumEpilogueWarpGroups_>,
    CtaTileMNK_,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    FusionCallbacks_,
    CopyOpG2S_,
    SmemLayoutAtomC_,
    CopyOpS2R_,
    CopyOpS2G_,
    SmemLayoutAtomD_,
    CopyOpR2S_,
    CopyAtomC_,
    CopyOpR2R_> {
 public:
  //
  // Type Aliases
  //
  using DispatchPolicy = Sm90PtrArrayTmaWarpSpecialized<
      StagesC_,
      StagesD_,
      FragmentSize_,
      ReuseSmemC_,
      DelayTmaStore_,
      NumEpilogueWarpGroups_>;
  using CtaTileMNK = CtaTileMNK_;
  using EpilogueTile = EpilogueTile_;
  using FusionCallbacks = FusionCallbacks_;
  using ElementC = ElementC_;
  using StrideC = StrideC_;
  using InternalStrideC = cute::remove_pointer_t<StrideC>;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using InternalStrideD = cute::remove_pointer_t<StrideD>;
  using CopyOpG2S = CopyOpG2S_;
  using SmemLayoutAtomC = SmemLayoutAtomC_;
  using CopyOpS2R = CopyOpS2R_;
  using CopyOpS2G = CopyOpS2G_;
  using SmemLayoutAtomD = SmemLayoutAtomD_;
  using CopyOpR2S = CopyOpR2S_;
  using CopyAtomC = CopyAtomC_;
  using CopyOpR2R = CopyOpR2R_;

  using ThreadEpilogueOp = typename fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;
  using GmemTiledCopyC = CopyOpG2S;
  using GmemTiledCopyD = CopyOpS2G;

  static_assert(
      !is_layout<EpilogueTile>::value && is_tuple<EpilogueTile>::value,
      "EpilogueTile must be a cute::Tile or cute::Shape");
  static_assert(cute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
  static_assert(cute::rank(EpilogueTile{}) == 2, "EpilogueTile must be rank-2: [EPI_TILE_M, EPI_TILE_N]");
  static_assert(size<0>(CtaTileMNK{}) % size<0>(shape(EpilogueTile{})) == 0, "EPI_TILE_M must divide CTA_M");
  static_assert(size<1>(CtaTileMNK{}) % size<1>(shape(EpilogueTile{})) == 0, "EPI_TILE_N must divide CTA_N");
  static_assert(cute::rank(InternalStrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
  static_assert(cute::rank(InternalStrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

  // SWIGLU: restrictions of this fork (see the file header for why each one is dropped).
  static_assert(cute::is_void_v<ElementC>, "SwiGLU-contract epilogue has no C source; build it with ElementC = void");
  static_assert(cute::is_void_v<CopyOpR2R>, "SwiGLU-contract epilogue does not support the R2R shuffle path");
  // Holds for both schedules we ship. Cooperative always gives EPI_TILE_M == CTA_M.
  // Pingpong gives EPI_TILE_M = CTA_M/2 for CTA_M >= 128, but the only Pingpong tiles
  // in the dispatch table are CTA_M = 64, where one warpgroup owns the whole tile in a
  // single subtile and EPI_TILE_M lands back on CTA_M. That is not a coincidence to
  // rely on so much as one to check: PP<128, N> would trip this assert, and it is
  // already excluded on its own merits (N accumulator registers per thread instead of
  // N/2, spilling under __launch_bounds__(384), and zero extra mainloop stages over
  // Cooperative once the fusion's freed epilogue smem is counted).
  //
  // Relaxing this to `EPI_TILE_M % 16 == 0` -- the gate/up pairing period, which is
  // what actually has to stay inside a fragment -- also compiles and is numerically
  // identical (relL1 0.203%, >1ulp = 0 on 397B M=512..4096); the D-side tiling at :811
  // already divides by EpilogueTileD and the epi_m -> mma_m mapping at :1079/:1090 is
  // stock CUTLASS. Kept strict because nothing shipping needs it.
  static_assert(
      size<0>(shape(EpilogueTile{})) == size<0>(CtaTileMNK{}), "SwiGLU contraction assumes EPI_TILE_M == CTA_M");
  static_assert(size<0>(shape(EpilogueTile{})) % 2 == 0, "EPI_TILE_M must be even to contract 2:1");
  static_assert(size<1>(shape(EpilogueTile{})) % 8 == 0, "contracted value layout needs EPI_TILE_N divisible by 8");

  // SWIGLU: the D side works on a HALF-HEIGHT tile -- the fused output is [T, I] fp8,
  // not [T, 2I]. Everything downstream of the accumulator (SmemLayoutD, TMA_D, gD)
  // is built from these instead of EpilogueTile / CtaTileMNK.
  static constexpr int EpiTileM = size<0>(shape(EpilogueTile{}));
  static constexpr int EpiTileN = size<1>(shape(EpilogueTile{}));
  static constexpr int EpiTileMo = EpiTileM / 2;  // contracted EPI_TILE_M
  using EpilogueTileD = Shape<Int<EpiTileMo>, Int<EpiTileN>>;
  using CtaTileD = decltype(replace<0>(CtaTileMNK{}, Int<size<0>(CtaTileMNK{}) / 2>{}));

 private:
  constexpr static bool is_source_supported = not cute::is_void_v<ElementC>;
  constexpr static bool is_destination_supported = not cute::is_void_v<ElementD>;
  using NonVoidElementD =
      cute::conditional_t<not is_destination_supported, fusion::get_element_aux_t<FusionCallbacks>, ElementD>;
  static_assert(not cute::is_void_v<NonVoidElementD>, "SmemElementD is void");
  using NonVoidElementC =
      cute::conditional_t<not is_source_supported, NonVoidElementD, ElementC>;  // prevents void ref breakages

  using SmemElementC = typename cutlass::detail::get_unpacked_element_type<NonVoidElementC>::type;
  using SmemElementD = typename cutlass::detail::get_unpacked_element_type<NonVoidElementD>::type;

  // SWIGLU: contracted register->smem copy for the D path.
  //
  // After the up slots are dropped, a thread owns ONE m_out row and EPI_TILE_N/2 values
  // along n, so the (m, m+8) register pair that CopyAtomC/STSM is built around is gone.
  // The replacement TV layout was read off the real accumulator partitioning and checked
  // against ground truth (0 mismatches, all 256 threads) for TileShape <128,64,256>,
  // <128,128,256> and <128,48,256> by w4a8_contract_tv_probe.cu:
  //
  //   thread (t0,t1,t2) -> 2*Mo*t0 + t1 + 8*t2       (tid = t0 + 4*t1 + 32*t2)
  //   value  (v0,v2)    -> Mo*v0 + 8*Mo*v2
  //
  // over a (Mo, EPI_TILE_N) tile with m contiguous (D is [T,Nout] row-major, so CUTLASS M
  // is the contiguous dim). The thread shape is (4, 8, NumContractWarps): the 4x8 is one
  // warp's quad/lane split, and the third mode is however many warps the EPI tile spans.
  // Each warp owns 16 accumulator rows -> 8 contracted, so that count is EpiTileMo/8 --
  // 8 warps (256 threads) for a cooperative TiledMma, 4 (128) for a pingpong one.
  // store() asserts the TiledMma it was handed agrees.
  static constexpr int NumContractWarps = EpiTileMo / 8;
  static_assert(EpiTileMo % 8 == 0, "each warp owns 16 accumulator rows -> 8 contracted");
  using ContractTV = decltype(make_layout(
      Layout<Shape<_4, _8, Int<NumContractWarps>>, Stride<Int<2 * EpiTileMo>, _1, _8>>{},
      Layout<Shape<_2, Int<EpiTileN / 8>>, Stride<Int<EpiTileMo>, Int<8 * EpiTileMo>>>{}));
  using TiledCopyR2SContract = decltype(make_tiled_copy_impl(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<cutlass::sizeof_bits<SmemElementD>::value>, SmemElementD>{},
      ContractTV{},
      Shape<Int<EpiTileMo>, Int<EpiTileN>>{}));
  static constexpr int NumContractThreads = size<0>(ContractTV{});

  // SWIGLU: the D store is one STS.U8 per value by design; butterfly-shuffle vectorization
  // was measured net-negative and reverted (see SWIGLU_FUSION.md B6).

  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static bool ReuseSmemC = ReuseSmemC_ and is_destination_supported;
  constexpr static bool DelayTmaStore = DelayTmaStore_;

  constexpr static bool is_m_major_C = detail::is_m_major<InternalStrideC>();
  constexpr static bool is_m_major_D = detail::is_m_major<InternalStrideD>();

  constexpr static bool is_im2col_C = cute::is_same_v<CopyOpG2S, SM90_TMA_LOAD_IM2COL>;
  constexpr static bool is_im2col_D = cute::is_same_v<CopyOpS2G, SM90_TMA_STORE_IM2COL>;

  // Check if register transformation is needed before copying register to shared memory.
  constexpr static bool IsUseR2R = !cute::is_void_v<CopyOpR2R>;

  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(size<0>(EpilogueTile{}), size<1>(EpilogueTile{}), Int<StagesC>{}),
      cute::conditional_t<is_m_major_C, Step<_2, _1, _3>, Step<_1, _2, _3>>{}));
  // SWIGLU: D's smem tile is half as tall, so the swizzle atom the builder chose for
  // (EPI_TILE_M, EPI_TILE_N) may no longer divide it -- e.g. an fp8 m-major SW128 atom
  // wants 128 contiguous bytes and only 64 remain. Re-run the stock selector on the
  // contracted tile instead of reusing SmemLayoutAtomD.
  using SmemLayoutAtomDContract =
      decltype(cutlass::epilogue::collective::detail::
                   sm90_get_epilogue_smem_swizzle_layout_atom<InternalStrideD, SmemElementD, EpilogueTileD>());
  using SmemLayoutD = decltype(tile_to_shape(
      SmemLayoutAtomDContract{},
      make_shape(Int<EpiTileMo>{}, Int<EpiTileN>{}, Int<ReuseSmemC ? StagesC : StagesD>{}),
      cute::conditional_t<is_m_major_D, Step<_2, _1, _3>, Step<_1, _2, _3>>{}));

  constexpr static bool support_smem_reuse = is_source_supported && is_destination_supported && StagesD <= StagesC &&
                                             cosize(take<0, 2>(SmemLayoutC{})) == cosize(take<0, 2>(SmemLayoutD{}));
  static_assert(not(ReuseSmemC && not support_smem_reuse), "Smem reuse requirements not met");

  constexpr static size_t SmemAlignmentD = cutlass::detail::alignment_for_swizzle(SmemLayoutD{});
  constexpr static size_t SmemAlignmentC = cutlass::detail::alignment_for_swizzle(SmemLayoutC{});
  constexpr static size_t MaxSmemAlignment = cute::max(SmemAlignmentC, SmemAlignmentD);

  using SmemArrayTypeC = cute::ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>>;
  using SmemArrayTypeD = cute::ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>>;

  using EmptyType = cute::tuple<>;
  using SmemCStorage = cute::conditional_t<is_source_supported and (not ReuseSmemC), SmemArrayTypeC, EmptyType>;
  using SmemDStorage = cute::conditional_t<is_destination_supported, SmemArrayTypeD, EmptyType>;

  struct CollectiveStorageWithC {
    alignas(SmemAlignmentC) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  union CollectiveStorageWithoutC {
    cute::array<SmemElementC, 0> smem_C;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  union CollectiveStorageReuseC {
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

 public:
  // TMA pipeline for loading C
  using LoadPipeline = cutlass::PipelineTransactionAsync<StagesC>;
  using LoadPipelineState = cutlass::PipelineState<StagesC>;
  constexpr static uint32_t TmaTransactionBytes =
      (size(take<0, 2>(SmemLayoutC{})) * static_cast<uint32_t>(sizeof_bits<SmemElementC>::value)) / 8;
  constexpr static bool RequiresTransactionBytes = true;

  constexpr static int NumEpilogueWarpGroups = NumEpilogueWarpGroups_;
  constexpr static uint32_t MinTensorMapWorkspaceAlignment = 64;

  // TMA pipeline for storing D
  using StorePipeline = cute::
      conditional_t<ReuseSmemC, cutlass::PipelineTmaStore<StagesC, StagesD - 1>, cutlass::PipelineTmaStore<StagesD>>;
  using StorePipelineState = cutlass::PipelineState<ReuseSmemC ? StagesC : StagesD>;

  struct SharedStorage {
    struct TensorStorage {
      using CollectiveStorage = cute::conditional_t<
          not is_source_supported,
          CollectiveStorageWithoutC,
          cute::conditional_t<ReuseSmemC, CollectiveStorageReuseC, CollectiveStorageWithC>>;
      CollectiveStorage collective;

      using FusionStorage = typename FusionCallbacks::SharedStorage;
      FusionStorage thread;
    } tensors;

    struct TensorMapStorage : cute::aligned_struct<128, _0> {
      cute::TmaDescriptor smem_tensormap_C;
      cute::array<cute::TmaDescriptor, NumEpilogueWarpGroups> smem_tensormap_D;
    } tensormaps;

    using PipelineStorage = typename LoadPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideC, StrideC>;

  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementC const** ptr_C = nullptr;
    StrideC dC;
    ElementD** ptr_D = nullptr;
    StrideD dD;
  };

  // Device side epilogue params
  struct Params {
    using TMA_C = decltype(make_tma_copy(
        CopyOpG2S{},
        make_tensor(
            make_gmem_ptr(static_cast<NonVoidElementC const*>(nullptr)),
            repeat_like(InternalStrideC{}, int32_t(0)),
            InternalStrideC{}),
        take<0, 2>(SmemLayoutC{}),
        EpilogueTile{},
        _1{}));

    // SWIGLU: D's TMA is built over the CONTRACTED epilogue tile.
    using TMA_D = decltype(make_tma_copy(
        CopyOpS2G{},
        make_tensor(
            make_gmem_ptr(static_cast<NonVoidElementD const*>(nullptr)),
            repeat_like(InternalStrideD{}, int32_t(0)),
            InternalStrideD{}),
        take<0, 2>(SmemLayoutD{}),
        EpilogueTileD{},
        _1{}));

    typename FusionCallbacks::Params thread{};
    TMA_C tma_load_c;
    TMA_D tma_store_d;
    cute::TmaDescriptor* tensormaps;
    ElementC const** ptr_C;
    StrideC dC;
    ElementD** ptr_D;
    StrideD dD;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, [[maybe_unused]] void* workspace) {
    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create tensormap/tma desc.
    // These will be replaced with correct values before the initial tma load.
    auto init_M = int32_t(size<0>(CtaTileMNK{}));
    auto init_N = int32_t(size<1>(CtaTileMNK{}));
    auto init_L = 1;

    static_assert(!is_im2col_C and !is_im2col_D, "Im2Col not supported on C or D");

    InternalStrideC stride_c;
    InternalStrideD stride_d;
    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      stride_c = InternalStrideC{};
      stride_d = InternalStrideD{};
    } else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNKL = append<4>(problem_shape.get_host_problem_shape(0), 1);
      init_M = get<0>(problem_shape_MNKL);
      init_N = get<1>(problem_shape_MNKL);
      stride_c = args.dC;
      stride_d = args.dD;
    }

    uint32_t transaction_bytes = TmaTransactionBytes;
    typename Params::TMA_C tma_load_c{};
    if constexpr (is_source_supported) {
      // NOTE: Since TMA desc creation with nullptr not possible until 12.6, we use an initial address even when tensor
      // addresses are on device. This address is never used.
      ElementC const* ptr_C_first_batch = reinterpret_cast<ElementC const*>(
          reinterpret_cast<uint64_t>(args.ptr_C) & 0xFFFFFFFFFFFFFFF0);  // Address must be 16B-aligned
      Tensor tensor_c =
          make_tensor(ptr_C_first_batch, make_layout(make_shape(init_M, init_N, init_L), append<3>(stride_c, _0{})));
      tma_load_c = make_tma_copy(CopyOpG2S{}, tensor_c, take<0, 2>(SmemLayoutC{}), EpilogueTile{}, _1{});
    }

    typename Params::TMA_D tma_store_d{};
    if constexpr (is_destination_supported) {
      // NOTE: Since TMA desc creation with nullptr not possible until 12.6, we use an initial address even when tensor
      // addresses are on device. This address is never used.
      ElementD const* ptr_D_first_batch = reinterpret_cast<ElementD const*>(
          reinterpret_cast<uint64_t>(args.ptr_D) & 0xFFFFFFFFFFFFFFF0);  // Address must be 16B-aligned
      // SWIGLU: D has half the M extent of the accumulator (2I channels -> I).
      Tensor tensor_d = make_tensor(
          ptr_D_first_batch, make_layout(make_shape(init_M / 2, init_N, init_L), append<3>(stride_d, _0{})));
      tma_store_d = make_tma_copy(CopyOpS2G{}, tensor_d, take<0, 2>(SmemLayoutD{}), EpilogueTileD{}, _1{});
    }

    auto fusion_workspace = static_cast<char*>(workspace);
    auto fusion_workspace_size =
        round_nearest(FusionCallbacks::get_workspace_size(problem_shape, args.thread), MinTensorMapWorkspaceAlignment);
    auto tma_descriptor_workspace =
        reinterpret_cast<cute::TmaDescriptor*>(static_cast<char*>(workspace) + fusion_workspace_size);

    return {
        FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, fusion_workspace),
        tma_load_c,
        tma_store_d,
        tma_descriptor_workspace,
        args.ptr_C,
        args.dC,
        args.ptr_D,
        args.dD,
        transaction_bytes,
    };
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    constexpr uint32_t NumInputTensors = NumEpilogueWarpGroups + (cute::is_void_v<ElementC> ? 0 : 1);
    auto descriptors_shape = cute::make_shape(sm_count, Int<NumInputTensors>{});
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);
    // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies
    return (size(descriptors_shape) * SizeOfCuTensorMap) +
           (round_nearest(
               FusionCallbacks::get_workspace_size(problem_shape, args.thread), MinTensorMapWorkspaceAlignment));
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace,
      cudaStream_t stream,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return FusionCallbacks::initialize_workspace(problem_shape, args.thread, workspace, stream, cuda_adapter);
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape problem_shape, [[maybe_unused]] Arguments const& args) {
    bool implementable = true;
    bool fusion_implementable = true;

    if (problem_shape.is_host_problem_shape_available()) {
      for (int i = 0; i < problem_shape.groups(); ++i) {
        auto problem_shape_MNKL = append<4>(problem_shape.get_host_problem_shape(i), 1);
        auto [M, N, K, L] = problem_shape_MNKL;

        if constexpr (is_destination_supported) {
          constexpr int tma_alignment_bits_D = cutlass::detail::get_output_alignment_bits<ElementD>();
          constexpr int min_tma_aligned_elements_D = tma_alignment_bits_D / cutlass::sizeof_bits<ElementD>::value;
          // SWIGLU: D is [T, M/2], so it is the CONTRACTED extent that must be TMA-aligned.
          implementable = implementable && (M % 2 == 0) &&
                          cutlass::detail::check_alignment<min_tma_aligned_elements_D>(
                              cute::make_shape(M / 2, N, L), InternalStrideD{});
        }

        if constexpr (is_source_supported) {
          constexpr int tma_alignment_bits_C = cutlass::detail::get_input_alignment_bits<ElementC>();
          constexpr int min_tma_aligned_elements_C = tma_alignment_bits_C / cutlass::sizeof_bits<ElementC>::value;
          implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_C>(
                                               cute::make_shape(M, N, L), InternalStrideC{});
        }

        fusion_implementable = fusion_implementable && FusionCallbacks::can_implement(problem_shape_MNKL, args.thread);
      }
    } else {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Ignoring check to can implement because host problem shape is not available.\n");
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }

    if (!fusion_implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum requirements for FusionCallbacks.\n");
    }

    bool beta_implementable = true;

    if (cute::is_void_v<ElementC> || args.ptr_C == nullptr) {
      if constexpr (detail::has_beta<Arguments>::value) {
        beta_implementable = args.thread.beta == 0.0;
      }
      if constexpr (detail::has_beta_ptr<Arguments>::value) {
        beta_implementable = beta_implementable && args.thread.beta_ptr == nullptr;
      }
      if constexpr (detail::has_beta_ptr_array<Arguments>::value) {
        beta_implementable = beta_implementable && args.thread.beta_ptr_array == nullptr;
      }
    }

    if (!beta_implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Beta/beta pointer was set, but epilogue is sourceless (void-C).\n");
    }

    return implementable && fusion_implementable && beta_implementable;
  }

  template <class TileShapeMNK>
  CUTLASS_HOST_DEVICE static constexpr int get_load_pipe_increment(TileShapeMNK tile_shape_MNK) {
    // Compute number of epilogue subtiles
    return size<1>(zipped_divide(make_layout(take<0, 2>(tile_shape_MNK)), EpilogueTile{}));
  }

  template <class TileShapeMNK>
  CUTLASS_HOST_DEVICE static constexpr int get_store_pipe_increment(TileShapeMNK tile_shape_MNK) {
    return get_load_pipe_increment(tile_shape_MNK);
  }

  CUTLASS_HOST_DEVICE
  Sm90SwiGLUContractEpilogueArray(Params const& params_, TensorStorage& shared_tensors)
      : params(params_), fusion_callbacks(params_.thread, shared_tensors.thread) {}

  CUTLASS_DEVICE
  bool is_producer_load_needed() const {
    return fusion_callbacks.is_producer_load_needed();
  }

  CUTLASS_DEVICE auto
  load_init(Params const& params, TensorMapStorage& shared_tensormaps, int32_t sm_count, int32_t sm_idx) {
    // Initialize tma for loading
    constexpr bool IsLoad = true;
    auto load_tensormaps = tensormaps_init<IsLoad>(params, shared_tensormaps, sm_count, sm_idx, 0);
    return load_tensormaps;
  }

  template <
      class ProblemShapeMNKL,
      class TileShapeMNK,
      class TileCoordMNKL,
      class TiledMma,
      class TensorMapC,
      __CUTE_REQUIRES(std::is_pointer_v<TensorMapC>)>
  CUTLASS_DEVICE auto load(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl,
      TiledMma tiled_mma,
      int thread_idx,
      TensorStorage& shared_tensors,
      TensorMapC const& load_tensormap,
      int subtile_idx = -1) {
    using namespace cute;

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;

    static_assert(!is_im2col_D, "Do not support im2col");

    auto coord_shape = append<3>(make_shape(m_coord, n_coord), Int<0>{});

    // Represent the full source tensor, slice to get the tile this CTA is currently responsible for
    Tensor mC_mn = params.tma_load_c.get_tma_tensor(append<3>(make_shape(M, N), Int<1>{}));  //       (M,N,L)
    Tensor mC = coalesce(mC_mn, take<0, 2>(CtaTileMNK{}));
    Tensor gC = local_tile(mC, take<0, 2>(CtaTileMNK{}), coord_shape);  // (CTA_M,CTA_N)

    // Apply epilogue subtile, get matching smem tensor
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    Tensor gC_epi = flat_divide(gC, EpilogueTile{});                    // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor sC_epi = make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{});  //      (EPI_TILE_M,EPI_TILE_N,PIPE_C)

    // Prepare the thread(b)lock's (G)mem to (S)mem TMA tiled copy (bGS_)
    ThrCopy thrblk_g2s = params.tma_load_c.get_slice(Int<0>{});
    Tensor bGS_gC = thrblk_g2s.partition_S(gC_epi);  // (G2S,G2S_M,G2S_N,EPI_M,EPI_N)
    Tensor bGS_sC = thrblk_g2s.partition_D(sC_epi);  // (G2S,G2S_M,G2S_N,PIPE_C)

    // Get the fusion callbacks for the producer load warp
    auto pld_args = cutlass::epilogue::fusion::detail::ProducerLoadArgs{
        problem_shape_mnkl, CtaTileMNK{}, tile_coord_mnkl, tiled_mma, EpilogueTile{}, thread_idx};
    auto pld_callbacks = fusion_callbacks.get_producer_load_callbacks(pld_args);
    bool is_C_load_needed = is_source_supported && fusion_callbacks.is_C_load_needed();

    LoadPipelineState last_load_producer_state = load_pipe_producer_state;

    // Predication for TMA load (one thread issues TMA load)
    bool issue_tma_load = cute::elect_one_sync();

    // Pre-loop fusion callback entry point
    pld_callbacks.begin();

    LoadPipelineState prior_state = load_pipe_producer_state;

    bool did_load = false;

    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gC_epi); ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < size<2>(gC_epi); ++epi_m) {
        if (subtile_idx != -1 && (epi_n * static_cast<int>(size<2>(gC_epi)) + epi_m) != subtile_idx) {
          continue;
        }

        // Acquire the lock for this stage
        constexpr uint16_t mcast_mask = 0;
        uint64_t* tma_barrier = load_pipeline.producer_get_barrier(load_pipe_producer_state);

        load_pipeline.producer_acquire(load_pipe_producer_state);

        // Loop fusion callback entry point
        pld_callbacks.step(tma_barrier, epi_m, epi_n, load_pipe_producer_state.count(), issue_tma_load);

        // Execute the TMA load for C if needed
        if (is_C_load_needed) {
          if (issue_tma_load) {
            copy(
                params.tma_load_c.with(load_tensormap, *tma_barrier, mcast_mask),
                bGS_gC(_, _, _, epi_m, epi_n),
                bGS_sC(_, _, _, load_pipe_producer_state.index()));
            load_pipeline.producer_expect_transaction(load_pipe_producer_state);
          }
          last_load_producer_state = load_pipe_producer_state;
          did_load = true;
        }

        // Commit TMA loads for this stage and release the lock
        load_pipeline.producer_commit(load_pipe_producer_state);
        ++load_pipe_producer_state;
      }
    }

    // Post-loop fusion callback entry point
    pld_callbacks.end();

    return load_pipe_producer_state;
  }

  CUTLASS_DEVICE auto load_tail(LoadPipeline load_pipeline, LoadPipelineState load_pipe_producer_state) {
    if (!fusion_callbacks.is_producer_load_needed()) {
      return load_pipe_producer_state;
    }

    bool issue_tma_load = cute::elect_one_sync();
    if (issue_tma_load) {
      load_pipeline.producer_tail(load_pipe_producer_state);
    }

    return load_pipe_producer_state;
  }

  template <
      class ProblemShapeMNKL,
      class TileShapeMNK,
      class TileCoordMNKL,
      class AccEngine,
      class AccLayout,
      class TiledMma,
      class TensorMapD>
  CUTLASS_DEVICE auto store(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl,
      cute::Tensor<AccEngine, AccLayout> accumulators,
      TiledMma tiled_mma,
      int thread_idx,
      TensorStorage& shared_tensors,
      TensorMapD const& store_tensormap,
      int subtile_idx = -1) {
    using namespace cute;
    using ElementAccumulator = typename AccEngine::value_type;
    using ElementCompute_ = typename fusion::FusionCallbacksTraits<FusionCallbacks>::ElementCompute;
    using ElementCompute = cute::conditional_t<cute::is_void_v<ElementCompute_>, ElementAccumulator, ElementCompute_>;

    static_assert(is_rmem<AccEngine>::value, "Accumulator must be RF resident.");
    static_assert(rank(AccLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA,MMA_M,MMA_N)");
    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<TileShapeMNK>::value, "TileShapeMNK must be static");
    static_assert(rank(TileShapeMNK{}) == 3, "TileShapeMNK must be rank 3");
    static_assert(rank(TileCoordMNKL{}) == 4, "TileCoordMNKL must be rank 4");

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;

    static_assert(!is_im2col_D, "Do not support im2col");

    auto coord_shape = append<3>(make_shape(m_coord, n_coord), Int<0>{});

    // Represent the full output tensor, slice to get the tile this CTA is responsible for
    // SWIGLU: D is contracted 2:1 along M -- the fused output is [T, I] fp8, one row per
    // intermediate channel, not [T, 2I]. The tile COORDINATE is unchanged: CTA_M/2 tiles
    // of an M/2-tall tensor index exactly like CTA_M tiles of an M-tall one.
    Tensor mD_mn = params.tma_store_d.get_tma_tensor(append<3>(make_shape(M / 2, N), Int<1>{}));  //     (M/2,N,L)

    Tensor mD = coalesce(mD_mn, take<0, 2>(CtaTileD{}));
    Tensor gD = local_tile(mD, take<0, 2>(CtaTileD{}), coord_shape);  // (CTA_M/2,CTA_N)

    // Apply epilogue subtiling
    Tensor gD_epi = flat_divide(gD, EpilogueTileD{});  // (EPI_TILE_M/2,EPI_TILE_N,EPI_M,EPI_N)

    // Construct the corresponding pipelined smem tensors
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    auto ptr_sD = shared_tensors.collective.smem_D.begin();
    Tensor sC_epi = cute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{}));  // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    Tensor sD_epi = cute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(ptr_sD), SmemLayoutD{}));  // (EPI_TILE_M,EPI_TILE_N,PIPE_D)

    TiledCopy tiled_copy_C_atom = make_tiled_copy_C_atom(CopyAtomC{}, tiled_mma);

    // thread-partition for register to register copy (tRR_)
    TiledCopy tiled_r2r = [&]() {
      if constexpr (IsUseR2R) {
        return make_tiled_copy_S(Copy_Atom<CopyOpR2R, ElementCompute>{}, tiled_copy_C_atom);
      } else {
        return make_tiled_copy_S(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementCompute>{}, tiled_copy_C_atom);
      }
    }();
    ThrCopy thread_r2r = tiled_r2r.get_slice(thread_idx);

    // thread-partition for register to smem copy (tRS_)
    //
    // SWIGLU: this splits in two. The ACCUMULATOR view stays exactly as CUTLASS built it,
    // because that is what the fusion node's visit() is defined against (a fragment reads
    // [G,G,U,U] repeating, up partner at slot +2). Only the D destination is contracted.
    TiledCopy tiled_r2s = make_tiled_copy_S(Copy_Atom<CopyOpR2S, SmemElementD>{}, tiled_copy_C_atom);
    ThrCopy thread_r2s = tiled_r2s.get_slice(thread_idx);
    Tensor tRS_rAcc = thread_r2s.retile_S(accumulators);  // ((R2S,R2S_V),MMA_M,MMA_N)

    // SWIGLU: contracted D destination. STSM is given up here -- after dropping the up
    // slots a thread no longer holds the (m, m+8) pair the atom needs -- so this is a
    // plain vectorizing copy over the TV layout derived in the design notes
    // (thread (t0,t1,t2) -> 2*Mo*t0 + t1 + 8*t2; value (v0,v2) -> Mo*v0 + 8*Mo*v2).
    static_assert(
        size(TiledMma{}) == NumContractThreads, "contracted TV layout warp count disagrees with the TiledMma");
    TiledCopyR2SContract tiled_r2s_D;
    ThrCopy thread_r2s_D = tiled_r2s_D.get_slice(thread_idx);
    Tensor tRS_sD = thread_r2s_D.partition_D(sD_epi);  // (R2S,R2S_M,R2S_N,PIPE_D)

    auto mma_tile_m = size<0>(TileShapeMNK{}) / size<1>(tRS_rAcc);
    auto mma_tile_n = size<1>(TileShapeMNK{}) / size<2>(tRS_rAcc);
    auto epi_tile_m = size<0>(EpilogueTile{});
    auto epi_tile_n = size<1>(EpilogueTile{});

    // Allocate D registers (contracted: half the values the accumulator view carries)
    Layout tRS_rD_layout = make_layout(take<0, 3>(shape(thread_r2s_D.partition_S(sD_epi))));
    Tensor tRS_rD = make_tensor<SmemElementD>(tRS_rD_layout);  // (R2S,R2S_M,R2S_N)

    // Vectorized fragment view. The accumulator keeps the stock FragmentSize; D gets half
    // of it, so the NUMBER of fragments is unchanged and the visit loop below stays 1:1 --
    // each visit() consumes one full acc fragment and yields one half-size D fragment.
    constexpr int FragmentSize = DispatchPolicy::FragmentSize;
    constexpr int FragmentSizeD = FragmentSize / 2;
    static_assert(FragmentSize % 4 == 0, "a fragment must be a whole number of [G,G,U,U] quads to contract");
    Tensor tRS_rAcc_frg = recast<Array<ElementAccumulator, FragmentSize>>(tRS_rAcc);
    Tensor tRS_rD_frg = recast<Array<SmemElementD, FragmentSizeD>>(tRS_rD);
    CUTE_STATIC_ASSERT(size<0>(tRS_rAcc) % FragmentSize == 0, "Fragment size does not vectorize properly");

    // SWIGLU: no C source (ElementC is void), so the whole smem->register C path is dead.
    // It is not merely unused: tSR_rC_layout is derived via thread_s2r.retile_D(tRS_rD),
    // and tRS_rD is now the CONTRACTED layout, which the C-side tiled copy cannot retile.
    // ConsumerStoreArgs still takes tRS_rC by reference, so keep a correctly-typed one.
    using RegisterElementC = ElementCompute;
    Tensor tRS_rC = make_tensor<RegisterElementC>(tRS_rD_layout);  // (R2S,R2S_M,R2S_N), unused

    // thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
    ThrCopy thrblk_s2g = params.tma_store_d.get_slice(Int<0>{});
    Tensor bSG_sD = thrblk_s2g.partition_S(sD_epi);  // (S2G,S2G_M,S2G_N,PIPE_D)
    Tensor bSG_gD = thrblk_s2g.partition_D(gD_epi);  // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)

    // OOB predication for tile quantization "residue"
    // SWIGLU: these describe the D geometry, so they use the CONTRACTED extents. Our
    // fusion node ignores them, but they must stay self-consistent with tRS_rD for any
    // node that does predicate (and for the partition shapes to match).
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M / 2, N));                               // (M/2,N)
    Tensor cD_mn = local_tile(mD_crd, take<0, 2>(CtaTileD{}), make_coord(m_coord, n_coord));  // (CTA_M/2,CTA_N)
    Tensor tRS_cD_mn = thread_r2s_D.partition_S(flat_divide(cD_mn, EpilogueTileD{}));  // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
    // Relative coordinate tensors (static)
    Tensor cD = make_coord_tensor(cD_mn.layout());          // (CTA_M/2,CTA_N)
    Tensor tRS_cD = make_coord_tensor(tRS_cD_mn.layout());  // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
    // Subtract the global "bottom right" corner from the local "top left" corner to get the max relative coordinate
    auto residue_cD = make_coord(M / 2, N) - cD_mn(_0{});          // (m,n)
    auto residue_tRS_cD = make_coord(M / 2, N) - tRS_cD_mn(_0{});  // (m,n)

    CUTE_STATIC_ASSERT(epi_tile_m % mma_tile_m == 0, "MMA_TILE_M must divide EPI_TILE_M");

    if constexpr (epi_tile_m * epi_tile_n > mma_tile_m * mma_tile_n) {
      // When the epilogue subtile is larger than the MMA tiles, loop over multiple MMA tiles
      CUTE_STATIC_ASSERT(epi_tile_n % mma_tile_n == 0, "MMA_TILE_N must divide EPI_TILE_N");
    } else {
      CUTE_STATIC_ASSERT(mma_tile_n % epi_tile_n == 0, "EPI_TILE_N must divide MMA_TILE_N");
    }

    // Get TiledCopy for partition reference when consumer store.
    TiledCopy tiled_copy_partition_ref = make_tiled_copy_S(Copy_Atom<CopyOpR2S, SmemElementD>{}, tiled_copy_C_atom);
    // Get the fusion callbacks for the consumer store warps
    constexpr bool RefSrc = true;  // Register tensors reference R2S copy src layout
    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
        problem_shape_mnkl,
        CtaTileMNK{},
        tile_coord_mnkl,
        tiled_mma,
        EpilogueTile{},
        tiled_copy_partition_ref,
        cD,
        residue_cD,
        tRS_cD,
        residue_tRS_cD,
        tRS_rC,
        thread_idx};
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<RefSrc>(cst_args);
    bool is_producer_load_needed = fusion_callbacks.is_producer_load_needed();
    bool is_C_load_needed = is_source_supported && fusion_callbacks.is_C_load_needed();

    // SWIGLU: visit() still returns a FULL-size fragment (it cannot contract); the
    // contraction happens when we compact its gate slots into tRS_rCompute below, which
    // is half as long.
    using FragmentVisit = decltype(cst_callbacks.visit(tRS_rAcc_frg(0), 0, 0, 0));
    constexpr bool IsDirectR2S = cute::is_same_v<FragmentVisit, Array<SmemElementD, FragmentSize>>;
    using RegisterElementD = cute::conditional_t<!IsDirectR2S, ElementCompute, SmemElementD>;
    Tensor tRS_rCompute = make_tensor<RegisterElementD>(tRS_rD_layout);  // (R2S,R2S_M,R2S_N)
    Tensor tRS_rCompute_frg = recast<Array<RegisterElementD, FragmentSizeD>>(tRS_rCompute);

    // SWIGLU: gate slots of a FragmentSize-long fragment, in order. The frag probe showed
    // a fragment reads [G,G,U,U] repeating every 4, so slot (b, b+1) of each quad is gate
    // and (b+2, b+3) is its up partner. Compaction index c = (b/4)*2 + j.
    auto contract_fragment = [](auto const& full) {
      Array<RegisterElementD, FragmentSizeD> half;
      CUTLASS_PRAGMA_UNROLL
      for (int b = 0; b < FragmentSize; b += 4) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 2; ++j) {
          half[(b / 4) * 2 + j] = static_cast<RegisterElementD>(full[b + j]);
        }
      }
      return half;
    };

    // Thread synchronizer for previously issued waits or fences
    // to ensure visibility of smem reads/writes to threads or TMA unit
    auto synchronize = [&]() {
      cutlass::arch::NamedBarrier::sync(size(TiledMma{}), cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    };

    // Predication for TMA store (a single thread from one warp issues TMA store)
    bool issue_tma_store = ((thread_idx / NumThreadsPerWarp) == 0) && cute::elect_one_sync();

    // In the reuse smem configuration we have StagesC smem buffers and at most StagesD committed TMA stores in flight.
    // The TMA store pipeline producer acquire returns when at most StagesD-1 committed stores are in-flight, so we can
    // only guarantee store completion after StagesD iterations, then we can begin issuing releases on the smem buffer
    // locks. store_pipe_producer_state tracks the acquire and load_pipe_consumer_state tracks the release, in circular
    // buffer fashion.
    LoadPipelineState load_wait_state = load_pipe_consumer_state;
    if constexpr (ReuseSmemC) {
      load_wait_state = store_pipe_producer_state;
      load_wait_state.phase_ ^= 1;
    }

    // We can delay issue of TMA store by one iteration to achieve better interleaving of non-TMA instructions
    // Sync requirements of smem reuse may preclude this optimization
    // Delayed stores cause delayed stage releases which causes deadlock when StagesC == StagesD
    int epi_m_prev = 0, epi_n_prev = 0;
    static_assert(
        not(DelayTmaStore and ReuseSmemC and StagesC <= StagesD), "This TMA epilogue configuration will deadlock");

    // The TMA store sequence for one subtile iteration
    auto tma_store_fn = [&](int epi_m, int epi_n) {
      // Write the tile from smem to gmem with TMA
      cutlass::arch::fence_view_async_shared();  // ensure smem writes are visible to TMA
      synchronize();                             // ensure all threads have issued their async fence
      if constexpr (is_destination_supported) {
        if (issue_tma_store) {
          copy(
              params.tma_store_d.with(store_tensormap),
              bSG_sD(_, _, _, store_pipe_producer_state.index()),
              bSG_gD(_, _, _, epi_m, epi_n));
        }
      }

      // Post async fence, pre TMA commit callback entry point
      cst_callbacks.tma_store(epi_m, epi_n, store_pipe_producer_state.count(), issue_tma_store);

      // Commit the TMA stores for this stage
      if (issue_tma_store) {
        store_pipeline.producer_commit(store_pipe_producer_state);
      }
      ++store_pipe_producer_state;
      ++issued_stores;

      // Wait for the next smem buffer to be available
      if (issue_tma_store) {
        store_pipeline.producer_acquire(store_pipe_producer_state);
      }
      synchronize();

      if constexpr (ReuseSmemC) {
        // producer_acquire returns when at most StagesD-1 committed stores are pending
        bool store_finished = issued_stores > StorePipeline::UnacquiredStages;
        // Let dma warp know earliest smem buffer is consumed and empty after StagesD producer commits
        if (store_finished) {
          if (is_producer_load_needed) {
            load_pipeline.consumer_release(load_pipe_consumer_state);
          }
          ++load_pipe_consumer_state;
        }
      }
    };

    //
    // BEGIN EPILOGUE
    //

    // Pre-loop fusion callback entry point
    cst_callbacks.begin();
    if (cst_callbacks.begin_sync_needed()) {
      synchronize();
    }

    // For each output tile
    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gD_epi); ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < size<2>(gD_epi); ++epi_m) {
        bool is_first_iteration = epi_m == 0 && epi_n == 0;
        bool is_last_iteration = epi_m == size<2>(gD_epi) - 1 && epi_n == size<3>(gD_epi) - 1;

        if (subtile_idx != -1 && (epi_n * static_cast<int>(size<2>(gD_epi)) + epi_m) != subtile_idx) {
          continue;
        }

        cst_callbacks.begin_loop(epi_m, epi_n);

        if (is_producer_load_needed) {
          // Wait for the producer load to fill smem
          load_pipeline.consumer_wait(load_wait_state);

          // SWIGLU: the smem->register C copy is gone with the C path (ElementC = void),
          // so is_C_load_needed is statically false here. A fusion node may still request
          // a producer load for its own auxiliary tensors, hence the wait above stays.
          static_assert(not is_source_supported, "SWIGLU fork has no C path");
        }

        // First loop fusion callback entry point
        cst_callbacks.previsit(epi_m, epi_n, load_wait_state.count(), is_producer_load_needed);

        if (is_producer_load_needed) {
          if constexpr (not ReuseSmemC) {
            // Let producer load warp know smem buffers are consumed and empty
            cutlass::arch::fence_view_async_shared();
            load_pipeline.consumer_release(load_pipe_consumer_state);
            ++load_pipe_consumer_state;
          }
          ++load_wait_state;
        }

        if constexpr (epi_tile_m * epi_tile_n > mma_tile_m * mma_tile_n) {
          // When the epilogue subtile is larger than the MMA tiles, loop over multiple
          // MMA tiles
          static constexpr int MmaMPerEpiM = epi_tile_m / mma_tile_m;
          static constexpr int MmaNPerEpiN = epi_tile_n / mma_tile_n;

          CUTLASS_PRAGMA_UNROLL
          for (int mma_n_in_epi = 0; mma_n_in_epi < MmaNPerEpiN; ++mma_n_in_epi) {
            int mma_n = (epi_n * MmaNPerEpiN) + mma_n_in_epi;

            CUTLASS_PRAGMA_UNROLL
            for (int mma_m_in_epi = 0; mma_m_in_epi < MmaMPerEpiM; ++mma_m_in_epi) {
              int mma_m = (epi_m * MmaMPerEpiM) + mma_m_in_epi;
              Tensor tRS_rAcc_frg_mn = tRS_rAcc_frg(_, mma_m, mma_n);
              int idx_in_epi_subtile = (mma_n_in_epi * MmaMPerEpiM + mma_m_in_epi);

              // SWIGLU: visit() yields a full fragment; keep only its gate slots.
              tRS_rCompute_frg(idx_in_epi_subtile) =
                  contract_fragment(cst_callbacks.visit(tRS_rAcc_frg_mn(0), idx_in_epi_subtile, epi_m, epi_n));
            }
          }
        } else {
          int mma_m = epi_m;
          int mma_n = (epi_n * size<1>(EpilogueTile{})) / mma_tile_n;
          Tensor tRS_rAcc_frg_mn = tRS_rAcc_frg(_, mma_m, mma_n);

          // Vectorized fragment loop with visitor callback entry point.
          // SWIGLU: size(tRS_rCompute_frg) == size(tRS_rAcc_frg_mn) still holds -- the D
          // fragments are half as LONG, not half as MANY -- so this stays 1:1.
          int epi_n_in_mma = epi_n % (mma_tile_n / epi_tile_n);
          int r2s_v = epi_n_in_mma * size(tRS_rCompute_frg);
          CUTLASS_PRAGMA_UNROLL
          for (int epi_v = 0; epi_v < size(tRS_rCompute_frg); ++epi_v) {
            tRS_rCompute_frg(epi_v) =
                contract_fragment(cst_callbacks.visit(tRS_rAcc_frg_mn(r2s_v + epi_v), epi_v, epi_m, epi_n));
          }
        }

        // The latest we can delay the TMA store is right before the smem store of the next iteration
        // since the current TMA store needs to be committed before we can acquire the next smem buffer
        if constexpr (DelayTmaStore) {
          // Issue TMA stores for the previous subtile
          if (not is_first_iteration and subtile_idx == -1) {
            tma_store_fn(epi_m_prev, epi_n_prev);
          }
          epi_m_prev = epi_m;
          epi_n_prev = epi_n;
        }

        // Smem reduction callback entry point using current store buffer for workspace
        cst_callbacks.reduce(
            sD_epi(_, _, store_pipe_producer_state.index()),
            synchronize,
            epi_m,
            epi_n,
            is_last_iteration,
            tRS_rCompute_frg);

        // SWIGLU: the R2R shuffle path is statically excluded (CopyOpR2R must be void) --
        // it retiles against the accumulator layout, which no longer matches tRS_rD.
        static_assert(not IsUseR2R, "SWIGLU fork does not support the R2R shuffle path");

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tRS_rD_frg); ++i) {
          tRS_rD_frg(i) =
              cutlass::NumericArrayConverter<SmemElementD, RegisterElementD, FragmentSizeD>{}(tRS_rCompute_frg(i));
        }

        // Copy tile from register to smem
        if constexpr (is_destination_supported) {
          // SWIGLU: contracted TiledCopy, not the stock STSM-backed tiled_r2s.
          copy(tiled_r2s_D, tRS_rD, tRS_sD(_, _, _, store_pipe_producer_state.index()));
        }

        // Post reduction, pre TMA store callback entry point
        constexpr bool issue_smem_store = true;  // No smem store predication
        cst_callbacks.postreduce(epi_m, epi_n, store_pipe_producer_state.count(), issue_smem_store);

        if constexpr (not DelayTmaStore) {
          // Issue TMA stores for this subtile
          tma_store_fn(epi_m, epi_n);
        }

        cst_callbacks.end_loop(epi_m, epi_n);

      }  // for epi_m
    }  // for epi_n

    if constexpr (DelayTmaStore) {
      // Issue TMA stores for the last subtile
      tma_store_fn(epi_m_prev, epi_n_prev);
    }

    // Post-loop fusion callback entry point
    cst_callbacks.end();

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  CUTLASS_DEVICE auto store_tail(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state) {
    // wait for all TMA stores to complete
    store_pipeline.producer_tail(store_pipe_producer_state);
    // reset store counter
    issued_stores = 0;

    if constexpr (ReuseSmemC) {
      if (fusion_callbacks.is_producer_load_needed()) {
        // Issue releases on up to StagesD-1 previously issued TMA stores
        constexpr int release_stages =
            cute::min(StorePipeline::UnacquiredStages, get_load_pipe_increment(CtaTileMNK{}));
        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < release_stages; ++stage) {
          load_pipeline.consumer_release(load_pipe_consumer_state);
          ++load_pipe_consumer_state;
        }
      }
    }

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  CUTLASS_DEVICE auto store_init(
      Params const& params,
      TensorMapStorage& shared_tensormaps,
      int32_t sm_count,
      int32_t sm_idx,
      int32_t warp_group_idx) {
    int warp_idx_in_warp_group = canonical_warp_idx_sync() % NumWarpsPerWarpGroup;
    // Since only one warp issues TMA store, we only need that one warp to initialize tensormaps
    if (warp_idx_in_warp_group == 0) {
      // Initialize tma
      constexpr bool IsLoad = false;
      auto store_tensormaps = tensormaps_init<IsLoad>(params, shared_tensormaps, sm_count, sm_idx, warp_group_idx);
      return store_tensormaps;
    }
    TmaDescriptor* null_tma_desc = nullptr;
    return cute::make_tuple(null_tma_desc);
  }

  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //

  template <bool IsLoad>
  CUTLASS_DEVICE auto tensormaps_init(
      Params const& params,
      TensorMapStorage& shared_tensormaps,
      int32_t sm_count,
      int32_t sm_idx,
      int32_t warp_group_idx) {
    constexpr uint32_t NumInputTensors = NumEpilogueWarpGroups + (cute::is_void_v<ElementC> ? 0 : 1);
    Layout desc_layout = make_layout(make_shape(sm_count, Int<NumInputTensors>{}));

    Tensor gmem_tensormap = make_tensor(params.tensormaps, desc_layout);  // (SMs, NumInputTensors)

    if constexpr (IsLoad) {
      if (is_source_supported) {
        constexpr int C_tensormap_index = NumEpilogueWarpGroups;
        Tensor pC_tensormap = make_tensor(params.tma_load_c.get_tma_descriptor(), Int<1>{}, Int<1>{});
        Tensor sC_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_C), Int<1>{}, Int<1>{});

        if (cute::elect_one_sync()) {
          // Bringing tensormaps from params to smem for modification later
          copy(recast<uint128_t>(pC_tensormap), recast<uint128_t>(sC_tensormap));
        }
        __syncwarp();
        return cute::make_tuple(&gmem_tensormap(sm_idx, C_tensormap_index));
      }
      TmaDescriptor* null_tma_desc = nullptr;
      return cute::make_tuple(null_tma_desc);
    } else {
      Tensor pD_tensormap = make_tensor(params.tma_store_d.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sD_tensormap =
          make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_D[warp_group_idx]), Int<1>{}, Int<1>{});

      if (cute::elect_one_sync()) {
        // Bringing tensormaps from params to smem for modification later
        copy(recast<uint128_t>(pD_tensormap), recast<uint128_t>(sD_tensormap));
      }
      __syncwarp();
      return cute::make_tuple(&gmem_tensormap(sm_idx, warp_group_idx));
    }
  }

  // Replace address for the global tensor (to be done by single thread)
  template <bool IsLoad>
  CUTLASS_DEVICE void tensormaps_replace_global_address(
      TensorMapStorage& shared_tensormaps, Params const& params, int32_t next_batch, int32_t warp_group_idx) {
    // Replacing global_address for the next batch
    if constexpr (IsLoad) {
      if constexpr (is_source_supported) {
        if (params.ptr_C != nullptr) {
          cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_C, params.ptr_C[next_batch]);
        }
      }
    } else if constexpr (is_destination_supported) {
      cute::tma_descriptor_replace_addr_in_shared_mem(
          shared_tensormaps.smem_tensormap_D[warp_group_idx], params.ptr_D[next_batch]);
    }
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by single thread)
  template <bool IsLoad, class ProblemShape_MNKL>
  CUTLASS_DEVICE void tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps,
      Params const& params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl,
      int32_t warp_group_idx) {
    const uint32_t M = get<0>(problem_shape_mnkl);
    const uint32_t N = get<1>(problem_shape_mnkl);
    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape = {1, 1, 1, 1, 1};
    cute::array<uint64_t, MaxTensorRank> prob_stride = {0, 0, 0, 0, 0};

    if constexpr (IsLoad) {
      if constexpr (is_source_supported) {
        if (params.dC != nullptr) {
          ElementC const* ptr_C = nullptr;
          Tensor tensor_c = make_tensor(ptr_C, make_layout(make_shape(M, N, Int<1>{}), params.dC[next_group]));

          cute::detail::fill_tma_gmem_shape_stride(params.tma_load_c, tensor_c, prob_shape, prob_stride);
          // Convert strides to byte strides
          for (uint64_t& stride : prob_stride) {
            stride = (stride * sizeof_bits_v<ElementC>) / 8;
          }
          cute::tma_descriptor_replace_dims_strides_in_shared_mem(
              shared_tensormaps.smem_tensormap_C, prob_shape, prob_stride);
        }
      }
    } else if constexpr (is_destination_supported) {
      ElementD const* ptr_D = nullptr;
      // SWIGLU: the per-group D tensormap describes the CONTRACTED [T, M/2] output.
      Tensor tensor_d = make_tensor(ptr_D, make_layout(make_shape(M / 2, N, Int<1>{}), params.dD[next_group]));

      cute::detail::fill_tma_gmem_shape_stride(params.tma_store_d, tensor_d, prob_shape, prob_stride);
      // Convert strides to byte strides
      for (uint64_t& stride : prob_stride) {
        stride = (stride * sizeof_bits_v<ElementD>) / 8;
      }

      cute::tma_descriptor_replace_dims_strides_in_shared_mem(
          shared_tensormaps.smem_tensormap_D[warp_group_idx], prob_shape, prob_stride);
    }
  }

  template <bool IsLoad, class ProblemShape_MNKL>
  CUTLASS_DEVICE void tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps,
      Params const& params,
      cute::TmaDescriptor const* tensormap,
      ProblemShape_MNKL problem_shape_mnkl,
      int32_t next_batch,
      int32_t warp_group_idx) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address<IsLoad>(shared_tensormaps, params, next_batch, warp_group_idx);

      if constexpr (IsGroupedGemmKernel) {
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties<IsLoad>(
            shared_tensormaps, params, next_batch, problem_shape_mnkl, warp_group_idx);
      }
    }
  }

  template <bool IsLoad>
  CUTLASS_DEVICE void tensormaps_cp_fence_release(
      TensorMapStorage& shared_tensormaps, cute::TmaDescriptor const* tensormap, const int32_t warp_group_idx = 0) {
    // Commit and wait for all TMA load/store instructions before updating the tensormap in gmem.
    // This operation only happens when the group/batch changes between consecutive tiles.
    // If there are no uncommitted instructions then tma_desc_commit_group results in an empty bulk async-group.
    auto tma_desc_wait_all_fn = []() CUTLASS_LAMBDA_FUNC_INLINE {
      if (cute::elect_one_sync()) {
        cute::tma_desc_commit_group();
        cute::tma_desc_wait_group();
      }
    };
    // Entire warp must do this (ie its aligned)
    if constexpr (IsLoad) {
      if constexpr (is_source_supported) {
        tma_desc_wait_all_fn();
        tma_descriptor_cp_fence_release(tensormap, shared_tensormaps.smem_tensormap_C);
      }
    } else if constexpr (is_destination_supported) {
      tma_desc_wait_all_fn();
      tma_descriptor_cp_fence_release(tensormap, shared_tensormaps.smem_tensormap_D[warp_group_idx]);
    }
  }

  template <bool IsLoad>
  CUTLASS_DEVICE void tensormaps_fence_acquire(cute::TmaDescriptor const* tensormap) {
    if constexpr (IsLoad) {
      if constexpr (is_source_supported) {
        cute::tma_descriptor_fence_acquire(tensormap);
      }
    } else {
      cute::tma_descriptor_fence_acquire(tensormap);
    }
  }

 private:
  Params const& params;
  FusionCallbacks fusion_callbacks;
  int issued_stores = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// SWIGLU: rebuild the fused epilogue from a stock CollectiveBuilder result, so the caller
// never has to spell out 17 template arguments (and stays in sync with whatever the stock
// builder picks for the copy atoms / smem layouts / stage counts).
//
//   using StockEpi = cutlass::epilogue::collective::CollectiveBuilder<...>::CollectiveOp;
//   using FusedEpi = MakeSwiGLUContractEpilogue<StockEpi>;
//
// Note the builder must be given ElementC = void (this fork has no C source) and the
// SwiGLU fusion callbacks as its 15th argument.
template <class StockEpilogue>
using MakeSwiGLUContractEpilogue = Sm90SwiGLUContractEpilogueArray<
    typename StockEpilogue::DispatchPolicy,
    typename StockEpilogue::CtaTileMNK,
    typename StockEpilogue::EpilogueTile,
    typename StockEpilogue::ElementC,
    typename StockEpilogue::StrideC,
    typename StockEpilogue::ElementD,
    typename StockEpilogue::StrideD,
    typename StockEpilogue::FusionCallbacks,
    typename StockEpilogue::CopyOpG2S,
    typename StockEpilogue::SmemLayoutAtomC,
    typename StockEpilogue::CopyOpS2R,
    typename StockEpilogue::CopyOpS2G,
    typename StockEpilogue::SmemLayoutAtomD,
    typename StockEpilogue::CopyOpR2S,
    typename StockEpilogue::CopyAtomC,
    typename StockEpilogue::CopyOpR2R>;

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace collective
}  // namespace epilogue
}  // namespace sglang

/////////////////////////////////////////////////////////////////////////////////////////////////
