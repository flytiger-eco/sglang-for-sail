/***************************************************************************************************
 * SwiGLU + fp8-quant fusion node for the W4A8 MoE gate_up GEMM.
 *
 * Pairs with sglang::epilogue::collective::Sm90SwiGLUContractEpilogueArray
 * (sm90_epilogue_array_tma_ws_swiglu_contract.hpp), which is what actually drops the
 * up-half of the accumulator and halves the D tile along M. This node only does the math;
 * it still returns a full-size fragment, so it is also usable with the stock epilogue if
 * you want the uncontracted [T, 2I] output for debugging (see w4a8_fused_harness_evt.cu).
 *
 * GEOMETRY -- why the partner slot is +2 and not +1.
 * The w4a8 kernel uses LayoutTranspose, so CUTLASS M = weight output channels and CUTLASS
 * N = tokens. A Hopper WGMMA accumulator thread owns exactly two M rows, m and m+8, so the
 * only register-local gate/up pairing is (m, m+8) and the weights must be laid out 8 gates
 * then 8 ups, repeating every 16 output channels. In the epilogue's visit() fragment that
 * lands as [G,G,U,U] repeating: the up partner of a gate slot is at slot +2, and slot +1 is
 * the next token. Verified over all 256 threads for tile shapes <128,48,256>, <128,64,256>
 * and <128,128,256>.
 **************************************************************************************************/
#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

namespace sglang::epilogue::fusion {

/// silu(gate * acc_scale) * (up * acc_scale) / out_scale, clamped to the e4m3 range.
///
/// Both scales arrive as device pointers to a single float, matching how the unfused path
/// passes them (a1_scale.float() into the alpha channel, a2_scale.float() into the separate
/// quant kernel). A null pointer means 1.0, which keeps the standalone harnesses simple.
template <class ElementCompute_>
struct Sm90FusedSwiGLUQuant {
  using ElementCompute = ElementCompute_;

  static constexpr int kPairSlot = 2;      // up partner offset inside a visit() fragment
  static constexpr float kFp8Max = 448.f;  // e4m3 max normal

  struct SharedStorage {};

  struct Arguments {
    float const* acc_scale_ptr = nullptr;  // a1_scale -- applied BEFORE silu
    float const* out_scale_ptr = nullptr;  // a2_scale -- output is divided by this
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return args;
  }
  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) {
    return true;
  }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }
  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const&, Arguments const&, void*, cudaStream_t, cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool is_producer_load_needed() const {
    return false;
  }
  CUTLASS_DEVICE bool is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE Sm90FusedSwiGLUQuant() {}
  CUTLASS_HOST_DEVICE Sm90FusedSwiGLUQuant(Params const& params, SharedStorage const&) : params(params) {}
  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(cutlass::epilogue::fusion::ProducerLoadArgs<Args...> const&) {
    return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
  }

  struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    float acc_scale, inv_out_scale;
    CUTLASS_DEVICE ConsumerStoreCallbacks(float a, float b) : acc_scale(a), inv_out_scale(b) {}

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<ElementCompute, FragmentSize> visit(//input：accumulator fragment(fp32)
        cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc, int, int, int) {
      static_assert(FragmentSize % 4 == 0, "fragment must be a whole number of [G,G,U,U] quads");
      cutlass::Array<ElementCompute, FragmentSize> out;
      CUTLASS_PRAGMA_UNROLL
      for (int b = 0; b < FragmentSize; b += 4) {  // frg_acc[i]
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kPairSlot; ++j) {
          float g = static_cast<float>(frg_acc[b + j]) * acc_scale;
          float u = static_cast<float>(frg_acc[b + j + kPairSlot]) * acc_scale;
          // This costs two MUFU under -use_fast_math (EX2 for the exponential, RCP for
          // the divide), and the epilogue is SFU-bound -- Hopper issues 128 FP32 ops per
          // SM-cycle but only 16 MUFU. The one-MUFU rewrite via the exact identity
          // silu(g) = h*(1 + tanh(h)), h = g/2, is therefore tempting. It is WRONG, and
          // was measured wrong, not reasoned wrong:
          //
          //   h*(1 + tanh(h)) for h < 0 subtracts two numbers of magnitude |h|. At
          //   g = -16 the true silu is -1.8e-6 and the tanh form computes -8 + 7.9999995,
          //   which has no correct digits; past |h| ~ 8.3 tanh.approx returns exactly
          //   1.0f and silu collapses to exactly 0. tests/test_w4a8_swiglu.py's
          //   saturation sweep catches it: 3848 elements over 1 ulp, and against a
          //   float64 reference the tanh path is the worse of the two on 100% of them.
          //
          // No one-MUFU form exists. sigmoid(g) for very negative g *is* e^g, and tanh
          // has already saturated that information away; the symmetry route
          // silu(-|g|) = silu(|g|) - |g| cancels identically. A threshold branch back to
          // the exp form gets predicated into three MUFU, i.e. slower than this.
          //
          // The whole tanh rewrite was worth ~26 us/layer, 0.25% of TTFT. Not a trade
          // worth making against the fusion's own selling point, which is that it stays
          // in fp32 and so is never less accurate than the Triton kernel it replaces.
          float y = (g / (1.f + __expf(-g))) * u * inv_out_scale;
          y = fmaxf(fminf(y, kFp8Max), -kFp8Max);
          out[b + j] = static_cast<ElementCompute>(y);
          // The up slot is dropped by the contracted epilogue, so this value is dead. It
          // must NOT be a copy of `y`: ptxas elides a store of the same variable into two
          // slots of the returned Array, and the elided lanes then read back as zero.
          out[b + j + kPairSlot] = static_cast<ElementCompute>(-y);
        }
      }
      return out;
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(cutlass::epilogue::fusion::ConsumerStoreArgs<Args...> const&) {
    // Dereference once per tile rather than once per fragment.
    float const acc_scale = params.acc_scale_ptr ? *params.acc_scale_ptr : 1.f;
    float const out_scale = params.out_scale_ptr ? *params.out_scale_ptr : 1.f;
    return ConsumerStoreCallbacks{acc_scale, 1.f / out_scale};
  }
};

}  // namespace sglang::epilogue::fusion
