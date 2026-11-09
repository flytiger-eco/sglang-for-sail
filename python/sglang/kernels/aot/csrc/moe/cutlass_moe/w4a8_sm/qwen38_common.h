// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

inline void check_qwen38_w4a8(
    torch::Tensor const& d,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scale,
    torch::Tensor const& b_scale,
    torch::Tensor const& offsets,
    torch::Tensor const& problems,
    torch::Tensor const& as,
    torch::Tensor const& bs,
    torch::Tensor const& ds,
    torch::Tensor const& ss,
    bool fused,
    int packed_scales) {
  TORCH_CHECK(a.is_cuda(), "Qwen38 W4A8 requires CUDA tensors");
  for (auto const* t : {&d, &b, &a_scale, &b_scale, &offsets, &problems, &as, &bs, &ds, &ss}) {
    TORCH_CHECK(t->device() == a.device(), "all W4A8 tensors must be on the same CUDA device");
    TORCH_CHECK(t->is_contiguous(), "all W4A8 tensors must be contiguous");
  }
  TORCH_CHECK(a.is_contiguous() && (a.dim() == 2 || a.dim() == 3), "A must be contiguous 2D/3D");
  TORCH_CHECK(
      d.dim() == a.dim() && d.sizes().slice(0, d.dim() - 1) == a.sizes().slice(0, a.dim() - 1),
      "A and D must have matching row dimensions");
  TORCH_CHECK(b.dim() == 3 && b.size(0) > 0 && b.size(0) <= 1024, "B must have 1..1024 experts");
  auto e = b.size(0), k = a.size(-1), n = b.size(1);
  TORCH_CHECK(
      (fused && n == 1280 && k == 2560 && d.size(-1) == 640) || (!fused && n == 2560 && k == 640 && d.size(-1) == 2560),
      "Qwen38 W4A8 supports gate-up N1280/K2560 and down N2560/K640 only");
  TORCH_CHECK(b.size(2) * 2 == k, "packed weight K does not match A");
  TORCH_CHECK(a.dim() != 3 || a.size(0) == e, "masked layout must have E leading groups");
  TORCH_CHECK(
      a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kInt8,
      "expected FP8 E4M3 activations and packed int8 weights");
  TORCH_CHECK(d.scalar_type() == (fused ? torch::kFloat8_e4m3fn : torch::kBFloat16), "wrong output dtype");
  TORCH_CHECK(
      a_scale.scalar_type() == torch::kFloat32 && a_scale.numel() == 1, "activation scale must be one FP32 value");
  TORCH_CHECK(
      b_scale.scalar_type() == torch::kBFloat16 && b_scale.dim() == 3 && b_scale.size(0) == e &&
          b_scale.size(1) == k / (128 * packed_scales) && b_scale.size(2) == n * packed_scales,
      "weight scale packing does not match selected tile");
  TORCH_CHECK(
      offsets.scalar_type() == torch::kInt32 && offsets.dim() == 1 && offsets.numel() == e,
      "expert_offsets must be int32 [E]");
  TORCH_CHECK(
      problems.scalar_type() == torch::kInt32 && problems.dim() == 2 && problems.size(0) == e && problems.size(1) == 3,
      "problem_sizes must be int32 [E,3] in N,M,K order");
  for (auto const* t : {&as, &bs, &ds, &ss}) {
    TORCH_CHECK(
        t->scalar_type() == torch::kInt64 && t->dim() == 2 && t->size(0) == e && t->size(1) == 3,
        "strides must be int64 [E,3]");
  }
  auto props = at::cuda::getDeviceProperties(a.get_device());
  TORCH_CHECK(props->major == 9, "Qwen38 W4A8 SM kernels require Hopper SM90");
}
