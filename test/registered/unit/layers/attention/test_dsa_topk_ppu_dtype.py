"""CPU checks for PPU BF16 prefill dispatch to the DeepSelect wheel adapter."""

import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.attention.dsa import dsa_topk_backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize("width", [355, 356, 357, 512])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ppu", [False, True])
def test_ragged_topk_preserves_scores_and_platform_boundary(width, dtype, ppu):
    rows, topk = 3, 8
    torch.manual_seed(width)
    logits = torch.randn(rows, 1024, dtype=dtype)[:, :width]
    before = logits.clone()
    starts = torch.tensor([1, 2, 3], dtype=torch.int32)
    lengths = torch.tensor([width - 1, width - 3, width - 5], dtype=torch.int32)
    offsets = torch.tensor([100, 200, 300], dtype=torch.int32)
    expected = torch.stack(
        [
            before[row, start : start + length].float().topk(topk).indices + offset
            for row, (start, length, offset) in enumerate(zip(starts, lengths, offsets))
        ]
    ).to(torch.int32)

    def kernel(scores, seq_lens, *, out_offsets, out_indices, row_starts):
        assert scores is logits
        assert seq_lens is lengths
        assert out_offsets is offsets
        assert row_starts is starts
        for row, (start, length, offset) in enumerate(zip(starts, lengths, offsets)):
            out_indices[row] = (
                scores[row, start : start + length].float().topk(topk).indices + offset
            )

    module_name = "sglang.kernels.ops.attention.dsv4.topk"
    kernel_module = ModuleType(module_name)
    kernel_module.topk_transform_ragged_v2 = kernel
    native_name = "sglang.kernels.ops.attention.dsa.deep_select"
    native_module = ModuleType(native_name)
    native_module.topk_transform_ragged = Mock(return_value=expected)
    with (
        patch.dict(
            sys.modules, {module_name: kernel_module, native_name: native_module}
        ),
        patch.object(dsa_topk_backend, "is_sail", return_value=ppu),
    ):
        out = dsa_topk_backend._topk_transform_v2_ragged(
            logits, lengths, topk, offsets, starts
        )

    assert out.dtype == torch.int32
    assert torch.equal(out, expected)
    assert torch.equal(logits, before)
    if ppu and dtype == torch.bfloat16:
        native_module.topk_transform_ragged.assert_called_once_with(
            logits, lengths, topk, offsets, starts
        )
    else:
        native_module.topk_transform_ragged.assert_not_called()


def test_ppu_ragged_topk_reported_shape():
    rows, width, topk = 356, 356, 2048
    logits = torch.randn(rows, 512, dtype=torch.bfloat16)[:, :width]
    lengths = torch.arange(1, rows + 1, dtype=torch.int32)
    starts = torch.zeros_like(lengths)
    offsets = torch.arange(rows, dtype=torch.int32) * width
    columns = torch.arange(topk, dtype=torch.int32).expand(rows, -1)
    expected = torch.where(columns < lengths[:, None], columns + offsets[:, None], -1)

    def kernel(scores, seq_lens, requested_topk, out_offsets, row_starts):
        assert scores.shape == (356, 356)
        assert scores is logits
        assert scores.dtype == torch.bfloat16
        assert scores.stride() == (512, 1)
        assert requested_topk == topk
        assert seq_lens is lengths
        assert out_offsets is offsets
        assert row_starts is starts
        return expected

    module_name = "sglang.kernels.ops.attention.dsa.deep_select"
    kernel_module = ModuleType(module_name)
    kernel_module.topk_transform_ragged = kernel
    with (
        patch.dict(sys.modules, {module_name: kernel_module}),
        patch.object(dsa_topk_backend, "is_sail", return_value=True),
        dsa_topk_backend.envs.SGLANG_DSA_FUSE_TOPK.override(True),
        dsa_topk_backend.envs.SGLANG_OPT_USE_TOPK_V2.override(True),
    ):
        out = dsa_topk_backend.DSATopKBackend.SGL_KERNEL.topk_transform(
            logits,
            lengths,
            topk,
            dsa_topk_backend.TopkTransformMethod.RAGGED,
            attn_metadata=None,
            topk_indices_offset=offsets,
            row_starts=starts,
        )

    assert torch.equal(out, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("bf16_topk", [False, True])
def test_paged_prefill_keeps_bf16_kernel_and_fp32_fallback(ppu, bf16_topk):
    from types import SimpleNamespace

    logits = torch.tensor([[1, 3, 2, 5, 4]], dtype=torch.bfloat16)
    lengths = torch.tensor([5], dtype=torch.int32)
    starts = torch.zeros_like(lengths)
    metadata = SimpleNamespace(
        page_table_1=torch.arange(5, dtype=torch.int32).view(1, -1),
        token_to_batch_idx=None,
    )
    expected = torch.tensor([[3, 4]], dtype=torch.int32)
    kernels = ModuleType("sgl_kernel")
    kernels.fast_topk_transform_fused = Mock(return_value=expected)
    kernels.fast_topk_transform_ragged_fused = Mock()
    with (
        patch.dict(sys.modules, {"sgl_kernel": kernels}),
        patch.object(dsa_topk_backend, "is_sail", return_value=ppu),
        patch.object(
            dsa_topk_backend.DSATopKBackend,
            "_topk_transform_dsv4_bf16",
            return_value=expected,
        ) as bf16_kernel,
        dsa_topk_backend.envs.SGLANG_DSA_FUSE_TOPK.override(True),
        dsa_topk_backend.envs.SGLANG_OPT_USE_TOPK_V2.override(False),
        dsa_topk_backend.envs.SGLANG_DSA_USE_DSV4_BF16_TOPK.override(bf16_topk),
    ):
        result = dsa_topk_backend.DSATopKBackend.SGL_KERNEL.topk_transform(
            logits,
            lengths,
            2,
            dsa_topk_backend.TopkTransformMethod.PAGED,
            metadata,
            row_starts=starts,
        )
    assert result is expected
    if ppu and bf16_topk:
        bf16_kernel.assert_called_once()
        assert bf16_kernel.call_args.kwargs["logits"] is logits
        kernels.fast_topk_transform_fused.assert_not_called()
    else:
        bf16_kernel.assert_not_called()
        kernels.fast_topk_transform_fused.assert_called_once()
        scores = kernels.fast_topk_transform_fused.call_args.kwargs["score"]
        if ppu:
            assert scores.dtype == torch.float32
            assert scores.stride(0) % 4 == 0
            assert torch.equal(scores, logits.float())
        else:
            assert scores is logits
