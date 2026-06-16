"""Tests for ``silu_and_mul_post_quant_mxfp4`` (TP, 2D MXFP4)."""
import sys

import pytest
import torch

from sglang.srt.utils import is_ppu

pytestmark = pytest.mark.skipif(not is_ppu(), reason="PPU-only kernel")


_TP_SHAPES = [
    (1, 1024),
    (8, 1024),
    (64, 1024),
    (128, 2048),
    (256, 4096),
]


@pytest.mark.parametrize("N,H", _TP_SHAPES)
@pytest.mark.parametrize("swiglu_limit", [None, 12.0])
def test_shapes(N: int, H: int, swiglu_limit: float | None) -> None:
    from sglang.jit_kernel.silu_mul_quant import silu_and_mul_post_quant_mxfp4

    torch.manual_seed(N * 17 + H)
    gateup = torch.randn((N, 2 * H), dtype=torch.bfloat16, device="cuda")
    out_quant, out_scale_view = silu_and_mul_post_quant_mxfp4(
        gateup, swiglu_limit=swiglu_limit
    )

    assert out_quant.shape == (N, H // 2)
    assert out_quant.dtype == torch.uint8

    s_pairs = (H // 32) // 2
    assert out_scale_view.shape == (N, s_pairs)
    assert out_scale_view.dtype == torch.uint16


def test_empty() -> None:
    from sglang.jit_kernel.silu_mul_quant import silu_and_mul_post_quant_mxfp4

    gateup = torch.empty((0, 2048), dtype=torch.bfloat16, device="cuda")
    out_quant, out_scale_view = silu_and_mul_post_quant_mxfp4(gateup)
    assert out_quant.shape == (0, 512)
    s_pairs = (1024 // 32) // 2
    assert out_scale_view.shape == (0, s_pairs)


def test_determinism() -> None:
    from sglang.jit_kernel.silu_mul_quant import silu_and_mul_post_quant_mxfp4

    torch.manual_seed(7)
    gateup = torch.randn((128, 4096), dtype=torch.bfloat16, device="cuda")
    a_q, a_s = silu_and_mul_post_quant_mxfp4(gateup, swiglu_limit=10.0)
    b_q, b_s = silu_and_mul_post_quant_mxfp4(gateup, swiglu_limit=10.0)
    assert torch.equal(a_q, b_q)
    assert torch.equal(a_s.contiguous(), b_s.contiguous())


def test_clamp_no_op() -> None:
    from sglang.jit_kernel.silu_mul_quant import silu_and_mul_post_quant_mxfp4

    torch.manual_seed(123)
    gateup = torch.randn((64, 2048), dtype=torch.bfloat16, device="cuda") * 0.1
    a_q, a_s = silu_and_mul_post_quant_mxfp4(gateup, swiglu_limit=None)
    b_q, b_s = silu_and_mul_post_quant_mxfp4(gateup, swiglu_limit=1000.0)
    assert torch.equal(a_q, b_q)
    assert torch.equal(a_s.contiguous(), b_s.contiguous())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
