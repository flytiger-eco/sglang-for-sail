"""Tests for ``silu_and_mul_masked_post_quant_mxfp4`` (EP, 3D MXFP4)."""
import sys

import pytest
import torch

from sglang.srt.utils import is_ppu

pytestmark = pytest.mark.skipif(not is_ppu(), reason="PPU-only kernel")


def _alloc_ep_outputs(E: int, T: int, twoN: int, device: torch.device):
    N = twoN // 2
    output = torch.empty((E, T, N // 2), dtype=torch.uint8, device=device)
    s_half = (N // 32) // 2
    output_scale = torch.empty(
        (E, s_half, T), dtype=torch.uint16, device=device
    )
    return output, output_scale


_EP_SHAPES = [
    (4, 32, 2048),
    (8, 64, 2048),
    (4, 128, 4096),
]


@pytest.mark.parametrize("E,T,twoN", _EP_SHAPES)
@pytest.mark.parametrize("swiglu_limit", [None, 8.0])
def test_shapes(E: int, T: int, twoN: int, swiglu_limit: float | None) -> None:
    from sglang.jit_kernel.silu_mul_quant import silu_and_mul_masked_post_quant_mxfp4

    torch.manual_seed(E * T)
    inp = torch.randn((E, T, twoN), dtype=torch.bfloat16, device="cuda")
    output, output_scale = _alloc_ep_outputs(E, T, twoN, inp.device)
    masked_m = torch.randint(
        low=1, high=T + 1, size=(E,), dtype=torch.int32, device="cuda"
    )

    silu_and_mul_masked_post_quant_mxfp4(
        inp, output, output_scale, masked_m, swiglu_limit=swiglu_limit
    )

    assert output.dtype == torch.uint8
    assert output_scale.dtype == torch.uint16
    assert output.shape == (E, T, twoN // 4)


def test_padded_rows_untouched() -> None:
    from sglang.jit_kernel.silu_mul_quant import silu_and_mul_masked_post_quant_mxfp4

    E, T, twoN = 4, 32, 2048
    torch.manual_seed(0)
    inp = torch.randn((E, T, twoN), dtype=torch.bfloat16, device="cuda")
    output, output_scale = _alloc_ep_outputs(E, T, twoN, inp.device)

    output.fill_(0xAA)
    output_scale.fill_(0xBEEF)

    valid = T // 2
    masked_m = torch.full((E,), valid, dtype=torch.int32, device="cuda")

    silu_and_mul_masked_post_quant_mxfp4(
        inp, output, output_scale, masked_m, swiglu_limit=None
    )

    pad_rows = output[:, valid:, :]
    assert torch.all(pad_rows == 0xAA), (
        "kernel wrote into padded rows of the EP output"
    )
    pad_scale = output_scale[:, :, valid:]
    assert torch.all(pad_scale == 0xBEEF), (
        "kernel wrote into padded columns of the EP output_scale"
    )


def test_determinism() -> None:
    from sglang.jit_kernel.silu_mul_quant import silu_and_mul_masked_post_quant_mxfp4

    E, T, twoN = 4, 32, 2048
    torch.manual_seed(11)
    inp = torch.randn((E, T, twoN), dtype=torch.bfloat16, device="cuda")
    masked_m = torch.randint(1, T + 1, (E,), dtype=torch.int32, device="cuda")

    out_a, scale_a = _alloc_ep_outputs(E, T, twoN, inp.device)
    out_b, scale_b = _alloc_ep_outputs(E, T, twoN, inp.device)
    silu_and_mul_masked_post_quant_mxfp4(
        inp, out_a, scale_a, masked_m, swiglu_limit=6.0
    )
    silu_and_mul_masked_post_quant_mxfp4(
        inp, out_b, scale_b, masked_m, swiglu_limit=6.0
    )

    masked_cpu = masked_m.cpu().tolist()
    for e, m in enumerate(masked_cpu):
        assert torch.equal(out_a[e, :m], out_b[e, :m])
        assert torch.equal(scale_a[e, :, :m], scale_b[e, :, :m])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
