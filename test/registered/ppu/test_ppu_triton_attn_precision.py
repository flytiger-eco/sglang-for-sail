"""PPU Triton attention non-power-of-2 head dim precision guard.

Guards SDK 2.1.1 regression #3 (internal bug report): the unified extend
attention kernel diverges from the 2-stage kernel beyond atol=0.15 at
config (B=8, N_CTX=256, H_Q=64, H_KV=8, D=80) -- a non-power-of-2 head
dim. D=128 configs pass. The regression has history: failed on sgl-kernel
0.4.1, fixed in 0.4.2.post2 (verified 7/7), regressed again in 0.4.3.

The shared test registered/attention/test_triton_attention_kernels.py
handles this by REMOVING the D=80 config from its list when
is_ppu_platform(). That hides the issue: nobody notices when the fix
lands, and nothing guards against another regression. Per decision doc
section 8.1 (W6-3), this native test instead runs D=80 EXPLICITLY:

- test_d80_* is marked xfail(strict=True) with the reason documented.
  While the bug exists it reports xfail; once the SDK fixes it, the test
  passes -> strict xfail turns that into an XPASS failure, which is the
  signal to delete the marker and restore the config upstream.
- The comparison is always done on a WARM Triton cache: a cold cache
  takes a different codegen path on first compile, which made the
  original measurements look flaky (3 consecutive runs: 2/1/1 failed).
  Each test therefore runs the kernel pair once to warm up, seeds, and
  only then asserts.

Usage:
python3 -m pytest test/registered/ppu/test_ppu_triton_attn_precision.py -v
"""

import sys

import pytest
import torch

from sglang.srt.layers.attention.triton_ops.extend_attention import (
    build_unified_kv_indices,
    extend_attention_fwd,
    extend_attention_fwd_unified,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=30, suite="nightly-1-ppu", nightly=True)

_ATOL = 0.15
_RTOL = 0.15


def _run_unified_vs_regular(B: int, N_CTX: int, H_Q: int, H_KV: int, D: int):
    """Run the 2-stage and unified extend attention kernels on one random
    instance (mirrors test_triton_attention_kernels.py) and return both
    outputs."""
    dtype = torch.bfloat16
    device = get_device()

    b_seq_len_prefix = torch.randint(
        1, N_CTX // 2, (B,), dtype=torch.int32, device=device
    )
    b_seq_len_extend = torch.randint(
        1, N_CTX // 2, (B,), dtype=torch.int32, device=device
    )
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
    kv_indices = torch.zeros(
        (b_seq_len_prefix.sum().item(),), dtype=torch.int64, device=device
    )
    for i in range(B):
        kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
            b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i]
        )

    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    k_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device=device
    ).normal_(mean=0.1, std=0.2)
    v_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device=device
    ).normal_(mean=0.1, std=0.2)

    k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
    v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
    q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)
    for i in range(B):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        v_extend[extend_start:extend_end] = v_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        q_extend[extend_start:extend_end] = torch.empty(
            (b_seq_len_extend[i], H_Q, D), dtype=dtype, device=device
        ).normal_(mean=0.1, std=0.2)

    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
    qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

    o_regular = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)
    extend_attention_fwd(
        q_extend,
        k_extend,
        v_extend,
        o_regular,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask=None,
        is_causal=True,
        mask_indptr=None,
        max_len_extend=max_len_extend,
        k_scale=1.0,
        v_scale=1.0,
    )

    extend_kv_indices = torch.arange(
        total_token_num - extend_token_num,
        total_token_num,
        dtype=torch.int64,
        device=device,
    )
    extend_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
    extend_start_loc[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    unified_kv_indptr, unified_kv_indices, prefix_lens = build_unified_kv_indices(
        kv_indptr,
        kv_indices,
        extend_start_loc,
        b_seq_len_extend,
        extend_kv_indices,
        B,
    )

    o_unified = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)
    extend_attention_fwd_unified(
        q_extend,
        o_unified,
        k_buffer,
        v_buffer,
        1.0,
        1.0,
        qo_indptr,
        unified_kv_indptr,
        unified_kv_indices,
        prefix_lens,
        max_len_extend=max_len_extend,
        custom_mask=None,
        mask_indptr=None,
        sm_scale=None,
        logit_cap=0.0,
        is_causal=True,
    )
    return o_regular, o_unified


class TestPPUTritonAttnPrecision(CustomTestCase):
    def _warm_then_assert(self, B, N_CTX, H_Q, H_KV, D):
        # Warm pass: populate the Triton cache. A cold cache takes a
        # different codegen path on first compile (documented in the bug
        # report as the cause of the 2/1/1 varying failure counts), so
        # asserting on the first run would present a false flaky.
        torch.manual_seed(0)
        _run_unified_vs_regular(B, N_CTX, H_Q, H_KV, D)

        # Assert pass on the warm cache.
        torch.manual_seed(0)
        o_regular, o_unified = _run_unified_vs_regular(B, N_CTX, H_Q, H_KV, D)
        max_diff = (o_regular - o_unified).abs().max().item()
        self.assertTrue(
            torch.allclose(o_regular, o_unified, rtol=_RTOL, atol=_ATOL),
            f"Unified kernel output differs from 2-stage kernel "
            f"(B={B}, N_CTX={N_CTX}, H_Q={H_Q}, H_KV={H_KV}, D={D}). "
            f"Max diff: {max_diff}",
        )

    def test_d128_control(self):
        """Power-of-2 head dim: passes on every stack measured; serves as
        the control that the comparison harness itself works."""
        self._warm_then_assert(4, 512, 32, 8, 128)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "SDK 2.1.1 regression #3: unified vs 2-stage kernel exceeds "
            "atol=0.15 at D=80 (non-power-of-2 head dim) on PPU. Failed on "
            "sgl-kernel 0.4.1, fixed in 0.4.2.post2, regressed in 0.4.3. "
            "Strict xfail: when this starts passing (XPASS), the SDK fix "
            "has landed -- delete this marker and restore the D=80 config "
            "in test_triton_attention_kernels.py."
        ),
    )
    def test_d80_non_power_of_two(self):
        """The exact regression config (B=8, N_CTX=256, H_Q=64, H_KV=8, D=80),
        run explicitly instead of being removed from the shared config list."""
        self._warm_then_assert(8, 256, 64, 8, 80)


# CI executes this file as ``python3 <file> -f`` (ci_utils.run_unittest_files),
# so the runner selected below decides whether the xfail(strict=True) marker
# above is honoured -- i.e. whether the strict-xfail semantics described in the
# module docstring actually hold.
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
