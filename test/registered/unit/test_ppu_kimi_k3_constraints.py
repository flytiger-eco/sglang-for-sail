"""CPU checks for PPU-only Kimi-K3 backend restrictions."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.kernels.jit.utils import arch as jit_arch
from sglang.kernels.ops.attention import kda_fused_decode
from sglang.srt.arg_groups import overrides as overrides_module
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear import kda_backend
from sglang.srt.layers.attention.linear.kernels import kda_flashkda
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16DeepGemmMoE,
)
from sglang.srt.models import kimi_k3


def test_kda_dispatcher_forces_unsupported_backends_to_triton_on_ppu():
    triton_kernel = type("TritonKernel", (), {"supports_packed_decode": False})()
    with (
        patch.object(kda_backend, "is_ppu", return_value=True),
        patch.object(kda_backend, "TritonKDAKernel", return_value=triton_kernel),
        patch.object(kda_backend, "rank0_log"),
    ):
        dispatcher = kda_backend.KDAKernelDispatcher(
            LinearAttnKernelBackend.FLASHINFER,
            LinearAttnKernelBackend.NVIDIA_KDA,
            LinearAttnKernelBackend.NV_CUTEDSL,
        )

    assert dispatcher.decode_kernel is triton_kernel
    assert dispatcher.extend_kernel is triton_kernel
    assert dispatcher.verify_kernel is triton_kernel


def test_kda_dispatcher_honors_flashkda_prefill_backend_on_ppu():
    triton_kernel = type("TritonKernel", (), {"supports_packed_decode": False})()
    flashkda_kernel = object()
    with (
        patch.object(kda_backend, "is_ppu", return_value=True),
        patch.object(kda_backend, "TritonKDAKernel", return_value=triton_kernel),
        patch.object(
            kda_flashkda, "FlashKDAKernel", return_value=flashkda_kernel
        ) as kernel_class,
        patch.object(kda_backend, "rank0_log"),
    ):
        dispatcher = kda_backend.KDAKernelDispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.FLASHKDA,
            LinearAttnKernelBackend.TRITON,
        )

    assert dispatcher.decode_kernel is triton_kernel
    assert dispatcher.extend_kernel is flashkda_kernel
    assert dispatcher.verify_kernel is triton_kernel
    kernel_class.assert_called_once_with()


def test_kda_extend_dispatcher_normalizes_kernel_return_contract():
    """Aligned-track batches request no intermediate states; CuteDSL still
    returns ``(out, None)`` while Triton/FlashKDA return a bare tensor. Both
    must come back as ``(out, h-or-None)`` so ``forward_extend`` can unpack
    unconditionally instead of leaving a tuple in tensor ops."""
    triton_kernel = type("TritonKernel", (), {"supports_packed_decode": False})()
    out = torch.zeros(1, 2, 4)
    with (
        patch.object(kda_backend, "is_ppu", return_value=False),
        patch.object(kda_backend, "TritonKDAKernel", return_value=triton_kernel),
        patch.object(kda_backend, "rank0_log"),
    ):
        dispatcher = kda_backend.KDAKernelDispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.TRITON,
        )

    extend_kwargs = dict(
        ssm_states=sentinel.ssm_states,
        cache_indices=sentinel.cache_indices,
        query_start_loc=sentinel.query_start_loc,
        return_intermediate_states=False,
    )

    dispatcher.extend_kernel = SimpleNamespace(
        supports_safe_gate=True, extend=lambda *a, **k: (out, None)
    )
    got, h = dispatcher.extend(
        sentinel.q, sentinel.k, sentinel.v, sentinel.g, sentinel.beta, **extend_kwargs
    )
    assert got is out
    assert h is None

    dispatcher.extend_kernel = SimpleNamespace(
        supports_safe_gate=True, extend=lambda *a, **k: out
    )
    got, h = dispatcher.extend(
        sentinel.q, sentinel.k, sentinel.v, sentinel.g, sentinel.beta, **extend_kwargs
    )
    assert got is out
    assert h is None


@pytest.mark.parametrize("extend_kwargs", [{}, {"return_intermediate_states": False}])
def test_ppu_flashkda_dispatches_to_platform_operator(extend_kwargs):
    num_heads = 12
    head_dim = 128
    packed_tokens = 64
    q = torch.zeros((1, packed_tokens, num_heads, head_dim), dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    g = torch.zeros_like(q)
    beta = torch.full((1, packed_tokens, num_heads), 0.5, dtype=torch.float32)
    ssm_states = torch.zeros((2, num_heads, head_dim, head_dim), dtype=torch.float32)
    cache_indices = torch.tensor([1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, packed_tokens], dtype=torch.int32)
    flashkda_fwd = Mock(
        side_effect=lambda **kwargs: (
            kwargs["out"].fill_(0.25),
            kwargs["final_state"].copy_(kwargs["initial_state"] + 1),
        )
    )
    pla_module = ModuleType("pla")
    prefill_module = ModuleType("pla.prefill")
    flashkda_module = ModuleType("pla.prefill.flashkdapro")
    pla_module.prefill = prefill_module
    prefill_module.flashkdapro = flashkda_module
    flashkda_module.flashkda_fwd = flashkda_fwd

    with (
        patch.object(kda_flashkda, "is_ppu", return_value=True),
        patch.object(kda_flashkda, "_load_flash_kda") as load_flash_kda,
        envs.SGLANG_SAIL_FLA_CUDA.override(True),
        envs.SGLANG_SAIL_PLA_CUDA.override(True),
        patch.dict(
            sys.modules,
            {
                "pla": pla_module,
                "pla.prefill": prefill_module,
                "pla.prefill.flashkdapro": flashkda_module,
            },
        ),
    ):
        output = kda_flashkda.FlashKDAKernel().extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=torch.zeros((1, 1, num_heads, 1), dtype=torch.float32),
            dt_bias=torch.zeros(num_heads * head_dim, dtype=torch.float32),
            lower_bound=-5.0,
            **extend_kwargs,
        )

    assert isinstance(output, torch.Tensor)
    assert output.shape == q.shape
    assert torch.all(output == 0.25)
    assert torch.all(ssm_states[1] == 1)
    assert torch.count_nonzero(ssm_states[0]) == 0
    load_flash_kda.assert_not_called()
    flashkda_fwd.assert_called_once()
    kwargs = flashkda_fwd.call_args.kwargs
    assert kwargs["q"].shape == (packed_tokens, num_heads, head_dim)
    assert kwargs["beta"].shape == (packed_tokens, num_heads)
    assert kwargs["beta"].dtype == torch.bfloat16
    assert kwargs["A_log"].shape == (num_heads,)
    assert kwargs["dt_bias"].shape == (num_heads, head_dim)
    assert kwargs["cu_seqlens"].dtype == torch.int64


@pytest.mark.parametrize(
    "track_ssm, lower_bound, is_spec_decode",
    [(True, -5.0, False), (False, None, False), (False, -5.0, True)],
)
def test_ppu_flashkda_fallback_preserves_return_contract(
    track_ssm, lower_bound, is_spec_decode
):
    kernel = kda_flashkda.FlashKDAKernel()
    output = torch.zeros((1, 64, 12, 128), dtype=torch.bfloat16)
    result = (output, sentinel.intermediate_states) if track_ssm else output
    track_state = sentinel.track_state if track_ssm else None
    track_chunk_idx = sentinel.track_chunk_idx if track_ssm else None

    with (
        patch.object(kda_flashkda, "is_ppu", return_value=True),
        patch.object(kernel, "_flashkda_extend") as fused_extend,
        envs.SGLANG_SAIL_FLA_CUDA.override(True),
        envs.SGLANG_SAIL_PLA_CUDA.override(True),
        patch(
            "sglang.kernels.ops.attention.fla.kda.chunk_kda", return_value=result
        ) as chunk_kda,
    ):
        actual = kernel.extend(
            sentinel.q,
            sentinel.k,
            sentinel.v,
            sentinel.g,
            sentinel.beta,
            ssm_states=sentinel.ssm_states,
            cache_indices=sentinel.cache_indices,
            query_start_loc=sentinel.query_start_loc,
            A_log=sentinel.A_log,
            dt_bias=sentinel.dt_bias,
            lower_bound=lower_bound,
            beta_is_raw=True,
            is_spec_decode=is_spec_decode,
            return_intermediate_states=track_ssm,
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
        )

    assert actual is result
    fused_extend.assert_not_called()
    chunk_kda.assert_called_once_with(
        q=sentinel.q,
        k=sentinel.k,
        v=sentinel.v,
        g=sentinel.g,
        beta=sentinel.beta,
        initial_state=sentinel.ssm_states,
        initial_state_indices=sentinel.cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=sentinel.query_start_loc,
        A_log=sentinel.A_log,
        dt_bias=sentinel.dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=True,
        output_intermediate_states=track_ssm,
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
    )


def test_ppu_flashkda_only_bypasses_cuda_sequence_length_fallback():
    kernel = kda_flashkda.FlashKDAKernel()
    short_query_start_loc = torch.tensor([0, 48], dtype=torch.int32)
    long_query_start_loc = torch.tensor([0, 4096], dtype=torch.int32)

    with patch.object(kda_flashkda, "is_ppu", return_value=True):
        assert not kernel._should_fall_back(-5.0, False, short_query_start_loc, [48])
        assert not kernel._should_fall_back(-5.0, False, long_query_start_loc, [4096])
        assert kernel._should_fall_back(None, False, short_query_start_loc, [48])
        assert kernel._should_fall_back(-5.0, True, short_query_start_loc, [48])

    with patch.object(kda_flashkda, "is_ppu", return_value=False):
        assert kernel._should_fall_back(-5.0, False, short_query_start_loc, [48])
        assert kernel._should_fall_back(-5.0, False, long_query_start_loc, [4096])


def test_ppu_pla_prefill_gate_requires_both_sail_gates():
    with patch.object(kda_flashkda, "is_ppu", return_value=True):
        with (
            envs.SGLANG_SAIL_FLA_CUDA.override(True),
            envs.SGLANG_SAIL_PLA_CUDA.override(True),
        ):
            assert kda_flashkda.ppu_pla_prefill_gate_open()
        with (
            envs.SGLANG_SAIL_FLA_CUDA.override(True),
            envs.SGLANG_SAIL_PLA_CUDA.override(False),
        ):
            assert not kda_flashkda.ppu_pla_prefill_gate_open()
        with (
            envs.SGLANG_SAIL_FLA_CUDA.override(False),
            envs.SGLANG_SAIL_PLA_CUDA.override(True),
        ):
            assert not kda_flashkda.ppu_pla_prefill_gate_open()

    # PLA prefill is PPU-only; other platforms use the flash_kda package.
    with patch.object(kda_flashkda, "is_ppu", return_value=False):
        with (
            envs.SGLANG_SAIL_FLA_CUDA.override(True),
            envs.SGLANG_SAIL_PLA_CUDA.override(True),
        ):
            assert not kda_flashkda.ppu_pla_prefill_gate_open()


def test_closed_pla_gate_falls_back_to_triton_on_ppu():
    kernel = kda_flashkda.FlashKDAKernel()
    with (
        patch.object(kda_flashkda, "is_ppu", return_value=True),
        envs.SGLANG_SAIL_PLA_CUDA.override(False),
        patch.object(kernel, "_flashkda_extend") as fused_extend,
        patch(
            "sglang.kernels.ops.attention.fla.kda.chunk_kda",
            return_value=sentinel.triton_output,
        ) as chunk_kda,
    ):
        actual = kernel.extend(
            sentinel.q,
            sentinel.k,
            sentinel.v,
            sentinel.g,
            sentinel.beta,
            ssm_states=sentinel.ssm_states,
            cache_indices=sentinel.cache_indices,
            query_start_loc=sentinel.query_start_loc,
            A_log=sentinel.A_log,
            dt_bias=sentinel.dt_bias,
            lower_bound=-5.0,
        )

    assert actual is sentinel.triton_output
    fused_extend.assert_not_called()
    chunk_kda.assert_called_once()


def _hf_model_config(linear_attn_config):
    return SimpleNamespace(
        hf_config=SimpleNamespace(linear_attn_config=linear_attn_config)
    )


def test_ppu_pla_prefill_default_is_per_runner_and_kda_only():
    with patch.object(kda_backend, "is_ppu", return_value=True):
        with (
            envs.SGLANG_SAIL_FLA_CUDA.override(True),
            envs.SGLANG_SAIL_PLA_CUDA.override(True),
        ):
            # KDA runner (target): opts in to the SAIL PLA prefill kernel.
            assert (
                kda_backend.ppu_pla_prefill_default(
                    _hf_model_config({"kda_layers": [0, 1]})
                )
                == "flashkda"
            )
            # Heterogeneous spec draft (GDN) must NOT inherit it: GDN has no
            # FlashKDA kernel and GDNKernelDispatcher rejects that backend.
            assert kda_backend.ppu_pla_prefill_default(_hf_model_config(None)) is None
        # Either master gate off turns the default off (same pair as decode).
        with (
            envs.SGLANG_SAIL_FLA_CUDA.override(True),
            envs.SGLANG_SAIL_PLA_CUDA.override(False),
        ):
            assert (
                kda_backend.ppu_pla_prefill_default(
                    _hf_model_config({"kda_layers": [0, 1]})
                )
                is None
            )
        with (
            envs.SGLANG_SAIL_FLA_CUDA.override(False),
            envs.SGLANG_SAIL_PLA_CUDA.override(True),
        ):
            assert (
                kda_backend.ppu_pla_prefill_default(
                    _hf_model_config({"kda_layers": [0, 1]})
                )
                is None
            )
    # PLA prefill is PPU-only.
    with (
        patch.object(kda_backend, "is_ppu", return_value=False),
        envs.SGLANG_SAIL_FLA_CUDA.override(True),
        envs.SGLANG_SAIL_PLA_CUDA.override(True),
    ):
        assert (
            kda_backend.ppu_pla_prefill_default(
                _hf_model_config({"kda_layers": [0, 1]})
            )
            is None
        )


def test_pla_prefill_default_resolves_per_runner_and_explicit_backend_wins():
    from sglang.srt.layers.attention.linear import utils as linear_utils

    def _resolve(prefill_default, explicit=None):
        mamba = SimpleNamespace(
            linear_attn_backend="triton",
            linear_attn_prefill_backend=explicit,
            linear_attn_decode_backend=None,
            linear_attn_verify_backend=None,
        )
        with (
            patch.object(
                linear_utils, "get_exec", return_value=SimpleNamespace(mamba=mamba)
            ),
            patch.object(linear_utils, "rank0_log"),
        ):
            return linear_utils.resolve_linear_attn_backends(
                prefill_default=prefill_default
            )

    # KDA runner: the per-runner default reaches the dispatcher.
    assert _resolve("flashkda").prefill is LinearAttnKernelBackend.FLASHKDA
    # GDN draft runner in the same process keeps its own default (Triton).
    assert _resolve(None).prefill.is_triton()
    # Explicit --linear-attn-prefill-backend still wins over the default.
    assert _resolve("flashkda", explicit="triton").prefill.is_triton()


def test_kda_track_path_requests_intermediate_states_only_for_unaligned_rows():
    from sglang.srt.layers.attention.linear.kda_backend import (
        needs_intermediate_states,
    )

    # Chunk-aligned tracked rows snapshot the final state from ssm_states, so
    # they must not force kernels that only exist on the per-chunk-state path.
    assert not needs_intermediate_states(True, torch.empty(0, dtype=torch.int64))
    # Chunk-unaligned tracked rows do read h at the last chunk boundary.
    assert needs_intermediate_states(True, torch.tensor([3], dtype=torch.int64))
    # Untracked batches and metadata without h rows never ask for h.
    assert not needs_intermediate_states(False, torch.tensor([3], dtype=torch.int64))
    assert not needs_intermediate_states(True, None)


def _fused_decode_coverage_inputs(num_heads):
    head_dim = 128
    seg = num_heads * head_dim
    batch_size = 2
    num_slots = 3
    return (
        torch.empty(batch_size, 3 * seg, dtype=torch.bfloat16),
        torch.empty(batch_size, seg, dtype=torch.bfloat16),
        torch.empty(batch_size, num_heads, dtype=torch.bfloat16),
        torch.empty(num_slots, 3, 3 * seg, dtype=torch.bfloat16),
        torch.empty(num_slots, num_heads, head_dim, head_dim, dtype=torch.float32),
        torch.arange(batch_size, dtype=torch.int32),
        torch.empty(batch_size, seg, dtype=torch.bfloat16),
    )


def test_ppu_fused_decode_bypasses_community_coverage():
    with (
        patch.object(kda_fused_decode, "is_ppu", return_value=True),
        patch.object(kda_fused_decode, "_SUPPORTED_HEADS", {12, 24, 48, 96}),
    ):
        for num_heads in (12, 24, 48, 96):
            assert kda_fused_decode.covered(*_fused_decode_coverage_inputs(num_heads))
        assert not kda_fused_decode.covered(*_fused_decode_coverage_inputs(6))

    with (
        patch.object(kda_fused_decode, "is_ppu", return_value=False),
        patch.object(kda_fused_decode, "_SUPPORTED_HEADS", {3, 6, 12}),
    ):
        assert kda_fused_decode.covered(*_fused_decode_coverage_inputs(6))
        assert not kda_fused_decode.covered(*_fused_decode_coverage_inputs(24))


def test_ppu_fused_decode_dispatches_tp4_without_graph_padding_rows():
    num_heads = 24
    head_dim = 128
    seg = num_heads * head_dim
    mixed_qkv, a, b, conv_states, ssm_states, cache_indices, onorm_gate = (
        _fused_decode_coverage_inputs(num_heads)
    )
    cache_indices[1] = -1
    pla_forward = Mock(side_effect=lambda **kwargs: kwargs["out"].zero_())
    pla_module = ModuleType("pla")
    decode_module = ModuleType("pla.decode")
    kda_module = ModuleType("pla.decode.kda")
    pla_module.decode = decode_module
    decode_module.kda = kda_module
    kda_module.fused_kda_decode_mega_forward = pla_forward

    with (
        patch.object(kda_fused_decode, "is_ppu", return_value=True),
        patch.dict(
            sys.modules,
            {
                "pla": pla_module,
                "pla.decode": decode_module,
                "pla.decode.kda": kda_module,
            },
        ),
    ):
        output = kda_fused_decode.kda_fused_decode(
            mixed_qkv,
            a,
            b,
            conv_states,
            torch.empty(4, seg, dtype=torch.float32),
            torch.empty(4, seg, dtype=torch.float32),
            torch.empty(4, seg, dtype=torch.float32),
            torch.empty(3 * seg, dtype=torch.float32),
            torch.empty(num_heads, dtype=torch.float32),
            torch.empty(seg, dtype=torch.float32),
            onorm_gate,
            torch.empty(head_dim, dtype=torch.float32),
            ssm_states,
            cache_indices,
            scale=head_dim**-0.5,
            onorm_eps=1e-5,
            lower_bound=-5.0,
            fused_weight=torch.empty(3, 4, seg, dtype=torch.float32),
            actual_batch_size=1,
        )

    assert output.shape == (1, 2, num_heads, head_dim)
    assert torch.count_nonzero(output[:, 1:]).item() == 0
    pla_forward.assert_called_once()
    kwargs = pla_forward.call_args.kwargs
    assert kwargs["x"].shape == (1, 3 * seg)
    assert kwargs["state_indices"].tolist() == [0]
    assert kwargs["out"].shape == (1, 1, num_heads, head_dim)


def _fused_decode_attention(num_heads):
    head_dim = 128
    seg = num_heads * head_dim
    layer = SimpleNamespace(
        conv_weights=torch.empty(3 * seg, 4, dtype=torch.float32),
        bias=None,
        A_log=torch.empty(num_heads, dtype=torch.float32),
        dt_bias=torch.empty(seg, dtype=torch.float32),
        num_v_heads=num_heads,
    )
    attention = SimpleNamespace(
        attn=layer,
        o_norm=SimpleNamespace(
            weight=torch.empty(head_dim, dtype=torch.float32), eps=1e-5
        ),
        _kda_fused_decode_ready=False,
    )
    return attention, layer


def test_ppu_prepares_fused_decode_handoff_for_k3_tp_layouts():
    with (
        patch.object(kimi_k3, "is_ppu", return_value=True),
        patch.object(kimi_k3, "_is_hip", False),
    ):
        for num_heads in (12, 24, 48, 96):
            attention, layer = _fused_decode_attention(num_heads)
            kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)

            assert attention._kda_fused_decode_ready
            assert layer._k3_fused_decode_weight.shape == (
                3,
                4,
                num_heads * 128,
            )


def test_non_ppu_keeps_community_fused_decode_preparation_limit():
    with (
        patch.object(kimi_k3, "is_ppu", return_value=False),
        patch.object(kimi_k3, "_is_hip", False),
        patch.object(kimi_k3, "rank0_log"),
    ):
        attention, layer = _fused_decode_attention(24)
        kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)
        assert not attention._kda_fused_decode_ready
        assert not hasattr(layer, "_k3_fused_decode_args")

        attention, layer = _fused_decode_attention(12)
        kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)
        assert attention._kda_fused_decode_ready
        assert not hasattr(layer, "_k3_fused_decode_weight")


def test_ppu_disables_pdl_and_megamoe():
    with patch.object(jit_arch, "is_ppu_runtime", return_value=True):
        assert not jit_arch.is_arch_support_pdl.__wrapped__()

    with patch("sglang.srt.utils.common.is_ppu", return_value=True):
        try:
            overrides_module._a2a_backend_overrides(
                type("View", (), {"moe_a2a_backend": "megamoe"})()
            )
        except ValueError as exc:
            assert "MegaMoE is not supported on PPU" in str(exc)
        else:
            raise AssertionError("PPU MegaMoE backend must be rejected")


def test_ppu_w4a16_deep_gemm_moe_accepts_k3_situ():
    scheme = object.__new__(CompressedTensorsWNA16DeepGemmMoE)
    scheme.moe_runner_config = SimpleNamespace(activation="situ")
    scheme.runner = Mock()
    scheme.runner.run.return_value = sentinel.combine_input
    scheme.get_deep_gemm_quant_info = Mock(return_value=sentinel.quant_info)

    result = scheme.apply_weights(sentinel.layer, sentinel.dispatch_output)

    assert result is sentinel.combine_input
    scheme.get_deep_gemm_quant_info.assert_called_once_with(sentinel.layer)
    scheme.runner.run.assert_called_once_with(
        sentinel.dispatch_output, sentinel.quant_info
    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
