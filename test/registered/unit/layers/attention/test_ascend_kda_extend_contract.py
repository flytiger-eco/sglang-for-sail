"""Contract test: AscendKDAAttnBackend.forward_extend must unpack the
dispatcher result for untracked prefill too.

``KDAKernelDispatcher.extend`` normalizes every kernel to ``(out, h-or-None)``
(Triton/FlashKDA return a bare tensor, the CuTe DSL and Ascend extend kernels
return a tuple). The Ascend backend used to unpack only when ``track_ssm`` was
set, so an untracked prefill returned the tuple itself and broke the tensor ops
downstream. This is a CPU test: ``sgl_kernel_npu`` is stubbed with a scoped
``patch.dict(sys.modules, ...)`` while the module is imported, and the NPU conv
op is replaced with an identity, so it runs in the regular CPU unit suite.
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ASCEND_BACKEND = "sglang.srt.hardware_backend.npu.attention.ascend_kda_backend"


def _fake_npu_kernels():
    """Stand-in for ``sgl_kernel_npu`` so the Ascend module imports on CPU."""
    pkg = ModuleType("sgl_kernel_npu")
    fla = ModuleType("sgl_kernel_npu.fla")
    gate = ModuleType("sgl_kernel_npu.fla.kda_gate")
    verify = ModuleType("sgl_kernel_npu.fla.kda_target_verify")
    gate.fused_kda_gate_npu = lambda g, *a, **k: g
    verify.kda_target_verify_npu = lambda *a, **k: a[0]
    pkg.fla = fla
    fla.kda_gate = gate
    fla.kda_target_verify = verify
    return {
        "sgl_kernel_npu": pkg,
        "sgl_kernel_npu.fla": fla,
        "sgl_kernel_npu.fla.kda_gate": gate,
        "sgl_kernel_npu.fla.kda_target_verify": verify,
    }


@pytest.fixture(scope="module")
def ascend_backend():
    # patch.dict restores sys.modules on teardown, so the fakes never outlive
    # this module. Dropping the cached backend afterwards keeps a stub-bound
    # import from leaking into suites that import it later.
    with patch.dict(sys.modules, _fake_npu_kernels()):
        module = importlib.import_module(_ASCEND_BACKEND)
        try:
            yield module
        finally:
            sys.modules.pop(_ASCEND_BACKEND, None)


def _make_ascend_backend(ascend_kda_backend, track_ssm, extend_result):
    num_tokens, head_dim = 4, 8
    backend = ascend_kda_backend.AscendKDAAttnBackend.__new__(
        ascend_kda_backend.AscendKDAAttnBackend
    )
    backend.forward_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
        mamba_cache_indices=torch.tensor([0], dtype=torch.int32),
        has_mamba_track_mask=track_ssm,
        track_conv_indices=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64),
        conv_states_mask_indices=torch.tensor([0, 1], dtype=torch.int64),
        track_ssm_h_src=(torch.tensor([0], dtype=torch.int64) if track_ssm else None),
        logical_num_tokens=num_tokens,
    )
    conv = torch.zeros(2, 3, 3 * head_dim)
    backend.req_to_token_pool = SimpleNamespace(
        mamba2_layer_cache=lambda layer_id: SimpleNamespace(
            conv=[conv], temporal=torch.zeros(2, 1, head_dim, head_dim)
        )
    )
    backend.kernel_dispatcher = SimpleNamespace(extend=Mock(return_value=extend_result))
    backend._prepare_extend_gate_inputs = Mock(
        side_effect=lambda layer, a, b: (a, b, None, None)
    )
    backend._track_mamba_state_extend = Mock()
    return backend, num_tokens, head_dim


def _make_layer(head_dim):
    return SimpleNamespace(
        layer_id=0,
        q_dim=head_dim,
        k_dim=head_dim,
        v_dim=head_dim,
        head_q_dim=head_dim,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        lower_bound=-5.0,
        conv_weights=torch.zeros(3 * head_dim, 3),
        bias=None,
    )


def _run(backend, num_tokens, head_dim):
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_target_verify=lambda: False, is_draft_extend_v2=lambda: False
        ),
        extend_prefix_lens=torch.zeros(1, dtype=torch.int32),
        extend_seq_lens_cpu=[num_tokens],
    )
    mixed_qkv = torch.zeros(num_tokens, 3 * head_dim)
    gate = torch.zeros(num_tokens, head_dim)
    beta = torch.zeros(num_tokens)
    fake_npu = SimpleNamespace(causal_conv1d=lambda mixed, *a, **k: mixed)
    with patch.object(torch, "ops", SimpleNamespace(npu=fake_npu)):
        return backend.forward_extend(
            _make_layer(head_dim), forward_batch, mixed_qkv, gate, beta
        )


def test_ascend_untracked_prefill_returns_a_tensor(ascend_backend):
    out = torch.zeros(1, 4, 1, 8)
    backend, num_tokens, head_dim = _make_ascend_backend(
        ascend_backend, track_ssm=False, extend_result=(out, None)
    )

    result = _run(backend, num_tokens, head_dim)

    # The dispatcher always returns (out, h); the caller must not hand the
    # tuple back as the layer output.
    assert isinstance(result, torch.Tensor)
    assert result is out
    backend._track_mamba_state_extend.assert_not_called()
    assert (
        backend.kernel_dispatcher.extend.call_args.kwargs["return_intermediate_states"]
        is False
    )


def test_ascend_tracked_prefill_still_snapshots_state(ascend_backend):
    out = torch.zeros(1, 4, 1, 8)
    states = torch.zeros(1, 1, 8, 8)
    backend, num_tokens, head_dim = _make_ascend_backend(
        ascend_backend, track_ssm=True, extend_result=(out, states)
    )

    result = _run(backend, num_tokens, head_dim)

    assert result is out
    backend._track_mamba_state_extend.assert_called_once()
    assert backend._track_mamba_state_extend.call_args.args[1] is states
    assert (
        backend.kernel_dispatcher.extend.call_args.kwargs["return_intermediate_states"]
        is True
    )


def test_ascend_backend_keeps_the_shared_dispatcher(ascend_backend):
    # Guard the assumption behind the contract: the Ascend backend reuses the
    # shared KDA dispatcher (only swapping extend_kernel), so it inherits the
    # normalized return shape.
    assert issubclass(
        ascend_backend.AscendKDAAttnBackend,
        ascend_backend.KDAAttnBackend,
    )
    assert LinearAttnKernelBackend.TRITON.is_triton()


if __name__ == "__main__":
    # run_suite.py launches each file as ``python3 <file> -f``; map that to
    # pytest's fail-fast flag so the registered suite actually runs the tests.
    args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    sys.exit(pytest.main([__file__, *args]))
