# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

pytest.importorskip("humming")

from sglang.srt.layers.linear import LinearBase  # noqa: E402
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # noqa: E402
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod  # noqa: E402
from sglang.srt.layers.quantization.humming import (  # noqa: E402
    HummingConfig,
    HummingMoEMethod,
)
from sglang.srt.layers.quantization.unquant import (  # noqa: E402
    UnquantizedLinearMethod,
)


def config(**overrides):
    return HummingConfig.from_config(
        dict(
            quant_method="wmxfp4afp8",
            weight_dtype="mxfp4",
            group_size=32,
            activation_dtype="fp8_e4m3",
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
            ignore=[
                "model.language_model.layers.0.mlp.gate",
                "model.language_model.layers.0.linear_attn.in_proj_a",
                "model.language_model.layers.0.linear_attn.in_proj_b",
            ],
            packed_modules_mapping={"in_proj_ba": ["in_proj_b", "in_proj_a"]},
        )
        | overrides
    )


def moe_method(cfg=None):
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    method = (cfg or config()).get_quant_method(layer, "model.layers.0.mlp.experts")
    assert isinstance(method, HummingMoEMethod)
    return layer, method


@pytest.mark.parametrize(
    "name,dtype", [("weight", torch.int32), ("weight_scale", torch.float8_e8m0fnu)]
)
def test_native_bytes_are_reinterpreted(name, dtype):
    layer, method = moe_method()
    raw = torch.arange(256, dtype=torch.uint8).reshape(4, 64)
    param = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    param.param_name = name
    captured = []
    loader = method.prepare_weight_loader(
        layer, lambda p, w, *a, **kw: captured.append(w)
    )
    loader(param, raw, name, "w1", 0)
    assert captured[0].dtype == dtype
    assert torch.equal(captured[0].view(torch.uint8), raw)


def test_rejects_unexpected_expert_dtype():
    layer, method = moe_method()
    param = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    param.param_name = "weight"
    loader = method.prepare_weight_loader(
        layer, lambda *a, **kw: pytest.fail("must reject before loading")
    )
    with pytest.raises(ValueError, match="native uint8"):
        loader(param, torch.zeros(2, 8, dtype=torch.bfloat16), "weight", "w1", 0)


def test_dense_fp8_and_ignored_router_keep_checkpoint_precision():
    class DummyLinear(LinearBase):
        def forward(self, x):
            return x

    layer = DummyLinear.__new__(DummyLinear)
    torch.nn.Module.__init__(layer)
    cfg = config()
    assert cfg.packed_modules_mapping["in_proj_ba"] == ["in_proj_b", "in_proj_a"]
    assert isinstance(
        cfg.get_quant_method(layer, "model.layers.0.linear_attn.in_proj_ba"),
        UnquantizedLinearMethod,
    )
    assert isinstance(
        cfg.get_quant_method(layer, "model.layers.0.mlp.gate"), UnquantizedLinearMethod
    )
    assert isinstance(
        cfg.get_quant_method(layer, "model.layers.0.linear_attn.in_proj_qkvz"),
        Fp8LinearMethod,
    )


@pytest.mark.parametrize(
    "key,value",
    [("group_size", 128), ("weight_dtype", "int4"), ("activation_dtype", "bf16")],
)
def test_rejects_incompatible_checkpoint(key, value):
    with pytest.raises(ValueError, match="Unsupported wmxfp4afp8"):
        moe_method(config(**{key: value}))


@pytest.mark.parametrize(
    "quant_method,intermediate,raises",
    [
        ("wmxfp4afp8", 640, False),
        ("fp8", 640, True),
        ("wmxfp4afp8", 642, True),
    ],
)
def test_moe_check_uses_expert_quantization(
    quant_method, intermediate, raises, monkeypatch
):
    from types import SimpleNamespace

    from sglang.srt.model_executor.model_runner_components import moe_ep_setup

    monkeypatch.setattr(moe_ep_setup, "_use_aiter", False)
    monkeypatch.setenv("SGLANG_SHARED_EXPERT_TP1", "0")
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            quantization_config={
                "quant_method": quant_method,
                "weight_block_size": [128, 128],
            }
        ),
        hf_text_config=SimpleNamespace(moe_intermediate_size=intermediate),
    )

    def check():
        moe_ep_setup.check_quantized_moe_compatibility(
            model_config=model_config, tp_size=4, moe_ep_size=1, moe_dp_size=1
        )

    if raises:
        with pytest.raises(ValueError):
            check()
    else:
        check()
