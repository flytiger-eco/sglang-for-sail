"""Baked Qwen4-Exp KV scales must reach the QSA attention backend."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.kv_cache import apply_qsa_kv_scale_weights
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layers():
    return {
        layer_id: SimpleNamespace(
            k_scale=torch.tensor(-1.0),
            v_scale=torch.tensor(-1.0),
            k_scale_float=None,
            v_scale_float=None,
        )
        for layer_id in (3, 7)
    }


def test_baked_kv_scales_update_backend_and_parameter_values():
    layers = _layers()
    apply_qsa_kv_scale_weights(
        layers,
        {(3, "k"): 0.023, (3, "v"): 0.03, (7, "k"): 0.027, (7, "v"): 0.04},
    )

    assert layers[3].k_scale_float == pytest.approx(0.023)
    assert layers[7].v_scale_float == pytest.approx(0.04)
    for layer in layers.values():
        assert layer.k_scale.item() == pytest.approx(layer.k_scale_float)
        assert layer.v_scale.item() == pytest.approx(layer.v_scale_float)


def test_incomplete_baked_kv_scales_fail_before_mutation():
    layers = _layers()
    with pytest.raises(ValueError, match="Incomplete QSA KV scales"):
        apply_qsa_kv_scale_weights(
            layers, {(3, "k"): 0.02, (3, "v"): 0.03, (7, "k"): 0.04}
        )

    assert all(layer.k_scale_float is None for layer in layers.values())
    assert all(layer.v_scale_float is None for layer in layers.values())


def test_invalid_baked_kv_scale_fails_before_mutation():
    layers = _layers()
    with pytest.raises(ValueError, match="finite and positive"):
        apply_qsa_kv_scale_weights(
            layers,
            {(3, "k"): 0.02, (3, "v"): 0.03, (7, "k"): 0.04, (7, "v"): 0},
        )

    assert all(layer.k_scale_float is None for layer in layers.values())
