"""Baked Qwen4-Exp KV scales must reach the QSA attention backend."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.kv_cache import (
    _qsa_pool_component,
    apply_qsa_kv_scale_weights,
    collect_qsa_kv_scale,
    load_qsa_kv_scales,
    merge_qsa_pp_components,
    protect_qsa_fp8_pd_weight_updates,
    qsa_fp8_transfer_contract,
)
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


@pytest.mark.parametrize(
    "prefix", ["model", "model.language_model", "mtp", "model.mtp"]
)
@pytest.mark.parametrize(
    "suffix",
    [
        "attn.k_scale",
        "self_attn.k_scale",
        "k_proj.k_scale",
        "self_attn.k_proj.k_scale",
        "k_scale",
    ],
)
def test_baked_scale_checkpoint_name_aliases(prefix, suffix):
    layers = _layers()
    scales = {}
    assert collect_qsa_kv_scale(
        f"{prefix}.layers.3.{suffix}", torch.tensor(0.02), layers, scales
    )
    assert scales[(3, "k")] == pytest.approx(0.02)
    assert layers[3].k_scale_float is None


@pytest.mark.parametrize(
    "weight",
    [
        torch.ones(2),
        torch.tensor(0.0),
        torch.tensor(float("nan")),
        torch.tensor(float("inf")),
    ],
)
def test_invalid_checkpoint_scale_is_rejected_before_application(weight):
    layers = _layers()
    with pytest.raises(ValueError):
        collect_qsa_kv_scale("model.layers.3.attn.k_scale", weight, layers, {})
    assert all(layer.k_scale_float is None for layer in layers.values())


def test_null_baked_scale_fails_before_mutating_other_layers():
    layers = _layers()
    with pytest.raises(ValueError, match="cannot be null"):
        apply_qsa_kv_scale_weights(
            layers, {(3, "k"): 1.0, (3, "v"): None, (7, "k"): 1.0, (7, "v"): 1.0}
        )
    assert all(layer.k_scale_float is None for layer in layers.values())


def test_duplicate_scale_aliases_are_rejected():
    layers = _layers()
    scales = {}
    collect_qsa_kv_scale(
        "model.layers.3.k_proj.k_scale", torch.tensor(1.0), layers, scales
    )
    with pytest.raises(ValueError, match="Duplicate"):
        collect_qsa_kv_scale(
            "model.layers.3.self_attn.k_scale", torch.tensor(2.0), layers, scales
        )
    assert not collect_qsa_kv_scale(
        "model.layers.8.attn.k_scale", torch.tensor(1.0), layers, scales
    )


def _pool(ids, dtype=torch.float8_e4m3fn, attach_scales=True):
    pool = SimpleNamespace(
        dtype=dtype,
        full_attention_layer_id_mapping={i: n for n, i in enumerate(ids)},
        page_size=64,
        head_num=2,
        head_dim=256,
        store_dtype=torch.uint8,
        kv_cache_layout="layer_first",
        index_state_dtype=torch.bfloat16,
        qsa_compress_ratio=4,
        get_contiguous_buf_infos=lambda: ([], [], [512] * len(ids) + [512] * len(ids)),
    )
    if attach_scales:
        pool.qsa_kv_scales = {i: (0.02, 0.03) for i in ids}
    return pool


def _envelope(main, draft=None):
    return {
        "fp8_qsa": True,
        "components": [_qsa_pool_component(main), _qsa_pool_component(draft)],
        "error": None,
    }


class _PPGroup:
    def __init__(self, parts):
        self.parts = parts
        self.world_size = len(parts)
        self.calls = []

    def broadcast_object(self, obj, src):
        self.calls.append(src)
        return self.parts[src]


def test_contract_pp_empty_stage_and_last_stage_draft_matches_pp1():
    empty, full, draft = _pool([], attach_scales=False), _pool([3, 7]), _pool([0])
    parts = [_envelope(empty), _envelope(full, draft)]
    expected = qsa_fp8_transfer_contract(full, draft)
    for main, local_draft in ((empty, None), (full, draft)):
        group = _PPGroup(parts)
        assert qsa_fp8_transfer_contract(main, local_draft, group) == expected
        assert group.calls == [0, 1]


def test_contract_decode_and_prefill_pp_shards_have_same_fingerprint():
    full, left, right = _pool([3, 7]), _pool([3]), _pool([7])
    expected = qsa_fp8_transfer_contract(full)
    parts = [_envelope(left), _envelope(right)]
    for pool in (left, right):
        assert qsa_fp8_transfer_contract(pool, pp_group=_PPGroup(parts)) == expected


def test_no_fp8_stage_still_participates_in_pp_collective():
    bf16 = _pool([], dtype=torch.bfloat16)
    full = _pool([3])
    parts = [_envelope(bf16), _envelope(full)]
    parts[0]["fp8_qsa"] = False
    group = _PPGroup(parts)
    with pytest.raises(ValueError, match="layout mismatch"):
        qsa_fp8_transfer_contract(bf16, pp_group=group)
    assert group.calls == [0, 1]


def test_contract_propagates_invalid_scale_after_all_pp_collectives():
    valid = _pool([3])
    parts = [
        _envelope(valid),
        {"fp8_qsa": True, "components": None, "error": "invalid peer scale"},
    ]
    group = _PPGroup(parts)
    with pytest.raises(ValueError, match="invalid peer scale"):
        qsa_fp8_transfer_contract(valid, pp_group=group)
    assert group.calls == [0, 1]


@pytest.mark.parametrize(
    "case",
    [
        "main_missing",
        "draft_wrong_owner",
        "overlap",
        "scale_missing",
        "item_lens",
        "layout",
    ],
)
def test_pp_merge_rejects_invalid_components(case):
    first = [_qsa_pool_component(_pool([3])), None]
    last = [_qsa_pool_component(_pool([7])), None]
    if case == "main_missing":
        first[0] = None
    elif case == "draft_wrong_owner":
        first[1] = _qsa_pool_component(_pool([0]))
    elif case == "overlap":
        last[0] = first[0].copy()
    elif case == "scale_missing":
        last[0]["scales"] = []
    elif case == "item_lens":
        last[0]["item_lens"] = [512]
    else:
        last[0]["head_dim"] = 128
    with pytest.raises(ValueError):
        merge_qsa_pp_components([first, last])


def test_explicit_scale_file_allows_pp_stage_without_qsa(tmp_path):
    path = tmp_path / "scales.json"
    path.write_text(
        json.dumps(
            {"kv_cache": {"dtype": "float8_e4m3fn", "scaling_factor": {"0": {}}}}
        )
    )
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen4_exp"), modules=lambda: []
    )
    load_qsa_kv_scales(model, path, tp_rank=0, tp_size=1)


@pytest.mark.parametrize(
    "mode,dtype,blocked",
    [
        ("prefill", torch.float8_e4m3fn, True),
        ("decode", torch.float8_e4m3fn, True),
        ("null", torch.float8_e4m3fn, False),
        ("prefill", torch.bfloat16, False),
    ],
)
def test_pd_fp8_scale_snapshot_blocks_online_updates_only_in_pd(mode, dtype, blocked):
    model = SimpleNamespace()
    protect_qsa_fp8_pd_weight_updates(model, _pool([3], dtype=dtype), mode)
    assert hasattr(model, "_derived_weight_cache_error") == blocked


@pytest.mark.parametrize("partial", [False, True])
def test_mtp_baked_scales_collected_before_inherited_loader_discards_them(
    monkeypatch, partial
):
    from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
    from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP

    model = Qwen4ExpForCausalLMMTP.__new__(Qwen4ExpForCausalLMMTP)
    torch.nn.Module.__init__(model)
    layer = SimpleNamespace(layer_id=0, k_scale_float=None, v_scale_float=None)
    model.modules = lambda: [SimpleNamespace(is_qsa=True, attn=layer)]
    forwarded = []

    def parent_load(self, weights, is_mtp=False):
        forwarded.extend(weights)
        return {"ordinary.weight"}

    monkeypatch.setattr(Qwen3_5ForCausalLMMTP, "load_weights", parent_load)
    weights = [
        ("model.layers.0.attn.k_scale", torch.tensor(9.0)),
        ("mtp.layers.0.self_attn.k_scale", torch.tensor(0.02)),
    ]
    if not partial:
        weights.append(("mtp.layers.0.v_proj.v_scale", torch.tensor(0.03)))
        loaded = model.load_weights(iter(weights))
        assert "ordinary.weight" in loaded
        assert layer.k_scale_float == pytest.approx(0.02)
        assert layer.v_scale_float == pytest.approx(0.03)
    else:
        with pytest.raises(ValueError, match="Incomplete"):
            model.load_weights(iter(weights))
        assert layer.k_scale_float is None
        assert layer.v_scale_float is None
    assert [name for name, _ in forwarded] == ["model.layers.0.attn.k_scale"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
