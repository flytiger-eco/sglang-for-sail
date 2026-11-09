"""Static QSA KV scales and the encoding contract used by PD transfers.

This module deliberately has no CUDA imports: configuration failures must be
testable without loading a model or allocating its KV pool.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from numbers import Real


def normalize_kv_scale(value):
    if value is None:
        return 1.0
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("QSA KV scale must be a finite positive host scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"QSA KV scale must be finite and positive, got {value}")
    return value


def qsa_attention_layers(model):
    return {
        module.attn.layer_id: module.attn
        for module in model.modules()
        if getattr(module, "is_qsa", False) and hasattr(module, "attn")
    }


def get_qsa_kv_scales(model):
    return {
        layer_id: (
            normalize_kv_scale(layer.k_scale_float),
            normalize_kv_scale(layer.v_scale_float),
        )
        for layer_id, layer in qsa_attention_layers(model).items()
    }


def apply_qsa_kv_scale_weights(layers, scales):
    """Apply baked checkpoint K/V descales after verifying every local QSA layer."""
    expected = {(layer_id, kind) for layer_id in layers for kind in ("k", "v")}
    missing = expected - scales.keys()
    unexpected = scales.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"Incomplete QSA KV scales: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    if any(value is None for value in scales.values()):
        raise ValueError("Baked QSA KV scales cannot be null")
    validated = {key: normalize_kv_scale(value) for key, value in scales.items()}
    for (layer_id, kind), value in validated.items():
        layer = layers[layer_id]
        setattr(layer, f"{kind}_scale_float", value)
        current = getattr(layer, f"{kind}_scale", None)
        if hasattr(current, "fill_"):
            current.fill_(value)
        else:
            setattr(layer, f"{kind}_scale", value)


def load_qsa_kv_scales(model, filename, *, tp_rank, tp_size, is_draft=False):
    """Load SGLang-style rank/layer scales, with optional separate K/V values.

    ``kv_cache.scaling_factor[rank][layer]`` is a positive scalar or
    ``{"k_scale": ..., "v_scale": ...}``. The draft uses a separate
    ``draft_kv_cache`` section with its own local layer ids. Explicit files
    must cover every local QSA layer; errors never fall back to unit scales.
    """
    with open(filename) as stream:
        config = json.load(stream)
    model_config = getattr(model.config, "text_config", model.config)
    if config.get("model_type") not in (None, model_config.model_type):
        raise ValueError("QSA KV scale model_type does not match the model")
    section_name = "draft_kv_cache" if is_draft else "kv_cache"
    section = config.get(section_name)
    if not isinstance(section, dict) or section.get("dtype") != "float8_e4m3fn":
        raise ValueError(f"{section_name} must specify dtype=float8_e4m3fn")
    ranks = section.get("scaling_factor", {})
    if not isinstance(ranks, dict) or set(ranks) != {
        str(rank) for rank in range(tp_size)
    }:
        raise ValueError(f"{section_name} must contain all {tp_size} TP ranks")
    scales = ranks[str(tp_rank)]
    if not isinstance(scales, dict):
        raise ValueError(f"{section_name} rank scales must map layer IDs to scales")
    layers = qsa_attention_layers(model)
    # PP 分片可能只含线性注意力层；仍验证文件格式，但不要求本地存在 QSA 层。
    pending = []
    for layer_id, layer in layers.items():
        key = str(layer_id)
        if key not in scales:
            raise ValueError(f"Missing {section_name} scale for layer {layer_id}")
        value = scales[key]
        if isinstance(value, dict):
            if set(value) != {"k_scale", "v_scale"}:
                raise ValueError(f"Layer {layer_id} requires k_scale and v_scale")
            k_scale, v_scale = value["k_scale"], value["v_scale"]
        else:
            k_scale = v_scale = value
        if k_scale is None or v_scale is None:
            raise ValueError(f"Layer {layer_id} has a null KV scale")
        pending.append(
            (layer, normalize_kv_scale(k_scale), normalize_kv_scale(v_scale))
        )
    # Validate every local QSA layer before changing any layer.
    for layer, k_scale, v_scale in pending:
        layer.k_scale_float, layer.v_scale_float = k_scale, v_scale
        for name, scale in (("k_scale", k_scale), ("v_scale", v_scale)):
            current = getattr(layer, name, None)
            if hasattr(current, "fill_"):
                current.fill_(scale)
            else:
                setattr(layer, name, scale)


def collect_qsa_kv_scale(name, weight, layers, scales):
    """收集目标或 MTP 的两种 checkpoint KV scale 命名，不分配设备内存。"""
    match = re.search(
        r"(?:^|\.)layers\.(\d+)\.(?:(?:self_attn\.|attn\.)?([kv])_proj\.([kv])_scale|(?:self_attn\.|attn\.)?([kv])_scale)$",
        name,
    )
    if match is None:
        return False
    layer_id = int(match[1])
    if layer_id not in layers:
        return False
    kind = match[4] or match[3]
    if match[2] is not None and match[2] != kind:
        raise ValueError(f"Invalid QSA KV scale name: {name}")
    key = (layer_id, kind)
    if key in scales:
        raise ValueError(f"Duplicate QSA KV scale for {key}")
    if weight.numel() != 1:
        raise ValueError(f"QSA KV scale {name} must be a scalar")
    value = weight.item()
    if value is None:
        raise ValueError(f"QSA KV scale {name} cannot be null")
    scales[key] = normalize_kv_scale(value)
    return True


def merge_qsa_pp_components(parts):
    """重建全局合同；允许空 QSA 分片和仅最后 PP stage 持有 draft。"""
    if not parts or any(len(part) != 2 for part in parts):
        raise ValueError("QSA PP contract component count mismatch")
    merged = []
    for component_idx in range(2):
        components = [part[component_idx] for part in parts]
        present = [i for i, c in enumerate(components) if c is not None]
        if not present:
            merged.append(None)
            continue
        if len(present) != len(parts):
            if component_idx == 0 or present != [len(parts) - 1]:
                raise ValueError(
                    "QSA PP contract missing component or invalid draft owner"
                )
        components = [components[i] for i in present]
        variable = {"layer_ids", "item_lens", "scales"}
        common = {k: v for k, v in components[0].items() if k not in variable}
        rows = {}
        has_scales = components[0]["scales"] is not None
        for c in components:
            if {k: v for k, v in c.items() if k not in variable} != common:
                raise ValueError("QSA PP layout mismatch")
            ids = c["layer_ids"]
            n = len(ids)
            if len(set(ids)) != n or len(c["item_lens"]) != 2 * n:
                raise ValueError("QSA PP requires layer-first K/V entries")
            if (c["scales"] is not None) != has_scales:
                raise ValueError("QSA PP scale presence mismatch")
            if has_scales and any(
                len(row) != 3 or row[1] is None or row[2] is None for row in c["scales"]
            ):
                raise ValueError("QSA PP requires a K/V scale pair for every layer")
            scales = {row[0]: row for row in c["scales"]} if has_scales else {}
            if has_scales and (len(c["scales"]) != n or set(scales) != set(ids)):
                raise ValueError("QSA PP scale coverage mismatch")
            for i, layer_id in enumerate(ids):
                if layer_id in rows:
                    raise ValueError("QSA PP layer overlap")
                if has_scales:
                    scales[layer_id] = [
                        layer_id,
                        *map(normalize_kv_scale, scales[layer_id][1:]),
                    ]
                rows[layer_id] = (
                    c["item_lens"][i],
                    c["item_lens"][n + i],
                    scales.get(layer_id),
                )
        ids = sorted(rows)
        merged.append(
            dict(
                common,
                layer_ids=ids,
                item_lens=[rows[i][0] for i in ids] + [rows[i][1] for i in ids],
                scales=[rows[i][2] for i in ids] if has_scales else None,
            )
        )
    return merged


def is_qsa_fp8_pool(pool):
    return (
        pool is not None
        and hasattr(pool, "qsa_compress_ratio")
        and str(pool.dtype).startswith("torch.float8_")
    )


def qsa_fp8_pd_weight_update_error(pool, disaggregation_mode):
    """先明确拒绝在线更新；重新启动 P/D 后才可重新校准并发布合同。"""
    mode = getattr(disaggregation_mode, "value", disaggregation_mode)
    if is_qsa_fp8_pool(pool) and mode in ("prefill", "decode"):
        return (
            "Online weight updates are not supported for FP8 QSA PD deployments; "
            "restart both P/D servers to reload KV scales and publish a new transfer contract."
        )
    return None


def protect_qsa_fp8_pd_weight_updates(model, pool, disaggregation_mode):
    """在线更新不支持协调已编码 KV、pool scale 快照及 P/D bootstrap 合同。"""
    reason = qsa_fp8_pd_weight_update_error(pool, disaggregation_mode)
    if reason is not None:
        model._derived_weight_cache_error = reason


def _qsa_pool_component(pool):
    if pool is None:
        return None
    dtype = str(pool.dtype)
    backing = getattr(pool, "full_kv_pool", pool)
    layer_ids = list(pool.full_attention_layer_id_mapping)
    if dtype.startswith("torch.float8_"):
        scales = getattr(pool, "qsa_kv_scales", {} if not layer_ids else None)
        if scales is None or set(scales) != set(layer_ids):
            raise ValueError("QSA FP8 PD requires effective scales for every KV layer")
        scales = [[i, *map(normalize_kv_scale, scales[i])] for i in layer_ids]
    else:
        scales = None
    return {
        "dtype": dtype,
        "layout": getattr(backing, "kv_cache_layout", "layer_first"),
        "store_dtype": str(getattr(backing, "store_dtype", pool.dtype)),
        "page_size": pool.page_size,
        "kv_heads": pool.head_num,
        "head_dim": pool.head_dim,
        "layer_ids": layer_ids,
        "item_lens": list(pool.get_contiguous_buf_infos()[2]),
        "index_dtype": str(getattr(pool, "index_state_dtype", None)),
        "compress_ratio": getattr(pool, "qsa_compress_ratio", None),
        "scales": scales,
    }


def qsa_fp8_transfer_contract(main_pool, draft_pool=None, pp_group=None):
    """聚合所有 PP stage 后再判断 FP8 QSA；TP 布局和 scale 仍须相同。"""
    pools = [main_pool, draft_pool]
    local = {
        "fp8_qsa": any(is_qsa_fp8_pool(p) for p in pools),
        "components": None,
        "error": None,
    }
    if any(p is not None and hasattr(p, "qsa_compress_ratio") for p in pools):
        try:
            local["components"] = [_qsa_pool_component(p) for p in pools]
        except (ValueError, TypeError, AttributeError) as exc:
            # 先传播错误，避免某个 stage 提前退出而其他 stage 等待 collective。
            local["error"] = str(exc)
    parts = [local]
    if pp_group is not None and pp_group.world_size > 1:
        parts = [
            pp_group.broadcast_object(local, src=i) for i in range(pp_group.world_size)
        ]
    if not any(p["fp8_qsa"] for p in parts):
        return None
    for part in parts:
        if part["error"] is not None:
            raise ValueError(part["error"])
        if part["components"] is None:
            raise ValueError("QSA PP contract missing main component")
    components = (
        merge_qsa_pp_components([p["components"] for p in parts])
        if len(parts) > 1
        else parts[0]["components"]
    )
    payload = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return "qsa-fp8-v1:" + hashlib.sha256(payload.encode()).hexdigest()


def validate_qsa_fp8_transfer_contract(local, peer):
    if local != peer and (local is not None or peer is not None):
        raise RuntimeError(
            "QSA FP8 KV transfer contract mismatch: both P/D servers must run "
            "the FP8 patch with identical main/draft scales and KV layouts. "
            f"Local={local}, peer={peer}"
        )
