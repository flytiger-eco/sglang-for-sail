"""Static QSA KV scales and the encoding contract used by PD transfers.

This module deliberately has no CUDA imports: configuration failures must be
testable without loading a model or allocating its KV pool.
"""

from __future__ import annotations

import hashlib
import json
import math
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
    if set(ranks) != {str(rank) for rank in range(tp_size)}:
        raise ValueError(f"{section_name} must contain all {tp_size} TP ranks")
    scales = ranks[str(tp_rank)]
    layers = qsa_attention_layers(model)
    if not layers:
        raise ValueError("No QSA attention layers found for KV scale loading")
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


def merge_qsa_pp_components(parts):
    """Reconstruct the full layer contract; keep all scale/layout checks."""
    if not parts or any(len(part) != len(parts[0]) for part in parts):
        raise ValueError("QSA PP contract component count mismatch")
    merged = []
    for component_idx in range(len(parts[0])):
        components = [part[component_idx] for part in parts]
        if all(c is None for c in components):
            merged.append(None)
            continue
        if any(c is None for c in components):
            raise ValueError("QSA PP contract missing component")
        variable = {"layer_ids", "item_lens", "scales"}
        common = {k: v for k, v in components[0].items() if k not in variable}
        rows = {}
        has_scales = components[0]["scales"] is not None
        for c in components:
            if {k: v for k, v in c.items() if k not in variable} != common:
                raise ValueError("QSA PP layout mismatch")
            ids = c["layer_ids"]
            n = len(ids)
            if not n or len(c["item_lens"]) != 2 * n:
                raise ValueError("QSA PP requires layer-first K/V entries")
            if (c["scales"] is not None) != has_scales:
                raise ValueError("QSA PP scale presence mismatch")
            scales = {row[0]: row for row in c["scales"]} if has_scales else {}
            if has_scales and set(scales) != set(ids):
                raise ValueError("QSA PP scale coverage mismatch")
            for i, layer_id in enumerate(ids):
                if layer_id in rows:
                    raise ValueError("QSA PP layer overlap")
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


def qsa_fp8_transfer_contract(main_pool, draft_pool=None, pp_group=None):
    """Fingerprint effective scales/layouts, excluding addresses and capacity.

    PP shards are reconstructed into the same global layer contract as PP1.
    Layout or scale differences still fail at bootstrap, before transferred
    FP8 bytes can be interpreted with the wrong descales.
    """
    pools = [main_pool, draft_pool]
    if not any(
        pool is not None
        and hasattr(pool, "qsa_compress_ratio")
        and str(pool.dtype).startswith("torch.float8_")
        for pool in pools
    ):
        return None
    components = []
    for pool in pools:
        if pool is None:
            components.append(None)
            continue
        dtype = str(pool.dtype)
        backing = getattr(pool, "full_kv_pool", pool)
        layer_ids = list(pool.full_attention_layer_id_mapping)
        if dtype.startswith("torch.float8_"):
            scales = getattr(pool, "qsa_kv_scales", None)
            if scales is None or set(scales) != set(layer_ids):
                raise ValueError(
                    "QSA FP8 PD requires effective scales for every KV layer"
                )
            scales = [
                [layer_id, *map(normalize_kv_scale, scales[layer_id])]
                for layer_id in layer_ids
            ]
        else:
            scales = None
        components.append(
            {
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
        )
    if pp_group is not None and pp_group.world_size > 1:
        parts = [
            pp_group.broadcast_object(components, src=i)
            for i in range(pp_group.world_size)
        ]
        components = merge_qsa_pp_components(parts)
    payload = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return "qsa-fp8-v1:" + hashlib.sha256(payload.encode()).hexdigest()


def validate_qsa_fp8_transfer_contract(local, peer):
    if local != peer and (local is not None or peer is not None):
        raise RuntimeError(
            "QSA FP8 KV transfer contract mismatch: both P/D servers must run "
            "the FP8 patch with identical main/draft scales and KV layouts. "
            f"Local={local}, peer={peer}"
        )
