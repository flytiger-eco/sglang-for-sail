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


def qsa_fp8_transfer_contract(main_pool, draft_pool=None):
    """Fingerprint effective scales/layouts, excluding addresses and capacity.

    The first PD implementation requires identical per-rank QSA layouts and
    scales. Differing PP shards or rank-specific scales fail at bootstrap,
    rather than interpreting transferred FP8 bytes with the wrong descales.
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
    payload = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return "qsa-fp8-v1:" + hashlib.sha256(payload.encode()).hexdigest()


def validate_qsa_fp8_transfer_contract(local, peer):
    if local != peer and (local is not None or peer is not None):
        raise RuntimeError(
            "QSA FP8 KV transfer contract mismatch: both P/D servers must run "
            "the FP8 patch with identical main/draft scales and KV layouts. "
            f"Local={local}, peer={peer}"
        )
