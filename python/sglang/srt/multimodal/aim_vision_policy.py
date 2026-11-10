"""Opt-in AIM vision configuration, independent of the model runtime."""

from copy import copy


def validate_vision_options(
    name,
    path,
    *,
    inherit_model=False,
    load_format="auto",
    startup_mode="serial",
    weight_cache_mode="off",
    enable_multimodal=None,
    model_impl="auto",
    modelopt_quant=None,
    model_quantization=None,
    pp_size=1,
):
    if name is None and path is None and not inherit_model:
        return
    if name not in {None, "fp8"}:
        raise ValueError("Only explicit fp8 vision quantization is supported")
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError("vision_model_path must be a nonempty checkpoint path")
    if path is not None and pp_size != 1:
        raise ValueError("Separate vision weights currently require pp_size=1")
    if load_format not in {"auto", "pt", "safetensors", "fastsafetensors", "npcache"}:
        raise ValueError("AIM vision options require the serial DefaultModelLoader")
    if startup_mode != "serial" or weight_cache_mode != "off":
        raise ValueError(
            "AIM vision options require serial loading without the weight cache"
        )
    if enable_multimodal is False or model_impl not in {"auto", "sglang"}:
        raise ValueError("AIM vision options require a native SGLang visual tower")
    if modelopt_quant or model_quantization in {
        "modelopt",
        "modelopt_fp8",
        "modelopt_fp4",
        "modelopt_mixed",
        "auto-round-int8",
        "mxfp8",
    }:
        raise ValueError(
            "AIM vision options do not support ModelOpt/INC or MXFP8 loading"
        )


def resolve_vision_quantization(
    name, model_quant, *, separate_weights, inherit_model, create_online
):
    """Keep the current unquantized vision default unless explicitly requested."""
    if name is not None:
        if name != "fp8":
            raise ValueError("Only explicit fp8 vision quantization is supported")
        return create_online(name)
    if separate_weights:
        return None
    return model_quant if inherit_model else None


def remap_vision_exclusions(config, mapper):
    """Copy vision exclusions without modifying the language-model policy."""
    if config is None:
        return None
    result = config
    for attribute in ("ignored_layers", "modules_to_not_convert"):
        original = getattr(config, attribute, None)
        if not original:
            continue
        expanded = []
        for name in original:
            if any(char in name for char in "*?[](){}|\\^$"):
                # The FP8 is_layer_skipped matcher uses literal dotted paths,
                # not glob/regex semantics. Retaining a pattern here could
                # silently quantize a module the checkpoint meant to exclude.
                raise ValueError(
                    "AIM vision exclusions require literal module paths; "
                    f"unsupported pattern: {name!r}"
                )
            if name.startswith(("visual.", "model.visual.")):
                # The loader may already have applied this mapper. Restrict
                # substring rules so qkv_proj never becomes qkv_proj_proj.
                if mapper is not None:
                    current_mapper = copy(mapper)
                    current_mapper.orig_to_new_substr = {
                        old: new
                        for old, new in mapper.orig_to_new_substr.items()
                        if new is None or new not in name
                    }
                    mapped = current_mapper.apply_list([name])
                    if len(mapped) != 1:
                        raise ValueError(
                            f"Vision exclusion mapping would drop {name!r}"
                        )
                    name = mapped[0]
                # Qwen3.5/Qwen4 use a manual load_weights mapper (class mapper
                # is None); apply only the same known visual namespace rules.
                normalized = visual_weight_name(name + ".weight")
                if normalized is None:
                    raise ValueError(
                        f"Vision exclusion mapped outside visual tower: {name!r}"
                    )
                name = normalized[: -len(".weight")]
            expanded.append(name)
            if name.startswith(("visual.", "model.visual.")) and name.endswith(
                ".attn.qkv_proj"
            ):
                prefix = name.rsplit(".", 1)[0]
                expanded.extend(
                    prefix + "." + part for part in ("q_proj", "k_proj", "v_proj")
                )
        expanded = list(dict.fromkeys(expanded))
        if expanded != list(original):
            if result is config:
                result = copy(config)
            setattr(result, attribute, expanded)
    return result


def visual_weight_name(name, *, allow_bare=False):
    """Normalize supported Qwen visual tensor names to the module namespace."""
    if name.startswith("model.visual."):
        name = name[len("model.") :]
    elif allow_bare and name.startswith(
        ("patch_embed.", "pos_embed.", "blocks.", "merger.", "deepstack_merger_list.")
    ):
        name = "visual." + name
    if not name.startswith("visual."):
        return None
    return name.replace(".attn.qkv.", ".attn.qkv_proj.")


def validate_online_vision_weights(weights, *, allow_bare=False):
    """Reject serialized quantized vision inputs before online FP8 conversion."""
    for name, tensor in weights:
        if visual_weight_name(name, allow_bare=allow_bare) is not None:
            if any(
                marker in name
                for marker in (
                    "weight_scale",
                    "input_scale",
                    "activation_scale",
                    "k_scale",
                    "v_scale",
                    "qweight",
                    "qzeros",
                    "zero_point",
                )
            ):
                raise ValueError(
                    "Online FP8 vision quantization cannot consume serialized "
                    f"quantized weights or scale metadata: {name!r}"
                )
            if str(tensor.dtype) not in {
                "torch.float16",
                "torch.bfloat16",
                "torch.float32",
                "torch.float64",
            }:
                raise ValueError(
                    "Online FP8 vision quantization requires unquantized visual "
                    f"tensors; {name!r} has dtype {tensor.dtype}"
                )
        yield name, tensor


def replace_vision_weights(
    primary, replacement, *, expected_names=None, required_names=()
):
    """Stream one primary checkpoint with an explicitly selected visual tower."""
    for name, tensor in primary:
        if visual_weight_name(name) is None:
            yield name, tensor
    found = set()
    for name, tensor in replacement:
        normalized = visual_weight_name(name, allow_bare=True)
        if normalized is not None:
            if expected_names is not None and normalized not in expected_names:
                raise ValueError(f"Unknown visual tensor: {name} -> {normalized}")
            if normalized in found:
                raise ValueError(f"Duplicate visual tensor: {normalized}")
            found.add(normalized)
            # Validate against parameter names, but emit checkpoint names.
            # Qwen's class mapper has a non-boundary substring rule for qkv;
            # feeding it qkv_proj would produce qkv_proj_proj. Both the class
            # mapper and Qwen3.5/Qwen4 manual loaders accept canonical qkv.
            yield normalized.replace(".attn.qkv_proj.", ".attn.qkv."), tensor
    if not found:
        raise ValueError("Separate vision checkpoint contains no visual tensors")
    missing = set(required_names) - found
    if missing:
        raise ValueError(
            f"Separate vision checkpoint is missing visual tensors: {sorted(missing)}"
        )
