"""PPU MegaGate routing for the GLM-5.3 text checkpoint."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.kernels import get_fused_op_backend
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_parallel

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


@functools.lru_cache(maxsize=None)
def _is_sail_ppu(device: torch.device) -> bool:
    # DeepGEMM handles architecture-specific dispatch on ZW devices.
    return "ZW" in torch.cuda.get_device_name(device)


def can_route(
    moe: DeepseekV2MoE,
    hidden: torch.Tensor,
    dispatch_info: ExpertLocationDispatchInfo | None,
) -> bool:
    from sglang.srt.layers.moe.topk import TopKOutputFormat

    # The DeepGEMM binding has no fake implementation for torch.compile.
    if torch.compiler.is_compiling():
        return False
    if (
        not envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.get()
        or moe.config.model_type != "glm_moe_dsa"
        or moe.is_hash
        or get_fused_op_backend() is not None
    ):
        return False
    # Waterfill appends the shared slot after TopK has reset its fused count.
    if moe.topk.enable_waterfill or moe.num_fused_shared_experts:
        return False
    # These modes need the original routing transform or its LP dispatch solver.
    if dispatch_info is not None and dispatch_info.ep_dispatch_algorithm in (
        "fake",
        "lp",
    ):
        return False
    cfg = moe.topk.topk_config
    weight = moe.gate.weight
    bias = moe.gate.e_score_correction_bias
    return (
        hidden.is_cuda
        and _is_sail_ppu(hidden.device)
        and hidden.ndim == 2
        and hidden.shape[1] == 6144
        and 0 < hidden.shape[0] <= 1 << 20
        and hidden.is_contiguous()
        and hidden.storage_offset() % 8 == 0
        and hidden.device == weight.device
        and hidden.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.shape == (256, 6144)
        and weight.is_contiguous()
        and weight.storage_offset() % 8 == 0
        and bias is not None
        and bias.device == hidden.device
        and bias.dtype == torch.float32
        and bias.shape == (256,)
        and bias.is_contiguous()
        and moe.gate.e_score_correction_bias_vl is None
        and moe.topk.resolve_output_format() == TopKOutputFormat.STANDARD
        and cfg.scoring_func == "sigmoid"
        and cfg.top_k == 8
        and cfg.renormalize
        and cfg.num_expert_group == cfg.topk_group == 1
        and not cfg.num_fused_shared_experts
        and cfg.custom_routing_function is None
        and not cfg.torch_native
        and not get_exec().moe.enable_eplb
        and not get_exec().deterministic.enable_deterministic_inference
        and not envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        and not envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    )


def route(
    moe: DeepseekV2MoE,
    hidden: torch.Tensor,
    batch: ForwardBatch | None,
    dispatch_info: ExpertLocationDispatchInfo | None,
) -> StandardTopKOutput:
    from sglang.kernels.ops.moe.mega_gate import bf16_mega_gate
    from sglang.srt.eplb.expert_distribution import (
        get_global_expert_distribution_recorder,
    )
    from sglang.srt.eplb.expert_location_updater import (
        get_global_expert_location_updater,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput, _post_process_topk_ids

    updater = get_global_expert_location_updater()
    if updater is not None:
        updater.set_current_layer_id(moe.layer_id, is_nextn=moe.is_nextn)
        updater.wait_gpu_stage()
    cfg = moe.topk.topk_config
    scale = (
        cfg.routed_scaling_factor if cfg.apply_routed_scaling_factor_on_output else 1.0
    )
    bound = batch.moe_num_token_non_padded() if batch is not None else None
    valid = (
        torch.arange(hidden.shape[0], device=hidden.device, dtype=torch.int32)
        < bound.reshape(-1)[:1]
        if bound is not None
        else None
    )
    weights, ids = bf16_mega_gate(
        hidden,
        moe.gate.weight,
        cfg.top_k,
        routed_scaling_factor=scale,
        ep_rank=get_parallel().moe_ep_rank,
        scoring_func="sigmoid",
        bias=moe.gate.e_score_correction_bias,
        valid_token_mask=valid,
    )
    # The standard result carries the expert dimension; this path never materializes logits.
    router_logits = hidden.new_empty((0, moe.gate.weight.shape[0]), dtype=torch.float32)
    ids, weights, recorder_ids = _post_process_topk_ids(
        topk_ids=ids,
        topk_weights=weights,
        topk_config=cfg,
        router_logits=router_logits,
        layer_id=moe.layer_id,
        num_token_non_padded=bound,
        expert_location_dispatch_info=dispatch_info,
        padded_rows_masked=True,
    )
    if recorder_ids is not None:
        get_global_expert_distribution_recorder().on_select_experts(
            topk_ids=recorder_ids
        )
    return StandardTopKOutput(weights, ids, router_logits)
