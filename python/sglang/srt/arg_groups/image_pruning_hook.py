import math
from sglang.srt.environ import envs
from sglang.srt.arg_groups.overrides import resolving_view

def handle_image_pruning(server_args):
    cfg = resolving_view(server_args)
    if not math.isfinite(cfg.image_pruning_rate) or not 0 <= cfg.image_pruning_rate < 1:
        raise ValueError("image_pruning_rate must be finite and in [0, 1)")
    if cfg.image_pruning_strategy != "attention_score":
        raise ValueError("Only attention_score image pruning is supported")
    if cfg.image_pruning_rate > 0 and not cfg.language_only:
        if cfg.mm_enable_dp_encoder:
            raise ValueError("Image pruning does not yet support DP vision encoders")
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            raise ValueError("Image pruning requires SGLANG_VIT_ENABLE_CUDA_GRAPH=0")
    if cfg.image_pruning_rate > 0 and cfg.disaggregation_mode == "prefill" and not cfg.disagg_decode_urls:
        raise ValueError("Image pruning on Prefill requires --disagg-decode-urls")
    if cfg.image_pruning_rate > 0 and cfg.disaggregation_mode == "decode" and not cfg.language_only:
        raise ValueError("Image pruning on Decode requires --language-only")

    if (
        cfg.image_pruning_rate > 0
        and cfg.disaggregation_mode in {"prefill", "decode"}
        and not envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.get()
    ):
        raise ValueError(
            "Image pruning on P/D requires SGLANG_DISAGG_EXPANDED_INPUT_LEN=1 "
            "on both workers for the atomic length and MRoPE handoff"
        )
