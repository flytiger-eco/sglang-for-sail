"""Shared contract for the opt-in, delta-only multimodal PD handoff.

Enable SGLANG_DISAGG_EXPANDED_INPUT_LEN on both Prefill and Decode. The
original dispatcher must assign a fresh bootstrap_pair_key per paired attempt
(and per batch item). Older gateways can carry an explicitly supplied unique
X-Request-ID header instead. Worker-local generated IDs cannot identify a remote pair.
Use the Python HTTP/tokenizer frontend; Rust and legacy SMG gRPC are unsupported.
"""

from sglang.srt.environ import envs

EXPANDED_INPUT_MAX_INFLIGHT = 4096

DELTA_POSITION_ARCHS = frozenset(
    {
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen4ExpForConditionalGeneration",
    }
)


def original_pair_key(request, raw_request=None):
    """Resolve only dispatcher-provided identity, before worker ID generation."""
    key = getattr(request, "bootstrap_pair_key", None)
    if key is not None:
        return key
    rid = getattr(request, "rid", None)
    if rid is not None:
        return rid
    if "request_id" in getattr(request, "model_fields_set", ()):
        return request.request_id
    if raw_request is not None:
        header = raw_request.headers.get("x-request-id")
        if isinstance(header, str):
            return header
    return None


def expanded_input_enabled():
    return envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.get()


def validate_expanded_input_config(cfg, architecture):
    if not expanded_input_enabled():
        return
    if cfg.disaggregation_mode not in ("prefill", "decode"):
        return
    if envs.SGLANG_RUST_SERVER.get() or getattr(cfg, "smg_grpc_mode", False):
        raise ValueError(
            "Expanded-input PD requires the Python HTTP/tokenizer frontend"
        )
    if getattr(cfg, "disaggregation_transfer_backend", "mooncake") != "mooncake":
        raise ValueError("Expanded-input PD requires the mooncake transfer backend")
    if getattr(cfg, "enable_pd_role_switch", False):
        raise ValueError("Expanded-input PD does not support --enable-pd-role-switch")
    if architecture not in DELTA_POSITION_ARCHS:
        raise ValueError(
            "Expanded-input PD only supports Qwen delta-based MRoPE models"
        )
    if cfg.disaggregation_mode == "decode" and cfg.tokenizer_worker_num > 1:
        raise ValueError("Expanded-input Decode requires --tokenizer-worker-num=1")
    if cfg.disaggregation_mode == "decode" and cfg.attn_cp_size > 1:
        raise ValueError("Expanded-input Decode does not support context parallelism")
    if cfg.pp_size > 1:
        raise ValueError("Expanded-input PD does not support pipeline parallelism")
    if (
        cfg.disaggregation_mode == "prefill"
        and cfg.language_only
        and cfg.encoder_transfer_backend != "zmq_to_tokenizer"
    ):
        raise ValueError(
            "Expanded-input PD requires the zmq_to_tokenizer encoder backend"
        )
    if cfg.disaggregation_mode == "prefill" and not cfg.disagg_decode_urls:
        raise ValueError("Expanded-input Prefill requires --disagg-decode-urls")


def finalize_expanded_input_positions(req):
    """Apply length-dependent options only after authoritative metadata arrives."""
    options = getattr(req, "_deferred_position_options", None)
    if options is None:
        return
    return_logprob, start, token_ids_logprob, routed, routed_start = options
    length = req.effective_input_len
    if not return_logprob:
        start = -1
    if start == -1:
        start = (
            length
            if ((return_logprob and token_ids_logprob is None) or req.is_prefill_only)
            else -1
        )
    if start > length:
        raise ValueError("logprob_start_len exceeds the expanded prompt length")
    if routed and not 0 <= routed_start <= length:
        raise ValueError("routed_experts_start_len is outside the expanded prompt")
    req.logprob_start_len = start
    req.routed_experts_start_len = routed_start
    req._deferred_position_options = None
