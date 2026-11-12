"""[FW_SPARSE] NVTX tagging for DSA (sparse attention) ops.

Tag format (aligned with the model_prof / platform team agreement):

    [FW_SPARSE]: op:<op>,fw:sglang,model:<model>,backend:<backend>,pd_type:<pd>,<k>:<v>,...

Two emission points, exactly one tag per sparse op:

* ``dsa_sparse_nvtx`` wraps ``DeepseekSparseAttnBackend.forward_extend`` /
  ``forward_decode`` (the unified entry). For ``flashmla_kv`` the entry does
  not emit a range: it hands ``pd_type`` / ``max_seqlen_k`` down through a
  ContextVar and the leaf emits the tag, so the range sits around the kernel.
* ``flashmla_kv_leaf_nvtx`` wraps ``flash_mla_with_kvcache`` and tags only when
  ``indices is not None`` (real sparse decode); dense MLA (``indices is None``)
  is never tagged. If a range is already open it stays silent, so entry-level
  tags for other backends are never duplicated.

``SGLANG_PROFILE_NVTX=1`` enables the ranges. While a [FW_SPARSE] range is open
the lower-level FA3 / FlashMLA wrappers skip their own ``[FW_FMHA]`` tag, so the
sparse kernels stay attributed to ``[FW_SPARSE]`` instead of being merged into
full attention by model_prof.
"""

from __future__ import annotations

import functools
import inspect
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from sglang.srt.environ import envs

# ==== NVTX profiling for DSA sparse attention ====
# Import-guarded NVTX helpers (mirror model_runner.py) so non-NVTX builds work.
SGLANG_PROFILE_NVTX = envs.SGLANG_PROFILE_NVTX.get()
if SGLANG_PROFILE_NVTX:
    try:
        from torch.cuda.nvtx import range_pop as th_nvtx_range_pop
        from torch.cuda.nvtx import range_push as th_nvtx_range_push

    except ImportError:
        SGLANG_PROFILE_NVTX = False

# True while a [FW_SPARSE] range is open on this thread. Lower-level FW_FMHA
# wrappers consult it to avoid a nested range overwriting the sparse op
# attribution in model_prof (nested framework ranges overwrite `op_name`).
_IN_SPARSE_RANGE: ContextVar[bool] = ContextVar("sglang_dsa_sparse_nvtx", default=False)

# pd_type / max_seqlen_k handed down from the unified entry to the leaf tag, so
# the leaf range carries the same fields as the entry-level tag it replaces.
_LEAF_HINT: ContextVar[Optional[dict]] = ContextVar(
    "sglang_dsa_leaf_hint", default=None
)

_OP_NAMES = {
    "flashmla_sparse": "flash_mla_sparse_fwd",
    "flashmla_sparse_q8": "flash_mla_sparse_fwd_q8",
    "flashmla_kv": "flash_mla_with_kvcache",
    "fa3": "flash_attn_with_kvcache",
    "flashinfer_sparse_mla": "flashinfer_sparse_mla",
    "tilelang": "tilelang_sparse_mla",
    "triton": "triton_sparse_mla",
    "trtllm": "trtllm_mha",
    "aiter": "aiter_mha",
    "intel_xpu": "intel_xpu_mha",
    "standard_mha": "standard_mha",
}

# Backends whose tag is emitted at the leaf wrapper instead of the entry.
_LEAF_TAGGED_BACKENDS = frozenset({"flashmla_kv"})


def in_sparse_range() -> bool:
    """True while a [FW_SPARSE] range is open on this thread."""
    return SGLANG_PROFILE_NVTX and _IN_SPARSE_RANGE.get()


@functools.lru_cache(maxsize=8)
def _model_basename(model_path: str) -> str:
    return os.path.basename(os.path.normpath(model_path)) or "default"


def _model_name() -> str:
    """Model directory name for the tag.

    Read through the ``model`` config namespace on every call rather than
    caching the name, so a config reload or override is reflected in later
    tags. Only the basename conversion is cached, keyed by path: the hot path
    stays a dict hit when the path is unchanged.
    """
    try:
        from sglang.srt.runtime_context import get_model

        model_path = get_model().model_path
    except Exception:
        return "default"
    return _model_basename(model_path) if model_path else "default"


def build_sparse_tag(op: str, backend: str, pd_type: str, **params) -> str:
    parts = [
        f"op:{op}",
        "fw:sglang",
        f"model:{_model_name()}",
        f"backend:{backend or 'default'}",
        f"pd_type:{pd_type or 'default'}",
    ]
    for key, value in params.items():
        if value is not None:
            parts.append(f"{key}:{value}")
    return "[FW_SPARSE]: " + ",".join(parts)


@contextmanager
def sparse_nvtx_range(op: str, backend: str, pd_type: str, **params):
    if not SGLANG_PROFILE_NVTX:
        yield
        return
    th_nvtx_range_push(build_sparse_tag(op, backend, pd_type, **params))
    token = _IN_SPARSE_RANGE.set(True)
    try:
        yield
    finally:
        _IN_SPARSE_RANGE.reset(token)
        th_nvtx_range_pop()


@contextmanager
def leaf_sparse_range(op: str, backend: str, pd_type: Optional[str] = None, **params):
    """Emit the [FW_SPARSE] range at a leaf wrapper.

    Skips when a range is already open (the caller owns the tag), which keeps
    the entry-level tags of other backends from being duplicated. Without an
    entry hint the phase is reported as ``default`` (unknown) instead of being
    guessed as decode: some paths (e.g. short prefill straight to
    ``flash_mla_with_kvcache``) never pass the decorated entry.
    """
    if not SGLANG_PROFILE_NVTX or _IN_SPARSE_RANGE.get():
        yield
        return
    hint = _LEAF_HINT.get() or {}
    merged = dict(params)
    if merged.get("max_seqlen_k") is None:
        merged["max_seqlen_k"] = hint.get("max_seqlen_k")
    tag = build_sparse_tag(
        op, backend, pd_type or hint.get("pd_type") or "default", **merged
    )
    th_nvtx_range_push(tag)
    token = _IN_SPARSE_RANGE.set(True)
    try:
        yield
    finally:
        _IN_SPARSE_RANGE.reset(token)
        th_nvtx_range_pop()


def _safe_max_seqlen_k(cache_seqlens) -> Optional[int]:
    """max_seqlen_k for the leaf tag without breaking CUDA graph capture.

    Reading ``cache_seqlens.max()`` would sync the stream, which is illegal
    during graph capture, so capture keeps the entry-provided value (or None).
    """
    if cache_seqlens is None:
        return None
    try:
        import torch

        if torch.cuda.is_current_stream_capturing():
            return None
        return int(cache_seqlens.max().item())
    except Exception:
        return None


def flashmla_kv_params(q, k_cache, cache_seqlens, head_dim_v, indices) -> dict:
    """Shape fields for the leaf tag.

    Values handed down by the unified entry win, so a leaf tag is identical to
    the entry-level tag it replaces (no CUDA sync, no semantic drift). When the
    leaf runs on its own, fall back to the tensors it actually received.
    """
    hint = _LEAF_HINT.get() or {}

    seqlen_q = hint.get("seqlen_q")
    if seqlen_q is None and q is not None:
        seqlen_q = q.shape[-3] if q.dim() == 4 else 1

    max_seqlen_k = hint.get("max_seqlen_k")
    if max_seqlen_k is None:
        max_seqlen_k = _safe_max_seqlen_k(cache_seqlens)

    head_dim = hint.get("head_dim")
    if head_dim is None and q is not None:
        head_dim = q.shape[-1]

    num_heads = hint.get("num_heads")
    if num_heads is None and q is not None:
        num_heads = q.shape[-2]

    topk = hint.get("topk")
    if topk is None and indices is not None and indices.dim() >= 1:
        topk = indices.shape[-1]

    dtype = hint.get("dtype")
    if dtype is None and q is not None:
        dtype = str(q.dtype).replace("torch.", "")

    bs = hint.get("bs")
    if bs is None and cache_seqlens is not None:
        bs = len(cache_seqlens)

    return {
        "seqlen_q": seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "topk": topk,
        "dtype": dtype,
        "bs": bs,
    }


def flashmla_kv_leaf_nvtx(fn):
    """Tag ``flash_mla_with_kvcache`` calls that run sparse decode.

    Only ``indices is not None`` is sparse attention; dense MLA KV decode is
    left untagged so it is not misreported as DSA.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not SGLANG_PROFILE_NVTX:
            return fn(*args, **kwargs)
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError:
            return fn(*args, **kwargs)
        bound.apply_defaults()
        indices = bound.arguments.get("indices")
        if indices is None:
            return fn(*args, **kwargs)
        with leaf_sparse_range(
            op="flash_mla_with_kvcache",
            backend="flashmla_kv",
            **flashmla_kv_params(
                bound.arguments.get("q"),
                bound.arguments.get("k_cache"),
                bound.arguments.get("cache_seqlens"),
                bound.arguments.get("head_dim_v"),
                indices,
            ),
        ):
            return fn(*args, **kwargs)

    return wrapper


def dsa_sparse_nvtx(fn):
    """Emit one [FW_SPARSE] range per DSA attention op.

    Wraps ``DeepseekSparseAttnBackend.forward_extend`` / ``forward_decode``;
    every DSA backend implementation dispatches from these two methods. For
    leaf-tagged backends (``flashmla_kv``) the pd_type is handed to the leaf
    instead of opening a range here.

    The wrapper forwards ``*args, **kwargs`` untouched, so the decorated
    methods keep their original call contract (positional or keyword). The
    fields the tag needs are bound from the signature at decoration time.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Disabled: one branch and the untouched call, as before.
        if not SGLANG_PROFILE_NVTX or sig is None:
            return fn(*args, **kwargs)

        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError:
            # Let the wrapped method raise its own TypeError.
            return fn(*args, **kwargs)
        arguments = bound.arguments
        self = arguments.get("self")
        forward_batch = arguments.get("forward_batch")
        if self is None or forward_batch is None:
            # Unexpected shape: never change the call, just skip the tag.
            return fn(*args, **kwargs)
        q = arguments.get("q")
        topk_indices = arguments.get("topk_indices")

        mode = forward_batch.forward_mode
        decode_like = (
            mode.is_decode() or mode.is_target_verify() or mode.is_draft_extend_v2()
        )
        if mode.is_mixed():
            pd_type = "default"
        else:
            pd_type = "decode" if decode_like else "prefill"

        configured_impl = self.dsa_decode_impl if decode_like else self.dsa_prefill_impl
        # kpool tails may rewrite flashmla_sparse -> fa3 / trtllm at runtime.
        backend = self._resolve_kpool_tail_backend(topk_indices, configured_impl)
        if not decode_like and getattr(self, "use_mha", False):
            backend = "standard_mha"

        metadata = getattr(self, "forward_metadata", None)
        topk = (
            topk_indices.shape[-1]
            if topk_indices is not None
            else getattr(self, "dsa_index_topk", None)
        )
        params = {
            "seqlen_q": q.shape[0] if q is not None else None,
            "max_seqlen_k": getattr(metadata, "max_seq_len_k", None),
            "head_dim": q.shape[-1] if q is not None else None,
            "num_heads": (q.shape[-2] if q is not None and q.dim() >= 3 else None),
            "topk": topk,
            "dtype": (str(q.dtype).replace("torch.", "") if q is not None else None),
            "bs": forward_batch.batch_size,
        }

        # flashmla_kv is tagged at the leaf; only the hint travels down here.
        if backend in _LEAF_TAGGED_BACKENDS:
            token = _LEAF_HINT.set(dict(params, pd_type=pd_type))
            try:
                return fn(*args, **kwargs)
            finally:
                _LEAF_HINT.reset(token)

        with sparse_nvtx_range(
            op=_OP_NAMES.get(backend, backend),
            backend=backend,
            pd_type=pd_type,
            **params,
        ):
            return fn(*args, **kwargs)

    return wrapper
