from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.triton_ops.gather_spec_extras import gather_spec_extras
from sglang.srt.utils import is_cuda, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.eagle_info import EagleDraftInput


def decide_needs_cpu_seq_lens(
    server_args: ServerArgs,
    attn_backends: Sequence[AttentionBackend],
) -> bool:
    """Whether FutureMap must publish seq_lens_cpu / sum.

    OR over per-backend needs_cpu_seq_lens; force True under TBO (it reads the
    CPU mirror outside the backend layer to split the batch) or ngram (its
    USE_FULL_MASK verify path reads the host mirror regardless of backend).
    """
    if server_args.enable_two_batch_overlap:
        # FIXME: support TBO without seq lens cpu value
        return True
    algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if algo.is_ngram():
        # ngram's USE_FULL_MASK verify path reads seq_lens_cpu per req to size
        # the tree mask, regardless of the attn backend (e.g. Triton opts out).
        return True
    # Skip unset slots (e.g. draft_extend_attn_backend on some spec configs);
    # missing flag -> True so undeclared backends stay on the legacy path.
    return any(
        getattr(b, "needs_cpu_seq_lens", True) for b in attn_backends if b is not None
    )


def decide_needs_confidence_relay(server_args: ServerArgs) -> bool:
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if not algo.is_dspark():
        return False
    return read_ragged_verify_mode() is not RaggedVerifyMode.STATIC


_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

# Token-buf consume tracking: init to -1, assert non-negative on gather,
# write -1 back. Catches "gather without intermediate stash" bugs. CI enables
# via the existing SGLANG_IS_IN_CI; off in production.
_DEBUG_ASSERT = envs.SGLANG_IS_IN_CI.get()


@torch.compile(dynamic=True, disable=_is_npu)
def _assert_nonneg_and_invalidate(
    values: torch.Tensor, buf: torch.Tensor, indices: torch.Tensor
) -> None:
    """Fused: assert all `values >= 0` and scatter -1 into `buf[indices]`.
    Compiled so the reduction + assert + scatter run as one kernel launch."""
    torch._assert_async((values >= 0).all())
    buf[indices] = -1


def resolve_forward_inputs(batch: ScheduleBatch, future_map: FutureMap) -> None:
    """Materialize input_ids at forward entry. Two sources:

    - Prefill: H2D copy from pinned CPU staging (prefill_input_ids_cpu).
    - Decode/spec_v2: gather from FutureMap (last iter's sampled token).
    """
    if batch.prefill_input_ids_cpu is not None:
        prefill_gpu = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        if batch.mix_running_indices is not None:
            decode_gpu = future_map.output_tokens_buf[batch.mix_running_indices]
            if _DEBUG_ASSERT:
                _assert_nonneg_and_invalidate(
                    decode_gpu,
                    future_map.output_tokens_buf,
                    batch.mix_running_indices,
                )
            batch.input_ids = torch.cat([prefill_gpu, decode_gpu])
        else:
            batch.input_ids = prefill_gpu
        batch.prefill_input_ids_cpu = None
        batch.mix_running_indices = None
    elif batch.input_ids is None and future_map.spec_algo.is_none():
        batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                batch.input_ids, future_map.output_tokens_buf, batch.req_pool_indices
            )

    # Only the overlap path relays spec extras through the future_map; the
    # synchronous (non-overlap) V2 path installs next_draft_input directly.
    if batch.enable_overlap and not batch.spec_algorithm.is_none():
        future_map._resolve_spec_extras(batch)


CONFIDENCE_RELAY_RING_LAG: int = 2
CONFIDENCE_RELAY_RING_DEPTH: int = CONFIDENCE_RELAY_RING_LAG + 1


class ResolvedConfidence(msgspec.Struct):

    confidence: torch.Tensor
    generation: torch.Tensor


@dataclass
class RelayPayload:
    """Per-iteration stash payload for the FutureMap bufs. Non-spec fills only
    `bonus_tokens`; which spec extras get relayed is decided by
    `FutureMap.spec_algo`, not by this payload's shape."""

    bonus_tokens: torch.Tensor
    topk_p: Optional[torch.Tensor] = None
    topk_index: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    draft_probs: Optional[torch.Tensor] = None
    dsa_topk_indices: Optional[torch.Tensor] = None

    @classmethod
    def from_draft_input(cls, draft_input: EagleDraftInput) -> RelayPayload:
        return cls(
            bonus_tokens=draft_input.bonus_tokens,
            topk_p=draft_input.topk_p,
            topk_index=draft_input.topk_index,
            hidden_states=draft_input.hidden_states,
            draft_probs=getattr(draft_input, "draft_probs", None),
            dsa_topk_indices=getattr(draft_input, "dsa_topk_indices", None),
        )


class ConfidenceRelay(msgspec.Struct):

    device: torch.device
    req_pool_size: int
    pool: Any
    confidence_buf: Optional[torch.Tensor] = None
    conf_ring: Optional[torch.Tensor] = None
    gen_ring: Optional[torch.Tensor] = None
    copy_done: Optional[list] = None
    ring_pos: int = 0
    initialized: bool = False

    def _lazy_init(self, confidence: torch.Tensor) -> None:
        self.initialized = True
        gamma = confidence.shape[-1]
        self.confidence_buf = torch.empty(
            (self.req_pool_size, gamma), dtype=torch.float32, device=self.device
        )
        if _is_cuda:
            depth = CONFIDENCE_RELAY_RING_DEPTH
            self.conf_ring = torch.empty(
                (depth, self.req_pool_size, gamma),
                dtype=torch.float32,
                pin_memory=True,
            )
            self.gen_ring = torch.zeros((depth, self.req_pool_size), dtype=torch.int64)
            self.copy_done = [
                torch.get_device_module(self.device).Event() for _ in range(depth)
            ]

    def scatter(self, indices: torch.Tensor, confidence: torch.Tensor) -> None:
        if not self.initialized:
            self._lazy_init(confidence)
        self.confidence_buf[indices] = confidence.to(self.confidence_buf.dtype)

    def issue_ring_copy(self, *, stream, publish_ready) -> None:
        if not self.initialized or stream is None or publish_ready is None:
            return
        slot = self.ring_pos % CONFIDENCE_RELAY_RING_DEPTH
        stream.wait_event(publish_ready)
        with torch.get_device_module(self.device).stream(stream):
            self.conf_ring[slot].copy_(self.confidence_buf, non_blocking=True)
            self.copy_done[slot].record()
        self.gen_ring[slot].copy_(self.pool.req_generation)
        self.ring_pos += 1

    def resolve(
        self, batch: ScheduleBatch, *, stream, publish_ready
    ) -> Optional[ResolvedConfidence]:
        if not self.initialized:
            return None
        draft_input = batch.spec_info
        if draft_input is None:
            return None
        fi = draft_input.future_indices
        if fi is None or fi.shape[0] == 0:
            return None

        if stream is None or publish_ready is None:
            idx = batch.req_pool_indices
            idx_cpu = batch.req_pool_indices_cpu
            return ResolvedConfidence(
                confidence=self.confidence_buf[idx].cpu(),
                generation=self.pool.req_generation[idx_cpu].clone(),
            )

        if self.ring_pos < CONFIDENCE_RELAY_RING_LAG:
            return None
        slot = (self.ring_pos - CONFIDENCE_RELAY_RING_LAG) % CONFIDENCE_RELAY_RING_DEPTH
        if not self.copy_done[slot].query():
            return None

        idx_cpu = batch.req_pool_indices_cpu
        return ResolvedConfidence(
            confidence=self.conf_ring[slot][idx_cpu],
            generation=self.gen_ring[slot][idx_cpu],
        )


class FutureMap:
    """Always-on pool-indexed relay for cross-iter values. Forward writes via
    publish/stash; next iter reads via resolve_forward_inputs / resolve_seq_lens_cpu.
    """

    def __init__(
        self,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
        req_to_token_pool: ReqToTokenPool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):
        # Bufs indexed by req_pool_idx; slot 0 mirrors KV padding row so
        # CUDA-graph padded batches (req_pool_idx == 0) are harmless.
        self.device = device
        self.spec_algo = spec_algo
        # Computed by decide_needs_cpu_seq_lens(); see that helper for the
        # full decision (per-backend flag + TBO / piecewise CG overrides).
        self.needs_cpu_seq_lens = needs_cpu_seq_lens
        self.needs_confidence_relay = needs_confidence_relay
        self.req_pool_size = req_to_token_pool.req_to_token.shape[0]

        self.output_tokens_buf = (
            torch.full((self.req_pool_size,), -1, dtype=torch.int64, device=self.device)
            if _DEBUG_ASSERT
            else torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )
        )
        self.new_seq_lens_buf = torch.empty(
            (self.req_pool_size,), dtype=torch.int64, device=self.device
        )
        # Pinned host copy of new_seq_lens_buf + private stream for fwd-prepare
        # D2H pulls (gated only on publish, off the schedule stream). CUDA-only:
        # recovers occupancy lost to the WAR barrier (also CUDA-only); other
        # platforms have no barrier and use the plain .cpu() bootstrap path.
        if _is_cuda:
            self.new_seq_lens_cpu_pinned = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, pin_memory=True
            )
            self.fwd_prepare_d2h_stream = torch.get_device_module(self.device).Stream()
        else:
            self.new_seq_lens_cpu_pinned = None
            self.fwd_prepare_d2h_stream = None
        if self.spec_algo.is_some():
            self._forward_buf_initialized = False

        self.publish_ready = None  # lazy device.Event(); only spec_v2 needs it

        self.confidence_relay = ConfidenceRelay(
            device=self.device,
            req_pool_size=self.req_pool_size,
            pool=req_to_token_pool,
        )

    def _lazy_init_forward_buf(self, payload: RelayPayload):
        # Local import (see decide_needs_cpu_seq_lens): keep module-level deps leaf.
        from sglang.srt.speculative.spec_utils import spec_need_hidden_states

        self._forward_buf_initialized = True

        self.need_verified_id = getattr(draft_input, "verified_id", None) is not None
        self.need_bonus_tokens = getattr(draft_input, "bonus_tokens", None) is not None
        self.need_topk = self.spec_algo.need_topk()
        self.need_hidden_states = (
            spec_need_hidden_states()
            and getattr(draft_input, "hidden_states", None) is not None
        )

        if self.need_verified_id:
            verified_id0 = draft_input.verified_id[0]
            self.verified_id_buf = (
                torch.full(
                    (self.req_pool_size, *verified_id0.shape),
                    -1,
                    dtype=verified_id0.dtype,
                    device=self.device,
                )
                if _DEBUG_ASSERT
                else torch.empty(
                    (self.req_pool_size, *verified_id0.shape),
                    dtype=verified_id0.dtype,
                    device=self.device,
                )
            )
        if self.need_topk:
            topk_p0 = draft_input.topk_p[0]
            topk_index0 = draft_input.topk_index[0]
            self.topk_p_buf = torch.empty(
                (self.req_pool_size, *topk_p0.shape),
                dtype=topk_p0.dtype,
                device=self.device,
            )
            self.topk_index_buf = torch.empty(
                (self.req_pool_size, *topk_index0.shape),
                dtype=topk_index0.dtype,
                device=self.device,
            )
        if self.need_hidden_states:
            hidden_states0 = draft_input.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.req_pool_size, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

        self.draft_probs_buf = None
        if getattr(draft_input, "draft_probs", None) is not None:
            draft_probs0 = draft_input.draft_probs[0]
            self.draft_probs_buf = torch.empty(
                (self.req_pool_size, *draft_probs0.shape),
                dtype=draft_probs0.dtype,
                device=self.device,
            )

        self.dsa_topk_indices_buf = None
        if payload.dsa_topk_indices is not None:
            seed0 = payload.dsa_topk_indices[0]
            self.dsa_topk_indices_buf = torch.empty(
                (self.req_pool_size, *seed0.shape),
                dtype=payload.dsa_topk_indices.dtype,
                device=self.device,
            )

    def resolve_confidence_cpu(
        self, batch: ScheduleBatch
    ) -> Optional[ResolvedConfidence]:
        if not self.needs_confidence_relay:
            return None
        return self.confidence_relay.resolve(
            batch,
            stream=self.fwd_prepare_d2h_stream,
            publish_ready=self.publish_ready,
        )

    def _resolve_spec_extras(self, batch: ScheduleBatch) -> None:
        if self.spec_algo.is_ngram():
            # FIXME: remove once precomputed draft is supported.
            return
        draft_input: EagleDraftInput = batch.spec_info
        if draft_input is None:
            # FIXME(lsyin): only prefill; not compatible with mixed mode
            return
        if self.spec_algo.is_dflash() and getattr(
            draft_input, "direct_carry_valid", False
        ):
            return
        indices = draft_input.future_indices
        if indices.shape[0] == 0:
            return
        # FIXME: indices = batch.req_pool_indices, pinned 2 iters via
        # record_batch_in_overlap; record_stream here is redundant.
        indices.record_stream(torch.get_device_module(self.device).current_stream())
        if self.need_verified_id:
            draft_input.verified_id = self.verified_id_buf[indices]
        if self.need_topk:
            hidden_states_buf = (
                self.hidden_states_buf if self.need_hidden_states else None
            )
            (
                draft_input.topk_p,
                draft_input.topk_index,
                bonus_tokens,
                hidden_states,
            ) = gather_spec_extras(
                indices,
                self.topk_p_buf,
                self.topk_index_buf,
                self.output_tokens_buf,
                hidden_states_buf,
            )
            if self.need_bonus_tokens:
                draft_input.bonus_tokens = bonus_tokens
            if hidden_states is not None:
                draft_input.hidden_states = hidden_states
            if self.draft_probs_buf is not None and draft_input.draft_probs is not None:
                draft_input.draft_probs = self.draft_probs_buf[indices]
        elif self.need_bonus_tokens:
            draft_input.bonus_tokens = self.output_tokens_buf[indices]
        if self.need_hidden_states and not self.need_topk:
            draft_input.hidden_states = self.hidden_states_buf[indices]
        if _DEBUG_ASSERT:
            if self.need_verified_id:
                _assert_nonneg_and_invalidate(
                    draft_input.verified_id, self.verified_id_buf, indices
                )
            if self.need_bonus_tokens:
                _assert_nonneg_and_invalidate(
                    draft_input.bonus_tokens, self.output_tokens_buf, indices
                )

    def resolve_seq_lens_cpu(self, batch: ScheduleBatch) -> None:
        # Lazy pull from new_seq_lens_buf for spec_v2 (accept_lens not known to
        # schedule). DFLASH intentionally keeps host-side lengths lagging and
        # uses its carried KV allocation watermark for planning, so only the GPU
        # seq_lens is resolved there. Other spec-v2 algorithms still need the CPU
        # mirror for host planning; use a private D2H stream for those copies.
        draft_input = batch.spec_info
        if draft_input is None:
            return
        if self.spec_algo.is_dflash() and getattr(
            draft_input, "direct_carry_valid", False
        ):
            batch.seq_lens = draft_input.new_seq_lens
            return

        fi = draft_input.future_indices
        if fi is None:
            return
        if self.publish_ready is not None:
            if _is_hip:
                # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
                self.publish_ready.synchronize()
            else:
                self.publish_ready.wait()
        batch.seq_lens = self.new_seq_lens_buf[fi]

        if self.spec_algo.is_dflash():
            # DFLASH keeps seq_lens_cpu as the lagging committed host view;
            # planning/reserved host lengths live on DFlashDraftInputV2.
            return

        if not self.needs_cpu_seq_lens:
            # GPU gather above is kept (SB.seq_lens must advance each verify);
            # skip the .cpu() D2H. Downstream takes the GPU-only path.
            batch.seq_lens_cpu = None
            batch.seq_lens_sum = None
            return

        if self.fwd_prepare_d2h_stream is None or self.publish_ready is None:
            batch.seq_lens_cpu = batch.seq_lens.cpu()  # bootstrap / non-CUDA
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            return

        # Mechanism: don't sync the schedule stream; gate a private stream on the
        # publish event and copy into the static pinned buffer.
        self.fwd_prepare_d2h_stream.wait_event(self.publish_ready)
        with torch.get_device_module(self.device).stream(self.fwd_prepare_d2h_stream):
            self.new_seq_lens_cpu_pinned.copy_(self.new_seq_lens_buf, non_blocking=True)
        self.fwd_prepare_d2h_stream.synchronize()

        # FIXME: fi == batch.req_pool_indices; unify future_indices and req_pool_indices.
        batch.seq_lens_cpu = self.new_seq_lens_cpu_pinned[batch.req_pool_indices_cpu]
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())

    def publish(
        self,
        future_indices: torch.Tensor,
        new_seq_lens: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> None:
        indices = future_indices
        if indices.shape[0] == 0:
            return  # DP idle
        self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)
        publish_confidence = self.needs_confidence_relay and confidence is not None
        if publish_confidence:
            self.confidence_relay.scatter(indices, confidence)
        # Only spec_v2 needs the event; it gates the seq_lens D2H on the private stream.
        if self.spec_algo.is_some():
            if self.publish_ready is None:
                self.publish_ready = torch.get_device_module(self.device).Event()
            self.publish_ready.record()
            self._publish_fresh = True
        if publish_confidence:
            self.confidence_relay.issue_ring_copy(
                stream=self.fwd_prepare_d2h_stream,
                publish_ready=self.publish_ready,
            )

    def stash(
        self,
        future_indices: torch.Tensor,
        payload: Union[torch.Tensor, EagleDraftInput],
    ) -> None:
        if self.spec_algo.is_ngram():
            # FIXME: remove once precomputed draft is supported.
            return
        indices = future_indices
        if indices.shape[0] == 0:
            # DP idle: payload is empty stub; lazy-init shape peek would IndexError.
            return
        # Dispatch by payload type, not spec_algo: non-spec decode passes a
        # token Tensor here.
        # FIXME(lsyin): unify this relay path with a dataclass instead of the
        # Tensor / EagleDraftInput type switch.
        if isinstance(payload, torch.Tensor):
            self.output_tokens_buf[indices] = payload.to(torch.int64)
            return

        draft_input: EagleDraftInput = payload
        if not self._forward_buf_initialized:
            self._lazy_init_forward_buf(draft_input)
        if self.need_verified_id:
            self.verified_id_buf[indices] = draft_input.verified_id.to(
                self.verified_id_buf.dtype
            )
        if self.need_bonus_tokens:
            self.output_tokens_buf[indices] = draft_input.bonus_tokens.to(
                self.output_tokens_buf.dtype
            )

        if self.need_topk:
            self.topk_p_buf[indices] = draft_input.topk_p.to(self.topk_p_buf.dtype)
            self.topk_index_buf[indices] = draft_input.topk_index.to(
                self.topk_index_buf.dtype
            )
        if self.need_hidden_states:
            self.hidden_states_buf[indices] = draft_input.hidden_states.to(
                self.hidden_states_buf.dtype
            )
        if self.draft_probs_buf is not None and draft_input.draft_probs is not None:
            self.draft_probs_buf[indices] = draft_input.draft_probs
