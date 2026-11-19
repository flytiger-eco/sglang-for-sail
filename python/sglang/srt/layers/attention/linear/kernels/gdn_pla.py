"""PLA packed GDN prefill with the existing Mamba radix snapshot contract."""

import logging
from dataclasses import dataclass
from itertools import accumulate

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.utils import is_pin_memory_available

logger = logging.getLogger(__name__)


def _load_pla():
    from pla.prefill import flashqla

    if not hasattr(flashqla.ops.flashqla_cute, "folded_coefficients"):
        raise RuntimeError(
            "PLA GDN prefill requires a PLA wheel with folded prefill support"
        )
    return flashqla


@dataclass(frozen=True)
class PlaPrefillLayout:
    lengths: tuple[int, ...]
    starts: tuple[int, ...]
    prefix_lengths: tuple[int, ...]
    prefix_rows: tuple[int, ...]
    tail_rows: tuple[int, ...]
    snapshot_rows: tuple[int, ...]

    @property
    def split(self):
        return bool(self.snapshot_rows)


def build_pla_prefill_layout(lengths, track_chunks, chunk_size):
    """Keep untracked/aligned rows whole; split tracked rows at the h boundary."""
    lengths, track_chunks = tuple(lengths), tuple(track_chunks)
    if len(lengths) != len(track_chunks) or chunk_size <= 0:
        raise ValueError("Invalid PLA prefill row metadata")
    prefix, snapshot = [], []
    for row, (length, chunk) in enumerate(zip(lengths, track_chunks)):
        cut = length if chunk < 0 else chunk * chunk_size
        if length < 0 or not 0 <= cut <= length:
            raise ValueError("PLA snapshot boundary is outside its input row")
        prefix.append(cut)
        if chunk >= 0:
            snapshot.append(row)
    return PlaPrefillLayout(
        lengths,
        tuple(accumulate(lengths, initial=0)),
        tuple(prefix),
        tuple(i for i, n in enumerate(prefix) if n),
        tuple(i for i, (n, p) in enumerate(zip(lengths, prefix)) if n > p),
        tuple(snapshot),
    )


@dataclass
class PlaPrefillPlan:
    layout: PlaPrefillLayout
    raw_cu: tuple[int, ...]
    cu: torch.Tensor
    prefix_cu: torch.Tensor | None = None
    prefix_raw_cu: tuple[int, ...] = ()
    prefix_tokens: torch.Tensor | None = None
    prefix_rows: torch.Tensor | None = None
    tail_cu: torch.Tensor | None = None
    tail_raw_cu: tuple[int, ...] = ()
    tail_tokens: torch.Tensor | None = None
    tail_rows: torch.Tensor | None = None


class PlaGDNKernel(TritonGDNKernel):
    """Ordinary packed prefill only; unsupported batches retain Triton."""

    def __init__(self):
        super().__init__()
        self.pla = _load_pla()
        self._logged = set()
        logger.info(
            "PLA GDN prefill extension: %s", self.pla.ops.flashqla_cute.__file__
        )

    def prepare_plan(self, lengths, track_chunks, chunk_size, device):
        layout = build_pla_prefill_layout(lengths, track_chunks, chunk_size)
        raw = layout.starts

        def tensor(values, dtype=torch.int64):
            host = torch.tensor(
                values, dtype=dtype, pin_memory=is_pin_memory_available(device)
            )
            return host.to(device, non_blocking=True)

        plan = PlaPrefillPlan(layout, raw, tensor(raw, torch.int32))
        if not layout.split:
            return plan
        for kind, rows in [("prefix", layout.prefix_rows), ("tail", layout.tail_rows)]:
            counts, tokens = [], []
            for row in rows:
                start, length, cut = (
                    layout.starts[row],
                    layout.lengths[row],
                    layout.prefix_lengths[row],
                )
                lo, hi = (
                    (start, start + cut)
                    if kind == "prefix"
                    else (start + cut, start + length)
                )
                counts.append(hi - lo)
                tokens.extend(range(lo, hi))
            offsets = tuple(accumulate(counts, initial=0))
            setattr(plan, kind + "_raw_cu", offsets)
            setattr(plan, kind + "_cu", tensor(offsets, torch.int32))
            setattr(plan, kind + "_tokens", tensor(tokens))
            setattr(plan, kind + "_rows", tensor(rows))
        return plan

    def can_use_packed_prefill(
        self, packed, pool, num_qk_heads, num_v_heads, head_dim, plan
    ):
        return (
            plan is not None
            and plan.raw_cu[-1] > 0
            and packed.is_cuda
            and packed.dtype == torch.bfloat16
            and packed.ndim == 2
            and packed.stride(1) == 1
            and packed.shape[0] >= plan.raw_cu[-1]
            and pool.dtype == torch.float32
            and pool.is_contiguous()
            and head_dim == 128
            and (num_v_heads, num_qk_heads) == (24, 8)
            and torch.cuda.get_device_capability(packed.device) == (8, 9)
            and not torch.cuda.is_current_stream_capturing()
        )

    def packed_extend(
        self,
        packed,
        a,
        b,
        *,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        num_qk_heads,
        num_v_heads,
        plan,
    ):
        cache_indices = cache_indices.to(torch.int32).contiguous()
        A_log, dt_bias = A_log.float(), dt_bias.float()

        def run(x, ra, rb, slots, cu, raw):
            return self.pla.gdn_prefill_packed_qkv_fwd(
                x,
                ra,
                rb,
                A_log,
                dt_bias,
                cu,
                num_qk_heads=num_qk_heads,
                num_v_heads=num_v_heads,
                initial_state=ssm_states,
                initial_state_indices=slots,
                raw_cu_seqlens=raw,
                output_final_state=False,
            )[2]

        layout = plan.layout
        label = "radix_snapshot" if layout.split else "whole_rows"
        if label not in self._logged:
            logger.info(
                "PLA GDN packed prefill active: %s, rows=%d", label, len(layout.lengths)
            )
            self._logged.add(label)
        if not layout.split:
            return run(packed, a, b, cache_indices, plan.cu, plan.raw_cu), None

        def part(kind):
            tokens = getattr(plan, kind + "_tokens")
            rows = getattr(plan, kind + "_rows")
            if not rows.numel():
                return None
            if len(layout.lengths) == 1:
                cut, valid = layout.prefix_lengths[0], layout.lengths[0]
                region = slice(0, cut) if kind == "prefix" else slice(cut, valid)
                x, ra, rb = packed[region], a[region], b[region]
                slots = cache_indices
            else:
                x, ra, rb = (t.index_select(0, tokens) for t in (packed, a, b))
                slots = cache_indices.index_select(0, rows)
            return run(
                x,
                ra,
                rb,
                slots,
                getattr(plan, kind + "_cu"),
                getattr(plan, kind + "_raw_cu"),
            )

        prefix_output = part("prefix")
        # Triton's h snapshots are activation-dtype states at chunk entry.
        # Preserve that rounding for radix restores; continue the live row
        # from the unrounded fp32 final state written by the prefix call.
        snapshots = (
            ssm_states.index_select(0, cache_indices.long()).to(packed.dtype).float()
        )
        tail_output = part("tail")
        output = packed.new_empty(1, packed.shape[0], num_v_heads, 128)
        if prefix_output is not None:
            output.index_copy_(1, plan.prefix_tokens, prefix_output)
        if tail_output is not None:
            output.index_copy_(1, plan.tail_tokens, tail_output)
        output[:, plan.raw_cu[-1] :].zero_()
        return output, snapshots
