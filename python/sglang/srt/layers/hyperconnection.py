from typing import Optional

import msgspec
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.hc_mix_triton import fused_hc_mix, fused_hc_mix_supported
from sglang.srt.layers.hc_mix_up_triton import (
    hc_mix_up,
    hc_mix_up_shape_ok,
    hc_mix_up_supported,
)
from sglang.srt.utils import is_ppu

_is_ppu = is_ppu()


_MAX_SPLIT_COMBINE_ROWS = 32


class HyperConnectionConfig(msgspec.Struct, frozen=True):
    hc_count: int = 4
    hidden_size: int = 64
    params_dtype: torch.dtype = torch.bfloat16
    mtp_hc: bool = False
    hc_lowrank: int = 16
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = False


class GroupedGemmaRMSNorm(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float = 1e-6, group_size: Optional[int] = None
    ):
        super().__init__()
        if group_size is not None and hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight.weight_loader = self._weight_loader
        # The JIT kernel requires group_size to be a multiple of 512; this is
        # init-static, so resolve it once here (device/dtype stay per-call).
        effective_group_size = group_size if group_size is not None else hidden_size
        self._jit_group_size = (
            effective_group_size if effective_group_size % 512 == 0 else None
        )

    def _weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._jit_group_size is not None
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
                grouped_gemma_rmsnorm,
            )

            return grouped_gemma_rmsnorm(
                x, self.weight, self._jit_group_size, self.variance_epsilon
            )
        input_dtype = x.dtype
        x_float = x.float()
        if self.group_size is None:
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x_float * torch.rsqrt(variance + self.variance_epsilon)
        else:
            x_grouped = x_float.reshape(
                *x_float.shape[:-1],
                x_float.shape[-1] // self.group_size,
                self.group_size,
            )
            variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
            x_norm = (
                x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            ).flatten(-2)
        return (x_norm * (1.0 + self.weight.float())).to(input_dtype)


class HyperConnectionBase(nn.Module):
    def __init__(
        self,
        config: HyperConnectionConfig,
        use_mix: bool = True,
        use_combine: bool = True,
        role: Optional[str] = None,
    ):
        super().__init__()

        self.config = config
        self.hc_count = config.hc_count
        if config.mtp_hc and role is not None and "mtp" in role:
            self.hc_count = self.hc_count + 1
        self.hidden_size = config.hidden_size
        self.params_dtype = config.params_dtype

    def mix(self, hyper_input: torch.Tensor):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        mixed_input = hyper_input.view(
            *hyper_input.shape[:-1], self.hc_count, self.hidden_size
        ).mean(dim=-2)
        return mixed_input, hyper_input

    def combine(
        self, block_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        assert residual.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        residual_reshaped = residual.view(
            *residual.shape[:-1], self.hc_count, self.hidden_size
        )
        combined_output = residual_reshaped + block_output.unsqueeze(-2)
        combined_output = combined_output.view(
            *residual.shape[:-1], self.hc_count * self.hidden_size
        )
        return combined_output


class GatedResidual(HyperConnectionBase):
    def __init__(
        self,
        config: HyperConnectionConfig,
        use_mix: bool = True,
        use_combine: bool = True,
        role: Optional[str] = None,
    ):
        super().__init__(config, use_mix, use_combine, role)

        norm_dim = (
            self.config.hidden_size * self.hc_count
            if self.config.hc_per_branch_norm
            else self.config.hidden_size
        )
        norm_group_size = (
            self.config.hidden_size if self.config.hc_per_branch_norm else None
        )
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_dim, eps=self.config.rms_norm_eps, group_size=norm_group_size
        )

        if use_mix:
            self.input_mix_weight_down = nn.Linear(
                self.hidden_size * self.hc_count,
                self.config.hc_lowrank,
                bias=False,
                device=torch.get_device_module().current_device(),
                dtype=config.params_dtype,
            )
            self.input_mix_weight_up = nn.Linear(
                self.config.hc_lowrank,
                self.hc_count * self.hidden_size,
                bias=False,
                device=torch.get_device_module().current_device(),
                dtype=config.params_dtype,
            )
            from sglang.srt.environ import envs

            lowrank = self.config.hc_lowrank
            self._jit_mix_ok = (
                envs.SGLANG_HC_MIX_CUDA.get()
                and torch.cuda.is_available()
                # The CuTe split-K pair is tcgen05 (sm_100 family) only; the
                # default-on env must not route Hopper/Ada to it.
                and torch.cuda.get_device_capability()[0] == 10
                and (self.hc_count * self.hidden_size) % 2048 == 0
                and self.hidden_size % 8 == 0
                and lowrank > 0
                and lowrank % 8 == 0
            )
            self._mix_up_weight_padded = None
            self._mix_up_fused_ok = envs.SGLANG_HC_MIX_UP_FUSED.get() and (
                hc_mix_up_shape_ok(self.hc_count, self.hidden_size, lowrank)
            )

        if use_combine:
            self.block_inject_weight = nn.Linear(
                self.hidden_size * self.hc_count,
                self.hc_count,
                bias=False,
                device=torch.get_device_module().current_device(),
                dtype=config.params_dtype,
            )
            # The JIT combine kernel requires hidden_size % 8 == 0 and
            # hc_count * hidden_size % 2048 == 0; this is init-static, so
            # resolve it once here (device/dtype stay per-call).
            self._jit_combine_ok = (
                self.hidden_size % 8 == 0
                and (self.hc_count * self.hidden_size) % 2048 == 0
            )
            from sglang.srt.environ import envs

            vecs = self.hc_count * self.hidden_size // 8
            self._split_combine_ok = (
                envs.SGLANG_HC_COMBINE_SPLIT.get()
                and self._jit_combine_ok
                and vecs % (8 * 160) == 0
                and (self.hidden_size // 8) % (vecs // 8) == 0
            )

        def _mix_compute(
            hyper_input_normed: torch.Tensor,
            input_mix_weight_down: torch.Tensor,
            input_mix_weight_up: torch.Tensor,
            hc: int,
            hs: int,
        ) -> torch.Tensor:
            input_mix_weight = F.silu(
                F.linear(hyper_input_normed, input_mix_weight_down) / hc
            )
            input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
            input_mix_weight = torch.sigmoid(input_mix_weight)
            input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
            output = (
                input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))
            ).mean(dim=-2)
            return output

        def _mix_and_gate_compute(
            hyper_input_normed: torch.Tensor,
            down_gate_weight: torch.Tensor,
            input_mix_weight_up: torch.Tensor,
            hc: int,
            hs: int,
            lowrank: int,
        ):
            # Tier-B fusion: one shared GEMM reads normed once and yields both
            # the mix down projection (columns [:lowrank]) and the combine gate
            # pre-activations dot(normed, W_inject) (columns [lowrank:]), so the
            # gate no longer needs its own read of normed at combine time.
            combined = F.linear(hyper_input_normed, down_gate_weight)
            down = combined[..., :lowrank]
            gate_pre = combined[..., lowrank:]
            input_mix_weight = F.silu(down / hc)
            input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
            input_mix_weight = torch.sigmoid(input_mix_weight)
            input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
            output = (
                input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))
            ).mean(dim=-2)
            gates = 2.0 * torch.sigmoid(gate_pre.to(torch.float32) / hc)
            return output, gates

        # Down-projection halves of the two mix chains above, stopping at the
        # up GEMM input; `hc_mix_up` then runs the up GEMM with the
        # sigmoid*normed/mean epilogue fused in.
        def _mix_down_compute(
            hyper_input_normed: torch.Tensor,
            input_mix_weight_down: torch.Tensor,
            hc: int,
        ) -> torch.Tensor:
            return F.silu(F.linear(hyper_input_normed, input_mix_weight_down) / hc)

        def _mix_down_and_gate_compute(
            hyper_input_normed: torch.Tensor,
            down_gate_weight: torch.Tensor,
            hc: int,
            lowrank: int,
        ):
            combined = F.linear(hyper_input_normed, down_gate_weight)
            act = F.silu(combined[..., :lowrank] / hc)
            gates = 2.0 * torch.sigmoid(combined[..., lowrank:].to(torch.float32) / hc)
            return act, gates

        def _combine_compute(
            block_output: torch.Tensor,
            residual: torch.Tensor,
            normed_residual: torch.Tensor,
            block_inject_weight: torch.Tensor,
            hc: int,
            hs: int,
        ) -> torch.Tensor:
            R = residual.unflatten(-1, (hc, hs))
            block_inject_weight_out = 2 * torch.sigmoid(
                F.linear(normed_residual, block_inject_weight) / hc
            )
            injection = block_output.unsqueeze(-2) * block_inject_weight_out.unsqueeze(
                -1
            )
            return (R + injection).flatten(-2)

        self._mix_compute = torch.compile(_mix_compute)
        self._mix_and_gate_compute = torch.compile(_mix_and_gate_compute)
        self._mix_down_compute = torch.compile(_mix_down_compute)
        self._mix_down_and_gate_compute = torch.compile(_mix_down_and_gate_compute)
        self._combine_compute = torch.compile(_combine_compute)

        # Tier-B gate fusion: when the prefill mix down projection and the
        # combine inject weight both exist and the fused combine kernel is
        # usable, fold the gate dot into the down GEMM so normed is read once.
        # The combined [lowrank + hc, K] weight is built lazily on first use
        # (after weights are loaded), mirroring `_mix_up_weight_padded`.
        self._down_gate_weight = None
        self._fuse_gate_into_mix = (
            use_mix and use_combine and getattr(self, "_jit_combine_ok", False)
        )

    def _use_split_combine(self, num_rows: int) -> bool:
        """Use the fused gate+apply path for launch-bound decode batches."""
        return (
            getattr(self, "_split_combine_ok", False)
            and num_rows <= _MAX_SPLIT_COMBINE_ROWS
        )

    def _use_fused_mix_up(self, hyper_input_normed: torch.Tensor) -> bool:
        return getattr(self, "_mix_up_fused_ok", False) and hc_mix_up_supported(
            hyper_input_normed, self.input_mix_weight_up.weight
        )

    def _use_persistent_mix(self, hyper_input_normed: torch.Tensor) -> bool:
        # On PPU the persistent grid-barrier mix costs ~40us at any row count,
        # slower than the down GEMM + hc_mix_up pair at every decode size
        # (1.2x at 2-16 rows, 2x at 1 row).
        if _is_ppu and self._use_fused_mix_up(hyper_input_normed):
            return False
        return fused_hc_mix_supported(
            hyper_input_normed,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
        )

    def _precompute_gates(self, hyper_input_normed: torch.Tensor):
        """Hoist the combine gate to mix time.

        combine() computes a[m, c] = 2 * sigmoid(dot(normed[m], W_inject[c]) / hc)
        and then out = residual + a * block. The gate depends only on normed and
        the inject weight, both available here in mix() where normed is already
        resident (it is also read by the mix down projection). Computing it now
        lets combine() run the apply-only kernel and skip re-reading the whole
        [rows, hc*hidden] normed tensor. Returns an fp32 [rows, hc] tensor, or
        None when the fused combine path is unavailable (then combine() falls
        back to the eager path that recomputes the gate from normed).
        """
        if not getattr(self, "_jit_combine_ok", False):
            return None
        if not hyper_input_normed.is_cuda or hyper_input_normed.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            return None
        if self.block_inject_weight.weight.dtype != hyper_input_normed.dtype:
            return None
        gate_pre = F.linear(hyper_input_normed, self.block_inject_weight.weight)
        return (
            2.0 * torch.sigmoid(gate_pre.to(torch.float32) / self.hc_count)
        ).contiguous()

    def mix(
        self,
        hyper_input: torch.Tensor,
        pre_normed: Optional[torch.Tensor] = None,
    ):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if hyper_input.shape[0] == 0:
            mixed_input = hyper_input.new_empty(
                (*hyper_input.shape[:-1], self.hidden_size), dtype=self.params_dtype
            )
            return mixed_input, (hyper_input, hyper_input, None)

        if pre_normed is not None:
            assert pre_normed.shape == hyper_input.shape
            assert pre_normed.dtype == hyper_input.dtype
            hyper_input_normed = pre_normed
        elif self.config.hc_per_branch_norm:
            hyper_input_normed = self.hc_norm(hyper_input)
        else:
            hyper_input_normed = self.hc_norm(
                hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
            ).flatten(-2)
        use_split_combine = self._use_split_combine(hyper_input_normed.shape[0])
        gates = None
        if (
            self._jit_mix_ok
            and hyper_input_normed.is_cuda
            and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
            and hyper_input_normed.shape[0] <= 24
        ):
            from sglang.kernels.ops.elementwise.hc_mix import (
                hc_mix,
                permute_pad_up_weight,
            )

            if self._mix_up_weight_padded is None:
                self._mix_up_weight_padded = permute_pad_up_weight(
                    self.input_mix_weight_up.weight, self.hc_count
                )
            mixed_input = hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight.data,
                self._mix_up_weight_padded,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        elif self._use_persistent_mix(hyper_input_normed):
            mixed_input = fused_hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        elif self._fuse_gate_into_mix and not use_split_combine:
            # Prefill: fold the combine gate dot into the mix down GEMM so
            # normed is read once. Small decode batches leave gates unset and
            # use the split combine kernel, whose fused gate+apply avoids the
            # fixed launch cost of a standalone tiny gate GEMV.
            if self._down_gate_weight is None:
                self._down_gate_weight = torch.cat(
                    [
                        self.input_mix_weight_down.weight,
                        self.block_inject_weight.weight,
                    ],
                    dim=0,
                ).contiguous()
            if self._use_fused_mix_up(hyper_input_normed):
                act, gates = self._mix_down_and_gate_compute(
                    hyper_input_normed,
                    self._down_gate_weight,
                    self.hc_count,
                    self.config.hc_lowrank,
                )
                mixed_input = hc_mix_up(
                    act,
                    self.input_mix_weight_up.weight,
                    hyper_input_normed,
                    self.hc_count,
                    self.hidden_size,
                )
            else:
                mixed_input, gates = self._mix_and_gate_compute(
                    hyper_input_normed,
                    self._down_gate_weight,
                    self.input_mix_weight_up.weight,
                    self.hc_count,
                    self.hidden_size,
                    self.config.hc_lowrank,
                )
            mixed_input = mixed_input.to(self.params_dtype)
        elif self._use_fused_mix_up(hyper_input_normed):
            act = self._mix_down_compute(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.hc_count,
            )
            mixed_input = hc_mix_up(
                act,
                self.input_mix_weight_up.weight,
                hyper_input_normed,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        else:
            mixed_input = self._mix_compute(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        if gates is None and not use_split_combine:
            gates = self._precompute_gates(hyper_input_normed)
        return mixed_input, (hyper_input, hyper_input_normed, gates)

    def combine(self, block_output: torch.Tensor, residuals) -> torch.Tensor:
        hyper_input, hyper_input_normed, gates = residuals
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        if block_output.shape[0] == 0:
            return hyper_input.to(self.params_dtype)

        jit_combine_inputs_ok = (
            self._jit_combine_ok
            and block_output.is_cuda
            and block_output.dtype in (torch.bfloat16, torch.float16)
            and hyper_input.dtype == block_output.dtype
            and hyper_input_normed.dtype == block_output.dtype
            and self.block_inject_weight.weight.dtype == block_output.dtype
        )
        if self._use_split_combine(block_output.shape[0]) and jit_combine_inputs_ok:
            from sglang.kernels.ops.elementwise.hc_combine import hc_combine_split

            return hc_combine_split(
                block_output,
                hyper_input,
                hyper_input_normed,
                self.block_inject_weight.weight.data,
                self.hc_count,
                self.hidden_size,
            )

        if jit_combine_inputs_ok and gates is not None:
            # Gate was hoisted into mix(); combine is a pure apply that never
            # re-reads normed_residual or inject_weight.
            from sglang.kernels.ops.elementwise.hc_combine import hc_combine_apply

            return hc_combine_apply(
                block_output,
                hyper_input,
                gates,
                self.hc_count,
                self.hidden_size,
            )

        updated_residuals = self._combine_compute(
            block_output,
            hyper_input,
            hyper_input_normed,
            self.block_inject_weight.weight,
            self.hc_count,
            self.hidden_size,
        ).to(self.params_dtype)
        return updated_residuals

    def combine_and_norm(
        self,
        block_output: torch.Tensor,
        residuals,
        next_norm_weight: torch.Tensor,
        next_norm_eps: float,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Combine and precompute the next HC mix normalization.

        The fused CUDA JIT path is used only when combine gates were hoisted
        into ``mix()`` and the next HC uses per-branch grouped RMSNorm.  Other
        paths return ``None`` for the normalized tensor so the next ``mix()``
        executes its regular normalization path.
        """
        hyper_input, hyper_input_normed, gates = residuals
        if block_output.shape[0] == 0:
            return hyper_input.to(self.params_dtype), None

        inputs_ok = (
            self._jit_combine_ok
            and gates is not None
            and block_output.is_cuda
            and block_output.dtype in (torch.bfloat16, torch.float16)
            and hyper_input.dtype == block_output.dtype
            and hyper_input_normed.dtype == block_output.dtype
            and next_norm_weight.dtype == block_output.dtype
            and next_norm_weight.is_cuda
            and next_norm_weight.numel() == self.hc_count * self.hidden_size
        )
        if not inputs_ok:
            return self.combine(block_output, residuals), None

        from sglang.kernels.ops.elementwise.hc_combine import hc_combine_norm

        updated, normed = hc_combine_norm(
            block_output,
            hyper_input,
            gates,
            next_norm_weight,
            self.hc_count,
            self.hidden_size,
            next_norm_eps,
        )
        return updated, normed


HYPERCONNECTION_CLASS_DICT = {
    "hyperconnection_average": HyperConnectionBase,
    "gated_residual_simple": GatedResidual,
}
