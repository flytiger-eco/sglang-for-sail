"""Stub for flashinfer autotune.

On PPU and other non-CUDA platforms, all functions are no-ops.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)


def should_run_flashinfer_autotune(
    model_runner: ModelRunner, *, for_speculative_draft: bool = False
) -> bool:
    """Check if flashinfer autotune should be run."""
    from sglang.srt.utils import is_cuda

    if not is_cuda():
        return False
    if model_runner.server_args.disable_flashinfer_autotune:
        return False
    return False


def maybe_flashinfer_autotune_speculative_draft(
    runner: BaseRunner,
    forward_fn: Callable[[], None],
    *,
    post_warmup_hook: Optional[Callable[[], None]] = None,
    skip_logits: bool = False,
) -> None:
    """Run speculative draft flashinfer autotune. No-op on non-CUDA platforms."""
    mr = runner.model_runner
    if not should_run_flashinfer_autotune(mr, for_speculative_draft=True):
        return
