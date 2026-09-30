"""Compatibility imports for the release kernel paths."""

from sglang.kernels.ops.gemm.small_gemm_bf16 import MAX_M as MAX_M
from sglang.kernels.ops.gemm.small_gemm_bf16 import (
    can_use_n32k5120_gemm as can_use_n32k5120_gemm,
)
from sglang.kernels.ops.gemm.small_gemm_bf16 import (
    n32k5120_gemm_bf16 as n32k5120_gemm_bf16,
)
