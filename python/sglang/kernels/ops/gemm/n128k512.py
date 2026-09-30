"""Compatibility imports for the release kernel paths."""

from sglang.kernels.ops.gemm.small_gemm_bf16 import (
    can_use_n128k512_gemm as can_use_n128k512_gemm,
)
from sglang.kernels.ops.gemm.small_gemm_bf16 import (
    n128k512_gemm_bf16 as n128k512_gemm_bf16,
)
