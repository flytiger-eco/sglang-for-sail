"""Explicit CPU adapters; source locations and their real assertions are preserved."""

import importlib
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ppu.unit.test_deepseek_v4_dspark_quant_metadata import (  # noqa: E402,F401
    TestDequantSupported,
    TestDsparkV4QuantMetadata,
    TestFusedCommitKvProjSupport,
)
from ppu.unit.test_dsa_prefill_graph_int8 import (  # noqa: E402,F401
    TestDsaPrefillGraphInt8,
)

register_cpu_ci(est_time=10, suite="stage-a-test-cpu-ppu")

# The standalone candidates mock dispatch boundaries/extract metadata. Require
# real checkout imports first so those techniques cannot hide broken imports in
# PR CI. Import failures are collection failures, never passing skips.
_checkout = Path(__file__).resolve().parents[4] / "python"
for _name in (
    "sglang.srt.layers.attention.dsa.dsa_prefill_cuda_graph",
    "sglang.srt.models.deepseek_v4_dspark",
    "sglang.srt.layers.quantization.compressed_tensors.utils",
    "sglang.kernels.ops.speculative.dspark.dspark_draft_model",
):
    _module = importlib.import_module(_name)
    assert Path(_module.__file__).resolve().is_relative_to(_checkout), _name

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
