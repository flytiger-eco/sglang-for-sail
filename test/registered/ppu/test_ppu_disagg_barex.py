"""PPU BAREX disaggregation smoke test.

Guards the PPU-specific AcclBarex RDMA path for prefill/decode
disaggregation (decision doc section 8.1, W6-5). Today the BAREX path is
only covered indirectly: disaggregation_fixture.apply_ppu_barex_env()
injects the BAREX env vars into six shared disaggregation tests whenever a
fic2 NIC is detected. If any of those tests stops running (model missing,
upstream refactor), BAREX silently loses all coverage.

This test is a direct fixture smoke test:

1. Asserts the BAREX transport env (USE_BAREX, SGLANG_MOONCAKE_CUSTOM_MEM_POOL)
   is actually in effect -- the thing the shared tests never check.
2. Launches a minimal prefill/decode/LB deployment and completes one
   end-to-end generate request through the load balancer.

Skips (does not fail) when the fic2 NIC is absent, per the W6-5 design.

Usage:
python3 -m pytest test/registered/ppu/test_ppu_disagg_barex.py -v
"""

import os
import unittest

import requests

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

# Disabled: 5c42d3f60c reverted the fixture's BAREX env injection and
# MC_LOCAL_HOSTNAME detection, so the KV transfer dies on PPU runners.
# Re-enable once the fixture logic is restored.
register_ppu_ci(
    est_time=115,
    suite="nightly-2-ppu",
    nightly=True,
    disabled="fixture BAREX env/MC_LOCAL_HOSTNAME reverted by 5c42d3f60c",
)


# Same small model as test_ppu_basic.py: known to load on the PPU runners.
PPU_CI_MODEL_PATH = (
    "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/Qwen2.5-0.5B-Instruct/"
)

_FIC2_NIC_SYSFS = "/sys/class/infiniband/fic2_soe_bond0"


class TestPPUDisaggBarex(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(_FIC2_NIC_SYSFS):
            raise unittest.SkipTest(
                f"fic2 NIC not present ({_FIC2_NIC_SYSFS} missing); "
                "PPU BAREX disaggregation path cannot be exercised here"
            )
        super().setUpClass()
        cls.model = PPU_CI_MODEL_PATH
        cls.launch_all()

    def test_barex_transport_env_in_effect(self):
        """The fixture must have injected the AcclBarex env on this machine.

        Without this assertion the test would degrade to a generic
        disaggregation smoke test if apply_ppu_barex_env() ever stops
        matching this NIC.
        """
        self.assertEqual(
            os.environ.get("USE_BAREX"),
            "1",
            "USE_BAREX not set: apply_ppu_barex_env() did not activate the "
            "BAREX transport even though the fic2 NIC is present",
        )
        self.assertEqual(
            os.environ.get("SGLANG_MOONCAKE_CUSTOM_MEM_POOL"),
            "BAREX",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL != BAREX: mooncake would not "
            "use the BAREX custom memory pool",
        )

    def test_end_to_end_generate_over_barex(self):
        """One request through the LB exercises prefill -> KV transfer over
        BAREX -> decode end to end."""
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=120,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"generate over BAREX disaggregation failed: {response.text[:500]}",
        )
        completion = response.json()["text"]
        self.assertTrue(
            completion.strip(),
            "empty completion from BAREX disaggregated generate",
        )
        print(f"BAREX disagg completion: {completion!r}")


if __name__ == "__main__":
    unittest.main()
