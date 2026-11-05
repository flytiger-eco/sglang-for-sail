"""CPU regressions for CP late-layer-tail routing IDs.

The late-layer tail keeps the last tokens of each request. Under prefill CP the
tail rows are selected from the rank's already-sharded hidden states, so their
routing IDs must come from the same layout. Re-sharding the compact tail from
zero shifts token ownership when the tail starts off the original CP boundary.
"""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV41TailCPRoutingIDs(unittest.TestCase):
    def _cp_tail(self, *, total: int):
        token_indices = torch.arange(1, total)
        forward_batch = NS(batch_size=1, positions=torch.arange(total))
        with patch(
            "sglang.srt.layers.attention.deepseek_v4_backend.get_parallel",
            return_value=NS(attn_cp_rank=0, attn_cp_size=2),
        ):
            return DeepseekV4AttnBackend._late_layer_tail_cp_layout(
                None,
                forward_batch,
                token_indices,
                torch.tensor([total - 1], dtype=torch.int32),
            )

    def test_off_boundary_tail_keeps_original_cp_ownership(self):
        total = 129
        cp_tail = self._cp_tail(total=total)
        full_ids = torch.arange(total)
        local_ids = full_ids[0::2]
        selected = local_ids[cp_tail["local_token_indices"]]
        expected = full_ids[2::2]
        torch.testing.assert_close(selected, expected)

        # The old compact re-shard starts at zero and loses the boundary parity.
        compact = full_ids[1:][0::2]
        self.assertFalse(torch.equal(compact, expected))

    def test_uneven_tail_pads_without_changing_ownership(self):
        total = 128
        cp_tail = self._cp_tail(total=total)
        self.assertEqual(cp_tail["pad_rows"], 1)
        full_ids = torch.arange(total)
        local_ids = full_ids[0::2]
        selected = local_ids[cp_tail["local_token_indices"]]
        torch.testing.assert_close(selected, full_ids[2::2])

    def test_routing_ids_use_local_rows_and_all_gather_for_no_a2a(self):
        local_ids = torch.arange(3)
        mode = MagicMock()
        with (
            patch.object(deepseek_v4, "get_moe_a2a_backend", return_value=mode),
            patch.object(deepseek_v4, "attn_cp_all_gather_into_tensor") as gather,
            patch.object(deepseek_v4, "get_parallel", return_value=NS(attn_cp_size=2)),
        ):
            mode.is_none.return_value = False
            input_ids, input_ids_global = deepseek_v4._tail_cp_routing_ids(
                local_input_ids=local_ids
            )
            torch.testing.assert_close(input_ids, local_ids)
            torch.testing.assert_close(input_ids_global, local_ids)
            gather.assert_not_called()

            mode.is_none.return_value = True

            def fake_all_gather(out, local):
                # Rank 0 keeps its local rows; rank 1 has distinct IDs.
                out.copy_(torch.cat([local, local + 100]))

            gather.side_effect = fake_all_gather
            input_ids, input_ids_global = deepseek_v4._tail_cp_routing_ids(
                local_input_ids=local_ids
            )
            torch.testing.assert_close(input_ids, local_ids)
            torch.testing.assert_close(
                input_ids_global, torch.tensor([0, 1, 2, 100, 101, 102])
            )
            gather.assert_called_once()


if __name__ == "__main__":
    unittest.main()
