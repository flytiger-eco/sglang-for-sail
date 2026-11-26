"""LayerSplit batch transport preserves untouched pages and reuses workspaces."""

import unittest

import torch

from sglang.srt.mem_cache.layer_split_batch_pages import BatchPageBroadcastPlan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestLayerSplitBatchPageBroadcast(CustomTestCase):
    def test_cpu_plan_reuses_workspace_and_preserves_other_pages(self):
        pages = torch.tensor([0, 3, 8])
        source = torch.arange(9 * 17).reshape(9, 17)
        destination = torch.full_like(source, -1)
        plan = BatchPageBroadcastPlan(pages, vector_scatter=True)
        addresses = []

        def collective(packed):
            addresses.append(packed.data_ptr())

        for _ in range(2):
            plan.broadcast(destination, source, collective)
        reference = torch.full_like(source, -1)
        reference.index_copy_(0, pages, source[pages])
        self.assertTrue(torch.equal(destination, reference))
        self.assertEqual(addresses[0], addresses[1])
        self.assertEqual(plan.collectives, 2)


if __name__ == "__main__":
    unittest.main()
