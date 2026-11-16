import os
from unittest.mock import patch

import numpy as np
import pytest
from test_replicated_kv_page_transfer import make_manager, send

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

REPLICATE = "SGLANG_QSA_ONLY_CP_REPLICATE"
PAGE = "SGLANG_MOONCAKE_REPLICATED_KV_PAGE_TRANSFER"
ALL = "SGLANG_MOONCAKE_REPLICATED_KV_PAGE_TRANSFER_ALL_RANKS"


def test_qsa_cp_defaults_send_disjoint_pages_from_all_ranks():
    manager, sources, targets = make_manager()
    sent = []
    with patch.dict(os.environ, {REPLICATE: "1"}):
        os.environ.pop(PAGE, None)
        os.environ.pop(ALL, None)
        assert envs.SGLANG_MOONCAKE_REPLICATED_KV_PAGE_TRANSFER.get()
        assert envs.SGLANG_MOONCAKE_REPLICATED_KV_PAGE_TRANSFER_ALL_RANKS.get()
        for rank in range(4):
            manager.kv_args.engine_rank = rank
            manager.engine.batch_transfer_sync.reset_mock()
            assert (
                send(manager, targets, pages=(0, 1, 2, 3), dst_pages=(0, 1, 3, 4)) == 0
            )
            calls = manager.engine.batch_transfer_sync.call_args_list
            sent.append(sum(sum(c.args[3]) for c in calls))
    assert sent == [64, 64, 64, 64]
    for src, target in zip(sources, targets):
        expected = np.full_like(target, -1)
        for a, b in zip(range(4), (0, 1, 3, 4)):
            expected[b * 8 : (b + 1) * 8] = src[a * 8 : (a + 1) * 8]
        np.testing.assert_array_equal(target, expected)


def test_non_qsa_defaults_are_unchanged():
    with patch.dict(os.environ):
        for key in [REPLICATE, PAGE, ALL]:
            os.environ.pop(key, None)
        assert not envs.SGLANG_MOONCAKE_REPLICATED_KV_PAGE_TRANSFER.get()
        assert not envs.SGLANG_MOONCAKE_REPLICATED_KV_PAGE_TRANSFER_ALL_RANKS.get()


@pytest.mark.parametrize("key", [PAGE, ALL])
def test_explicit_zero_overrides_qsa_cp_default(key):
    with patch.dict(os.environ, {REPLICATE: "1", key: "0"}):
        assert not getattr(envs, key).get()


def test_explicit_all_ranks_zero_preserves_single_sender():
    manager, _, targets = make_manager(rank=1)
    with patch.dict(os.environ, {REPLICATE: "1", ALL: "0"}):
        os.environ.pop(PAGE, None)
        assert send(manager, targets) == 0
    manager.engine.batch_transfer_sync.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
