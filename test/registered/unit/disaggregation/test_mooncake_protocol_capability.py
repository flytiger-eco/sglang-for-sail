"""CPU regressions for mixed-KV sender capability negotiation."""

import asyncio
import dataclasses
import json
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeProtocolCapability(CustomTestCase):
    def manager(self, tp=8, draft=2, dcp=1, rank=0):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.kv_args = KVArgs()
        mgr.kv_args.draft_kv_entry_count = draft
        mgr.kv_args.page_size = 64
        mgr.attn_tp_size = tp
        mgr.dcp_size = dcp
        mgr.dcp_rank = rank
        mgr.is_mla_backend = True
        mgr.is_hybrid_mla_backend = False
        mgr.kv_cache_dtype_str = "float16"
        mgr.prefill_info_table = {}
        mgr.connection_pool = {}
        mgr.connection_lock = threading.Lock()
        mgr.record_failure = Mock()
        mgr.update_status = Mock()
        mgr._resolve_rank_mapping = Mock()
        return mgr

    def info(self, **kwargs):
        args = dict(
            attn_tp_size=16,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=64,
            kv_cache_dtype="float16",
            follow_bootstrap_room=True,
        )
        args.update(kwargs)
        return PrefillServerInfo(**args)

    def receiver(self, mgr, info, rank_info):
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = mgr
        receiver.prefill_info = info
        receiver.bootstrap_addr = "localhost"
        receiver.bootstrap_room = 42
        receiver.prefill_dp_rank = 0
        receiver.target_cp_ranks = [0]
        receiver.target_tp_ranks = [0]
        receiver.target_tp_rank = 0
        receiver.target_pp_ranks = [0]
        receiver.conclude_state = None
        receiver._connection_pool_entries = {}
        receiver._get_bootstrap_info_from_server = Mock(return_value=rank_info)
        receiver._register_kv_args = Mock(return_value=True)
        return receiver

    def test_old_sender_is_rejected_before_parallel_info_cache(self):
        for count in (None, 2):
            with self.subTest(draft_count=count):
                mgr = self.manager()
                response = NS(
                    status_code=200,
                    json=lambda: dataclasses.asdict(
                        self.info(draft_kv_entry_count=count)
                    ),
                )
                with patch(
                    "sglang.srt.disaggregation.common.conn.requests.get",
                    return_value=response,
                ) as get:
                    with self.assertRaisesRegex(RuntimeError, "every prefill rank"):
                        mgr.try_ensure_parallel_info("localhost")
                self.assertIn("kv_transfer_capabilities=1", get.call_args.args[0])
                self.assertEqual(mgr.prefill_info_table, {})
                mgr._resolve_rank_mapping.assert_not_called()

    def test_cached_global_info_is_revalidated(self):
        mgr = self.manager()
        mgr.prefill_info_table["localhost"] = self.info()
        with patch("sglang.srt.disaggregation.common.conn.requests.get") as get:
            with self.assertRaisesRegex(RuntimeError, "every prefill rank"):
                mgr.try_ensure_parallel_info("localhost")
        get.assert_not_called()

    def test_supported_sender_passes(self):
        mgr = self.manager()
        info = self.info(mixed_kv_transfer_version=1, dcp_size=1)
        mgr._validate_prefill_transfer_capability(info)
        mgr._validate_prefill_transfer_capability(
            info, dict(mixed_kv_transfer_version=1, dcp_size=1, dcp_rank=0)
        )

    def test_legacy_whole_page_and_target_only_are_preserved(self):
        self.manager(tp=16)._validate_prefill_transfer_capability(self.info())
        self.manager(draft=0)._validate_prefill_transfer_capability(self.info())
        self.manager(tp=16)._validate_prefill_transfer_capability(self.info(), {})
        self.manager(draft=0)._validate_prefill_transfer_capability(self.info(), {})

    def test_remote_draft_also_requires_capability(self):
        with self.assertRaisesRegex(RuntimeError, "mixed_kv_transfer_version"):
            self.manager(draft=0)._validate_prefill_transfer_capability(
                self.info(draft_kv_entry_count=2)
            )

    def test_dcp_remapping_requires_capability_even_with_equal_tp(self):
        mgr = self.manager(tp=16, dcp=8, rank=3)
        for size in (None, 1):
            with self.subTest(source_dcp=size):
                with self.assertRaisesRegex(RuntimeError, "decode_dcp=8"):
                    mgr._validate_prefill_transfer_capability(self.info(dcp_size=size))
        info = self.info(dcp_size=8)
        mgr._validate_prefill_transfer_capability(info)
        mgr._validate_prefill_transfer_capability(info, dict(dcp_size=8, dcp_rank=3))
        for rank_info in ({}, dict(dcp_size=8, dcp_rank=2)):
            with self.assertRaisesRegex(RuntimeError, "mixed_kv_transfer_version"):
                mgr._validate_prefill_transfer_capability(info, rank_info)

    def test_old_rank_is_rejected_before_registration_and_cache(self):
        mgr = self.manager()
        receiver = self.receiver(
            mgr,
            self.info(mixed_kv_transfer_version=1),
            dict(rank_ip="localhost", rank_port=1234),
        )
        receiver._setup_bootstrap_infos()
        receiver._register_kv_args.assert_not_called()
        self.assertEqual(mgr.connection_pool, {})
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        mgr.record_failure.assert_called_once()

    def test_cached_old_rank_cannot_bypass_capability(self):
        mgr = self.manager()
        receiver = self.receiver(mgr, self.info(mixed_kv_transfer_version=1), None)
        mgr.connection_pool["localhost_0_0_0"] = [
            dict(rank_ip="localhost", rank_port=1234)
        ]
        receiver._setup_bootstrap_infos()
        receiver._get_bootstrap_info_from_server.assert_not_called()
        receiver._register_kv_args.assert_not_called()
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        self.assertEqual(mgr.connection_pool, {})

    def test_supported_rank_is_registered_and_cached(self):
        mgr = self.manager()
        receiver = self.receiver(
            mgr,
            self.info(mixed_kv_transfer_version=1),
            dict(rank_ip="localhost", rank_port=1234, mixed_kv_transfer_version=1),
        )
        receiver._setup_bootstrap_infos()
        receiver._register_kv_args.assert_called_once()
        self.assertIsNone(receiver.conclude_state)
        self.assertIn("localhost_0_0_0", mgr.connection_pool)

    def test_cache_race_winner_is_also_validated(self):
        mgr = self.manager()
        receiver = self.receiver(
            mgr,
            self.info(mixed_kv_transfer_version=1),
            dict(rank_ip="localhost", rank_port=1234, mixed_kv_transfer_version=1),
        )

        def register():
            mgr.connection_pool["localhost_0_0_0"] = [
                dict(rank_ip="localhost", rank_port=9999)
            ]
            return True

        receiver._register_kv_args.side_effect = register
        receiver._setup_bootstrap_infos()
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        self.assertEqual(mgr.connection_pool, {})

    def bootstrap(self):
        with patch.object(CommonKVBootstrapServer, "run"):
            return CommonKVBootstrapServer("127.0.0.1", 0)

    def register(self, server, tp, pp, version=1):
        data = dict(
            attn_tp_size=2,
            attn_tp_rank=tp,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
            pp_size=2,
            pp_rank=pp,
            system_dp_size=1,
            system_dp_rank=0,
            rank_ip="localhost",
            rank_port=1234 + 2 * tp + pp,
            page_size=64,
            kv_cache_dtype="float16",
            draft_kv_entry_count=2 if pp else 0,
        )
        if version is not None:
            data.update(mixed_kv_transfer_version=version, dcp_size=1, dcp_rank=0)
        response = asyncio.run(
            server._handle_route_put(NS(json=AsyncMock(return_value=data)))
        )
        self.assertEqual(response.status, 200)

    def query(self, server, opt_in=True, rank=None):
        query = dict(
            prefill_dp_rank="-1",
            prefill_cp_rank="-1",
            target_tp_rank="-1",
            target_pp_rank="-1",
        )
        if rank is not None:
            query.update(
                prefill_dp_rank="0",
                prefill_cp_rank="0",
                target_tp_rank=str(rank[0]),
                target_pp_rank=str(rank[1]),
            )
        if opt_in:
            query["kv_transfer_capabilities"] = "1"
        response = asyncio.run(server._handle_route_get(NS(query=query)))
        self.assertEqual(response.status, 200)
        return json.loads(response.text)

    def test_capability_requires_every_stage_and_rank(self):
        for old_rank in ((0, 0), (0, 1), (1, 0), (1, 1)):
            with self.subTest(old_rank=old_rank):
                server = self.bootstrap()
                for tp in range(2):
                    for pp in range(2):
                        self.register(
                            server, tp, pp, None if (tp, pp) == old_rank else 1
                        )
                info = self.query(server)
                self.assertEqual(info["draft_kv_entry_count"], 2)
                self.assertEqual(info["mixed_kv_transfer_version"], 0)
                self.assertEqual(
                    self.query(server, rank=old_rank)["mixed_kv_transfer_version"], 0
                )

    def test_missing_rank_cannot_be_hidden_by_duplicate_registration(self):
        server = self.bootstrap()
        for _ in range(4):
            self.register(server, 0, 0)
        self.assertEqual(self.query(server)["mixed_kv_transfer_version"], 0)

    def test_reregistration_downgrades_capability(self):
        server = self.bootstrap()
        for tp in range(2):
            for pp in range(2):
                self.register(server, tp, pp)
        self.assertEqual(self.query(server)["mixed_kv_transfer_version"], 1)
        self.register(server, 0, 0, None)
        self.assertEqual(self.query(server)["mixed_kv_transfer_version"], 0)

    def test_capability_response_is_opt_in_for_legacy_clients(self):
        # Exact field contract before this PR (0844d76dd3), not a permissive
        # **kwargs mock: new response keys must not break old dataclasses.
        legacy_server_info = dataclasses.make_dataclass(
            "LegacyPrefillServerInfo",
            [
                (name, object)
                for name in (
                    "attn_tp_size",
                    "attn_cp_size",
                    "dp_size",
                    "pp_size",
                    "page_size",
                    "kv_cache_dtype",
                    "follow_bootstrap_room",
                )
            ]
            + [
                (name, object, dataclasses.field(default=None))
                for name in (
                    "enable_dsa_cache_layer_split",
                    "dsv41_spec_layout",
                    "prefill_http_port",
                    "target_tp_rank",
                    "target_tp_ranks",
                    "target_cp_ranks",
                    "target_pp_ranks",
                    "required_dst_info_num",
                    "required_prefill_response_num",
                )
            ],
        )
        legacy_rank_info = dataclasses.make_dataclass(
            "LegacyPrefillRankInfo", [("rank_ip", str), ("rank_port", int)]
        )
        server = self.bootstrap()
        for tp in range(2):
            for pp in range(2):
                self.register(server, tp, pp)
        for rank in (None, (0, 0)):
            payload = self.query(server, opt_in=False, rank=rank)
            for field in (
                "mixed_kv_transfer_version",
                "dcp_size",
                "dcp_rank",
                "draft_kv_entry_count",
            ):
                self.assertNotIn(field, payload)
            legacy = legacy_server_info if rank is None else legacy_rank_info
            legacy(**payload)
        self.assertEqual(
            self.query(server, rank=(0, 0))["mixed_kv_transfer_version"], 1
        )
        self.assertEqual(self.query(server)["dcp_size"], 1)


if __name__ == "__main__":
    unittest.main()
