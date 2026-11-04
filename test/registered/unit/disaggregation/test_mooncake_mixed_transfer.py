"""Mooncake mixed-transfer regressions with CPU pools and mocked transport."""

import struct
import threading
import unittest
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch

import sglang.srt.disaggregation.mooncake.conn as mc
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import PrefillServerInfo
from sglang.srt.disaggregation.utils import set_draft_kv_transfer_metadata
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class EndMessages(BaseException):
    """Stop the synchronous bootstrap loop after consuming the fake socket."""


def run_loop(manager, method, messages, observe=None):
    # Execute the real thread body synchronously and record escaped exceptions.
    events = []

    def recv():
        if observe:
            observe()
        if not messages:
            raise EndMessages()
        return messages.pop(0)

    manager.server_socket = NS(recv_multipart=recv)

    def start(target):
        try:
            target()
        except EndMessages:
            pass
        except Exception as e:
            events.append(e)

    with patch.object(
        mc.threading,
        "Thread",
        side_effect=lambda target: NS(start=lambda: start(target)),
    ):
        method()
    return events


def manager(draft=2, tp=16):
    m = mc.MooncakeKVManager.__new__(mc.MooncakeKVManager)
    m.kv_args = KVArgs()
    a = m.kv_args
    for k, v in dict(
        page_size=64,
        engine_rank=0,
        kv_item_lens=[64, 1024, 1024],
        kv_data_ptrs=[10000, 20000, 30000],
        kv_layer_ids=[0, 1, 1],
        draft_kv_entry_count=draft,
        draft_kv_head_num=1,
        draft_total_kv_head_num=16,
        draft_kv_token_major=True,
    ).items():
        setattr(a, k, v)
    m.attn_tp_size = tp
    m.attn_cp_size = 1
    m.attn_cp_rank = 0
    m.attn_tp_rank = 0
    m.pp_size = 1
    m.pp_rank = 0
    m.dcp_size = 1
    m.dcp_rank = 0
    m.is_mla_backend = True
    m.is_hybrid_mla_backend = False
    m.enable_all_cp_ranks_for_transfer = False
    m.decode_kv_args_table = {}
    m.rejected_registrations = {}
    m.transfer_infos = {}
    m.req_to_decode_prefix_len = {}
    m.failed_sessions = set()
    m.session_failures = defaultdict(int)
    m.session_lock = threading.Lock()
    m.failure_lock = threading.Lock()
    m._status_lock = threading.Lock()
    m.failure_records = {}
    m.request_status = {}
    m.enable_staging = False
    m.sync_status_to_decode_endpoint = Mock()
    m._start_heartbeat_checker_thread = Mock()
    m.required_prefill_response_num_table = {}
    m.prefill_response_tracker = defaultdict(set)
    return m


def registration(session="ok", count=2, token_major=True):
    return [
        b"None",
        b"127.0.0.1",
        b"1234",
        session.encode(),
        struct.pack("3Q", 40000, 50000, 60000),
        b"",
        b"",
        b"0",
        b"8",
        b"64",
        b"",
        b"",
        struct.pack("3I", 0, 1, 1),
        b"",
        b"",
        b"",
        b"1",
        b"0",
        struct.pack("3Q", 64, 2048, 2048),
        str(count).encode(),
        b"2",
        b"16",
        b"64",
        b"1" if token_major else b"0",
    ]


def request(session, room):
    return [
        str(room).encode(),
        b"127.0.0.1",
        b"1234",
        session.encode(),
        np.array([2], dtype=np.int32).tobytes(),
        b"0",
        b"",
        b"1",
    ]


def pool(layout, heads=1):
    # Exercise the real allocator method and descriptors on CPU; no fake tensor layout.
    p = MHATokenToKVPool.__new__(MHATokenToKVPool)
    p.size = 128
    p.page_size = 64
    p.head_num = heads
    p.head_dim = 8
    p.v_head_dim = 8
    p.layer_num = 1
    p.store_dtype = torch.float16
    p.device = "cpu"
    p.kv_cache_layout = layout
    p._kv_vector_x = 8
    p.use_hnd = False
    p.enable_custom_mem_pool = False
    p.memory_saver_adapter = NS(region=lambda *a: nullcontext())
    p._create_buffers_normal()
    p._kv_buffer_descs = p._build_kv_buffer_descs()
    return p


class TestMooncakeMixedTransfer(CustomTestCase):
    def test_mla_pool_layer_ids_match_exported_buffers(self):
        from sglang.srt.mem_cache.dsa_cache_layer_split import (
            LayerSplitDSATokenToKVPool,
        )
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool

        for cls in (MLATokenToKVPool, DSATokenToKVPool, LayerSplitDSATokenToKVPool):
            for shard in (0, 1):
                with self.subTest(pool=cls.__name__, shard=shard):
                    p = cls.__new__(cls)
                    p.start_layer = 7
                    p.layer_num = 5
                    p.page_size = 64
                    p.layer_shard_rank = shard
                    p.layer_shard_size = 2
                    p.kv_buffer = [
                        torch.zeros((64, 1, 16), dtype=torch.uint8) for _ in range(5)
                    ]
                    ptrs, _, _ = p.get_contiguous_buf_infos()
                    ids = p.get_kv_layer_ids()
                    self.assertEqual(ptrs, [p.kv_buffer[i - 7].data_ptr() for i in ids])
                    if cls is LayerSplitDSATokenToKVPool:
                        lo, hi = p._owned_local_layer_range()
                        self.assertEqual(ids, list(range(7 + lo, 7 + hi)))
                    else:
                        self.assertEqual(ids, [7, 8, 9, 10, 11])

    def test_mla_layer_ids_preserve_subclass_tensor_groups(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        class GroupedPool(MLATokenToKVPool):
            def get_contiguous_buf_infos(self):
                return list(range(self.entry_count)), [], []

        p = GroupedPool.__new__(GroupedPool)
        p.start_layer = 3
        p.layer_num = 2
        for groups in (1, 2, 3):
            p.entry_count = groups * 2
            self.assertEqual(p.get_kv_layer_ids(), [3, 4] * groups)
        p.entry_count = 0
        self.assertEqual(p.get_kv_layer_ids(), [])
        p.entry_count = 3
        with self.assertRaisesRegex(ValueError, "complete layer groups"):
            p.get_kv_layer_ids()

    def test_layer_split_mixed_pool_mapping_and_addresses(self):
        import ctypes

        from sglang.srt.disaggregation.utils import get_transfer_kv_layer_ids
        from sglang.srt.mem_cache.dsa_cache_layer_split import (
            LayerSplitDSATokenToKVPool,
        )
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        def target(split, shard=1):
            cls = LayerSplitDSATokenToKVPool if split else DSATokenToKVPool
            p = cls.__new__(cls)
            p.start_layer = 0
            p.layer_num = 4
            p.page_size = 64
            if split:
                p.layer_shard_rank = shard
                p.layer_shard_size = 2
            p.kv_buffer = [
                torch.zeros(
                    (192 if not split or i // 2 == shard else 0, 1, 16),
                    dtype=torch.uint8,
                )
                for i in range(4)
            ]
            return p

        # Real exported pool geometry; only the native transport is replaced.
        for split in (False, True):
            for dst_tp in (16, 8):
                for count in (65, 128):
                    with self.subTest(split=split, dst_tp=dst_tp, count=count):
                        dt, dd = target(False), pool("nhd", heads=16 // dst_tp)
                        dd.start_layer = 0
                        dest = dt.kv_buffer + dd.k_buffer + dd.v_buffer
                        for t in dest:
                            t.zero_()
                        writes = {
                            t.data_ptr(): np.zeros(t.nbytes, dtype=np.uint8)
                            for t in dest
                        }
                        dst_ptrs, _, dst_sizes = dt.get_contiguous_buf_infos()
                        dp, _, ds = dd.get_contiguous_buf_infos()
                        dst_ptrs += dp
                        dst_sizes += ds
                        dst_ids = get_transfer_kv_layer_ids(dt, dd, 4)
                        self.assertEqual(dst_ids, [0, 1, 2, 3, 4, 4])
                        for rank in range(16 // dst_tp):
                            st, sd = target(split), pool("nhd")
                            sd.start_layer = 0
                            for layer, t in enumerate(st.kv_buffer):
                                t.fill_(layer + 1)
                            for t in sd.k_buffer + sd.v_buffer:
                                t.fill_(rank + 10)
                            m = manager()
                            m.attn_cp_size = 2 if split else 1
                            m.attn_cp_rank = 1 if split else 0
                            a = m.kv_args
                            a.engine_rank = rank
                            a.prefill_start_layer = 2 if split else 0
                            a.prefill_end_layer = 4
                            sp, _, ss = st.get_contiguous_buf_infos()
                            kp, _, ks = sd.get_contiguous_buf_infos()
                            a.kv_data_ptrs = sp + kp
                            a.kv_item_lens = ss + ks
                            a.kv_layer_ids = get_transfer_kv_layer_ids(st, sd, 4)
                            self.assertEqual(
                                a.kv_layer_ids,
                                ([2, 3] if split else [0, 1, 2, 3]) + [4, 4],
                            )
                            wire = registration()
                            wire[4] = struct.pack(f"{len(dst_ptrs)}Q", *dst_ptrs)
                            wire[8] = str(dst_tp).encode()
                            wire[9] = b"1024"
                            wire[12] = struct.pack("6I", *dst_ids)
                            wire[18] = struct.pack("6Q", *dst_sizes)
                            wire[20] = str(16 // dst_tp).encode()
                            m._register_decode_peer(wire, "s")
                            self.assertNotIn("s", m.rejected_registrations)
                            info = m.decode_kv_args_table["s"]

                            def transfer(session, blocks):
                                for src, dst, size in blocks:
                                    owners = [
                                        t
                                        for t in dest
                                        if t.data_ptr() <= dst
                                        and dst + size <= t.data_ptr() + t.nbytes
                                    ]
                                    self.assertEqual(len(owners), 1)
                                    base = owners[0].data_ptr()
                                    writes[base][dst - base : dst - base + size] += 1
                                    ctypes.memmove(dst, src, size)
                                return 0

                            m._transfer_data = transfer
                            m._validate_envelope_kv_layout = Mock()
                            self.assertEqual(
                                m.send_kvcache_mixed(
                                    "s",
                                    np.array([0, 1]),
                                    np.array([1, 2]),
                                    info=info,
                                    src_page_offset=0,
                                    decode_prefix_len=0,
                                    num_kv_tokens=count,
                                ),
                                0,
                            )
                        for layer, t in enumerate(dt.kv_buffer):
                            expected = torch.zeros_like(t)
                            if not split or layer >= 2:
                                expected[64 : 64 + count] = layer + 1
                            self.assertTrue(torch.equal(t, expected))
                            self.assertTrue(
                                np.array_equal(
                                    writes[t.data_ptr()],
                                    (expected.numpy().reshape(-1) != 0).astype(
                                        np.uint8
                                    ),
                                )
                            )
                        for t in dd.k_buffer + dd.v_buffer:
                            expected = torch.zeros_like(t)
                            for head in range(16 // dst_tp):
                                expected[64 : 64 + count, head] = head + 10
                            self.assertTrue(torch.equal(t, expected))
                            expected_writes = np.zeros(t.nbytes, dtype=np.uint8)
                            row_bytes = t[0].nbytes
                            expected_writes[
                                64 * row_bytes : (64 + count) * row_bytes
                            ] = 1
                            self.assertTrue(
                                np.array_equal(writes[t.data_ptr()], expected_writes)
                            )

        # A source without draft can still map into a draft-equipped Decode.
        for shard in (0, 1):
            st, dt = target(True, shard), target(False)
            sd = pool("nhd")
            sd.start_layer = 0
            src_ids = get_transfer_kv_layer_ids(st, None, 4)
            dst_ids = get_transfer_kv_layer_ids(dt, sd, 4)
            self.assertEqual(
                mc.resolve_dcp_dst_entry_indices(src_ids, dst_ids, 2, 6),
                [2 * shard, 2 * shard + 1],
            )

    def test_bootstrap_http_advertises_source_draft_presence(self):
        import asyncio
        import json
        from unittest.mock import AsyncMock

        from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer

        for count in (0, 2):
            with patch.object(CommonKVBootstrapServer, "run"):
                server = CommonKVBootstrapServer("127.0.0.1", 0)
            data = dict(
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                attn_dp_size=1,
                attn_dp_rank=0,
                pp_size=1,
                pp_rank=0,
                system_dp_size=1,
                system_dp_rank=0,
                rank_ip="127.0.0.1",
                rank_port=1234,
                page_size=64,
                kv_cache_dtype="float16",
                load_balance_method="follow_bootstrap_room",
                draft_kv_entry_count=count,
            )
            response = asyncio.run(
                server._handle_route_put(NS(json=AsyncMock(return_value=data)))
            )
            self.assertEqual(response.status, 200)
            response = asyncio.run(
                server._handle_route_get(
                    NS(
                        query={
                            k: "-1"
                            for k in [
                                "prefill_dp_rank",
                                "prefill_cp_rank",
                                "target_tp_rank",
                                "target_pp_rank",
                            ]
                        }
                    )
                )
            )
            info = PrefillServerInfo(**json.loads(response.text))
            self.assertEqual(getattr(info, "draft_kv_entry_count", None), count)

    def test_layer_split_draft_presence_is_registration_order_independent(self):
        import asyncio
        import json
        from unittest.mock import AsyncMock

        from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer

        for counts in ((0, 2), (2, 0), (0, 0)):
            with self.subTest(counts=counts):
                with patch.object(CommonKVBootstrapServer, "run"):
                    server = CommonKVBootstrapServer("127.0.0.1", 0)
                for cp, count in enumerate(counts):
                    for tp in range(16):
                        data = dict(
                            attn_tp_size=16,
                            attn_tp_rank=tp,
                            attn_cp_size=2,
                            attn_cp_rank=cp,
                            attn_dp_size=1,
                            attn_dp_rank=0,
                            pp_size=1,
                            pp_rank=0,
                            system_dp_size=1,
                            system_dp_rank=0,
                            rank_ip="127.0.0.1",
                            rank_port=1234 + cp * 16 + tp,
                            page_size=64,
                            kv_cache_dtype="float16",
                            load_balance_method="follow_bootstrap_room",
                            draft_kv_entry_count=count,
                            enable_dsa_cache_layer_split=True,
                        )
                        response = asyncio.run(
                            server._handle_route_put(
                                NS(json=AsyncMock(return_value=data))
                            )
                        )
                        self.assertEqual(response.status, 200)
                response = asyncio.run(
                    server._handle_route_get(
                        NS(
                            query={
                                k: "-1"
                                for k in (
                                    "prefill_dp_rank",
                                    "prefill_cp_rank",
                                    "target_tp_rank",
                                    "target_pp_rank",
                                )
                            }
                        )
                    )
                )
                self.assertEqual(response.status, 200)
                info = PrefillServerInfo(**json.loads(response.text))
                self.assertEqual(info.draft_kv_entry_count, max(counts))
                for plain in (True, False):
                    m = manager(tp=8)
                    m.is_mla_backend = plain
                    m.is_hybrid_mla_backend = not plain
                    m._resolve_rank_mapping(info)
                    expected = 2 if plain and not any(counts) else 4
                    self.assertEqual(info.required_prefill_response_num, expected)
                    r = mc.MooncakeKVReceiver.__new__(mc.MooncakeKVReceiver)
                    r.kv_mgr = m
                    r.prefill_info = info
                    for key in (
                        "target_tp_rank",
                        "target_tp_ranks",
                        "target_cp_ranks",
                        "target_pp_ranks",
                    ):
                        setattr(r, key, getattr(info, key))
                    r.bootstrap_addr = "localhost"
                    r.prefill_dp_rank = 0
                    r.bootstrap_room = 99
                    r.conclude_state = None
                    m.connection_pool = {}
                    r._get_bootstrap_info_from_server = lambda dp, cp, tp, pp: dict(
                        rank_ip="localhost",
                        rank_port=1000 + cp * 16 + tp,
                        sender=tp * 2 + cp,
                    )
                    r._register_kv_args = Mock(return_value=True)
                    r._setup_bootstrap_infos()
                    real = [b["sender"] for b in r.bootstrap_infos if not b["is_dummy"]]
                    self.assertEqual(
                        set(real), {0, 1} if expected == 2 else {0, 1, 2, 3}
                    )
                    m.request_status[99] = KVPoll.WaitingForInput
                    m.required_prefill_response_num_table[99] = expected
                    states = []
                    # Duplicate responses must not cause early completion.
                    ids = real[:-1] + real[:-1] + real[-1:]
                    msgs = [
                        [b"99", str(int(KVPoll.Success)).encode(), str(rank).encode()]
                        for rank in ids
                    ]
                    self.assertEqual(
                        run_loop(
                            m,
                            m.start_decode_thread,
                            msgs,
                            lambda: states.append(m.request_status[99]),
                        ),
                        [],
                    )
                    self.assertTrue(
                        all(state != KVPoll.Success for state in states[:-1])
                    )
                    self.assertEqual(states[-1], KVPoll.Success)

    def test_cp_filtered_chunks_keep_token_count_and_destination_alignment(self):
        import sglang.srt.disaggregation.common.conn as common

        for total_tokens in (450, 512):
            for chunks in ((8,), (3, 5), (5, 3), (1,) * 8):
                for cp in (0, 1):
                    with self.subTest(tokens=total_tokens, chunks=chunks, cp=cp):
                        sender = mc.MooncakeKVSender.__new__(mc.MooncakeKVSender)
                        sender.curr_idx = 0
                        sender.num_kv_indices = 8
                        sender.kv_mgr = NS(
                            enable_all_cp_ranks_for_transfer=True,
                            attn_cp_size=2,
                            attn_cp_rank=cp,
                            is_dummy_cp_rank=False,
                            kv_args=NS(page_size=64),
                            add_transfer_request=Mock(),
                        )
                        sender.bootstrap_room = 9
                        sender.aux_index = 0
                        sender.trace_ctx = Mock()
                        sender._transfer_num_kv_indices = 0
                        sender._transfer_num_state_indices = 0
                        send = getattr(
                            mc.MooncakeKVSender.send,
                            "__wrapped__",
                            mc.MooncakeKVSender.send,
                        )
                        start = 0
                        observed = []
                        for pages in chunks:
                            count = min(pages * 64, max(0, total_tokens - start * 64))
                            with patch.object(
                                common,
                                "get_parallel",
                                return_value=NS(enable_dsa_cache_layer_split=False),
                            ), patch.object(
                                mc,
                                "get_parallel",
                                return_value=NS(enable_dsa_cache_layer_split=False),
                            ):
                                send(
                                    sender,
                                    np.arange(start, start + pages, dtype=np.int32),
                                    num_kv_tokens=count,
                                )
                            call = sender.kv_mgr.add_transfer_request.call_args
                            source, index_slice = call.args[1:3]
                            dest = np.arange(20, 28, dtype=np.int32)[index_slice]
                            plan = mc.build_dcp_token_transfer_plan(
                                source,
                                dest,
                                physical_page_size=64,
                                dcp_size=1,
                                dcp_rank=0,
                                num_kv_tokens=call.kwargs["num_kv_tokens"],
                            )
                            observed.extend(
                                zip(
                                    plan.src_token_indices.tolist(),
                                    plan.dst_token_indices.tolist(),
                                )
                            )
                            start += pages
                        expected = [
                            (t, 20 * 64 + t)
                            for t in range(cp * 256, min((cp + 1) * 256, total_tokens))
                        ]
                        self.assertEqual(observed, expected)
                        self.assertTrue(
                            sender.kv_mgr.add_transfer_request.call_args.args[3]
                        )

    def test_receiver_mapping_dummy_and_completion(self):
        for plain, hybrid, draft, source in [
            (True, False, 2, 2),
            (False, True, 2, 2),
            (True, False, 0, 0),
            (True, False, 2, 0),
            (False, True, 0, 0),
        ]:
            for rank in range(8):
                with self.subTest(
                    plain=plain, hybrid=hybrid, draft=draft, source=source, rank=rank
                ):
                    m = manager()
                    m.attn_tp_size = 8
                    m.kv_args.engine_rank = rank
                    m.kv_args.draft_kv_entry_count = draft
                    m.is_mla_backend = plain
                    m.is_hybrid_mla_backend = hybrid
                    info = PrefillServerInfo(16, 1, 1, 1, 64, None, True)
                    info.draft_kv_entry_count = source
                    m._resolve_rank_mapping(info)
                    expected = 1 if plain and not source else 2
                    self.assertEqual(info.required_prefill_response_num, expected)
                    r = mc.MooncakeKVReceiver.__new__(mc.MooncakeKVReceiver)
                    r.kv_mgr = m
                    r.prefill_info = info
                    for key in [
                        "target_tp_rank",
                        "target_tp_ranks",
                        "target_cp_ranks",
                        "target_pp_ranks",
                    ]:
                        setattr(r, key, getattr(info, key))
                    r.bootstrap_addr = "localhost"
                    r.prefill_dp_rank = 0
                    r.bootstrap_room = 99
                    r.conclude_state = None
                    m.connection_pool = {}
                    r._get_bootstrap_info_from_server = lambda dp, cp, tp, pp: dict(
                        rank_ip="localhost", rank_port=1000 + tp, tp=tp
                    )
                    r._register_kv_args = Mock(return_value=True)
                    r._setup_bootstrap_infos()
                    real = [b["tp"] for b in r.bootstrap_infos if not b["is_dummy"]]
                    self.assertEqual(
                        real, [2 * rank] if expected == 1 else [2 * rank, 2 * rank + 1]
                    )
                    m.request_status[99] = KVPoll.WaitingForInput
                    m.required_prefill_response_num_table[99] = expected
                    states = []
                    msgs = [
                        [b"99", str(int(KVPoll.Success)).encode(), str(tp).encode()]
                        for tp in real
                    ]
                    self.assertEqual(
                        run_loop(
                            m,
                            m.start_decode_thread,
                            msgs,
                            lambda: states.append(m.request_status[99]),
                        ),
                        [],
                    )
                    self.assertEqual(states[-1], KVPoll.Success)
                    if expected == 2:
                        self.assertNotEqual(states[1], KVPoll.Success)

    def test_invalid_registration_then_valid_and_request_failure(self):
        for malformed in [
            "counts",
            "bytes",
            "integer",
            "packed",
            "topology",
            "missing_layer",
            "one_sided_layers",
            "layer_count",
        ]:
            with self.subTest(malformed=malformed):
                m = manager()
                m.request_status = {1: KVPoll.Bootstrapping, 2: KVPoll.Bootstrapping}
                bad = registration("bad", count=1)
                if malformed == "bytes":
                    bad = registration("bad")
                    bad[4] = b"x"
                if malformed == "integer":
                    bad = registration("bad")
                    bad[20] = b"oops"
                if malformed == "packed":
                    bad = registration("bad", token_major=False)
                if malformed == "topology":
                    bad = registration("bad")
                    bad[17] = b"1"
                if malformed == "missing_layer":
                    bad = registration("bad")
                    bad[12] = struct.pack("3I", 9, 1, 1)
                if malformed == "one_sided_layers":
                    bad = registration("bad")
                    bad[12] = b""
                if malformed == "layer_count":
                    bad = registration("bad")
                    bad[12] = struct.pack("2I", 0, 1)
                messages = [bad, request("bad", 1), registration(), request("ok", 2)]
                errors = run_loop(m, m.start_prefill_thread, messages)
                self.assertEqual(errors, [])
                self.assertIn("ok", m.decode_kv_args_table)
                self.assertNotIn("bad", m.decode_kv_args_table)
                self.assertEqual(m.request_status[1], KVPoll.Failed)
                self.assertEqual(m.request_status[2], KVPoll.WaitingForInput)
                self.assertIn("Rejected Mooncake registration", m.failure_records[1])
                self.assertEqual(m.sync_status_to_decode_endpoint.call_count, 1)

    def test_rejection_during_transfer_remains_failed(self):
        for hybrid in (False, True):
            for phase in ("kv", "aux"):
                with self.subTest(hybrid=hybrid, phase=phase):
                    m = manager()
                    m.is_mla_backend = not hybrid
                    m.is_hybrid_mla_backend = hybrid
                    m.enable_trace = False
                    m.bootstrap_port = 8998
                    m._staging_outstanding = defaultdict(int)
                    m._validate_envelope_kv_layout = Mock()
                    m.sync_status_to_decode_endpoint = (
                        mc.MooncakeKVManager.sync_status_to_decode_endpoint.__get__(m)
                    )
                    m._status_messages = Mock()
                    m._connect = Mock(
                        return_value=NS(send_multipart=m._status_messages)
                    )
                    m._socket_send_locks = defaultdict(threading.Lock)
                    for session, room in (("bad", 1), ("ok", 2)):
                        m._register_decode_peer(registration(session), session)
                        m.request_status[room] = KVPoll.WaitingForInput
                        m.transfer_infos[room] = {
                            session: mc.TransferInfo.from_zmq(request(session, room))
                        }
                    entered, resume = threading.Event(), threading.Event()
                    errors = []

                    def block(session, at):
                        if session == "bad" and phase == at:
                            entered.set()
                            if not resume.wait(10):
                                raise TimeoutError("Transfer was not resumed")
                        return 0

                    m._transfer_data = lambda session, blocks: block(session, "kv")
                    m.send_aux = lambda req, *args: block(
                        req.mooncake_session_id, "aux"
                    )
                    chunks = [
                        NS(
                            room=room,
                            staging_counted=False,
                            is_last_chunk=True,
                            index_slice=slice(0, 1),
                            prefill_kv_indices=np.array([0]),
                            num_kv_tokens=1,
                            state_indices=[],
                            prefill_aux_index=0,
                        )
                        for room in (1, 2)
                    ]

                    def get_chunk():
                        if not chunks:
                            raise EndMessages()
                        return chunks.pop(0)

                    def worker():
                        try:
                            m.transfer_worker(NS(get=get_chunk), None)
                        except EndMessages:
                            pass
                        except BaseException as exc:
                            errors.append(exc)

                    thread = threading.Thread(target=worker, daemon=True)
                    thread.start()
                    try:
                        self.assertTrue(entered.wait(10), errors)
                        m._register_decode_peer(registration("bad", count=1), "bad")
                        self.assertEqual(m.check_status(1), KVPoll.Failed)
                    finally:
                        resume.set()
                        thread.join(10)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(errors, [])
                    self.assertEqual(m.check_status(1), KVPoll.Failed)
                    self.assertEqual(m.check_status(2), KVPoll.Success)
                    sender = mc.MooncakeKVSender.__new__(mc.MooncakeKVSender)
                    sender.kv_mgr = m
                    sender.bootstrap_room = 1
                    sender.conclude_state = None
                    sender.trace_ctx = Mock()
                    self.assertEqual(sender.poll(), KVPoll.Failed)
                    notifications = [
                        (int(c.args[0][0]), int(c.args[0][1]))
                        for c in m._status_messages.call_args_list
                    ]
                    self.assertIn((1, KVPoll.Failed), notifications)
                    self.assertNotIn((1, KVPoll.Success), notifications)
                    self.assertIn((2, KVPoll.Success), notifications)
                    self.assertFalse(m._staging_outstanding)

    def test_success_notification_is_ordered_before_concurrent_failure(self):
        m = manager()
        m.request_status[1] = KVPoll.Success
        sending, release = threading.Event(), threading.Event()
        failing, failed = threading.Event(), threading.Event()
        transitioned = threading.Event()
        notifications, errors = [], []

        def send(message):
            status = int(message[1])
            if status == KVPoll.Success:
                sending.set()
                if not release.wait(10):
                    raise TimeoutError("Notification was not released")
            notifications.append(status)

        m._connect = Mock(return_value=NS(send_multipart=send))
        m._socket_send_locks = defaultdict(threading.Lock)

        def notify_success():
            try:
                mc.MooncakeKVManager.sync_status_to_decode_endpoint(
                    m, "localhost", 1234, 1, KVPoll.Success, 0
                )
            except BaseException as exc:
                errors.append(exc)

        def reject():
            try:
                failing.set()
                m.update_status(1, KVPoll.Failed)
                transitioned.set()
                mc.MooncakeKVManager.sync_status_to_decode_endpoint(
                    m, "localhost", 1234, 1, KVPoll.Failed, 0
                )
            except BaseException as exc:
                errors.append(exc)
            finally:
                failed.set()

        success_thread = threading.Thread(target=notify_success, daemon=True)
        failure_thread = threading.Thread(target=reject, daemon=True)
        success_thread.start()
        try:
            self.assertTrue(sending.wait(10), errors)
            failure_thread.start()
            self.assertTrue(failing.wait(10))
            # State updates do not wait for the network send.
            self.assertTrue(transitioned.wait(10))
            self.assertEqual(m.check_status(1), KVPoll.Failed)
            # Failure notification cannot overtake an already-started success.
            self.assertFalse(failed.wait(0.1))
        finally:
            release.set()
            success_thread.join(10)
            if failure_thread.ident is not None:
                failure_thread.join(10)
        self.assertFalse(success_thread.is_alive())
        self.assertFalse(failure_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(notifications, [KVPoll.Success, KVPoll.Failed])
        self.assertEqual(m.check_status(1), KVPoll.Failed)

    def test_failed_status_blocks_late_updates_and_notifications(self):
        m = manager()
        m.request_status[1] = KVPoll.WaitingForInput
        m.update_status(1, KVPoll.Success)
        m.update_status(1, KVPoll.Failed)
        for status in (KVPoll.WaitingForInput, KVPoll.Transferring, KVPoll.Success):
            m.update_status(1, status)
            self.assertEqual(m.check_status(1), KVPoll.Failed)
        m._status_messages = Mock()
        m._connect = Mock(return_value=NS(send_multipart=m._status_messages))
        m._socket_send_locks = defaultdict(threading.Lock)
        mc.MooncakeKVManager.sync_status_to_decode_endpoint(
            m, "localhost", 1234, 1, KVPoll.Success, 0
        )
        m._status_messages.assert_not_called()
        sender = mc.MooncakeKVSender.__new__(mc.MooncakeKVSender)
        sender.kv_mgr = m
        sender.bootstrap_room = 1
        sender.clear()
        m.update_status(1, KVPoll.Success)
        self.assertNotIn(1, m.request_status)
        m.update_status(1, KVPoll.Bootstrapping)
        self.assertEqual(m.check_status(1), KVPoll.Bootstrapping)

    def test_target_groups_are_reused_across_entries(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                m = manager()
                m.kv_args.engine_rank = rank
                m.attn_tp_rank = rank
                m.is_mla_backend = False
                m.is_hybrid_mla_backend = True
                m.kv_args.kv_data_ptrs = [0x100000 + i * 0x100000 for i in range(26)]
                m.kv_args.kv_item_lens = [64] * 24 + [1024] * 2
                layers = list(range(24)) + [24, 24]
                m.kv_args.kv_layer_ids = layers
                wire = registration()
                wire[4] = struct.pack(
                    "26Q", *[0x10000000 + i * 0x100000 for i in range(26)]
                )
                wire[12] = struct.pack("26I", *layers)
                wire[18] = struct.pack("26Q", *([64] * 24 + [2048] * 2))
                m._register_decode_peer(wire, "ok")
                info = m.decode_kv_args_table["ok"]
                m._validate_envelope_kv_layout = Mock()
                m._transfer_data = Mock(return_value=0)
                with patch.object(
                    mc,
                    "group_concurrent_contiguous",
                    wraps=mc.group_concurrent_contiguous,
                ) as grouping:
                    m.send_kvcache_mixed(
                        "ok",
                        np.array([1, 3]),
                        np.array([4, 7]),
                        info=info,
                        src_page_offset=0,
                        decode_prefix_len=0,
                        num_kv_tokens=65,
                    )
                self.assertEqual(grouping.call_count, 1 if rank == 0 else 0)
                blocks = m._transfer_data.call_args.args[1]
                expected = []
                if rank == 0:
                    for src, dst in zip(
                        m.kv_args.kv_data_ptrs[:24], info.dst_kv_ptrs[:24]
                    ):
                        expected.extend(
                            [(src + 64, dst + 256, 64), (src + 192, dst + 448, 1)]
                        )
                # Compare target descriptors, including discontiguous pages and a one-token tail.
                target_blocks = [b for b in blocks if b[0] < m.kv_args.kv_data_ptrs[24]]
                self.assertEqual(target_blocks, expected)

    def test_staging_prefetch_uses_one_registration_lookup(self):
        class RemovedAfterRead(dict):
            def get(self, key, default=None):
                return self.pop(key, default)

        for present in (True, False):
            with self.subTest(present=present):
                m = manager()
                m.enable_staging = True
                m.kv_buffer_tensors = []
                m._staging_ctx = NS(prefetch_requested=set(), prefetch_sockets={})
                m.transfer_infos[1] = {"s": mc.TransferInfo.from_zmq(request("s", 1))}
                m.decode_kv_args_table = RemovedAfterRead(
                    {"s": NS(dst_attn_tp_size=8)} if present else {}
                )
                with patch.object(
                    mc, "get_schedule", return_value=NS(chunked_prefill_size=64)
                ), patch(
                    "sglang.srt.disaggregation.common.staging_handler.prefetch_staging_reqs"
                ) as prefetch:
                    m._prefetch_staging_reqs(1)
                self.assertEqual(prefetch.call_count, int(present))

    def test_registration_changes_do_not_terminate_transfer_worker(self):
        for change in ("reject", "recover", "missing"):
            with self.subTest(change=change):
                m = manager()
                m.enable_trace = False
                m.bootstrap_port = 8998
                m._staging_outstanding = defaultdict(int)
                m.send_kvcache_mixed = Mock(return_value=0)
                m.send_aux = Mock(return_value=0)
                for session, room in (("changing", 1), ("ok", 2)):
                    m._register_decode_peer(registration(session), session)
                    m.request_status[room] = KVPoll.WaitingForInput
                    m.transfer_infos[room] = {
                        session: mc.TransferInfo.from_zmq(request(session, room))
                    }
                if change == "recover":
                    m.rejected_registrations["changing"] = "layout rejected"
                elif change == "missing":
                    m.decode_kv_args_table.pop("changing")

                reached = threading.Event()
                resume = threading.Event()

                class InterleavedDict(dict):
                    """Pause a reader at a controlled registration-update boundary."""

                    def pause(self, key):
                        if key == "changing" and not reached.is_set():
                            reached.set()
                            if not resume.wait(10):
                                raise TimeoutError("Registration update did not finish")

                    def __contains__(self, key):
                        present = super().__contains__(key)
                        if change == "recover":
                            self.pause(key)
                        return present

                    def __getitem__(self, key):
                        if change == "reject":
                            self.pause(key)
                        return super().__getitem__(key)

                    def get(self, key, default=None):
                        if change == "reject":
                            self.pause(key)
                        value = super().get(key, default)
                        if change == "recover":
                            self.pause(key)
                        return value

                if change == "reject":
                    m.decode_kv_args_table = InterleavedDict(m.decode_kv_args_table)
                elif change == "recover":
                    m.rejected_registrations = InterleavedDict(m.rejected_registrations)

                chunks = [
                    NS(
                        room=room,
                        staging_counted=False,
                        is_last_chunk=True,
                        index_slice=slice(0, 1),
                        prefill_kv_indices=np.array([0]),
                        num_kv_tokens=1,
                        state_indices=[],
                        prefill_aux_index=0,
                    )
                    for room in (1, 2)
                ]
                errors = []

                def get_chunk():
                    if not chunks:
                        raise EndMessages()
                    return chunks.pop(0)

                def worker():
                    try:
                        m.transfer_worker(NS(get=get_chunk), None)
                    except EndMessages:
                        pass
                    except BaseException as exc:
                        errors.append(exc)

                thread = threading.Thread(target=worker, daemon=True)
                thread.start()
                try:
                    if change != "missing":
                        self.assertTrue(reached.wait(10), errors)
                        wire = registration(
                            "changing", count=1 if change == "reject" else 2
                        )
                        m._register_decode_peer(wire, "changing")
                finally:
                    resume.set()
                    thread.join(10)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(m.request_status[1], KVPoll.Failed)
                self.assertEqual(m.request_status[2], KVPoll.Success)
                self.assertIn(1, m.failure_records)
                self.assertEqual(m.send_kvcache_mixed.call_count, 1)
                self.assertEqual(m.send_kvcache_mixed.call_args.args[0], "ok")
                self.assertEqual(m.send_aux.call_count, 1)
                self.assertNotIn(1, m.transfer_infos)
                self.assertNotIn(2, m.transfer_infos)
                self.assertFalse(m._staging_outstanding)
                statuses = [
                    (c.args[2], c.args[3])
                    for c in m.sync_status_to_decode_endpoint.call_args_list
                ]
                self.assertIn((1, KVPoll.Failed), statuses)
                self.assertIn((2, KVPoll.Success), statuses)
                self.assertNotIn((1, KVPoll.Success), statuses)

    def test_rejected_session_recovers_only_after_valid_registration(self):
        m = manager()
        m.request_status = {1: KVPoll.Bootstrapping}
        m.transfer_infos[1] = {"bad": mc.TransferInfo.from_zmq(request("bad", 1))}
        m.engine = NS(send_probe=lambda session: 0)
        m._register_decode_peer(registration("bad"), "bad")
        self.assertIn("bad", m.decode_kv_args_table)
        bad = registration("bad")
        bad[12] = struct.pack("3I", 9, 1, 1)
        errors = run_loop(m, m.start_prefill_thread, [bad])
        self.assertEqual(errors, [])
        self.assertEqual(m.request_status[1], KVPoll.Failed)
        m._run_one_probe_pass()
        self.assertIn("bad", m.rejected_registrations)
        self.assertNotIn("bad", m.decode_kv_args_table)
        errors = run_loop(m, m.start_prefill_thread, [registration("bad")])
        self.assertEqual(errors, [])
        self.assertIn("bad", m.decode_kv_args_table)
        self.assertNotIn("bad", m.rejected_registrations)

    def test_failed_room_is_not_revived_by_late_success(self):
        m = manager()
        m.request_status[1] = KVPoll.Failed
        m.required_prefill_response_num_table[1] = 1
        self.assertEqual(run_loop(m, m.start_decode_thread, [[b"1", b"4", b"0"]]), [])
        self.assertEqual(m.request_status[1], KVPoll.Failed)
        self.assertEqual(
            run_loop(m, m.start_prefill_thread, [registration(), request("ok", 1)]), []
        )
        self.assertEqual(m.request_status[1], KVPoll.Failed)
        self.assertNotIn(1, m.transfer_infos)

    def test_rejection_before_sender_creation_fails_without_timeout(self):
        import sglang.srt.disaggregation.common.conn as common

        m = manager()
        m.is_dummy_cp_rank = False
        m.enable_trace = False
        errors = run_loop(
            m, m.start_prefill_thread, [registration("bad", count=1), request("bad", 7)]
        )
        self.assertEqual(errors, [])
        self.assertIn(7, m.failure_records)
        with patch.object(common, "get_parallel", return_value=NS(dp_size=1)):
            sender = mc.MooncakeKVSender(m, "localhost", 7, [0], 0)
        self.assertEqual(sender.poll(), KVPoll.Failed)

    def test_packed_layout_is_rejected_on_either_peer(self):
        for tp in [8, 16]:
            for side in ["source", "destination"]:
                with self.subTest(tp=tp, side=side):
                    m = manager(tp=tp)
                    if tp == 8:
                        m.kv_args.draft_kv_head_num = 2
                        m.kv_args.kv_item_lens = [64, 2048, 2048]
                    info = mc.KVArgsRegisterInfo.from_zmq(registration())
                    if side == "source":
                        m.kv_args.draft_kv_token_major = False
                    else:
                        info.dst_draft_token_major = False
                    with self.assertRaisesRegex(ValueError, "token-major"):
                        m.prepare_dcp_entry_geometry(info)

    def test_metadata_real_pool_buffers(self):
        for layout in ["nhd", "vectorized_5d"]:
            with self.subTest(layout=layout):
                p = pool(layout)
                a = KVArgs()  # Production publishes target page_size after metadata.
                set_draft_kv_transfer_metadata(
                    a, p, draft_kv_entry_count=2, tp_size=16, total_head_num=16
                )
                self.assertEqual(a.draft_kv_entry_count, 2)
                self.assertEqual(
                    getattr(a, "draft_kv_token_major", True), layout == "nhd"
                )

    def test_packed_tail_repro_equal_tp(self):
        import ctypes

        for count in (1, 65):
            with self.subTest(count=count):
                src = pool("vectorized_5d", heads=2)
                dst = pool("vectorized_5d", heads=2)
                for t in src.k_buffer + src.v_buffer:
                    t.fill_(7)
                for t in dst.k_buffer + dst.v_buffer:
                    t.fill_(-1)
                target = torch.zeros(192, dtype=torch.uint8)
                source_target = target.clone()
                m = manager(tp=8)
                sp, sl, si = src.get_contiguous_buf_infos()
                dp, dl, di = dst.get_contiguous_buf_infos()
                m.kv_args.kv_data_ptrs = [source_target.data_ptr()] + sp
                m.kv_args.kv_item_lens = [64] + si
                set_draft_kv_transfer_metadata(
                    m.kv_args, src, draft_kv_entry_count=2, tp_size=8, total_head_num=16
                )
                da = KVArgs()
                da.page_size = 64
                set_draft_kv_transfer_metadata(
                    da, dst, draft_kv_entry_count=2, tp_size=8, total_head_num=16
                )
                wire = registration()
                wire[4] = struct.pack("3Q", target.data_ptr(), *dp)
                wire[18] = struct.pack("3Q", 64, *di)
                wire[23] = b"1" if getattr(da, "draft_kv_token_major", True) else b"0"
                reg = mc.KVArgsRegisterInfo.from_zmq(wire)
                try:
                    m.prepare_dcp_entry_geometry(reg)
                except ValueError as exc:
                    self.assertIn("token-major", str(exc))
                    continue
                m._validate_envelope_kv_layout = lambda *a: None

                def transfer(session, blocks):
                    for a, b, n in blocks:
                        ctypes.memmove(b, a, n)
                    return 0

                m._transfer_data = transfer
                m.send_kvcache_mixed(
                    "s",
                    np.array([0, 1]),
                    np.array([1, 2]),
                    info=reg,
                    src_page_offset=0,
                    decode_prefix_len=0,
                    num_kv_tokens=count,
                )
                missing = 0
                for token in range(count):
                    slot = token + 64
                    missing += int(
                        (
                            dst.v_buffer[0][slot // 64, :, slot % 64 // 8, :, slot % 8]
                            != 7
                        ).sum()
                    )
                self.fail(
                    f"Packed layout accepted: {missing} required V elements missing for {count} tokens at equal TP8"
                )

    def test_real_pool_transfer_heads_and_tail(self):
        import ctypes

        for plain in (True, False):
            for src_tp in (8, 16):
                for count in (1, 65):
                    for dst_rank in range(8):
                        with self.subTest(
                            plain=plain, src_tp=src_tp, count=count, dst_rank=dst_rank
                        ):
                            dst = pool("nhd", heads=2)
                            for t in dst.k_buffer + dst.v_buffer:
                                t.fill_(-1)
                            target = torch.zeros(192, dtype=torch.uint8)
                            receiver = manager()
                            receiver.attn_tp_size = 8
                            receiver.kv_args.engine_rank = dst_rank
                            receiver.is_mla_backend = plain
                            receiver.is_hybrid_mla_backend = not plain
                            info = PrefillServerInfo(src_tp, 1, 1, 1, 64, None, True)
                            info.draft_kv_entry_count = 2
                            receiver._resolve_rank_mapping(info)
                            writes = set()
                            for src_rank in info.target_tp_ranks:
                                src = pool("nhd", heads=16 // src_tp)
                                for t in src.k_buffer + src.v_buffer:
                                    for h in range(16 // src_tp):
                                        t[:, h, :] = (
                                            torch.arange(192)[:, None]
                                            + (src_rank * (16 // src_tp) + h) * 100
                                        )
                                source_target = torch.full((192,), 7, dtype=torch.uint8)
                                m = manager(tp=src_tp)
                                m.is_mla_backend = plain
                                m.is_hybrid_mla_backend = not plain
                                m.attn_tp_rank = src_rank
                                m.kv_args.engine_rank = src_rank
                                ptrs, lens, items = src.get_contiguous_buf_infos()
                                m.kv_args.kv_data_ptrs = [
                                    source_target.data_ptr()
                                ] + ptrs
                                m.kv_args.kv_item_lens = [64] + items
                                set_draft_kv_transfer_metadata(
                                    m.kv_args,
                                    src,
                                    draft_kv_entry_count=2,
                                    tp_size=src_tp,
                                    total_head_num=16,
                                )
                                dp, dl, di = dst.get_contiguous_buf_infos()
                                da = KVArgs()
                                da.page_size = 64
                                set_draft_kv_transfer_metadata(
                                    da,
                                    dst,
                                    draft_kv_entry_count=2,
                                    tp_size=8,
                                    total_head_num=16,
                                )
                                wire = registration()
                                wire[4] = struct.pack("3Q", target.data_ptr(), *dp)
                                wire[7] = str(dst_rank).encode()
                                wire[18] = struct.pack("3Q", 64, *di)
                                wire[23] = (
                                    b"1"
                                    if getattr(da, "draft_kv_token_major", True)
                                    else b"0"
                                )
                                reg = mc.KVArgsRegisterInfo.from_zmq(wire)
                                m.prepare_dcp_entry_geometry(reg)
                                m._validate_envelope_kv_layout = lambda *a: None

                                def transfer(session, blocks):
                                    for a, b, n in blocks:
                                        for address in range(b, b + n):
                                            self.assertNotIn(address, writes)
                                            writes.add(address)
                                        ctypes.memmove(b, a, n)
                                    return 0

                                m._transfer_data = transfer
                                self.assertEqual(
                                    m.send_kvcache_mixed(
                                        "s",
                                        np.array([0, 1]),
                                        np.array([1, 2]),
                                        info=reg,
                                        src_page_offset=0,
                                        decode_prefix_len=0,
                                        num_kv_tokens=count,
                                    ),
                                    0,
                                )
                            self.assertTrue(torch.all(target[64 : 64 + count] == 7))
                            self.assertTrue(torch.all(target[64 + count :] == 0))
                            for t in dst.k_buffer + dst.v_buffer:
                                for h in range(2):
                                    torch.testing.assert_close(
                                        t[64 : 64 + count, h, :],
                                        (
                                            torch.arange(count)[:, None]
                                            + (2 * dst_rank + h) * 100
                                        )
                                        .expand(count, 8)
                                        .half(),
                                        rtol=0,
                                        atol=0,
                                    )
                                self.assertTrue(torch.all(t[64 + count :] == -1))
                            self.assertEqual(len(writes), count * (1 + 2 * 2 * 8 * 2))

    def test_mha_not_rejected_for_packed_metadata(self):
        m = manager()
        m.is_mla_backend = False
        m.kv_args.draft_kv_token_major = False
        info = mc.KVArgsRegisterInfo.from_zmq(registration(token_major=False))
        info.requires_dcp_relayout = False
        self.assertFalse(m._requires_dcp_entry_geometry(info))


if __name__ == "__main__":
    unittest.main(verbosity=2)
