"""Generation-scoped PD metadata, admission and supported deployment boundaries."""

import asyncio
import json
import time
from collections import OrderedDict
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.expanded_input import (
    validate_expanded_input_config,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_ABORT, MultimodalInputs, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.req_double import ReqDouble

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestExpandedInputMetadata(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.q = DecodePreallocQueue.__new__(DecodePreallocQueue)
        self.q.queue = []
        self.q.gloo_group = None
        self.q._pending_expanded_lens = OrderedDict()
        self.q._seen_expanded_metadata = OrderedDict()
        self.q._expanded_len_cache_ttl = 600
        self.q._expanded_len_cache_max = 4
        self.req = self.request()

    def request(self, rid="d1", room=7, length=2):
        return ReqDouble(
            rid=rid,
            bootstrap_room=room,
            origin_input_ids=[1] * length,
            multimodal_inputs=None,
            needs_expanded_input_len=True,
            expanded_input_len=None,
            skip_radix_cache_insert=False,
            _expanded_input_len_wait_start=time.monotonic(),
            finished_reason=None,
            return_logprob=False,
        )

    def enqueue(self, req=None):
        req = req or self.req
        self.q.queue.append(NS(req=req))
        return req

    def test_early_duplicate_and_conflict(self):
        self.q.set_expanded_input_len("7", 8, -3, "g1")
        self.enqueue()
        self.q._sync_expanded_input_metadata()
        self.assertEqual(self.req.expanded_input_len, 8)
        self.assertEqual(
            self.req.multimodal_inputs.mrope_position_delta.tolist(), [[-3]]
        )
        self.q.set_expanded_input_len("7", 8, -3, "g1")
        self.assertFalse(self.q._pending_expanded_lens)
        self.q.set_expanded_input_len("7", 9, -3, "g1")
        self.q._sync_expanded_input_metadata()
        self.assertIsInstance(self.req.finished_reason, FINISH_ABORT)

    def test_room_reuse_with_equal_metadata_and_stale_retry(self):
        self.enqueue()
        self.q.set_expanded_input_len("7", 8, 0, "old-generation")
        self.q._sync_expanded_input_metadata()
        self.q.queue.clear()
        fresh = self.enqueue(self.request(rid="d2"))
        self.q.set_expanded_input_len("7", 8, 0, "old-generation")
        self.q._sync_expanded_input_metadata()
        self.assertIsNone(fresh.expanded_input_len)
        self.q.set_expanded_input_len("7", 8, 0, "new-generation")
        self.q._sync_expanded_input_metadata()
        self.assertEqual(fresh.expanded_input_len, 8)
        self.assertEqual(fresh._expanded_input_generation, "new-generation")

    def test_generation_bound_lookup_does_not_scan_pending_keys(self):
        class DirectLookupOnly(dict):
            def __iter__(self):
                raise AssertionError("generation-bound lookup scanned pending keys")

        self.req._expanded_input_generation = "known"
        self.q._pending_expanded_lens = DirectLookupOnly(
            {("7", "known"): ((8, 0, 2), time.monotonic())}
        )
        self.enqueue()
        self.q._sync_expanded_input_metadata()
        self.assertEqual(self.req.expanded_input_len, 8)

    def test_equal_length_still_disables_radix_cache(self):
        self.q._apply_expanded_input_len(self.req, (2, 0, 2))
        self.assertTrue(self.req.skip_radix_cache_insert)

    def test_invalid_values_and_versions_abort(self):
        for i, (length, delta, version) in enumerate(
            [(True, 0, 2), (8, True, 2), (8, 0, 1), (8, -8, 2), (8, 2**80, 2)]
        ):
            self.q.queue.clear()
            req = self.enqueue(self.request(rid=str(i), room=i))
            self.q.set_expanded_input_len(str(i), length, delta, "g", version)
            self.q._sync_expanded_input_metadata()
            self.assertIsInstance(req.finished_reason, FINISH_ABORT)
            self.assertIsNone(req.expanded_input_len)

    def test_pending_conflict_never_releases_admission(self):
        self.q.set_expanded_input_len("7", 8, 0, "g")
        self.q.set_expanded_input_len("7", 9, 0, "g")
        self.enqueue()
        self.q._sync_expanded_input_metadata()
        self.assertIsNone(self.req.expanded_input_len)
        self.assertIsInstance(self.req.finished_reason, FINISH_ABORT)

    def test_bounded_early_cache_preserves_accepted_deliveries(self):
        accepted = [self.q.set_expanded_input_len(str(i), 8, 0, "g") for i in range(10)]
        self.assertEqual(accepted, [True] * 4 + [False] * 6)
        self.assertIn(("0", "g"), self.q._pending_expanded_lens)
        self.assertEqual(len(self.q._pending_expanded_lens), 4)
        self.assertFalse(self.q._seen_expanded_metadata)

    def test_pending_metadata_survives_tombstone_expiry(self):
        self.q.set_expanded_input_len("7", 8, 0, "g")
        self.q._expanded_len_cache_ttl = -1
        self.enqueue()
        self.q._sync_expanded_input_metadata()
        self.assertIsNone(self.req.finished_reason)
        self.assertEqual(self.req.expanded_input_len, 8)

    def test_follower_uses_leader_metadata_not_local_expiry(self):
        self.enqueue()
        decisions = [("d1", "g", (8, -3, 2), None)]

        def receive(objects, **kwargs):
            objects[0] = decisions

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_rank", return_value=1),
            patch("torch.distributed.get_global_rank", return_value=4),
            patch("torch.distributed.broadcast_object_list", side_effect=receive),
        ):
            self.q._sync_expanded_input_metadata()
        self.assertEqual(self.req.expanded_input_len, 8)

    def test_delta_only_extend_positions(self):
        fb = ForwardBatch.__new__(ForwardBatch)
        fb.seq_lens_cpu = torch.tensor([10])
        batch = NS(
            multimodal_inputs=[
                MultimodalInputs(mm_items=[], mrope_position_delta=torch.tensor([[-3]]))
            ],
            extend_lens=[2],
            prefix_lens=[8],
            reqs=[NS(session=None)],
        )
        fb._compute_mrope_positions_extend(NS(device="cpu"), batch)
        self.assertEqual(fb.mrope_positions.tolist(), [[5, 6]] * 3)

    def test_batch_children_and_text_item_flags(self):
        obj = GenerateReqInput(
            bootstrap_pair_key="pair", text=["a", "b"], image_data=[["x"], None]
        )
        obj.normalize_batch_and_arguments()
        tm = TokenizerManager.__new__(TokenizerManager)
        tm._expanded_input_tasks = {}
        with (
            get_context().override_server_args(disaggregation_mode="decode"),
            envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True),
        ):
            tm._handle_epd_disaggregation_encode_request(obj)
        self.assertTrue(obj[0]._needs_expanded_input_len)
        self.assertFalse(obj[1]._needs_expanded_input_len)

    def test_legacy_cached_batch_children_use_encoder_only_for_media(self):
        """Cached children created before EPD dispatch must inherit its decision."""
        for adaptive in (False, True):
            obj = GenerateReqInput(text=["image", "text"], image_data=[["x"], None])
            obj.normalize_batch_and_arguments()
            children = [obj[i] for i in range(len(obj.rid))]
            tm = TokenizerManager.__new__(TokenizerManager)
            tm._should_dispatch_to_encoder = lambda request: not adaptive
            with (
                get_context().override_server_args(
                    disaggregation_mode="prefill",
                    language_only=True,
                    encoder_transfer_backend="zmq_to_tokenizer",
                    enable_adaptive_dispatch_to_encoder=adaptive,
                ),
                envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(False),
            ):
                tm._handle_epd_disaggregation_encode_request(obj)
            self.assertEqual(children[0].need_wait_for_mm_inputs, not adaptive)
            self.assertFalse(children[1].need_wait_for_mm_inputs)

    def test_parallel_sampling_is_rejected_before_dispatch(self):
        obj = GenerateReqInput(
            bootstrap_pair_key="pair",
            text="a",
            image_data=["x"],
            sampling_params={"n": 2},
        )
        obj.normalize_batch_and_arguments()
        tm = TokenizerManager.__new__(TokenizerManager)
        tm._expanded_input_tasks = {}
        with (
            get_context().override_server_args(disaggregation_mode="decode"),
            envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True),
        ):
            with self.assertRaisesRegex(ValueError, "parallel sampling"):
                tm._handle_epd_disaggregation_encode_request(obj)

    def test_feature_off_preserves_legacy_decode_dispatch(self):
        obj = GenerateReqInput(bootstrap_pair_key="pair", text="a", image_data=["x"])
        obj.normalize_batch_and_arguments()
        tm = TokenizerManager.__new__(TokenizerManager)
        tm._expanded_input_tasks = {}
        tm.enable_trace = False
        tm.mm_receiver = Mock()
        tm.mm_receiver.send_encode_request.return_value = None
        with (
            get_context().override_server_args(
                disaggregation_mode="decode",
                language_only=True,
                encoder_transfer_backend="mooncake",
            ),
            envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(False),
        ):
            tm._handle_epd_disaggregation_encode_request(obj)
        tm.mm_receiver.send_encode_request.assert_called_once()

    def test_session_is_rejected_before_dispatch(self):
        obj = GenerateReqInput(
            bootstrap_pair_key="pair", text="a", image_data=["x"], session_id="session"
        )
        obj.normalize_batch_and_arguments()
        tm = TokenizerManager.__new__(TokenizerManager)
        tm._expanded_input_tasks = {}
        with (
            get_context().override_server_args(disaggregation_mode="decode"),
            envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True),
        ):
            with self.assertRaisesRegex(ValueError, "sessions"):
                tm._handle_epd_disaggregation_encode_request(obj)

    def test_multimodal_rebootstrap_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "rebootstrap"):
            Req.build_rebootstrap_payload(self.req)

    def test_scoped_configuration_guards(self):
        with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True):
            cfg = resolving_view(
                ServerArgs(
                    model_path="dummy",
                    disaggregation_mode="decode",
                )
            )
            validate_expanded_input_config(cfg, "Qwen4ExpForConditionalGeneration")
            decoder = resolving_view(
                ServerArgs(
                    model_path="dummy",
                    disaggregation_mode="decode",
                    language_only=True,
                    encoder_transfer_backend="mooncake",
                )
            )
            validate_expanded_input_config(decoder, "Qwen4ExpForConditionalGeneration")
            with self.assertRaisesRegex(ValueError, "Qwen"):
                validate_expanded_input_config(cfg, "GlmImageForConditionalGeneration")
            for changes, error in [
                ({"pp_size": 2}, "pipeline"),
                (
                    {
                        "language_only": True,
                        "disaggregation_mode": "prefill",
                        "disagg_decode_urls": ["http://decode"],
                        "encoder_transfer_backend": "mooncake",
                    },
                    "zmq_to_tokenizer",
                ),
                ({"disaggregation_mode": "prefill"}, "decode-urls"),
            ]:
                cfg = resolving_view(
                    ServerArgs(
                        model_path="dummy",
                        disaggregation_mode=changes.get(
                            "disaggregation_mode", "decode"
                        ),
                    )
                )
                with get_context().override_server_args(**changes):
                    # resolving_view is attached to a concrete argument record.
                    args = ServerArgs(
                        model_path="dummy",
                        **dict({"disaggregation_mode": "decode"}, **changes),
                    )
                    with self.assertRaisesRegex(ValueError, error):
                        validate_expanded_input_config(
                            resolving_view(args), "Qwen4ExpForConditionalGeneration"
                        )

    def test_all_decode_urls_are_queried_but_only_owner_gets_metadata(self):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm._expanded_input_tasks = {}
        calls = []

        def post(url, **kwargs):
            calls.append((url, kwargs["json"]))
            query = kwargs["json"].get("query_generation")
            response = NS(
                status=202 if query and "d2" in url else 200,
                json=AsyncMock(return_value={"request_generation": "d-nonce"}),
            )
            cm = AsyncMock()
            cm.__aenter__.return_value = response
            return cm

        tm._get_decode_control_session = AsyncMock(return_value=NS(post=post))
        with get_context().override_server_args(
            disagg_decode_urls=["http://d1", "http://d2"]
        ):
            asyncio.run(tm._notify_decode_expanded_len("7", 8, "p1", 0, "pair"))
        self.assertEqual(
            {url for url, body in calls if body.get("query_generation")},
            {"http://d1/v1/expanded_input_len", "http://d2/v1/expanded_input_len"},
        )
        notifications = [
            (url, body) for url, body in calls if not body.get("query_generation")
        ]
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0][0], "http://d1/v1/expanded_input_len")
        self.assertEqual(notifications[0][1]["request_generation"], "d-nonce")

    def test_decode_control_key_is_independent(self):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm._expanded_input_tasks = {}
        tm._decode_control_session = None
        with (
            get_context().override_server_args(
                api_key="p-key", disagg_decode_control_key="d-key"
            ),
            patch("aiohttp.ClientSession", return_value=NS(closed=False)) as factory,
        ):
            asyncio.run(tm._get_decode_control_session())
        factory.assert_called_once_with(headers={"Authorization": "Bearer d-key"})

    def test_bad_json_and_nonobject_metadata_return_400(self):
        from sglang.srt.entrypoints.http_server import expanded_input_len

        for value in [[], None, 1, "bad"]:
            response = asyncio.run(
                expanded_input_len(NS(json=AsyncMock(return_value=value)))
            )
            self.assertEqual(response.status_code, 400)
        response = asyncio.run(
            expanded_input_len(
                NS(json=AsyncMock(side_effect=json.JSONDecodeError("bad", "", 0)))
            )
        )
        self.assertEqual(response.status_code, 400)

    def prepare_admission(self):
        self.q._resolve_pending_reqs = Mock()
        self.q._update_handshake_waiters = Mock()
        self.q._uses_swa_tail_prealloc = Mock(return_value=False)
        self.q._uses_swa_reservation = Mock(return_value=False)
        self.q._allocatable_token_budgets = Mock(return_value=0)
        self.q._hicache_pending_restore_tokens = Mock(return_value=0)
        self.q._check_if_req_exceed_kv_capacity = Mock(return_value=False)
        self.q.req_to_metadata_buffer_idx_allocator = NS(available_size=lambda: 1)
        self.q.req_to_token_pool = NS(available_size=lambda: 0)
        self.q.pp_size = 1
        self.q.enable_host_receive = False
        self.q.pending_reqs = []
        self.q.retracted_queue = []
        scheduler = NS(
            running_batch=NS(reqs=[]),
            enable_priority_scheduling=False,
            enable_hisparse=False,
            enable_lora=False,
            max_req_input_len=16,
            max_req_len=16,
            max_new_tokens_limit=None,
            max_total_num_tokens=128,
            kv_shard_widening=1,
            page_size=1,
            sliding_window_size=None,
            chunked_prefill_size=None,
            token_to_kv_pool_allocator=NS(
                max_new_tokens_for_memory=lambda input_len, tokens, **kw: tokens
            ),
            output_streamer=Mock(),
        )
        scheduler.init_req_max_new_tokens = lambda req: (
            Scheduler.init_req_max_new_tokens(scheduler, req)
        )
        self.q.scheduler = scheduler
        self.req.sampling_params = NS(max_new_tokens=100, min_new_tokens=100)
        dr = NS(
            req=self.req,
            kv_receiver=Mock(),
            waiting_for_input=True,
            is_rebootstrap=False,
        )
        self.q.queue = [dr]
        return dr

    def test_pop_waits_without_allocating_before_metadata(self):
        self.prepare_admission()
        good, failed = self.q.pop_preallocated()
        self.assertFalse(good or failed)
        self.assertIsNone(self.req.expanded_input_len)
        self.q._check_if_req_exceed_kv_capacity.assert_not_called()

    def test_pop_recomputes_output_allowance_for_early_and_late_metadata(self):
        for early in (True, False):
            self.q.queue = []
            self.q._pending_expanded_lens.clear()
            self.q._seen_expanded_metadata.clear()
            self.req = self.request()
            if early:
                self.q.set_expanded_input_len("7", 12, -3, "generation")
            self.prepare_admission()
            if not early:
                self.q.set_expanded_input_len("7", 12, -3, "generation")
            good, failed = self.q.pop_preallocated()
            self.assertFalse(good or failed)
            self.assertEqual(self.req.sampling_params.max_new_tokens, 3)
            self.assertEqual(self.req.sampling_params.min_new_tokens, 3)
            self.assertTrue(self.req.skip_radix_cache_insert)

    def test_pop_rejects_context_overflow_and_shorter_expansion(self):
        for length in (16, 1):
            self.q.queue = []
            self.req = self.request(rid=str(length))
            dr = self.prepare_admission()
            self.q.set_expanded_input_len("7", length, 0, str(length))
            good, failed = self.q.pop_preallocated()
            self.assertFalse(good)
            self.assertEqual(failed, [dr])
            self.assertIsInstance(self.req.finished_reason, FINISH_ABORT)
            self.assertFalse(self.q.queue)

    def test_allocator_pending_abort_is_terminal_before_preallocation(self):
        import types

        from sglang.srt.runtime_context import get_parallel

        dr = self.prepare_admission()
        self.req.session = None
        self.req.set_finish_with_abort = types.MethodType(
            Req.set_finish_with_abort, self.req
        )
        self.q.scheduler.token_to_kv_pool_allocator.max_new_tokens_for_memory = (
            lambda *a, **kw: None
        )
        self.q.set_expanded_input_len("7", 12, 0, "g")
        with get_parallel().override(tp_rank=0):
            good, failed = self.q.pop_preallocated()
        self.assertFalse(good)
        self.assertEqual(failed, [dr])
        self.assertIsInstance(self.req.finished_reason, FINISH_ABORT)
        self.assertFalse(self.q.queue)
        self.q._check_if_req_exceed_kv_capacity.assert_not_called()

    def test_pop_timeout_cleans_receiver_without_allocation(self):
        dr = self.prepare_admission()
        self.req._expanded_input_len_wait_start -= 10
        with envs.SGLANG_DISAGG_EXPANDED_LEN_WAIT_TIMEOUT.override(1):
            good, failed = self.q.pop_preallocated()
        self.assertFalse(good)
        self.assertEqual(failed, [dr])
        self.assertIsNone(dr.kv_receiver)

    def test_native_ipc_preserves_request_generation(self):
        from sglang.srt.managers.io_struct import msgpack_decode, msgpack_encode

        tm = TokenizerManager.__new__(TokenizerManager)
        tm._expanded_input_tasks = {}
        sent = []
        tm._dispatch_to_scheduler = sent.append
        tm.notify_expanded_input_len("room", 556, 512, "generation", 2)
        decoded = msgpack_decode(msgpack_encode(sent[0]))
        self.assertEqual(
            (
                decoded.rid,
                decoded.expanded_input_len,
                decoded.mrope_position_delta,
                decoded.metadata_version,
                decoded.request_generation,
            ),
            ("room", 556, 512, 2, "generation"),
        )

    def test_http_rejects_missing_generation_and_integer_overflow(self):
        from sglang.srt.entrypoints.http_server import expanded_input_len

        base = {
            "rid": "7",
            "expanded_input_len": 8,
            "mrope_position_delta": 0,
            "metadata_version": 2,
            "request_generation": "g",
            "bootstrap_pair_key": "pair",
        }
        for change in [
            {"bootstrap_pair_key": None},
            {"request_generation": None},
            {"rid": {}},
            {"mrope_position_delta": 2**80},
            {"metadata_version": 1},
        ]:
            body = dict(base, **change)
            response = asyncio.run(
                expanded_input_len(NS(json=AsyncMock(return_value=body)))
            )
            self.assertEqual(response.status_code, 400)

    def test_prefill_batch_dispatch_flags_are_per_child(self):
        obj = GenerateReqInput(
            bootstrap_pair_key="pair", text=["a", "b"], image_data=[["x"], None]
        )
        obj.normalize_batch_and_arguments()
        tm = TokenizerManager.__new__(TokenizerManager)
        tm._expanded_input_tasks = {}
        with (
            get_context().override_server_args(
                disaggregation_mode="prefill",
                language_only=True,
                encoder_transfer_backend="zmq_to_tokenizer",
            ),
            envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True),
        ):
            tm._handle_epd_disaggregation_encode_request(obj)
        self.assertTrue(obj[0].need_wait_for_mm_inputs)
        self.assertFalse(obj[1].need_wait_for_mm_inputs)

    def test_unconsumed_old_generation_cannot_attach_to_reused_room(self):
        self.req._expanded_input_generation = "current-decode-nonce"
        self.q.set_expanded_input_len("7", 8, 0, "old-decode-nonce")
        self.enqueue()
        self.q._sync_expanded_input_metadata()
        self.assertIsNone(self.req.expanded_input_len)
        self.q.set_expanded_input_len("7", 9, 0, "current-decode-nonce")
        self.q._sync_expanded_input_metadata()
        self.assertEqual(self.req.expanded_input_len, 9)

    def test_owner_nonce_is_scoped_to_live_decode_request(self):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm._expanded_input_tasks = {}
        obj = NS(
            bootstrap_room=7,
            bootstrap_pair_key="pair",
            _needs_expanded_input_len=True,
            _expanded_input_generation="decode-nonce",
        )
        state = NS(obj=obj, finished=False)
        tm.rid_to_state = {"d-rid": state}
        tm._expanded_input_owners = {("7", "pair"): state}
        with get_context().override_server_args(disaggregation_mode="decode"):
            self.assertEqual(
                tm.get_expanded_input_generation("7", "pair"), "decode-nonce"
            )
            state.finished = True
            self.assertIsNone(tm.get_expanded_input_generation("7", "pair"))

    def test_metadata_query_and_stale_post_are_generation_checked(self):
        from sglang.srt.entrypoints import http_server

        manager = NS(
            get_expanded_input_generation=Mock(return_value=None),
            notify_expanded_input_len=Mock(),
        )
        with patch.object(http_server, "_global_state", NS(tokenizer_manager=manager)):
            request = NS(
                json=AsyncMock(
                    return_value={
                        "rid": "7",
                        "query_generation": True,
                        "bootstrap_pair_key": "pair",
                    }
                )
            )
            response = asyncio.run(http_server.expanded_input_len(request))
            self.assertEqual(response.status_code, 202)
            manager.get_expanded_input_generation.return_value = "new-nonce"
            request = NS(
                json=AsyncMock(
                    return_value={
                        "rid": "7",
                        "expanded_input_len": 8,
                        "mrope_position_delta": 0,
                        "metadata_version": 2,
                        "request_generation": "old-nonce",
                        "bootstrap_pair_key": "pair",
                    }
                )
            )
            response = asyncio.run(http_server.expanded_input_len(request))
            self.assertEqual(response.status_code, 409)
            manager.notify_expanded_input_len.assert_not_called()

    def test_request_retirement_cancels_owner_query_task(self):
        async def check():
            tm = TokenizerManager.__new__(TokenizerManager)
            tm._expanded_input_tasks = {}
            tm.asyncio_tasks = set()
            tm.rid_to_state = {}
            pending = asyncio.Event()
            tm._track_expanded_len_task(pending.wait(), "rid")
            tm._cancel_expanded_input_tasks("rid")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertFalse(tm.asyncio_tasks)
            self.assertFalse(tm._expanded_input_tasks)

        asyncio.run(check())

    def test_owner_query_authentication_errors_fail_without_timeout(self):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm._expanded_input_tasks = {}
        cm = AsyncMock()
        cm.__aenter__.return_value = NS(status=401)
        session = NS(post=Mock(return_value=cm))
        tm._get_decode_control_session = AsyncMock(return_value=session)
        with get_context().override_server_args(disagg_decode_urls=["http://d"]):
            with self.assertRaisesRegex(PermissionError, "authentication failed"):
                asyncio.run(tm._notify_decode_expanded_len("7", 8, "p", 0, "pair"))
        self.assertEqual(session.post.call_count, 1)

    def test_pd_session_guard_does_not_change_standalone_encoder_dispatch(self):
        obj = GenerateReqInput(
            bootstrap_pair_key="pair", text="a", image_data=["x"], session_id="session"
        )
        obj.normalize_batch_and_arguments()
        tm = TokenizerManager.__new__(TokenizerManager)
        tm._expanded_input_tasks = {}
        with (
            envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True),
            get_context().override_server_args(
                disaggregation_mode="null",
                language_only=True,
                encoder_transfer_backend="zmq_to_tokenizer",
            ),
        ):
            tm._handle_epd_disaggregation_encode_request(obj)
        self.assertTrue(obj.need_wait_for_mm_inputs)


if __name__ == "__main__":
    import unittest

    unittest.main()
