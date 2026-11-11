"""Admission, interface and failure-boundary regressions for multimodal PD."""

import asyncio
import pickle
import time
import unittest
from array import array
from collections import OrderedDict
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import aiohttp
import fastapi
import msgspec

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.expanded_input import (
    finalize_expanded_input_positions,
    original_pair_key,
    validate_expanded_input_config,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    ExpandedInputLenReleaseReq,
    GenerateReqInput,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.config_dump import capture_engine_command
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.req_double import ReqDouble

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestExpandedInputBoundaries(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def manager(self):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm._expanded_input_tasks = {}
        tm._expanded_input_owners = {}
        tm.rid_to_state = {}
        tm.asyncio_tasks = set()
        tm._dispatch_to_scheduler = Mock()
        return tm

    def owner(self, tm, rid="r", room=7, pair="pair", generation="g"):
        obj = GenerateReqInput(
            text="a",
            image_data=["x"],
            rid=rid,
            bootstrap_room=room,
            bootstrap_pair_key=pair,
        )
        obj._expanded_input_generation = generation
        obj._needs_expanded_input_len = True
        state = NS(obj=obj, finished=False, expanded_input_pair=None)
        tm.rid_to_state[rid] = state
        token = TokenizedGenerateReqInput(
            rid=rid,
            input_text="a",
            input_ids=array("q", [1]),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(max_new_tokens=1),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            bootstrap_room=room,
            bootstrap_pair_key=pair,
            needs_expanded_input_len=True,
            expanded_input_generation=generation,
        )
        return state, token

    def test_owner_index_admission_and_release_protect_pending_deliveries(self):
        tm = self.manager()
        q = DecodePreallocQueue.__new__(DecodePreallocQueue)
        q._pending_expanded_lens = OrderedDict()
        q._seen_expanded_metadata = OrderedDict()
        q._expanded_len_cache_max = 4

        def dispatch(message):
            if isinstance(message, ExpandedInputLenReleaseReq):
                q.release_expanded_input_len(message.rid, message.request_generation)
            else:
                self.assertTrue(
                    q.set_expanded_input_len(
                        message.rid,
                        message.expanded_input_len,
                        message.mrope_position_delta,
                        message.request_generation,
                        message.metadata_version,
                    )
                )

        tm._dispatch_to_scheduler = dispatch
        with (
            patch(
                "sglang.srt.managers.tokenizer_manager.EXPANDED_INPUT_MAX_INFLIGHT", 4
            ),
            get_context().override_server_args(disaggregation_mode="decode"),
        ):
            for i in range(4):
                _, token = self.owner(tm, str(i), i, generation=str(i))
                tm._register_expanded_input_owner(token)
                self.assertEqual(
                    tm.get_expanded_input_generation(str(i), "pair"), str(i)
                )
                tm.notify_expanded_input_len(str(i), 8, 0, str(i))
            _, fifth = self.owner(tm, "fifth", 4, generation="fifth")
            with self.assertRaises(fastapi.HTTPException) as error:
                tm._register_expanded_input_owner(fifth)
            self.assertEqual(error.exception.status_code, 503)
            self.assertIsNone(tm.get_expanded_input_generation("4", "pair"))
            self.assertIn(("0", "0"), q._pending_expanded_lens)
            tm._cancel_expanded_input_tasks("0")
            self.assertNotIn(("0", "0"), q._pending_expanded_lens)
            tm._register_expanded_input_owner(fifth)
            tm.notify_expanded_input_len("4", 8, 0, "fifth")
            self.assertEqual(len(q._pending_expanded_lens), 4)

    def test_duplicate_owner_and_old_retirement_are_safe(self):
        tm = self.manager()
        first, token = self.owner(tm)
        tm._register_expanded_input_owner(token)
        _, duplicate = self.owner(tm, "duplicate")
        with self.assertRaisesRegex(ValueError, "Concurrent reuse"):
            tm._register_expanded_input_owner(duplicate)
        replacement = NS(
            obj=first.obj, finished=False, expanded_input_pair=("7", "pair")
        )
        tm._expanded_input_owners[("7", "pair")] = replacement
        tm._cancel_expanded_input_tasks("r")
        self.assertIs(tm._expanded_input_owners[("7", "pair")], replacement)
        tm._dispatch_to_scheduler.assert_not_called()

    def test_old_handler_failure_does_not_retire_a_reused_rid(self):
        tm = self.manager()
        old, old_token = self.owner(tm)
        tm._register_expanded_input_owner(old_token)
        tm._cancel_expanded_input_tasks("r")
        replacement, token = self.owner(tm, pair="replacement", generation="new")
        tm._register_expanded_input_owner(token)
        tm._dispatch_to_scheduler.reset_mock()
        tm._release_req_states_on_failure(["r"], {"r": old})
        self.assertIs(tm._expanded_input_owners[("7", "replacement")], replacement)
        self.assertIs(tm.rid_to_state["r"], replacement)
        tm._dispatch_to_scheduler.assert_not_called()

    def test_owner_lookup_does_not_scan_request_states(self):
        tm = self.manager()
        state, token = self.owner(tm)
        tm._register_expanded_input_owner(token)

        class NoScan(dict):
            def values(self):
                raise AssertionError("owner lookup scanned requests")

        tm.rid_to_state = NoScan(tm.rid_to_state)
        with get_context().override_server_args(disaggregation_mode="decode"):
            self.assertEqual(tm.get_expanded_input_generation("7", "pair"), "g")
            state.finished = True
            self.assertIsNone(tm.get_expanded_input_generation("7", "pair"))

    def test_release_control_roundtrip(self):
        message = ExpandedInputLenReleaseReq(rid="7", request_generation="g")
        decoded = msgpack_decode(msgpack_encode(message))
        self.assertEqual((decoded.rid, decoded.request_generation), ("7", "g"))

    def test_expiry_never_scans_pending_or_unexpired_tombstones(self):
        q = DecodePreallocQueue.__new__(DecodePreallocQueue)

        class NoScan(OrderedDict):
            def items(self):
                raise AssertionError("full cache scan")

        q._pending_expanded_lens = NoScan({("live", "g"): ((8, 0, 2), 0)})
        q._seen_expanded_metadata = NoScan(
            {("old", "g"): ((8, 0, 2), 0), ("new", "g"): ((8, 0, 2), 100)}
        )
        q._expanded_len_cache_ttl = 10
        q._cleanup_pending_expanded_lens(105)
        self.assertIn(("live", "g"), q._pending_expanded_lens)
        self.assertEqual(list(q._seen_expanded_metadata), [("new", "g")])

    def test_context_parallel_guard_is_decode_and_feature_scoped(self):
        cfg = NS(
            disaggregation_mode="decode",
            tokenizer_worker_num=1,
            pp_size=1,
            attn_cp_size=2,
            language_only=False,
            disagg_decode_urls=["http://d"],
        )
        with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(False):
            validate_expanded_input_config(cfg, "Qwen4ExpForConditionalGeneration")
        with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True):
            with self.assertRaisesRegex(ValueError, "context parallelism"):
                validate_expanded_input_config(cfg, "Qwen4ExpForConditionalGeneration")
            cfg.disaggregation_mode = "prefill"
            validate_expanded_input_config(cfg, "Qwen4ExpForConditionalGeneration")

    def test_position_options_use_expanded_length_and_defaults(self):
        for start, expected in ((9, 9), (-1, 12)):
            req = ReqDouble(
                origin_input_ids=[1, 2],
                expanded_input_len=12,
                is_prefill_only=False,
                _deferred_position_options=(True, start, None, True, 10),
            )
            finalize_expanded_input_positions(req)
            self.assertEqual(req.logprob_start_len, expected)
            self.assertEqual(req.routed_experts_start_len, 10)
        for start, routed in ((13, 0), (9, 13), (9, -1)):
            req._deferred_position_options = (True, start, None, True, routed)
            with self.assertRaises(ValueError):
                finalize_expanded_input_positions(req)

    def test_deferred_decode_rejects_placeholder_prompt_ids(self):
        tm = self.manager()
        tm.disaggregation_mode = DisaggregationMode.DECODE
        obj = GenerateReqInput(text="a", image_data=["x"], return_prompt_token_ids=True)
        obj._needs_expanded_input_len = True
        with self.assertRaisesRegex(ValueError, "actual Prefill prompt token IDs"):
            asyncio.run(tm._tokenize_one_request(obj))

    def test_native_identity_does_not_require_header_overrides(self):
        from sglang.srt.entrypoints import http_server

        seen = []

        async def generate(obj, request):
            seen.append(obj.bootstrap_pair_key)
            yield {"text": "ok"}

        state = NS(tokenizer_manager=NS(generate_request=generate))
        with (
            patch.object(http_server, "_global_state", state),
            patch.object(http_server, "apply_header_overrides") as apply,
            envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.override(False),
        ):
            obj = GenerateReqInput(text="a")
            response = asyncio.run(
                http_server.generate_request(
                    obj, NS(headers={"x-request-id": "original"})
                )
            )
            self.assertEqual(response.status_code, 200)
            apply.assert_not_called()
        self.assertEqual(seen, ["original"])

    def test_responses_identity_ignores_auto_generated_request_id(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

        request = ResponsesRequest(model="dummy", input="a")
        self.assertIsNone(original_pair_key(request))
        self.assertEqual(
            original_pair_key(request, NS(headers={"x-request-id": "original"})),
            "original",
        )
        explicit = ResponsesRequest(model="dummy", input="a", request_id="caller")
        self.assertEqual(original_pair_key(explicit), "caller")

    def test_engine_sync_and_async_forward_original_identity(self):
        engine = Engine.__new__(Engine)
        engine._resolve_routed_dp_rank = lambda rank, alias: rank
        captured = []

        async def generate(obj, request):
            captured.append(obj)
            yield {"text": "ok"}

        engine.tokenizer_manager = NS(generate_request=generate)
        loop = asyncio.new_event_loop()
        engine.loop = loop
        try:
            engine.generate(prompt="a", rid="sync")
            asyncio.run(engine.async_generate(prompt="a", rid="async"))
            engine.generate(prompt="a", rid="other", bootstrap_pair_key="explicit")
            self.assertEqual(
                [obj.bootstrap_pair_key for obj in captured],
                ["sync", "async", "explicit"],
            )
        finally:
            loop.close()

    def test_request_and_crash_dump_config_is_sanitized_but_ipc_is_raw(self):
        tm = self.manager()
        args = ServerArgs(
            model_path="dummy",
            api_key="p-secret",
            admin_api_key="admin-secret",
            disagg_decode_control_key="d-secret",
        )
        msgspec.Struct.__setattr__(
            args,
            "_launch_command",
            capture_engine_command(
                {"disagg_decode_control_key": "d-secret", "api_key": "p-secret"}
            ),
        )
        tm.server_args = args
        tm.model_path = "dummy"
        tm.served_model_name = "dummy"
        publish(args, role="tokenizer")
        with get_context().override_server_args(
            disagg_decode_control_key="rotated-secret"
        ):
            snapshot = tm._dump_configuration()
            self.assertEqual(
                snapshot["resolved_config"]["disagg_decode_control_key"], "<redacted>"
            )
            dumped = pickle.dumps(snapshot)
        for secret in (b"p-secret", b"admin-secret", b"d-secret", b"rotated-secret"):
            self.assertNotIn(secret, dumped)
        restored = pickle.loads(pickle.dumps(args))
        self.assertEqual(restored.disagg_decode_control_key, "d-secret")

    def test_unserializable_config_does_not_prevent_a_sanitized_dump(self):
        tm = self.manager()
        tm.server_args = NS(
            api_key=None,
            admin_api_key=None,
            disagg_decode_control_key="secret",
            launch_command=capture_engine_command(
                {"disagg_decode_control_key": "secret"}
            ),
            resolved_dict=Mock(side_effect=RuntimeError("cannot snapshot")),
        )
        snapshot = tm._dump_configuration()
        self.assertIsNone(snapshot["server_args"])
        self.assertIsNone(snapshot["resolved_config"])
        self.assertNotIn(b"secret", pickle.dumps(snapshot))

    def test_mixed_batch_wait_flag_is_media_specific_with_feature_off(self):
        obj = GenerateReqInput(text=["image", "text"], image_data=[["x"], None])
        obj.normalize_batch_and_arguments()
        obj.need_wait_for_mm_inputs = True
        with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(False):
            self.assertTrue(obj[0].need_wait_for_mm_inputs)
            self.assertFalse(obj[1].need_wait_for_mm_inputs)

    def test_transient_owner_and_notification_errors_retry_until_recovery(self):
        async def run(query_failures, notify_failures):
            tm = self.manager()
            counts = {"query": 0, "notify": 0}

            def post(url, **kw):
                key = "query" if kw["json"].get("query_generation") else "notify"
                counts[key] += 1
                failures = query_failures if key == "query" else notify_failures
                status = 503 if counts[key] <= failures else 200
                cm = AsyncMock()
                cm.__aenter__.return_value = NS(
                    status=status,
                    json=AsyncMock(return_value={"request_generation": "g"}),
                )
                return cm

            tm._get_decode_control_session = AsyncMock(return_value=NS(post=post))
            with (
                get_context().override_server_args(disagg_decode_urls=["http://d"]),
                envs.SGLANG_DISAGG_EXPANDED_LEN_WAIT_TIMEOUT.override(2),
            ):
                await tm._notify_decode_expanded_len("7", 8, "p", 0, "pair")
            self.assertGreater(counts["query"], query_failures)
            self.assertGreater(counts["notify"], notify_failures)

        asyncio.run(run(2, 0))
        asyncio.run(run(0, 4))

    def test_connection_failure_retries_but_is_bounded_by_deadline(self):
        async def run():
            tm = self.manager()
            counts = []

            def post(url, **kw):
                counts.append(url)
                cm = AsyncMock()
                cm.__aenter__.side_effect = aiohttp.ClientConnectionError("temporary")
                return cm

            tm._get_decode_control_session = AsyncMock(return_value=NS(post=post))
            started = time.monotonic()
            with (
                get_context().override_server_args(disagg_decode_urls=["http://d"]),
                envs.SGLANG_DISAGG_EXPANDED_LEN_WAIT_TIMEOUT.override(0.15),
            ):
                with self.assertRaisesRegex(RuntimeError, "No live Decode owner"):
                    await tm._notify_decode_expanded_len("7", 8, "p", 0, "pair")
            self.assertGreater(len(counts), 1)
            self.assertLess(time.monotonic() - started, 0.6)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
