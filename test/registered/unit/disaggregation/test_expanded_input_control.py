"""Compatibility and request-lifecycle regressions for expanded-input control."""

import asyncio
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import msgspec

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.disaggregation.expanded_input import validate_expanded_input_config
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.config_dump import capture_engine_command
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestExpandedInputControl(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def manager(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager._expanded_input_tasks = {}
        manager._expanded_input_owners = {}
        manager.asyncio_tasks = set()
        manager.rid_to_state = {}
        return manager

    def test_control_key_is_redacted_without_changing_authentication(self):
        args = ServerArgs(model_path="dummy", disagg_decode_control_key="decode-secret")
        msgspec.Struct.__setattr__(
            args,
            "_launch_command",
            capture_engine_command({"disagg_decode_control_key": "decode-secret"}),
        )
        self.assertEqual(args.disagg_decode_control_key, "decode-secret")
        self.assertEqual(
            args.resolved_dict()["disagg_decode_control_key"], "<redacted>"
        )
        self.assertNotIn("decode-secret", args.launch_command)
        publish(args, role="tokenizer")
        with get_context().override_server_args(
            disagg_decode_control_key="rotated-secret"
        ):
            self.assertEqual(
                get_context().resolved_server_args_dict()["disagg_decode_control_key"],
                "<redacted>",
            )

    def test_engine_launch_readback_redacts_escaped_control_key(self):
        key = "decode\nsecret\\tail"
        args = ServerArgs(model_path="dummy", disagg_decode_control_key=key)
        msgspec.Struct.__setattr__(
            args,
            "_launch_command",
            capture_engine_command({"disagg_decode_control_key": key}),
        )
        self.assertIn("<redacted>", args.launch_command)
        self.assertNotIn(repr(key), args.launch_command)

    def test_rust_positional_tail_is_unchanged(self):
        fields = list(TokenizedGenerateReqInput.__struct_fields__)
        self.assertEqual(fields[fields.index("decode_tp_size") + 1], "routed_dp_rank")
        self.assertEqual(
            fields[-2:], ["needs_expanded_input_len", "expanded_input_generation"]
        )
        # Match the Rust sender's short array, including optional integer DP ranks.
        values = [TokenizedGenerateReqInput.__struct_config__.tag] + [None] * (
            fields.index("disagg_prefill_dp_rank") + 1
        )
        concrete = {
            "rid": "rust",
            "input_ids": None,
            "sampling_params": [],
            "return_logprob": False,
            "logprob_start_len": -1,
            "top_logprobs_num": 0,
            "stream": False,
            "return_sampling_mask": False,
            "return_flat_raw_top_logprobs": False,
            "return_hidden_states": False,
            "return_routed_experts": False,
            "routed_experts_start_len": 0,
            "return_indexer_topk": False,
            "routed_dp_rank": 2,
            "disagg_prefill_dp_rank": 1,
        }
        for name, value in concrete.items():
            values[fields.index(name) + 1] = value
        from sglang.srt.managers.io_struct import msgpack_decode

        decoded = msgpack_decode(msgspec.msgpack.encode(values))
        self.assertEqual(
            (decoded.routed_dp_rank, decoded.disagg_prefill_dp_rank), (2, 1)
        )
        self.assertFalse(decoded.needs_expanded_input_len)
        self.assertIsNone(decoded.expanded_input_generation)

    def test_original_pair_identity_uses_forwarded_header_without_local_ids(self):
        from sglang.srt.disaggregation.expanded_input import original_pair_key

        request = NS(bootstrap_pair_key=None, rid=None)
        raw = NS(headers={"x-request-id": "original-dispatch"})
        self.assertEqual(original_pair_key(request, raw), "original-dispatch")
        self.assertIsNone(original_pair_key(request))
        request.rid = "explicit-rid"
        self.assertEqual(original_pair_key(request, raw), "explicit-rid")
        request.bootstrap_pair_key = "explicit-pair"
        self.assertEqual(original_pair_key(request, raw), "explicit-pair")

    def test_old_pair_cannot_adopt_replacement_owner_nonce(self):
        tm = self.manager()
        obj = NS(
            bootstrap_room=7,
            bootstrap_pair_key="new-pair",
            _needs_expanded_input_len=True,
            _expanded_input_generation="new-nonce",
        )
        tm.rid_to_state["new"] = NS(obj=obj, finished=False)
        tm._expanded_input_owners[("7", "new-pair")] = tm.rid_to_state["new"]
        with get_context().override_server_args(disaggregation_mode="decode"):
            self.assertIsNone(tm.get_expanded_input_generation("7", "old-pair"))
            self.assertEqual(
                tm.get_expanded_input_generation("7", "new-pair"), "new-nonce"
            )

    def test_decode_multiworker_guard_is_feature_scoped(self):
        args = ServerArgs(
            model_path="dummy", disaggregation_mode="decode", tokenizer_worker_num=2
        )
        with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(False):
            validate_expanded_input_config(
                resolving_view(args), "Qwen4ExpForConditionalGeneration"
            )
        with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True):
            with self.assertRaisesRegex(ValueError, "tokenizer-worker-num"):
                validate_expanded_input_config(
                    resolving_view(args), "Qwen4ExpForConditionalGeneration"
                )

    def test_parallel_multimodal_and_missing_identity_rejected_before_dispatch(self):
        tm = self.manager()
        with (
            get_context().override_server_args(disaggregation_mode="decode"),
            envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True),
        ):
            parallel = GenerateReqInput(
                text="a", image_data=["x"], sampling_params={"n": 2}
            )
            parallel.normalize_batch_and_arguments()
            with self.assertRaisesRegex(ValueError, "parallel sampling"):
                tm._handle_epd_disaggregation_encode_request(parallel)
            single = GenerateReqInput(text="a", image_data=["x"])
            single.normalize_batch_and_arguments()
            with self.assertRaisesRegex(ValueError, "bootstrap_pair_key"):
                tm._handle_epd_disaggregation_encode_request(single)
            text = GenerateReqInput(text="a", sampling_params={"n": 2})
            text.normalize_batch_and_arguments()
            tm._handle_epd_disaggregation_encode_request(text)

    def test_multimodal_embedding_does_not_read_generation_session_fields(self):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.io_struct import EmbeddingReqInput

        tm = self.manager()
        tm.disaggregation_mode = DisaggregationMode.DECODE
        tm.mm_processor = None
        tm.tokenizer = Mock()
        tm._tokenize_texts = AsyncMock(
            side_effect=ValueError("reached text tokenization")
        )
        for enabled in (False, True):
            with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(enabled):
                obj = EmbeddingReqInput(text="a", image_data=["x"])
                with self.assertRaisesRegex(ValueError, "reached text tokenization"):
                    asyncio.run(tm._tokenize_one_request(obj))

    def test_healthy_owner_is_not_delayed_by_slow_replica(self):
        async def check():
            tm = self.manager()
            slow_cancelled = asyncio.Event()
            delivered = asyncio.Event()

            class Response:
                status = 200

                async def json(self):
                    return {"request_generation": "nonce"}

            class Request:
                def __init__(self, url, body):
                    self.url, self.body = url, body

                async def __aenter__(self):
                    if "slow" in self.url:
                        try:
                            await asyncio.sleep(60)
                        except asyncio.CancelledError:
                            slow_cancelled.set()
                            raise
                    if not self.body.get("query_generation"):
                        delivered.set()
                    return Response()

                async def __aexit__(self, *args):
                    return False

            session = NS(post=lambda url, **kw: Request(url, kw["json"]))
            tm._get_decode_control_session = AsyncMock(return_value=session)
            with get_context().override_server_args(
                disagg_decode_urls=["http://slow", "http://healthy"]
            ):
                await asyncio.wait_for(
                    tm._notify_decode_expanded_len("7", 8, "p", 0, "pair"), 1
                )
            self.assertTrue(delivered.is_set())
            self.assertTrue(slow_cancelled.is_set())

        asyncio.run(check())

    def test_terminal_notification_failure_aborts_dispatched_request(self):
        async def check(dispatched):
            tm = self.manager()
            state = NS(finished=False, dispatched=dispatched, expanded_input_error=None)
            tm.rid_to_state["p"] = state
            tm.abort_request = Mock()

            async def fail():
                raise PermissionError("control authentication failed")

            tm._track_expanded_len_task(fail(), "p")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertIn("authentication", state.expanded_input_error)
            if not dispatched:
                tm._mark_state_dispatched("p")
            tm.abort_request.assert_called_once_with(
                "p",
                exact=True,
                finished_reason={
                    "type": "abort",
                    "message": state.expanded_input_error,
                    "status_code": 502,
                },
            )
            self.assertFalse(tm._expanded_input_tasks)
            self.assertFalse(tm.asyncio_tasks)

        asyncio.run(check(True))
        asyncio.run(check(False))

    def test_control_failure_status_does_not_depend_on_error_wording(self):
        """Editing a public error message must not change its HTTP status."""

        async def check(error, expected):
            tm = self.manager()
            state = NS(finished=False, dispatched=False, expanded_input_error=None)
            tm.rid_to_state["p"] = state
            reasons = []
            tm.abort_request = lambda rid, **kwargs: reasons.append(
                kwargs["finished_reason"]
            )

            async def fail():
                raise error

            tm._track_expanded_len_task(fail(), "p")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            state.expanded_input_error = "rewritten detail without classification words"
            tm._mark_state_dispatched("p")
            self.assertEqual(reasons[0]["status_code"], expected)
            self.assertEqual(reasons[0]["message"], state.expanded_input_error)

        for error, status in [
            (TimeoutError("deadline"), 504),
            (RuntimeError("no owner"), 504),
            (PermissionError("denied"), 502),
            (ValueError("invalid"), 502),
        ]:
            asyncio.run(check(error, status))

    def test_old_failure_callback_cannot_abort_replacement_request(self):
        async def check():
            tm = self.manager()
            old = NS(finished=False, dispatched=True, expanded_input_error=None)
            replacement = NS(finished=False, dispatched=True, expanded_input_error=None)
            tm.rid_to_state["reused"] = old
            tm.abort_request = Mock()

            async def fail():
                raise RuntimeError("old notification failed")

            tm._track_expanded_len_task(fail(), "reused")
            tm.rid_to_state["reused"] = replacement
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            tm.abort_request.assert_not_called()
            self.assertIsNone(replacement.expanded_input_error)

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
