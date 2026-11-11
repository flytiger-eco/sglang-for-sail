"""Control-plane failure isolation, authentication and shutdown regressions."""

import asyncio
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import aiohttp
import fastapi

from sglang.srt.disaggregation.expanded_input import validate_expanded_input_config
from sglang.srt.disaggregation.utils import DisaggregationMode, prepare_abort
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    AbortReq,
    GenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.auth import AuthLevel, decide_request_auth
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestExpandedInputReview(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def manager(self):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm._decode_control_session = None
        tm._expanded_input_tasks = {}
        tm.asyncio_tasks = set()
        tm.rid_to_state = {}
        tm._dispatch_to_scheduler = Mock()
        tm.enable_metrics = False
        return tm

    def test_control_route_preserves_optional_admin_authentication(self):
        from sglang.srt.entrypoints.http_server import app

        route = next(
            r
            for r in app.routes
            if getattr(r, "path", None) == "/v1/expanded_input_len"
        )
        self.assertEqual(route.endpoint._auth_level, AuthLevel.ADMIN_OPTIONAL)
        for api, admin, header, allowed in (
            (None, None, None, True),
            ("inference", None, None, False),
            ("inference", None, "Bearer inference", True),
            ("inference", "admin", "Bearer inference", False),
            ("inference", "admin", "Bearer admin", True),
        ):
            result = decide_request_auth(
                method="POST",
                path=route.path,
                authorization_header=header,
                api_key=api,
                admin_api_key=admin,
                auth_level=route.endpoint._auth_level,
            )
            self.assertEqual(result.allowed, allowed)

    def test_control_http_query_works_without_admin_configuration(self):
        import httpx

        from sglang.srt.entrypoints import http_server
        from sglang.srt.utils.auth import add_api_key_middleware

        async def run():
            manager = NS(get_expanded_input_generation=lambda room, pair: None)
            with patch.object(
                http_server, "_global_state", NS(tokenizer_manager=manager)
            ):
                for api in (None, "inference"):
                    app = fastapi.FastAPI()
                    app.post("/v1/expanded_input_len")(http_server.expanded_input_len)
                    add_api_key_middleware(app, api_key=api, admin_api_key=None)
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app), base_url="http://test"
                    ) as client:
                        response = await client.post(
                            "/v1/expanded_input_len",
                            json={
                                "rid": "7",
                                "query_generation": True,
                                "bootstrap_pair_key": "pair",
                            },
                            headers={"Authorization": "Bearer inference"}
                            if api
                            else {},
                        )
                        self.assertEqual(response.status_code, 202, response.text)

        asyncio.run(run())

    def test_frontend_guards_allow_pd_without_control_credentials(self):
        cfg = NS(
            disaggregation_mode="decode",
            tokenizer_worker_num=1,
            attn_cp_size=1,
            pp_size=1,
            language_only=False,
            admin_api_key="admin",
            smg_grpc_mode=False,
        )
        with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True):
            with envs.SGLANG_RUST_SERVER.override(True):
                with self.assertRaisesRegex(ValueError, "Python HTTP"):
                    validate_expanded_input_config(
                        cfg, "Qwen4ExpForConditionalGeneration"
                    )
            cfg.smg_grpc_mode = True
            with self.assertRaisesRegex(ValueError, "Python HTTP"):
                validate_expanded_input_config(cfg, "Qwen4ExpForConditionalGeneration")
            cfg.smg_grpc_mode = False
            cfg.admin_api_key = None
            validate_expanded_input_config(cfg, "Qwen4ExpForConditionalGeneration")
            cfg.disaggregation_mode = "prefill"
            cfg.disagg_decode_urls = ["http://decode"]
            cfg.disagg_decode_control_key = None
            validate_expanded_input_config(cfg, "Qwen4ExpForConditionalGeneration")
            cfg.disagg_decode_control_key = "admin"
            validate_expanded_input_config(cfg, "Qwen4ExpForConditionalGeneration")

    def test_expanded_input_only_accepts_registration_aware_transfer_backend(self):
        """Accepted backends must support cancellation before P admits a room."""
        for role in ("prefill", "decode"):
            for backend in ("mooncake", "nixl", "mori", "fake"):
                cfg = NS(
                    disaggregation_mode=role,
                    disaggregation_transfer_backend=backend,
                    tokenizer_worker_num=1,
                    attn_cp_size=1,
                    pp_size=1,
                    language_only=False,
                    disagg_decode_urls=["http://decode"],
                    smg_grpc_mode=False,
                )
                with self.subTest(role=role, backend=backend):
                    with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(False):
                        validate_expanded_input_config(
                            cfg, "Qwen4ExpForConditionalGeneration"
                        )
                    with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True):
                        if backend == "mooncake":
                            validate_expanded_input_config(
                                cfg, "Qwen4ExpForConditionalGeneration"
                            )
                        else:
                            with self.assertRaisesRegex(
                                ValueError, "mooncake transfer backend"
                            ):
                                validate_expanded_input_config(
                                    cfg, "Qwen4ExpForConditionalGeneration"
                                )

    def test_pd_role_switch_is_rejected_only_with_expanded_input(self):
        cfg = NS(
            disaggregation_mode="decode",
            tokenizer_worker_num=1,
            attn_cp_size=1,
            pp_size=1,
            language_only=False,
            disagg_decode_urls=[],
            enable_pd_role_switch=True,
        )
        with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(False):
            validate_expanded_input_config(cfg, "Qwen4ExpForConditionalGeneration")
        with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True):
            with self.assertRaisesRegex(ValueError, "enable-pd-role-switch"):
                validate_expanded_input_config(cfg, "Qwen4ExpForConditionalGeneration")

    def test_expanded_multimodal_rejects_untransferred_expert_captures(self):
        """Prompt expert rows cannot be read from unpopulated Decode slots."""
        tm = self.manager()
        for role in ("prefill", "decode"):
            with (
                get_context().override_server_args(disaggregation_mode=role),
                envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True),
            ):
                obj = GenerateReqInput(
                    text="image",
                    image_data="dummy-image",
                    bootstrap_pair_key="paired",
                    return_routed_experts=True,
                )
                obj.normalize_batch_and_arguments()
                with self.assertRaisesRegex(ValueError, "return_routed_experts"):
                    tm._handle_epd_disaggregation_encode_request(obj)
                text = GenerateReqInput(text="text only", return_routed_experts=True)
                text.normalize_batch_and_arguments()
                tm._handle_epd_disaggregation_encode_request(text)

    def test_fake_host_prefill_skips_mooncake_pairing_hook(self):
        from sglang.srt.disaggregation.fake.conn import FakeKVSender
        from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(enabled),
            ):
                queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
                queue._check_if_req_exceed_kv_capacity = Mock(return_value=False)
                queue.transfer_backend = "mooncake"
                queue.tp_rank = queue.pp_rank = 0
                queue.bootstrap_port = 8998
                queue.kv_manager = NS()
                queue._process_req = Mock()
                req = NS(
                    rid="explicit-rid",
                    bootstrap_host=FAKE_BOOTSTRAP_HOST,
                    bootstrap_room=83,
                    disagg_prefill_dp_rank=None,
                    _expanded_input_pair_key="explicit-rid",
                )
                self.assertTrue(queue.create_sender(req, 1))
                self.assertIsInstance(req.disagg_kv_sender, FakeKVSender)
                self.assertTrue(req.pending_bootstrap)
                queue._process_req.assert_called_once_with(req)

    def test_cli_redaction_accepts_tuple_and_list_argparse_results(self):
        import argparse

        from sglang.srt.utils.config_dump import SECRET_FIELDS, redact_cli_secrets

        for use_list in (False, True):
            with self.subTest(list_result=use_list):
                parser = argparse.ArgumentParser()
                parser.add_argument("--model-path")
                argv = ["--model-path", "unchanged"]
                values = []
                for field in sorted(SECRET_FIELDS):
                    option = "--" + field.replace("_", "-")
                    parser.add_argument(option)
                    previous, current = field + "-old", field + "-current"
                    argv.extend([option, previous, option + "=" + current])
                    values.extend([previous, current])
                argv.extend(["--unknown-option", "ordinary-value"])
                original = list(argv)
                native = parser._parse_optional

                def parse(argument):
                    option = native(argument)
                    if use_list and option is not None and not isinstance(option, list):
                        return [option]
                    return option

                with patch.object(parser, "_parse_optional", side_effect=parse):
                    redacted = redact_cli_secrets(argv, parser)
                for value in values:
                    self.assertNotIn(value, " ".join(redacted))
                self.assertEqual(redacted[:2], ["--model-path", "unchanged"])
                self.assertEqual(redacted[-2:], ["--unknown-option", "ordinary-value"])
                self.assertEqual(argv, original)

    def test_cancel_e2e_supplies_only_configured_control_credentials(self):
        """Run the actual E2E method against requests that enforce control auth."""
        import os
        from urllib.parse import urlsplit

        module = self._load_epd_fixture()
        method = "test_decode_cancel_before_prefill_registration_unblocks_both_peers"
        for key in (None, "test-admin"):
            with self.subTest(admin_configured=key is not None):
                case = module.TestQwen4ExpandedInputEPD(method)
                case.admin_key = key
                case.decode_url = "http://decode"
                case._paired_native_request = Mock(
                    return_value={"bootstrap_pair_key": "pair", "bootstrap_room": 83}
                )
                seen = []

                def post(url, **kwargs):
                    path = urlsplit(url).path
                    if path in ("/v1/expanded_input_len", "/abort_request"):
                        headers = kwargs.get("headers", {})
                        seen.append((path, headers))
                        expected = {"Authorization": "Bearer " + key} if key else {}
                        if headers != expected:
                            return NS(
                                status_code=401,
                                text="missing configured control authorization",
                            )
                        return NS(status_code=200, text="ok")
                    if url.startswith("http://decode"):
                        return NS(
                            status_code=200,
                            text="cancelled",
                            json=lambda: {
                                "meta_info": {"finish_reason": {"type": "abort"}},
                                "output_ids": [],
                            },
                        )
                    return NS(status_code=500, text="prefill aborted")

                with (
                    patch.dict(os.environ, EPD_TEST_PREFILL_URL="http://prefill"),
                    patch.object(module.requests, "post", side_effect=post),
                    patch.object(
                        module.requests, "get", return_value=NS(status_code=200)
                    ),
                    patch("time.sleep"),
                ):
                    getattr(case, method)()
                self.assertEqual(
                    [path for path, _ in seen],
                    ["/v1/expanded_input_len", "/abort_request"],
                )

    def test_http_abort_preserves_exact_and_legacy_prefix_matching(self):
        import httpx

        from sglang.srt.entrypoints import http_server

        async def run():
            active = {"a", "ab"}

            def abort(rid, abort_all=False, exact=False):
                request = AbortReq(rid=rid, abort_all=abort_all, exact=exact)
                active.difference_update(
                    [value for value in active if request.matches(value)]
                )

            app = fastapi.FastAPI()
            app.post("/abort_request")(http_server.abort_request)
            with patch.object(
                http_server,
                "_global_state",
                NS(tokenizer_manager=NS(abort_request=abort)),
            ):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/abort_request", json={"rid": "a", "exact": True}
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(active, {"ab"})
                    active.add("a")
                    response = await client.post("/abort_request", json={"rid": "a"})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(active, set())
                    active.update({"a", "b"})
                    response = await client.post(
                        "/abort_request",
                        json={"rid": "a", "exact": True, "abort_all": True},
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(active, set())

        asyncio.run(run())

    def test_cli_capture_redacts_overridden_credential_values(self):
        import pickle

        from sglang.srt.server_args import prepare_server_args
        from sglang.srt.utils.config_dump import SECRET_FIELDS, redact_config_for_dump

        argv = ["--model-path", "dummy"]
        secrets = []
        for field in sorted(SECRET_FIELDS):
            option = "--" + field.replace("_", "-")
            previous, current = field + "-previous-'\\quoted", field + "-current"
            argv.extend([option, previous, option + "=" + current])
            secrets.extend([previous, current])
        args = ServerArgs(
            model_path="dummy", **{field: field + "-current" for field in SECRET_FIELDS}
        )
        # Keep real parsing/capture; isolate model/platform initialization.
        with patch.object(ServerArgs, "from_cli_args", return_value=args):
            parsed = prepare_server_args(argv)
        self.assertIs(parsed, args)
        for value in secrets:
            self.assertNotIn(value, args._launch_command)
            self.assertNotIn(value, args.launch_command)
            self.assertNotIn(
                value,
                str(
                    redact_config_for_dump(
                        {"launch_command": args.launch_command}, args, {}
                    )
                ),
            )
        self.assertIn("--model-path dummy", args.launch_command)
        raw = pickle.loads(pickle.dumps(args))
        for field in SECRET_FIELDS:
            self.assertEqual(getattr(raw, field), field + "-current")

    def test_cli_capture_redacts_abbreviated_secret_options_and_keeps_argv(self):
        import argparse

        from sglang.srt.utils.config_dump import redact_cli_secrets

        parser = argparse.ArgumentParser()
        parser.add_argument("--admin-api-key")
        parser.add_argument("--model-path")
        argv = [
            "--admin-api-k=first-value",
            "--admin-api-k",
            "second value",
            "--model-path",
            "unchanged",
        ]
        original = list(argv)
        self.assertEqual(
            redact_cli_secrets(argv, parser),
            [
                "--admin-api-k=<redacted>",
                "--admin-api-k",
                "<redacted>",
                "--model-path",
                "unchanged",
            ],
        )
        self.assertEqual(argv, original)

    def test_tls_password_is_redacted_without_mutating_runtime_or_ipc(self):
        """TLS credentials must follow API/control-key readback protections."""
        import pickle

        from sglang.srt.utils.config_dump import (
            capture_engine_command,
            redact_config_for_dump,
        )

        password = "tls-sentinel-password"
        args = ServerArgs(model_path="dummy", ssl_keyfile_password=password)
        import msgspec

        msgspec.Struct.__setattr__(
            args,
            "_launch_command",
            capture_engine_command({"ssl_keyfile_password": password}),
        )
        publish(args, role="tokenizer")
        self.assertEqual(args.resolved_dict()["ssl_keyfile_password"], "<redacted>")
        self.assertEqual(
            get_context().resolved_server_args_dict()["ssl_keyfile_password"],
            "<redacted>",
        )
        self.assertNotIn(password, args.launch_command)
        snapshot = redact_config_for_dump(
            {
                "ssl_keyfile_password": password,
                "command": "--ssl-keyfile-password " + password,
            },
            args,
            {},
        )
        self.assertNotIn(password, str(snapshot))
        self.assertEqual(
            pickle.loads(pickle.dumps(args)).ssl_keyfile_password, password
        )

    def test_every_secret_is_scrubbed_from_readbacks_but_ipc_stays_raw(self):
        import pickle

        import msgspec

        from sglang.srt.utils.config_dump import SECRET_FIELDS, capture_engine_command

        values = {name: name + "-sentinel\nquoted'\\tail" for name in SECRET_FIELDS}
        args = ServerArgs(model_path="dummy", **values)
        msgspec.Struct.__setattr__(
            args,
            "_launch_command",
            capture_engine_command(values),
        )
        publish(args, role="tokenizer")
        for name, key in values.items():
            self.assertEqual(args.resolved_dict()[name], "<redacted>")
            self.assertNotIn(key, args.launch_command)
            self.assertNotIn(repr(key), args.launch_command)
        rotated = {name: "rotated-" + name for name in SECRET_FIELDS}
        with get_context().override_server_args(**rotated):
            snapshot = get_context().resolved_server_args_dict()
            for name in SECRET_FIELDS:
                self.assertEqual(snapshot[name], "<redacted>")
        # A short credential must not corrupt unrelated config identifiers.
        short = ServerArgs(model_path="dummy", api_key="dummy")
        self.assertEqual(short.resolved_dict()["model_path"], "dummy")
        raw = pickle.loads(pickle.dumps(args))
        for name, key in values.items():
            self.assertEqual(getattr(raw, name), key)

    def _load_epd_fixture(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).parents[2] / "disaggregation/test_expanded_input_epd.py"
        spec = importlib.util.spec_from_file_location("epd_fixture_for_unit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_fixture_identity_survives_a_router_that_drops_headers(self):
        from sglang.srt.disaggregation.expanded_input import original_pair_key
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        fixture = self._load_epd_fixture()
        pairs = []

        def post(url, json, **kwargs):
            # Real MiniLB forwards JSON without forwarding request headers.
            forwarded = ChatCompletionRequest(**json)
            pairs.append(original_pair_key(forwarded))
            return NS(
                raise_for_status=lambda: None,
                json=lambda: {
                    "choices": [{"message": {"content": "red"}}],
                    "usage": {"prompt_tokens": 127},
                },
            )

        with patch.object(fixture.requests, "post", side_effect=post):
            for _ in range(2):
                text, tokens = fixture.answer(
                    "http://fixture", fixture.payload((224, 448))
                )
                self.assertEqual((text, tokens), ("red", 127))
        self.assertTrue(all(isinstance(pair, str) and pair for pair in pairs))
        self.assertNotEqual(pairs[0], pairs[1])
        self.assertIsNone(
            original_pair_key(ChatCompletionRequest(**fixture.payload((224, 448))))
        )

    def test_fixture_decode_bootstrap_fallback_targets_its_prefill(self):
        import os

        from sglang.test.server_fixtures import disaggregation_fixture

        fixture = self._load_epd_fixture()
        cls = fixture.TestExpandedInputEPD
        launches = []

        def launch(model, **kwargs):
            launches.append(kwargs["other_args"])
            return NS(pid=-1)

        with (
            patch.dict(os.environ, {"EPD_TEST_URL": ""}),
            patch("sglang.test.test_utils._wait_for_gpu_idle_in_ci"),
            patch.object(disaggregation_fixture, "is_in_ci", return_value=False),
            patch.object(
                disaggregation_fixture, "_get_available_ib_devices", return_value=None
            ),
            patch.object(fixture, "popen_launch_server", side_effect=launch),
            patch.object(fixture, "terminate_and_kill_process_tree"),
            patch.object(
                cls,
                "launch_lb",
                classmethod(lambda cls: setattr(cls, "process_lb", NS(pid=-1))),
            ),
        ):
            try:
                cls.setUpClass()
                decode = next(args for args in launches if "decode" in args)
                prefill = next(args for args in launches if "prefill" in args)
                option = "--disaggregation-bootstrap-port"
                decode_port = (
                    int(decode[decode.index(option) + 1])
                    if option in decode
                    else ServerArgs(model_path="dummy").disaggregation_bootstrap_port
                )
                prefill_port = int(prefill[prefill.index(option) + 1])
                # MiniLB sends null bootstrap_port; Decode uses its configured fallback.
                self.assertEqual(decode_port, prefill_port)
                self.assertNotEqual(prefill_port, 8998)
            finally:
                cls.doClassCleanups()

    def test_internal_abort_does_not_match_another_rid_prefix(self):
        tm = self.manager()
        tm.rid_to_state["a"] = NS(abort_sent=False)
        tm.abort_request(
            "a",
            exact=True,
            finished_reason={
                "type": "abort",
                "message": "metadata failure",
                "status_code": 502,
            },
        )
        req = msgpack_decode(
            msgpack_encode(tm._dispatch_to_scheduler.call_args.args[0])
        )
        self.assertTrue(req.matches("a"))
        self.assertFalse(req.matches("ab"))
        # Older Rust/Python senders omit the appended exact flag.
        import msgspec

        legacy = msgspec.msgpack.decode(msgpack_encode(AbortReq(rid="a")))[:-1]
        legacy_req = msgpack_decode(msgspec.msgpack.encode(legacy))
        self.assertFalse(legacy_req.exact)
        self.assertTrue(legacy_req.matches("ab"))
        self.assertTrue(AbortReq(rid="a").matches("ab"))
        self.assertTrue(AbortReq(abort_all=True, exact=True).matches("ab"))

    def test_exact_scheduler_abort_isolates_running_and_transfer_requests(self):
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.chunked_req = None
            scheduler.mm_receiver = None
            scheduler.waiting_queue = []
            scheduler.dllm_config = None
            scheduler.grammar_manager = NS(abort_requests=Mock())
            scheduler.disaggregation_mode = mode
            a, ab = [
                NS(rid=rid, finished=lambda: False, to_finish=None)
                for rid in ("a", "ab")
            ]
            scheduler.collect_inflight_reqs = lambda: [a, ab]
            scheduler._release_aborted_request = Mock()
            if mode == DisaggregationMode.PREFILL:
                for req in (a, ab):
                    req.disagg_kv_sender = NS(abort=Mock())
                scheduler.disagg_prefill_bootstrap_queue = NS(queue=[])
                scheduler.disagg_prefill_inflight_queue = [a, ab]
            else:
                items = [
                    NS(req=req, kv_receiver=NS(abort=Mock()), host_staged=False)
                    for req in (a, ab)
                ]
                scheduler.disagg_decode_prealloc_queue = NS(
                    queue=items, retracted_queue=[]
                )
                scheduler.disagg_decode_transfer_queue = NS(queue=items)
            reason = {
                "type": "abort",
                "message": "metadata authentication failed",
                "status_code": 502,
            }
            scheduler.abort_request(
                AbortReq(rid="a", exact=True, finished_reason=reason)
            )
            self.assertEqual(a.to_finish.to_json()["status_code"], 502)
            self.assertIsNone(ab.to_finish)
            if mode == DisaggregationMode.PREFILL:
                a.disagg_kv_sender.abort.assert_called_once()
                ab.disagg_kv_sender.abort.assert_not_called()
            else:
                items[0].kv_receiver.abort.assert_called()
                items[1].kv_receiver.abort.assert_not_called()
            a.return_logprob = False
            prepare_abort(a, "generic transfer failure", 500)
            self.assertEqual(a.finished_reason.to_json()["message"], reason["message"])

    def test_504_and_502_propagate_as_nonstream_http_errors(self):
        async def run():
            for status in (502, 503, 504):
                tm = self.manager()
                tm.enable_lora = False
                state = NS(obj=NS(rid="a"))
                tm.rid_to_state["a"] = state
                out = {
                    "meta_info": {
                        "finish_reason": {
                            "type": "abort",
                            "status_code": status,
                            "message": "metadata failure",
                        }
                    }
                }
                with self.assertRaises(fastapi.HTTPException) as raised:
                    await tm._handle_abort_finish_reason(out, state, False)
                self.assertEqual(raised.exception.status_code, status)
                self.assertNotIn("a", tm.rid_to_state)

        asyncio.run(run())

    def test_exact_abort_isolates_grammar_and_encoder_waiters(self):
        from sglang.srt.constrained.grammar_manager import GrammarManager
        from sglang.srt.disaggregation.encoder.receiver import (
            MMReceiverBase,
            WaitingMMRequestStatus,
        )

        grammar = GrammarManager.__new__(GrammarManager)
        grammar.grammar_queue = [
            NS(rid=rid, grammar=None, set_finish_with_abort=Mock())
            for rid in ("a", "ab")
        ]
        reason = {"type": "abort", "message": "metadata failed", "status_code": 502}
        message = AbortReq(rid="a", exact=True, finished_reason=reason)
        grammar.abort_requests(message)
        grammar.grammar_queue[0].set_finish_with_abort.assert_called_once_with(
            "metadata failed", status_code=502
        )
        grammar.grammar_queue[1].set_finish_with_abort.assert_not_called()
        waiting = [
            NS(
                rid=rid,
                status=WaitingMMRequestStatus.PENDING,
                _fail_and_release=Mock(),
                release_resources=Mock(),
            )
            for rid in ("a", "ab")
        ]
        MMReceiverBase.abort_waiting_requests(NS(waiting_list=waiting), message)
        waiting[0]._fail_and_release.assert_called_once_with(
            "metadata failed", error_code=502
        )
        waiting[1]._fail_and_release.assert_not_called()

    def test_control_session_uses_control_admin_then_inference_credentials(self):
        async def run():
            tm = self.manager()
            with get_context().override_server_args(
                api_key="inference", admin_api_key="admin"
            ):
                session = await tm._get_decode_control_session()
                self.assertEqual(session.headers["Authorization"], "Bearer admin")
                await tm.close_decode_control()
            with get_context().override_server_args(
                api_key="inference", admin_api_key=None
            ):
                session = await tm._get_decode_control_session()
                self.assertEqual(session.headers["Authorization"], "Bearer inference")
                await tm.close_decode_control()

        asyncio.run(run())

    def test_native_streaming_admission_failure_has_retryable_error_and_done(self):
        from sglang.srt.entrypoints import http_server

        async def fail(*args):
            raise fastapi.HTTPException(503, "admission full")
            yield

        async def run():
            tm = NS(generate_request=fail, create_abort_task=lambda obj: None)
            with patch.object(http_server, "_global_state", NS(tokenizer_manager=tm)):
                response = await http_server.generate_request(
                    GenerateReqInput(text="hi", stream=True), NS(headers={})
                )
                data = b"".join([chunk async for chunk in response.body_iterator])
                self.assertIn(b'"code":503', data)
                self.assertIn(b'"retryable":true', data)
                self.assertTrue(data.endswith(b"data: [DONE]\n\n"))

        asyncio.run(run())

    def test_202_owner_retries_while_another_replica_is_stalled(self):
        async def run():
            tm = self.manager()
            counts = {"query": 0, "notify": 0, "cancel": 0}

            class Response:
                def __init__(self, status):
                    self.status = status

                async def json(self):
                    return {"request_generation": "g"}

            class Request:
                def __init__(self, url, body):
                    self.url = url
                    self.body = body

                async def __aenter__(self):
                    if "stalled" in self.url:
                        try:
                            await asyncio.Event().wait()
                        finally:
                            counts["cancel"] += 1
                    if self.body.get("query_generation"):
                        counts["query"] += 1
                        return Response(202 if counts["query"] == 1 else 200)
                    counts["notify"] += 1
                    return Response(200)

                async def __aexit__(self, *args):
                    return False

            tm._get_decode_control_session = AsyncMock(
                return_value=NS(post=lambda url, **kw: Request(url, kw["json"]))
            )
            with get_context().override_server_args(
                disagg_decode_urls=["http://healthy", "http://stalled"]
            ):
                await asyncio.wait_for(
                    tm._notify_decode_expanded_len("7", 8, "p", 0, "pair"), 0.5
                )
            self.assertEqual(counts, {"query": 2, "notify": 1, "cancel": 1})

        asyncio.run(run())

    def test_shutdown_drains_tasks_before_closing_session(self):
        async def run():
            tm = self.manager()
            tm._decode_control_session = aiohttp.ClientSession()
            session = tm._decode_control_session
            cancelled = asyncio.Event()

            async def pending():
                try:
                    await asyncio.Event().wait()
                finally:
                    self.assertFalse(session.closed)
                    cancelled.set()

            task = asyncio.create_task(pending())
            tm._expanded_input_tasks["a"] = {task}
            await asyncio.sleep(0)
            await tm.close_decode_control()
            self.assertTrue(cancelled.is_set())
            self.assertTrue(session.closed)

        asyncio.run(run())

    def test_attached_engine_shutdown_on_running_loop_closes_control_session(self):
        async def run():
            tm = self.manager()
            session = tm._decode_control_session = aiohttp.ClientSession()
            tm.mm_processor = None
            tm.cuda_vmm_feature_transport = NS(shutdown=lambda: None)
            engine = Engine.__new__(Engine)
            engine._attached = True
            engine.tokenizer_manager = tm
            engine.shutdown()
            await tm._decode_control_close_task
            self.assertTrue(session.closed)

        asyncio.run(run())

    def test_sync_shutdown_with_idle_or_closed_session_loop(self):
        for closed in (False, True):
            with self.subTest(closed=closed):
                loop = asyncio.new_event_loop()
                tm = self.manager()

                async def create():
                    return aiohttp.ClientSession()

                session = tm._decode_control_session = loop.run_until_complete(create())
                if closed:
                    loop.close()
                try:
                    tm.close_decode_control_sync()
                    self.assertTrue(session.closed)
                finally:
                    if not loop.is_closed():
                        loop.close()

    def test_cancelled_shutdown_still_closes_the_session(self):
        async def run():
            tm = self.manager()
            session = tm._decode_control_session = aiohttp.ClientSession()
            started = asyncio.Event()

            async def pending():
                try:
                    await asyncio.Event().wait()
                finally:
                    started.set()
                    await asyncio.sleep(1)

            task = asyncio.create_task(pending())
            tm._expanded_input_tasks["a"] = {task}
            await asyncio.sleep(0)
            close = asyncio.create_task(tm.close_decode_control())
            await started.wait()
            close.cancel()
            await asyncio.gather(close, return_exceptions=True)
            self.assertTrue(session.closed)
            self.assertTrue(task.done())

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
