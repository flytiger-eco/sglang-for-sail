"""Structured credential capture and paired request input validation."""

import argparse
import ast
import asyncio
import pickle
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import urlunsplit

import msgspec

from sglang.srt.disaggregation.expanded_input import expanded_input_pair_key
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.config_dump import (
    capture_engine_command,
    redact_cli_secrets,
    redact_config_for_dump,
    redact_url_userinfo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEPDReadbackContract(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_only_media_prefill_and_deferred_decode_keep_private_pair_key(self):
        for role in ("prefill", "decode", "null"):
            for deferred in (False, True):
                for media, waiting in (
                    (None, False),
                    ({"images": []}, False),
                    (None, True),
                ):
                    request = NS(
                        bootstrap_pair_key="from-x-request-id",
                        needs_expanded_input_len=deferred,
                        mm_inputs=media,
                        need_wait_for_mm_inputs=waiting,
                    )
                    expected = (
                        "from-x-request-id"
                        if deferred
                        or (role == "prefill" and (media is not None or waiting))
                        else None
                    )
                    with self.subTest(
                        role=role, deferred=deferred, media=media, waiting=waiting
                    ):
                        self.assertEqual(
                            expanded_input_pair_key(request, role), expected
                        )
                        self.assertEqual(
                            expanded_input_pair_key(request, DisaggregationMode(role)),
                            expected,
                        )
                        self.assertEqual(
                            request.bootstrap_pair_key, "from-x-request-id"
                        )

    def test_bootstrap_pair_list_must_match_unexpanded_prompt_count(self):
        for keys in ([], ["one"], ["one", "two", "extra"]):
            with self.subTest(keys=keys):
                request = GenerateReqInput(text=["a", "b"], bootstrap_pair_key=keys)
                with self.assertRaisesRegex(
                    ValueError, "bootstrap_pair_key.*batch size"
                ):
                    request.normalize_batch_and_arguments()

    def test_valid_pair_lists_and_scalar_broadcast_preserve_parallel_sampling(self):
        for keys, expected in (
            (["a", "b"], ["a", "b", "a", "b"]),
            ("scalar", ["scalar"] * 4),
        ):
            request = GenerateReqInput(
                text=["a", "b"], bootstrap_pair_key=keys, sampling_params={"n": 2}
            )
            request.normalize_batch_and_arguments()
            self.assertEqual(request.bootstrap_pair_key, expected)
            self.assertEqual(
                [request[i].bootstrap_pair_key for i in range(4)], expected
            )

    def test_url_userinfo_redaction_preserves_authority_path_and_query(self):
        for source, expected in (
            (
                urlunsplit(("http", "user:pass@decode:8000", "/path", "a=b", "")),
                "http://<redacted>@decode:8000/path?a=b",
            ),
            (
                urlunsplit(
                    ("https", "user%40mail:pass%2Fword@[::1]:8000", "/a", "", "")
                ),
                "https://<redacted>@[::1]:8000/a",
            ),
            (
                urlunsplit(("http", "user@host", "/a", "", "")),
                "http://<redacted>@host/a",
            ),
            (
                urlunsplit(("http", "user:pass word@host", "/a", "", "")),
                "http://<redacted>@host/a",
            ),
            (
                "--disagg-decode-urls="
                + urlunsplit(("http", "u:pass word@host", "/a", "", "")),
                "--disagg-decode-urls=http://<redacted>@host/a",
            ),
            ("http://host/a@b?mail=x@y", "http://host/a@b?mail=x@y"),
        ):
            with self.subTest(source=source):
                self.assertEqual(redact_url_userinfo(source), expected)
                self.assertEqual(redact_url_userinfo(expected), expected)

    def test_cli_capture_redacts_url_values_in_both_option_forms(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-path")
        parser.add_argument("--api-key")
        parser.add_argument("--disagg-decode-urls", nargs="+")
        url = urlunsplit(("http", "user:url-password@decode:8000", "/a", "", ""))
        argv = [
            "--model-path",
            "/models/test-vl",
            "--api-key",
            "test",
            "--disagg-decode-urls=" + url,
            "--disagg-decode-urls",
            url,
        ]
        before = list(argv)
        capture = redact_cli_secrets(argv, parser)
        self.assertEqual(argv, before)
        self.assertIn("/models/test-vl", capture)
        self.assertEqual(capture[3], "<redacted>")
        self.assertNotIn("url-password", str(capture))
        self.assertEqual(sum("<redacted>@decode:8000/a" in item for item in capture), 2)

    def test_engine_capture_is_field_aware_even_when_secret_matches_model_path(self):
        kwargs = {
            "model_path": "/models/test-vl",
            "api_key": "test",
            "ssl_keyfile_password": "quoted'\npassword",
            "disagg_decode_urls": [
                urlunsplit(("http", "u:private@decode:8000", "", "", ""))
            ],
        }
        captured = capture_engine_command(kwargs)
        fields = {
            field.arg: ast.literal_eval(field.value)
            for field in ast.parse(captured, mode="eval").body.keywords
        }
        self.assertEqual(fields["model_path"], kwargs["model_path"])
        self.assertEqual(fields["api_key"], "<redacted>")
        self.assertEqual(fields["ssl_keyfile_password"], "<redacted>")
        self.assertEqual(
            fields["disagg_decode_urls"], ["http://<redacted>@decode:8000"]
        )
        self.assertEqual(kwargs["api_key"], "test")
        self.assertEqual(
            kwargs["disagg_decode_urls"],
            [urlunsplit(("http", "u:private@decode:8000", "", "", ""))],
        )

    def test_command_readback_and_dump_keep_model_paths_after_secret_rotation(self):
        args = ServerArgs(
            model_path="dummy",
            api_key="test",
            disagg_decode_urls=[
                urlunsplit(("http", "u:private@decode:8000", "", "", ""))
            ],
        )
        command = capture_engine_command(
            {
                "model_path": "/models/test-vl",
                "api_key": "test",
                "disagg_decode_urls": args.disagg_decode_urls,
            }
        )
        msgspec.Struct.__setattr__(args, "_launch_command", command)
        self.assertEqual(args.launch_command, command)
        dump = redact_config_for_dump(
            {
                "model_path": "/models/test-vl",
                "launch_command": args.launch_command,
                "disagg_decode_urls": args.disagg_decode_urls,
            },
            args,
            [("rotate", {"api_key": "old-secret"})],
        )
        self.assertEqual(dump["model_path"], "/models/test-vl")
        self.assertEqual(dump["launch_command"], command)
        self.assertNotIn("private", str(dump))
        raw = pickle.loads(pickle.dumps(args))
        self.assertEqual(raw.api_key, "test")
        self.assertEqual(
            raw.disagg_decode_urls,
            [urlunsplit(("http", "u:private@decode:8000", "", "", ""))],
        )
        self.assertEqual(
            args.resolved_dict()["disagg_decode_urls"],
            ["http://<redacted>@decode:8000"],
        )

    def test_remaining_resources_close_after_control_error_or_cancellation(self):
        async def run(error):
            manager = TokenizerManager.__new__(TokenizerManager)
            manager.close_decode_control = AsyncMock(side_effect=error)
            sidecar, grpc, warmup = Mock(), Mock(), Mock()
            tool = NS(aclose=AsyncMock())
            with patch.object(http_server, "_shutdown_native_grpc_server") as stop_grpc:
                if isinstance(error, asyncio.CancelledError):
                    with self.assertRaises(asyncio.CancelledError):
                        await http_server._close_runtime_resources(
                            manager, sidecar, grpc, tool, warmup
                        )
                else:
                    await http_server._close_runtime_resources(
                        manager, sidecar, grpc, tool, warmup
                    )
                sidecar.stop.assert_called_once()
                stop_grpc.assert_called_once_with(grpc)
                tool.aclose.assert_awaited_once()
                warmup.join.assert_called_once()

        for error in (None, RuntimeError("control close"), asyncio.CancelledError()):
            with self.subTest(error=error):
                asyncio.run(run(error))

    def test_control_query_and_delivery_errors_redact_url_credentials(self):
        # Synthetic userinfo is assembled separately from the URL scheme so
        # secret scanners do not mistake a redaction fixture for a credential.
        url = urlunsplit(("http", "user:control-fixture@decode:8000", "/a", "", ""))

        async def run(query_status, delivery_status, error_type):
            calls = []

            class Response:
                def __init__(self, status):
                    self.status = status

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    return None

                async def json(self):
                    return {"request_generation": "generation"}

            class Session:
                def post(self, target, *, json, timeout):
                    calls.append(target)
                    return Response(
                        query_status
                        if json.get("query_generation")
                        else delivery_status
                    )

            manager = TokenizerManager.__new__(TokenizerManager)
            manager._get_decode_control_session = AsyncMock(return_value=Session())
            manager._expanded_input_tasks = {}
            manager.asyncio_tasks = set()
            manager.rid_to_state = {}
            with (
                get_context().override_server_args(disagg_decode_urls=[url]),
                self.assertLogs(
                    "sglang.srt.managers.tokenizer_manager", level="ERROR"
                ) as captured,
            ):
                manager._track_expanded_len_task(
                    manager._notify_decode_expanded_len("7", 8, "request", 0, "pair"),
                    "request",
                )
                task = next(iter(manager.asyncio_tasks))
                await asyncio.gather(task, return_exceptions=True)
                await asyncio.sleep(0)
            self.assertIsInstance(task.exception(), error_type)
            self.assertNotIn("control-fixture", str(task.exception()))
            self.assertIn("<redacted>@decode:8000", str(task.exception()))
            self.assertNotIn("control-fixture", "\n".join(captured.output))
            self.assertTrue(calls)
            self.assertTrue(all(target.startswith(url) for target in calls))

        for query, delivery, error in (
            (401, None, PermissionError),
            (400, None, ValueError),
            (200, 400, ValueError),
        ):
            with self.subTest(query=query, delivery=delivery):
                asyncio.run(run(query, delivery, error))

    def test_control_error_logging_does_not_include_unsafe_exception_chains(self):
        raw_url = urlunsplit(("http", "user:chained-fixture@decode:8000", "/a", "", ""))

        async def run():
            manager = TokenizerManager.__new__(TokenizerManager)
            manager._expanded_input_tasks = {}
            manager.asyncio_tasks = set()
            manager.rid_to_state = {}

            async def fail():
                try:
                    raise ValueError(raw_url)
                except ValueError as error:
                    raise RuntimeError("control delivery failed") from error

            with self.assertLogs(
                "sglang.srt.managers.tokenizer_manager", level="ERROR"
            ) as captured:
                manager._track_expanded_len_task(fail(), "request")
                await asyncio.gather(
                    *list(manager.asyncio_tasks), return_exceptions=True
                )
                await asyncio.sleep(0)
            output = "\n".join(captured.output)
            self.assertIn("control delivery failed", output)
            self.assertNotIn("chained-fixture", output)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
