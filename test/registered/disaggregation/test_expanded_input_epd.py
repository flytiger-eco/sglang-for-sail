"""Real E/P/D+gateway multimodal regression with a standalone reference.

By default this provisions Qwen2.5-VL on four CUDA GPUs (E/P/D/reference).
For an existing PPU 890 cluster set EPD_TEST_URL, EPD_TEST_DECODE_URL,
EPD_TEST_REFERENCE_URL and, when configured, EPD_TEST_ADMIN_KEY. Reference comparison can also
be replayed after saving EPD outputs using EPD_TEST_CAPTURE_PATH: run the EPD
suite, replace the provisioned deployment with the same standalone model, then
run TestStandaloneReplay with EPD_TEST_REFERENCE_URL and that capture path.
For Qwen4-Exp/Qwen3.8 FlashNext use TestQwen4ExpandedInputEPD with
EPD_TEST_QWEN4_MODEL_PATH and EPD_TEST_PREFILL_URL. It verifies the actual model,
QSA and PLE config instead of treating the request's model alias as deployment.
Match preprocessing limits on Encoder (vision_config.image) and standalone
(image) before comparing prompt lengths. No gateway changes are required.
"""

import base64
import io
import json
import os
import secrets
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import requests
from PIL import Image, ImageDraw

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-h100")


def payload(size, reverse=False):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    colors = ("blue", "red") if reverse else ("red", "blue")
    draw.rectangle((0, 0, size[0] // 2 - 1, size[1]), fill=colors[0])
    draw.rectangle((size[0] // 2, 0, size[0], size[1]), fill=colors[1])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return {
        "model": os.getenv("EPD_TEST_MODEL", "default"),
        "temperature": 0,
        "max_tokens": 96,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {
                        "type": "text",
                        "text": "What color is on the LEFT half? Reply with only red or blue.",
                    },
                ],
            }
        ],
    }


def answer(url, body):
    # MiniLB forwards JSON but not X-Request-ID. Older deployed routers may
    # forward the header instead; both identities must describe the same pair.
    pair_key = uuid.uuid4().hex
    body = dict(body, bootstrap_pair_key=pair_key)
    response = requests.post(
        url.rstrip("/") + "/v1/chat/completions",
        json=body,
        headers={"X-Request-ID": pair_key},
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()
    assert result.get("choices"), result
    return result["choices"][0]["message"]["content"], result["usage"]["prompt_tokens"]


class TestExpandedInputEPD(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        cls.external = bool(os.getenv("EPD_TEST_URL"))
        cls.processes = []
        cls.records = []
        if cls.external:
            cls.lb_url = os.environ["EPD_TEST_URL"]
            cls.decode_url = os.environ["EPD_TEST_DECODE_URL"]
            cls.reference_url = os.getenv("EPD_TEST_REFERENCE_URL")
            cls.admin_key = os.getenv("EPD_TEST_ADMIN_KEY")
            return
        super().setUpClass()
        cls.addClassCleanup(cls.cleanup_servers)
        cls.admin_key = secrets.token_urlsafe(32)
        model = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
        encoder_url = f"http://{cls.base_host}:{int(cls.lb_port) + 600}"
        cls.reference_url = f"http://{cls.base_host}:{int(cls.lb_port) + 700}"
        common = ["--trust-remote-code", "--mem-fraction-static", "0.7"]
        feature_env = dict(os.environ, SGLANG_DISAGG_EXPANDED_INPUT_LEN="1")

        def launch(url, gpu, args, env=None):
            process = popen_launch_server(
                model,
                base_url=url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=common + ["--base-gpu-id", str(gpu)] + args,
                env=env,
            )
            cls.processes.append(process)
            return process

        launch(
            encoder_url,
            0,
            ["--encoder-only", "--encoder-transfer-backend", "zmq_to_tokenizer"],
        )
        cls.process_decode = launch(
            cls.decode_url,
            2,
            [
                "--disaggregation-mode",
                "decode",
                "--disaggregation-bootstrap-port",
                cls.bootstrap_port,
                "--language-only",
                "--admin-api-key",
                cls.admin_key,
            ]
            + cls.transfer_backend
            + cls.rdma_devices_for(range(2, 3)),
            feature_env,
        )
        cls.process_prefill = launch(
            cls.prefill_url,
            1,
            [
                "--disaggregation-mode",
                "prefill",
                "--language-only",
                "--encoder-transfer-backend",
                "zmq_to_tokenizer",
                "--encoder-urls",
                encoder_url,
                "--disaggregation-bootstrap-port",
                cls.bootstrap_port,
                "--disagg-decode-urls",
                cls.decode_url,
                "--disagg-decode-control-key",
                cls.admin_key,
            ]
            + cls.transfer_backend
            + cls.rdma_devices_for(range(1, 2)),
            feature_env,
        )
        launch(
            cls.reference_url,
            3,
            [],
            dict(os.environ, SGLANG_DISAGG_EXPANDED_INPUT_LEN="0"),
        )
        cls.launch_lb()
        cls.processes.append(cls.process_lb)

    @classmethod
    def cleanup_servers(cls):
        for process in reversed(cls.processes):
            terminate_and_kill_process_tree(process)

    @classmethod
    def tearDownClass(cls):
        # Parent teardown assumes its own process/watcher initialization.
        if cls.external:
            return
        cls.cleanup_servers()
        cls.processes = []
        cls.process_lb = cls.process_decode = cls.process_prefill = None
        super().tearDownClass()

    def test_00_greeting(self):
        text, _ = answer(
            self.lb_url,
            {
                "model": os.getenv("EPD_TEST_MODEL", "default"),
                "max_tokens": 512,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": "你好，请介绍一下你自己。"}],
            },
        )
        self.assertTrue(text.strip())

    def _control_headers(self):
        return {"Authorization": f"Bearer {self.admin_key}"} if self.admin_key else {}

    def test_control_respects_configured_auth_and_rejects_invalid_metadata(self):
        url = self.decode_url + "/v1/expanded_input_len"
        if self.admin_key:
            for headers in ({}, {"Authorization": "Bearer inference-only"}):
                response = requests.post(
                    url, json={"rid": "7"}, headers=headers, timeout=10
                )
                self.assertIn(response.status_code, (401, 403))
        else:
            response = requests.post(
                url,
                json={
                    "rid": "7",
                    "query_generation": True,
                    "bootstrap_pair_key": "unregistered-pair",
                },
                timeout=10,
            )
            self.assertEqual(response.status_code, 202, response.text)
        response = requests.post(
            url,
            json={
                "rid": "7",
                "bootstrap_pair_key": "invalid-pair",
                "request_generation": "g",
                "expanded_input_len": -1,
                "mrope_position_delta": 0,
                "metadata_version": 2,
            },
            headers=self._control_headers(),
            timeout=10,
        )
        self.assertEqual(response.status_code, 400, response.text)

    def test_missing_dispatcher_identity_is_rejected(self):
        response = requests.post(
            self.lb_url + "/v1/chat/completions", json=payload((448, 448)), timeout=60
        )
        self.assertEqual(response.status_code, 400, response.text)

    def test_variable_grids_and_mrope_match_standalone(self):
        records = []
        for size in ((224, 448), (672, 224), (448, 672)):
            for reverse in (False, True):
                with self.subTest(size=size, reverse=reverse):
                    body = payload(size, reverse)
                    text, tokens = answer(self.lb_url, body)
                    expected = "blue" if reverse else "red"
                    self.assertIn(expected, text.lower())
                    self.assertNotIn("red" if reverse else "blue", text.lower())
                    self.assertGreater(tokens, 32)
                    records.append(
                        {
                            "body": body,
                            "expected": expected,
                            "text": text,
                            "prompt_tokens": tokens,
                        }
                    )
                    if self.reference_url:
                        reference, ref_tokens = answer(self.reference_url, body)
                        self.assertIn(expected, reference.lower())
                        self.assertEqual(tokens, ref_tokens)
        self.assertGreater(len({r["prompt_tokens"] for r in records}), 1)
        path = os.getenv("EPD_TEST_CAPTURE_PATH")
        if path:
            with open(path, "w") as output:
                json.dump(records, output)


def qwen4_state_model(model_path):
    from sglang.srt.layers.attention.qsa.config import parse_qsa_profile

    config = json.loads((Path(model_path) / "config.json").read_text())
    if "Qwen4ExpForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("Qwen4 EPD regression requires an actual Qwen4-Exp model")
    text_config = config.get("text_config", config)
    profile = parse_qsa_profile(SimpleNamespace(**text_config))
    if profile is None or not text_config.get("ple_layer_ids"):
        raise ValueError("Qwen4 state-handoff regression requires both QSA and PLE")
    return {
        "model_path": model_path,
        "architectures": config["architectures"],
        "qsa_compress_ratio": profile.compress_ratio,
        "ple_layer_ids": text_config["ple_layer_ids"],
    }


def assert_served_model(test_case, url, model_path, role=None):
    response = requests.get(url.rstrip("/") + "/get_server_info", timeout=15)
    response.raise_for_status()
    config = response.json()
    test_case.assertEqual(config["model_path"], model_path)
    if role is not None:
        test_case.assertEqual(config["disaggregation_mode"], role)
        test_case.assertTrue(config["language_only"])
    return config


class TestQwen4ExpandedInputEPD(TestExpandedInputEPD):
    """Opt-in Qwen4/Qwen3.8 FlashNext QSA+PLE handoff against the real model.

    Set EPD_TEST_QWEN4_MODEL_PATH plus EPD_TEST_URL, EPD_TEST_PREFILL_URL and
    EPD_TEST_DECODE_URL for the provisioned 4E1P1D deployment. Supply a reference
    URL or capture outputs for the sequential standalone replay. This class
    cannot fall back to launching Qwen2.5 on a single GPU per role.
    """

    @classmethod
    def setUpClass(cls):
        cls.external = True
        model_path = os.getenv("EPD_TEST_QWEN4_MODEL_PATH")
        if not model_path:
            raise unittest.SkipTest(
                "Qwen4 EPD requires an explicitly provisioned model and deployment"
            )
        for name in ("EPD_TEST_URL", "EPD_TEST_PREFILL_URL", "EPD_TEST_DECODE_URL"):
            if not os.getenv(name):
                raise ValueError(f"Qwen4 EPD requires {name}")
        cls.state_model = qwen4_state_model(model_path)
        super().setUpClass()
        for url, role in (
            (os.environ["EPD_TEST_PREFILL_URL"], "prefill"),
            (cls.decode_url, "decode"),
        ):
            response = requests.get(url.rstrip("/") + "/get_server_info", timeout=15)
            response.raise_for_status()
            config = response.json()
            if (
                config["model_path"] != model_path
                or config["disaggregation_mode"] != role
            ):
                raise ValueError(
                    f"Qwen4 EPD endpoint {url} is serving a different model or role"
                )

    def _paired_native_request(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.state_model["model_path"], trust_remote_code=True
        )
        image_url = payload((224, 448))["messages"][0]["content"][0]["image_url"]["url"]
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_url},
                        {"type": "text", "text": "What color is on the left?"},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        pair = uuid.uuid4().hex
        body = {
            "text": prompt,
            "image_data": [image_url],
            "stream": False,
            "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            "bootstrap_pair_key": pair,
            "bootstrap_room": uuid.uuid4().int % (2**31 - 1) + 1,
            "bootstrap_host": os.environ["EPD_TEST_PREFILL_URL"]
            .split("://", 1)[1]
            .split(":", 1)[0],
            "bootstrap_port": int(os.getenv("EPD_TEST_BOOTSTRAP_PORT", "8998")),
        }
        return body

    def test_decode_validation_failure_unblocks_prefill(self):
        """A Decode-only deferred-position rejection must not strand Prefill."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        body = self._paired_native_request()
        pair = body["bootstrap_pair_key"]
        decode = dict(
            body, rid="d-" + pair, return_logprob=True, logprob_start_len=8192
        )
        prefill = dict(body, rid="p-" + pair)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            d_future = pool.submit(
                requests.post, self.decode_url + "/generate", json=decode, timeout=30
            )
            time.sleep(0.15)
            p_future = pool.submit(
                requests.post,
                os.environ["EPD_TEST_PREFILL_URL"] + "/generate",
                json=prefill,
                timeout=30,
            )
            d_response = d_future.result(timeout=35)
            p_response = p_future.result(timeout=35)
        self.assertEqual(d_response.status_code, 400, d_response.text)
        self.assertIn("logprob_start_len", d_response.text)
        self.assertGreaterEqual(p_response.status_code, 400, p_response.text)
        self.assertLess(
            time.monotonic() - started, 15, "Prefill waited for its bootstrap timeout"
        )
        for url in (self.decode_url, os.environ["EPD_TEST_PREFILL_URL"]):
            self.assertEqual(requests.get(url + "/health", timeout=5).status_code, 200)

    def test_decode_cancel_before_prefill_registration_unblocks_both_peers(self):
        """A public exact cancel must survive late Prefill registration."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        body = self._paired_native_request()
        pair = body["bootstrap_pair_key"]
        decode_rid = "d-cancel-" + pair
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            d_future = pool.submit(
                requests.post,
                self.decode_url + "/generate",
                json=dict(body, rid=decode_rid),
                timeout=30,
            )
            # Wait for this HTTP owner rather than cancelling before dispatch.
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                query = requests.post(
                    self.decode_url + "/v1/expanded_input_len",
                    json={
                        "rid": str(body["bootstrap_room"]),
                        "bootstrap_pair_key": pair,
                        "query_generation": True,
                    },
                    headers=self._control_headers(),
                    timeout=5,
                )
                if query.status_code == 200:
                    break
                self.assertEqual(query.status_code, 202, query.text)
                time.sleep(0.02)
            else:
                self.fail("Decode owner did not register before cancellation")
            aborted = requests.post(
                self.decode_url + "/abort_request",
                json={"rid": decode_rid, "exact": True},
                headers=self._control_headers(),
                timeout=5,
            )
            self.assertEqual(aborted.status_code, 200, aborted.text)
            # Prefill has not registered any sender when the cancel is issued.
            time.sleep(0.15)
            p_future = pool.submit(
                requests.post,
                os.environ["EPD_TEST_PREFILL_URL"] + "/generate",
                json=dict(body, rid="p-cancel-" + pair),
                timeout=30,
            )
            d_response = d_future.result(timeout=35)
            p_response = p_future.result(timeout=35)
        self.assertEqual(d_response.status_code, 200, d_response.text)
        cancelled = d_response.json()
        self.assertEqual(cancelled["meta_info"]["finish_reason"]["type"], "abort")
        self.assertEqual(cancelled["output_ids"], [])
        self.assertGreaterEqual(p_response.status_code, 400, p_response.text)
        self.assertLess(
            time.monotonic() - started,
            15,
            "Cancelled pair waited for bootstrap timeout",
        )
        for url in (self.decode_url, os.environ["EPD_TEST_PREFILL_URL"]):
            self.assertEqual(requests.get(url + "/health", timeout=5).status_code, 200)

    def test_model_and_state_handoff_configuration(self):
        model_path = self.state_model["model_path"]
        prefill = assert_served_model(
            self, os.environ["EPD_TEST_PREFILL_URL"], model_path, "prefill"
        )
        decode = assert_served_model(self, self.decode_url, model_path, "decode")
        self.assertTrue(prefill["encoder_urls"])
        self.assertIn(self.decode_url, prefill["disagg_decode_urls"])
        self.assertFalse(decode["encoder_urls"])
        if self.reference_url:
            assert_served_model(self, self.reference_url, model_path)
        path = os.getenv("EPD_TEST_CAPTURE_PATH")
        if path:
            Path(path + ".model.json").write_text(json.dumps(self.state_model))


class TestStandaloneReplay(CustomTestCase):
    @unittest.skipUnless(
        os.getenv("EPD_TEST_CAPTURE_PATH") and os.getenv("EPD_TEST_REFERENCE_URL"),
        "Sequential PPU reference replay requires captured EPD outputs and a standalone URL",
    )
    def test_captured_epd_matches_standalone(self):
        model_path = os.getenv("EPD_TEST_QWEN4_MODEL_PATH")
        if model_path:
            model = qwen4_state_model(model_path)
            proof = Path(os.environ["EPD_TEST_CAPTURE_PATH"] + ".model.json")
            self.assertEqual(json.loads(proof.read_text()), model)
            assert_served_model(self, os.environ["EPD_TEST_REFERENCE_URL"], model_path)
        with open(os.environ["EPD_TEST_CAPTURE_PATH"]) as source:
            records = json.load(source)
        self.assertEqual(len(records), 6)
        for row in records:
            with self.subTest(
                expected=row["expected"], prompt_tokens=row["prompt_tokens"]
            ):
                text, tokens = answer(os.environ["EPD_TEST_REFERENCE_URL"], row["body"])
                self.assertIn(row["expected"], text.lower())
                self.assertEqual(tokens, row["prompt_tokens"])


if __name__ == "__main__":
    unittest.main()
