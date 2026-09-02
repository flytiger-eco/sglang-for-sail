"""PPU nightly Answer MVP for Qwen3.5-397B-A17B-W8A8-INT8.

This ports the ten public Answer prompts into the GitHub-native CI path.  The
current gate is deliberately limited to L0 request integrity and L1
deterministic facts/quality checks.  LLM-as-Judge is deferred and its absence is
recorded explicitly in every result.
"""

import json
import os
import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.answer_eval_kit import (
    CandidateRequestError,
    answer_expected_hardware,
    build_answer_server_args,
    build_report,
    canonical_digest,
    default_provenance,
    load_json,
    render_candidates,
    render_summary,
    request_chat_completion,
    validate_test_config,
    write_report_files,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server

DATA_DIR = Path(__file__).with_name("answer_eval") / "data"
DEFAULT_TEST_CONFIG_PATH = DATA_DIR / "qwen3_5_397b_a17b_w8a8_int8_test_config.json"
TEST_CONFIG_PATH_ENV = "SGLANG_PPU_ANSWER_TEST_CONFIG"

register_ppu_ci(est_time=3600, suite="nightly-answer-8-ppu", nightly=True)


def _load_test_config():
    configured_path = os.environ.get(TEST_CONFIG_PATH_ENV)
    config_path = Path(configured_path) if configured_path else DEFAULT_TEST_CONFIG_PATH
    config = load_json(config_path)
    validate_test_config(config)
    return config, config_path


class TestPPUQwen35Answer(unittest.TestCase):
    @classmethod
    def _provenance(cls, accelerator):
        provenance = default_provenance(
            cls.model_config["served_model_name"],
            cls.model_path or "",
            server_config=cls.server_config,
            generation_config=cls.request_config["generation"],
            expected_hardware=answer_expected_hardware(cls.test_config),
            accelerator=accelerator,
        )
        provenance["test_config_id"] = cls.test_config["test_id"]
        provenance["test_config_sha256"] = canonical_digest(cls.test_config)
        return provenance

    @classmethod
    def _write_setup_failure(cls, stage, failure_class, exc):
        try:
            visible_device_count = torch.cuda.device_count()
        except Exception:
            visible_device_count = None
        responses = {
            case["id"]: {
                "error": type(exc).__name__,
                "reason_code": f"{stage}_failed",
                "failure_class": failure_class,
            }
            for case in cls.dataset["cases"]
        }
        provenance = cls._provenance({"visible_device_count": visible_device_count})
        provenance["setup_stage"] = stage
        report = build_report(cls.dataset, cls.profile, responses, provenance)
        write_report_files(report, cls.output_dir)
        print(render_summary(report), flush=True)

    @classmethod
    def setUpClass(cls):
        cls.test_config, cls.test_config_path = _load_test_config()
        cls.model_config = cls.test_config["model"]
        cls.server_config = cls.test_config["server"]["parameters"]
        cls.request_config = cls.test_config["request"]
        evaluation = cls.test_config["evaluation"]
        cls.dataset = load_json(cls.test_config_path.parent / evaluation["dataset"])
        cls.profile = load_json(
            cls.test_config_path.parent / evaluation["quality_profile"]
        )
        cls.output_dir = Path(
            os.environ.get("SGLANG_PPU_ANSWER_RESULTS_DIR", "ppu-answer-artifacts")
        )
        cls.model_path = cls.model_config["path"]
        stage = "runner_preflight"
        failure_class = "runner_error"
        try:
            if not Path(cls.model_path).is_dir():
                raise RuntimeError(
                    f"the model directory configured by {cls.test_config_path} is missing"
                )
            model_dir = Path(cls.model_path)
            if model_dir.name != cls.model_config["checkpoint_name"]:
                raise RuntimeError(
                    "model.path must identify " f"{cls.model_config['checkpoint_name']}"
                )
            config_path = model_dir / "config.json"
            if not config_path.is_file():
                raise RuntimeError(f"{cls.model_path} does not contain config.json")
            model_config = load_json(config_path)
            if (
                model_config.get("model_type")
                not in cls.model_config["accepted_model_types"]
            ):
                raise RuntimeError(
                    "checkpoint config model_type is not a supported Qwen3.5 MoE type"
                )
            expected_device_count = len(cls.test_config["hardware"]["visible_devices"])
            if torch.cuda.device_count() != expected_device_count:
                raise RuntimeError(
                    f"Answer MVP requires {expected_device_count} visible PPU devices; "
                    f"found {torch.cuda.device_count()}"
                )

            cls.base_url = DEFAULT_URL_FOR_TEST
            stage = "server_start"
            failure_class = "server_error"
            cls.process = popen_launch_server(
                model=cls.model_path,
                base_url=cls.base_url,
                timeout=cls.test_config["server"]["startup_timeout_seconds"],
                other_args=build_answer_server_args(cls.test_config),
            )
        except Exception as exc:
            try:
                cls._write_setup_failure(stage, failure_class, exc)
            except Exception as report_error:
                print(
                    "failed to write structured Answer setup evidence: "
                    f"{type(report_error).__name__}",
                    flush=True,
                )
            raise

    @classmethod
    def tearDownClass(cls):
        process = getattr(cls, "process", None)
        if process is not None:
            kill_process_tree(process.pid)

    def test_public_answer_suite(self):
        responses = {}
        for case in self.dataset["cases"]:
            try:
                responses[case["id"]] = request_chat_completion(
                    self.base_url,
                    self.model_config["served_model_name"],
                    case["prompt"],
                    self.dataset["request_suffix"],
                    timeout_seconds=self.request_config["timeout_seconds"],
                    generation_config=self.request_config["generation"],
                )
            except CandidateRequestError as exc:
                responses[case["id"]] = {
                    "error": str(exc),
                    "reason_code": exc.reason_code,
                }

        report = build_report(
            self.dataset,
            self.profile,
            responses,
            self._provenance(
                {
                    "visible_device_count": torch.cuda.device_count(),
                    "devices": [
                        {
                            "name": torch.cuda.get_device_name(index),
                            "total_memory_bytes": torch.cuda.get_device_properties(
                                index
                            ).total_memory,
                        }
                        for index in range(torch.cuda.device_count())
                    ],
                }
            ),
        )
        include_raw_outputs = os.environ.get(
            "SGLANG_PPU_ANSWER_INCLUDE_RAW_OUTPUTS", "0"
        ) in {"1", "true", "TRUE"}
        write_report_files(
            report, self.output_dir, include_raw_outputs=include_raw_outputs
        )
        print(render_summary(report), flush=True)
        if include_raw_outputs:
            # The log is a disclosure surface like the artifact, so the same
            # switch governs both.  ci_utils runs this file as a plain
            # subprocess with an inherited stdout, so the block lands in the job
            # log whether or not the assertion below fails.  Relaxing the error
            # handler keeps a candidate that stdout cannot encode from turning
            # the nightly red over log formatting alone.
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(errors="backslashreplace")
            print(render_candidates(report, self.dataset), flush=True)

        self.assertEqual(
            report["summary"]["failed"],
            0,
            json.dumps(
                {
                    result["case_id"]: [
                        finding["reason_code"]
                        for finding in result["findings"]
                        if finding["action"] == "hard_fail"
                    ]
                    for result in report["cases"]
                    if result["verdict"] == "failed"
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
