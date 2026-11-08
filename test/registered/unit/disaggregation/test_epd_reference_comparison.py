"""Exercise the live EPD suite's output checks without registering GPU suites."""

import ast
import copy
import math
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def reference_helpers():
    path = (
        Path(__file__).resolve().parents[2]
        / "disaggregation/test_expanded_input_epd.py"
    )
    # Importing the whole module would register a CUDA suite in the CPU runner.
    # Compile its maintained pure functions, with no replacement implementation.
    tree = ast.parse(path.read_text())
    names = {"generation_details", "assert_reference_matches"}
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ],
        type_ignores=[],
    )
    namespace = {"math": math}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["generation_details"], namespace["assert_reference_matches"]


class TestEPDReferenceComparison(CustomTestCase):
    def setUp(self):
        self.details, self.compare = reference_helpers()
        self.result = {
            "text": "red",
            "prompt_tokens": 128,
            "generated_tokens": ["red"],
            "output_logprobs": [-0.01],
        }

    def test_matching_full_generation_passes(self):
        self.compare(self, self.result, copy.deepcopy(self.result))

    def test_matching_color_and_prompt_length_do_not_hide_changed_generation(self):
        changed = dict(
            self.result, text="red extra", generated_tokens=["red", " extra"]
        )
        with self.assertRaises(AssertionError):
            self.compare(self, self.result, changed)
        changed = dict(self.result, generated_tokens=["r", "ed"])
        with self.assertRaises(AssertionError):
            self.compare(self, self.result, changed)

    def test_explicit_calibrated_tolerance_checks_logprobs(self):
        reference = dict(self.result, output_logprobs=[-0.02])
        self.compare(self, self.result, reference, logprob_atol=0.02)
        with self.assertRaises(AssertionError):
            self.compare(self, self.result, reference, logprob_atol=0.001)
        for tolerance in (-1, float("nan"), float("inf")):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                self.compare(self, self.result, reference, logprob_atol=tolerance)

    def test_legacy_capture_and_missing_or_nonfinite_logprobs_are_rejected(self):
        with self.assertRaises(AssertionError):
            self.compare(self, {"text": "red", "prompt_tokens": 128}, self.result)
        response = {
            "choices": [{"message": {"content": "red"}, "logprobs": None}],
            "usage": {"prompt_tokens": 128},
        }
        with self.assertRaises(AssertionError):
            self.details(response)
        response["choices"][0]["logprobs"] = {
            "content": [{"token": "red", "logprob": float("nan")}]
        }
        with self.assertRaises(AssertionError):
            self.details(response)

    def test_response_preserves_all_generated_tokens_and_probabilities(self):
        response = {
            "choices": [
                {
                    "message": {"content": "red"},
                    "logprobs": {"content": [{"token": "red", "logprob": -0.01}]},
                }
            ],
            "usage": {"prompt_tokens": 128},
        }
        self.assertEqual(self.details(response), self.result)


if __name__ == "__main__":
    unittest.main()
