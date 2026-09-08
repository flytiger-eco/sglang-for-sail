import unittest
from array import array

import torch

from sglang.srt.utils.common import LazyValue, flatten_arrays_to_int64_tensor
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFlattenArraysToInt64Tensor(CustomTestCase):
    """`flatten_arrays_to_int64_tensor` is invoked by `prepare_for_extend`
    to build the per-batch input_ids tensor (pinned, async H2D) from a
    list of array.array('q') per-req get_fill_ids() slices. Tests the
    full matrix of (device, pin) the production code paths through.
    """

    DEVICES = ("cpu", "cuda")
    PIN_OPTIONS = (False, True)

    def _check(self, parts: list, expected: list[int]) -> None:
        for device in self.DEVICES:
            for pin in self.PIN_OPTIONS:
                with self.subTest(device=device, pin=pin):
                    out = flatten_arrays_to_int64_tensor(parts, device, pin)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    self.assertEqual(out.dtype, torch.int64)
                    self.assertEqual(out.device.type, device)
                    self.assertEqual(out.shape, (len(expected),))
                    self.assertEqual(out.cpu().tolist(), expected)

    def test_single_part(self):
        parts = [array("q", [1, 2, 3, 4, 5])]
        self._check(parts, [1, 2, 3, 4, 5])

    def test_multiple_parts(self):
        parts = [
            array("q", [10, 20, 30]),
            array("q", [100, 200]),
            array("q", [1000]),
        ]
        self._check(parts, [10, 20, 30, 100, 200, 1000])


class TestLazyValue(CustomTestCase):
    """`LazyValue` defers a model's expert-weight gather until EPLB asks for
    it, and the ask arrives as `getattr(model, "routed_experts_weights_of_layer",
    None)` in ModelRunner.initialize. Both paths below were reached in
    production: run 34132812028 lost every rank of a two-node Kimi K2.6 to a
    RecursionError raised 981 rounds away from the missing attribute that
    caused it.
    """

    def test_creates_once_and_forwards(self):
        calls = []

        def creator():
            calls.append(1)
            return {"a": 1}

        lazy = LazyValue(creator)
        self.assertEqual(calls, [])
        self.assertEqual(list(lazy.keys()), ["a"])
        self.assertEqual(lazy["a"], 1)
        lazy["b"] = 2
        self.assertEqual(lazy.value, {"a": 1, "b": 2})
        self.assertEqual(calls, [1])

    def test_creator_attribute_error_does_not_recurse(self):
        calls = []

        def creator():
            calls.append(1)
            raise AttributeError("'PPMissingLayer' object has no attribute 'mlp'")

        lazy = LazyValue(creator)
        with self.assertRaises(RuntimeError) as caught:
            lazy.value
        self.assertIn("PPMissingLayer", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, AttributeError)
        # The creator runs once. Forwarding "value" back into the property is
        # what turned one failure into a stack's worth of them.
        self.assertEqual(calls, [1])

    def test_creator_attribute_error_survives_getattr_default(self):
        lazy = LazyValue(lambda: (_ for _ in ()).throw(AttributeError("mlp")))

        class Model:
            @property
            def routed_experts_weights_of_layer(self):
                return lazy.value

        # A bare AttributeError here would be spent on the default and the
        # broken model would look like a model that has no experts at all.
        with self.assertRaises(RuntimeError):
            getattr(Model(), "routed_experts_weights_of_layer", None)

    def test_absent_attribute_of_created_value_still_raises(self):
        lazy = LazyValue(lambda: {"a": 1})
        with self.assertRaises(AttributeError):
            lazy.no_such_method

    def test_uninitialised_instance_fails_loudly(self):
        # Reached if an instance is ever produced without __init__ running.
        # Without the guard this is a RecursionError as well.
        with self.assertRaises(AttributeError):
            LazyValue.__new__(LazyValue).value


if __name__ == "__main__":
    unittest.main()
