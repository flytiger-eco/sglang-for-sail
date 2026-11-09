"""Online updates must reject model-owned derived buffers before writing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _unsupported_derived_weight_cache_error,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDerivedWeightCache(unittest.TestCase):
    def test_nested_model_cache_rejects_updates(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False, device="cpu"),
            torch.nn.Sequential(torch.nn.Module()),
        )
        model[1][0]._derived_weight_cache_error = "derived scales require restart"
        original = model[0].weight.detach().clone()
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )
        with patch(
            "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
            return_value=SimpleNamespace(weight_cache_mode="off"),
        ):
            self.assertEqual(
                updater.update_weights_from_tensor(
                    [("0.weight", torch.zeros_like(original))], load_format="direct"
                ),
                (False, "derived scales require restart"),
            )
        self.assertTrue(torch.equal(model[0].weight, original))

    def test_rejects_begin_before_restoring_weights(self):
        model = torch.nn.Module()
        model._derived_weight_cache_error = "prepared W4AFP8 requires restart"
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda **kw: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )
        with (
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater.DefaultModelLoader.restore_weights_before_loading"
            ) as restore,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires restart"):
                updater.begin_weight_update()
            restore.assert_not_called()
            for result in (
                updater.update_weights_from_disk("other", "auto"),
                updater.update_weights_from_tensor([], load_format="direct"),
                updater.load_weights_from_distributed([]),
                updater.update_weights_from_ipc(SimpleNamespace()),
            ):
                self.assertEqual(result, (False, "prepared W4AFP8 requires restart"))

    def test_qsa_fp8_pd_rejects_every_update_before_mutation(self):
        from sglang.srt.disaggregation.utils import DisaggregationMode

        for mode in (
            "prefill",
            "decode",
            DisaggregationMode.PREFILL,
            DisaggregationMode.DECODE,
        ):
            for ids in ([3], []):
                with self.subTest(mode=mode, ids=ids):
                    model = torch.nn.Linear(2, 2, bias=False, device="cpu")
                    original = model.weight.detach().clone()
                    pool = SimpleNamespace(
                        dtype=torch.float8_e4m3fn,
                        qsa_compress_ratio=4,
                        full_attention_layer_id_mapping={
                            i: n for n, i in enumerate(ids)
                        },
                    )
                    runner = SimpleNamespace(token_to_kv_pool=pool)
                    update = Mock()
                    updater = WeightUpdater(
                        tp_rank=0,
                        device="cpu",
                        gpu_id=0,
                        model_config=None,
                        custom_weight_loaders={},
                        get_model=lambda: model,
                        update_model_fields=update,
                        recapture_cuda_graph=lambda: None,
                        get_model_runner=lambda: runner,
                    )
                    prefix = "sglang.srt.model_executor.model_runner_components.weight_updater."
                    with (
                        patch(
                            prefix + "get_model",
                            return_value=SimpleNamespace(weight_cache_mode="off"),
                        ),
                        patch(
                            prefix + "get_disagg",
                            return_value=SimpleNamespace(disaggregation_mode=mode),
                        ),
                        patch(
                            prefix + "DefaultModelLoader.restore_weights_before_loading"
                        ) as restore,
                        patch.object(model, "load_weights", create=True) as load,
                    ):
                        with self.assertRaisesRegex(RuntimeError, "FP8 QSA PD"):
                            updater.begin_weight_update()
                        for success, message in (
                            updater.update_weights_from_disk("replacement", "auto"),
                            updater.update_weights_from_tensor(
                                [("weight", torch.zeros_like(original))],
                                load_format="direct",
                            ),
                            updater.update_weights_from_tensor(
                                {}, load_format="flattened_bucket"
                            ),
                            updater.update_weights_from_tensor(
                                [], load_format="custom-loader"
                            ),
                            updater.load_weights_from_distributed([]),
                            updater.update_weights_from_ipc(SimpleNamespace()),
                        ):
                            self.assertFalse(success)
                            self.assertIn("restart both P/D servers", message)
                        restore.assert_not_called()
                        load.assert_not_called()
                    update.assert_not_called()
                    self.assertTrue(torch.equal(model.weight, original))

    def test_qsa_update_guard_does_not_reject_bf16_or_non_pd(self):
        prefix = "sglang.srt.model_executor.model_runner_components.weight_updater."
        model = torch.nn.Module()
        for mode, dtype in (("null", torch.float8_e4m3fn), ("prefill", torch.bfloat16)):
            with (
                self.subTest(mode=mode, dtype=dtype),
                patch(
                    prefix + "get_disagg",
                    return_value=SimpleNamespace(disaggregation_mode=mode),
                ),
                patch(
                    "sglang.kernels.ops.gemm.bf16_fp32.hpc_bf16xfp32_gemm_enabled",
                    return_value=False,
                ),
            ):
                runner = SimpleNamespace(
                    token_to_kv_pool=SimpleNamespace(dtype=dtype, qsa_compress_ratio=4)
                )
                self.assertIsNone(
                    _unsupported_derived_weight_cache_error(model, runner)
                )

    def test_model_without_derived_cache_keeps_updates_enabled(self):
        model = torch.nn.Module()
        with patch(
            "sglang.kernels.ops.gemm.bf16_fp32.hpc_bf16xfp32_gemm_enabled",
            return_value=False,
        ):
            self.assertIsNone(_unsupported_derived_weight_cache_error(model))


if __name__ == "__main__":
    unittest.main()
