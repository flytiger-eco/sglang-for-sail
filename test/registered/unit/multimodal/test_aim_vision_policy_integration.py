"""AIM vision policy exercised with real SGLang config/loader and CPU tensors."""

import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.model_loader.loader import DefaultModelLoader, _validate_aim_vision_load
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.multimodal.aim_vision_policy import (
    remap_vision_exclusions,
    replace_vision_weights,
)
from sglang.srt.runtime_context import get_context, get_mm, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestVisionConfigurationIntegration(unittest.TestCase):
    def test_separate_source_rejects_pipeline_parallel_before_loading(self):
        with self.assertRaisesRegex(ValueError, "pp_size=1"):
            ServerArgs(model_path="dummy", vision_model_path="/vision", pp_size=2).resolve_once()
        ServerArgs(model_path="dummy", pp_size=2)
        ServerArgs(model_path="dummy", vision_quantization="fp8", pp_size=2)

    def test_options_publish_in_current_multimodal_namespace(self):
        with get_context().override_server_args(
            vision_quantization="fp8", vision_model_path="/vision"
        ):
            self.assertEqual(get_mm().vision_quantization, "fp8")
            self.assertEqual(get_mm().vision_model_path, "/vision")

    def test_unsupported_mode_rejected_even_for_dummy_config(self):
        with self.assertRaisesRegex(ValueError, "DefaultModelLoader"):
            ServerArgs(
                model_path="dummy", vision_model_path="/vision", load_format="gguf"
            ).resolve_once()


class TestExclusionMapperIntegration(unittest.TestCase):
    def test_mapper_drop_is_not_silently_accepted(self):
        mapper = WeightsMapper(orig_to_new_prefix={"model.visual.": None})
        with self.assertRaisesRegex(ValueError, "would drop"):
            remap_vision_exclusions(
                SimpleNamespace(ignored_layers=["model.visual.blocks.0.attn.qkv"]),
                mapper,
            )

    def test_incompatible_patterns_are_rejected_without_mutating_config(self):
        config = SimpleNamespace(ignored_layers=["model.visual.*.attn.qkv"])
        mapper = WeightsMapper(orig_to_new_substr={"attn.qkv": "attn.qkv_proj"})
        with self.assertRaisesRegex(ValueError, "literal module paths"):
            remap_vision_exclusions(config, mapper)
        self.assertEqual(config.ignored_layers, ["model.visual.*.attn.qkv"])

    def test_real_mapper_is_idempotent_and_does_not_mutate_language_config(self):
        mapper = WeightsMapper(
            orig_to_new_substr={"attn.qkv": "attn.qkv_proj"},
            orig_to_new_prefix={"model.visual.": "visual."},
        )
        config = SimpleNamespace(ignored_layers=["model.visual.blocks.0.attn.qkv"])
        mapped = remap_vision_exclusions(config, mapper)
        self.assertEqual(config.ignored_layers, ["model.visual.blocks.0.attn.qkv"])
        self.assertIn("visual.blocks.0.attn.q_proj", mapped.ignored_layers)
        self.assertTrue(
            is_layer_skipped(
                "model.visual.blocks.0.attn.qkv_proj", mapped.ignored_layers
            )
        )
        self.assertIs(remap_vision_exclusions(mapped, mapper), mapped)


class TestRealQwenWeightMappings(unittest.TestCase):
    def test_replacement_survives_actual_qwen3vl_class_mapper(self):
        from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration

        mapper = Qwen3VLForConditionalGeneration.hf_to_sglang_mapper
        tensor = torch.ones(3, 1)
        target_name = "visual.blocks.0.attn.qkv_proj.weight"
        for source_name in (
            "model.visual.blocks.0.attn.qkv.weight",
            "visual.blocks.0.attn.qkv_proj.weight",
            "blocks.0.attn.qkv.weight",
        ):
            with self.subTest(source_name=source_name):
                weights = replace_vision_weights(
                    [],
                    [(source_name, tensor)],
                    expected_names={target_name},
                    required_names={target_name},
                )
                mapped = list(mapper.apply(weights))
                self.assertEqual([name for name, _ in mapped], [target_name])
                self.assertIs(mapped[0][1], tensor)

    def test_actual_model_loaders_load_qkv_replacement_once(self):
        from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
        from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

        for model_class in (
            Qwen3VLForConditionalGeneration,
            Qwen3_5ForConditionalGeneration,
            Qwen4ExpForConditionalGeneration,
        ):
            with self.subTest(model_class=model_class.__name__):
                # Exercise the real load_weights implementation without building
                # the GPU forward graph or allocating a full language model.
                model = model_class.__new__(model_class)
                torch.nn.Module.__init__(model)
                model.config = SimpleNamespace(
                    tie_word_embeddings=False, encoder_only=False, language_only=False
                )
                model.pp_group = SimpleNamespace(is_last_rank=False)
                model.language_model_only = False
                model.visual = torch.nn.Module()
                block = torch.nn.Module()
                block.attn = torch.nn.Module()
                block.attn.qkv_proj = torch.nn.Linear(1, 3, bias=False)
                model.visual.blocks = torch.nn.ModuleList([block])
                expected = torch.full((3, 1), 7.0)
                name = "visual.blocks.0.attn.qkv_proj.weight"
                with torch.no_grad():
                    model.load_weights(
                        replace_vision_weights(
                            [
                                (
                                    "model.visual.blocks.0.attn.qkv.weight",
                                    torch.zeros(3, 1),
                                )
                            ],
                            [("model.visual.blocks.0.attn.qkv.weight", expected)],
                            expected_names={name},
                            required_names={name},
                        )
                    )
                torch.testing.assert_close(block.attn.qkv_proj.weight, expected)


class TinyVisualModel(torch.nn.Module):
    """Small model exercising the real loader/post-load interface on CPU."""

    supports_aim_vision_policy = True

    def __init__(self):
        super().__init__()
        self.language = torch.nn.Linear(1, 1, bias=False)
        self.visual = torch.nn.Linear(1, 1, bias=False)
        self.loaded_names = []
        self.post_load_values = []
        self.visual.quant_method = SimpleNamespace(
            process_weights_after_loading=self.process_weights_after_loading
        )

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        with torch.no_grad():
            for name, tensor in weights:
                self.loaded_names.append(name)
                parameters[name].copy_(tensor)

    def process_weights_after_loading(self, module):
        self.post_load_values.append(module.weight.item())


class TestSeparateVisionLoaderIntegration(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.contexts.enter_context(get_parallel().override(pp_size=1))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.primary = Path(self.directory.name, "primary")
        self.vision = Path(self.directory.name, "vision")
        self.primary.mkdir()
        self.vision.mkdir()
        save_file(
            {
                "language.weight": torch.tensor([[3.0]]),
                "model.visual.weight": torch.tensor([[4.0]]),
            },
            self.primary / "model.safetensors",
        )
        save_file(
            {
                "language.weight": torch.tensor([[98.0]]),
                "model.visual.weight": torch.tensor([[7.0]]),
            },
            self.vision / "model.safetensors",
        )
        self.config = SimpleNamespace(
            model_path=str(self.primary),
            revision=None,
            is_draft_model=False,
            hf_config=SimpleNamespace(architectures=["TinyVisualModel"]),
            quantization=None,
        )
        self.loader = DefaultModelLoader(
            LoadConfig(
                load_format="safetensors",
                model_loader_extra_config={"enable_multithread_load": False},
            )
        )

    def test_separate_source_rejects_resolved_pipeline_parallel_in_loader(self):
        with (
            get_context().override_server_args(vision_model_path=str(self.vision)),
            get_parallel().override(pp_size=2),
        ):
            with self.assertRaisesRegex(ValueError, "pp_size=1"):
                list(self.loader._get_all_weights(self.config, TinyVisualModel()))

    def test_online_fp8_rejects_serialized_main_visual_source(self):
        for visual in (
            {"model.visual.weight": torch.tensor([[2.0]]).to(torch.float8_e4m3fn)},
            {
                "model.visual.weight": torch.tensor([[2.0]]),
                "model.visual.weight_scale": torch.tensor(0.25),
            },
        ):
            with self.subTest(tensors=list(visual)):
                save_file(
                    {"language.weight": torch.tensor([[3.0]]), **visual},
                    self.primary / "model.safetensors",
                )
                with get_context().override_server_args(vision_quantization="fp8"):
                    with self.assertRaisesRegex(ValueError, "Online FP8 vision"):
                        list(
                            self.loader._get_all_weights(self.config, TinyVisualModel())
                        )

    def test_online_fp8_rejects_serialized_separate_visual_source(self):
        for visual in (
            {"model.visual.weight": torch.tensor([[2.0]]).to(torch.float8_e4m3fn)},
            {
                "model.visual.weight": torch.tensor([[2.0]]),
                "model.visual.weight_scale": torch.tensor(0.25),
            },
        ):
            with self.subTest(tensors=list(visual)):
                save_file(visual, self.vision / "model.safetensors")
                model = TinyVisualModel()
                with get_context().override_server_args(
                    vision_quantization="fp8", vision_model_path=str(self.vision)
                ):
                    with self.assertRaisesRegex(ValueError, "Online FP8 vision"):
                        self.loader.load_weights_and_postprocess(
                            model,
                            self.loader._get_all_weights(self.config, model),
                            torch.device("cpu"),
                        )
                self.assertEqual(model.post_load_values, [])

    def test_serialized_main_source_is_unchanged_without_explicit_online_policy(self):
        save_file(
            {
                "model.visual.weight": torch.tensor([[2.0]]).to(torch.float8_e4m3fn),
                "model.visual.weight_scale": torch.tensor(0.25),
            },
            self.primary / "model.safetensors",
        )
        for inherited in ("0", "1"):
            with (
                self.subTest(inherited=inherited),
                patch.dict(os.environ, {"SGLANG_QUANTIZE_VIT": inherited}),
                get_context().override_server_args(),
            ):
                weights = dict(
                    self.loader._get_all_weights(self.config, TinyVisualModel())
                )
                self.assertEqual(
                    weights["model.visual.weight"].dtype, torch.float8_e4m3fn
                )
                self.assertEqual(weights["model.visual.weight_scale"].item(), 0.25)

    def test_online_fp8_allows_float_vision_and_preserves_quantized_language(self):
        save_file(
            {
                "language.weight": torch.tensor([[3.0]]).to(torch.float8_e4m3fn),
                "model.visual.weight": torch.tensor([[4.0]], dtype=torch.bfloat16),
            },
            self.primary / "model.safetensors",
        )
        with get_context().override_server_args(vision_quantization="fp8"):
            weights = dict(self.loader._get_all_weights(self.config, TinyVisualModel()))
        self.assertEqual(weights["language.weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(weights["model.visual.weight"].dtype, torch.bfloat16)

    def test_online_fp8_checks_only_selected_separate_visual_source(self):
        save_file(
            {
                "language.weight": torch.tensor([[3.0]]).to(torch.float8_e4m3fn),
                "model.visual.weight": torch.tensor([[2.0]]).to(torch.float8_e4m3fn),
                "model.visual.weight_scale": torch.tensor(0.25),
            },
            self.primary / "model.safetensors",
        )
        with get_context().override_server_args(
            vision_quantization="fp8", vision_model_path=str(self.vision)
        ):
            weights = dict(self.loader._get_all_weights(self.config, TinyVisualModel()))
        self.assertEqual(weights["language.weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(weights["visual.weight"].item(), 7.0)
        self.assertNotIn("model.visual.weight_scale", weights)

    def test_real_disk_iterator_loads_replacement_before_existing_postprocess(self):
        with get_context().override_server_args(vision_model_path=str(self.vision)):
            model = TinyVisualModel()
            self.loader.load_weights_and_postprocess(
                model,
                self.loader._get_all_weights(self.config, model),
                torch.device("cpu"),
            )
        self.assertEqual(model.language.weight.item(), 3.0)
        self.assertEqual(model.visual.weight.item(), 7.0)
        self.assertEqual(model.loaded_names, ["language.weight", "visual.weight"])
        self.assertEqual(model.post_load_values, [7.0])

    def test_incomplete_replacement_prevents_postprocessing(self):
        save_file(
            {"language.weight": torch.tensor([[98.0]])},
            self.vision / "model.safetensors",
        )
        model = TinyVisualModel()
        with get_context().override_server_args(vision_model_path=str(self.vision)):
            with self.assertRaisesRegex(ValueError, "no visual tensors"):
                self.loader.load_weights_and_postprocess(
                    model,
                    self.loader._get_all_weights(self.config, model),
                    torch.device("cpu"),
                )
        self.assertEqual(model.post_load_values, [])

    def test_draft_weights_are_not_replaced_by_target_visual_checkpoint(self):
        self.config.is_draft_model = True
        with get_context().override_server_args(vision_model_path=str(self.vision)):
            weights = dict(self.loader._get_all_weights(self.config, TinyVisualModel()))
        self.assertEqual(weights["model.visual.weight"].item(), 4.0)
        self.assertNotIn("visual.weight", weights)

    def test_no_visual_tower_and_unsupported_model_fail_early(self):
        with get_context().override_server_args(vision_model_path=str(self.vision)):
            with self.assertRaisesRegex(ValueError, "Qwen3VL"):
                _validate_aim_vision_load(
                    self.config, self.loader.load_config, torch.nn.Linear
                )
            model = TinyVisualModel()
            model.visual = None
            with self.assertRaisesRegex(ValueError, "initialized visual tower"):
                list(self.loader._get_all_weights(self.config, model))


if __name__ == "__main__":
    unittest.main()
