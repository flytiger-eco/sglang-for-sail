"""Dependency-light tests for opt-in AIM visual-weight policy."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[4]
_SPEC = importlib.util.spec_from_file_location(
    "aim_vision_policy_under_test",
    _ROOT / "python/sglang/srt/multimodal/aim_vision_policy.py",
)
policy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(policy)


class TestVisionQuantizationPolicy(unittest.TestCase):
    def test_explicit_fp8_overrides_separate_weight_default(self):
        self.assertEqual(
            policy.resolve_vision_quantization(
                "fp8",
                object(),
                separate_weights=True,
                inherit_model=True,
                create_online=lambda name: ("online", name),
            ),
            ("online", "fp8"),
        )

    def test_default_does_not_construct_online_quantization(self):
        self.assertIsNone(
            policy.resolve_vision_quantization(
                None,
                object(),
                separate_weights=False,
                inherit_model=False,
                create_online=lambda name: self.fail("Unexpected online factory"),
            )
        )

    def test_unsupported_online_quantization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Only explicit fp8"):
            policy.resolve_vision_quantization(
                "awq",
                object(),
                separate_weights=False,
                inherit_model=False,
                create_online=lambda name: ("online", name),
            )

    def test_separate_weights_do_not_inherit_language_quantization(self):
        self.assertIsNone(
            policy.resolve_vision_quantization(
                None,
                object(),
                separate_weights=True,
                inherit_model=True,
                create_online=lambda name: self.fail("Unexpected online factory"),
            )
        )

    def test_explicit_inheritance_reuses_selected_model_policy(self):
        model_quant = object()
        self.assertIs(
            policy.resolve_vision_quantization(
                None,
                model_quant,
                separate_weights=False,
                inherit_model=True,
                create_online=lambda name: self.fail("Unexpected online factory"),
            ),
            model_quant,
        )

    def test_explicit_fp8_uses_online_factory(self):
        self.assertEqual(
            policy.resolve_vision_quantization(
                "fp8",
                object(),
                separate_weights=False,
                inherit_model=False,
                create_online=lambda name: ("online", name),
            ),
            ("online", "fp8"),
        )

    def test_default_keeps_vision_unquantized(self):
        model_quant = object()
        self.assertIsNone(
            policy.resolve_vision_quantization(
                None,
                model_quant,
                separate_weights=False,
                inherit_model=False,
                create_online=lambda name: ("online", name),
            )
        )


class TestVisionConfiguration(unittest.TestCase):
    def test_separate_weights_reject_pipeline_parallel_but_other_modes_unchanged(self):
        with self.assertRaisesRegex(ValueError, "pp_size=1"):
            policy.validate_vision_options(None, "/vision", pp_size=2)
        policy.validate_vision_options(None, None, pp_size=2)
        policy.validate_vision_options("fp8", None, pp_size=2)
        policy.validate_vision_options(None, "/vision", pp_size=1)

    def test_invalid_scheme_and_empty_path_are_not_silently_ignored(self):
        for name, path in (("int8", None), (None, ""), (None, "  ")):
            with self.subTest(name=name, path=path), self.assertRaises(ValueError):
                policy.validate_vision_options(name, path)

    def test_overlap_weight_cache_and_disabled_tower_are_rejected(self):
        for option in (
            {"startup_mode": "overlap"},
            {"weight_cache_mode": "client"},
            {"enable_multimodal": False},
            {"model_impl": "transformers"},
            {"modelopt_quant": True},
            {"model_quantization": "mxfp8"},
        ):
            with self.subTest(option=option), self.assertRaises(ValueError):
                policy.validate_vision_options(None, "/vision", **option)

    def test_unsupported_loader_is_rejected_only_when_feature_is_requested(self):
        policy.validate_vision_options(None, None, load_format="gguf")
        with self.assertRaisesRegex(ValueError, "DefaultModelLoader"):
            policy.validate_vision_options("fp8", None, load_format="gguf")


class TestVisionExclusions(unittest.TestCase):
    def test_pattern_exclusions_fail_fast_without_mutating_original(self):
        for name in (
            "model.visual.*.attn.qkv",
            "re:visual.*",
            "visual.blocks.[0-9].attn.qkv",
        ):
            original = [name]
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "literal module paths"),
            ):
                policy.remap_vision_exclusions(
                    SimpleNamespace(ignored_layers=original), None
                )
            self.assertEqual(original, [name])

    def test_mapperless_qwen_manual_visual_names_are_still_respected(self):
        config = SimpleNamespace(ignored_layers=["model.visual.blocks.0.attn.qkv"])
        mapped = policy.remap_vision_exclusions(config, None)
        self.assertIn("visual.blocks.0.attn.qkv_proj", mapped.ignored_layers)
        self.assertIn("visual.blocks.0.attn.q_proj", mapped.ignored_layers)
        self.assertEqual(config.ignored_layers, ["model.visual.blocks.0.attn.qkv"])

    def test_both_exclusion_lists_are_idempotent(self):
        config = SimpleNamespace(
            ignored_layers=["visual.blocks.0.attn.qkv_proj"] * 2,
            modules_to_not_convert=[
                "visual.blocks.1.attn.qkv_proj",
                "visual.patch_embed",
            ],
        )
        mapped = policy.remap_vision_exclusions(config, None)
        self.assertIn("visual.patch_embed", mapped.modules_to_not_convert)
        self.assertEqual(len(mapped.ignored_layers), 4)
        self.assertEqual(len(mapped.modules_to_not_convert), 5)
        self.assertIs(policy.remap_vision_exclusions(mapped, None), mapped)

    def test_no_mapping_or_exclusions_keeps_identity(self):
        config = SimpleNamespace(ignored_layers=["visual.patch_embed"])
        self.assertIsNone(policy.remap_vision_exclusions(None, None))
        self.assertIs(policy.remap_vision_exclusions(config, None), config)

    def test_language_exclusions_are_not_expanded(self):
        config = SimpleNamespace(ignored_layers=["model.layers.0.attn.qkv_proj"])
        self.assertIs(policy.remap_vision_exclusions(config, None), config)

    def test_exclusion_expansion_does_not_mutate_language_policy(self):
        original = ["visual.blocks.0.attn.qkv_proj"]
        config = SimpleNamespace(ignored_layers=original)
        mapped = policy.remap_vision_exclusions(config, None)
        self.assertIsNot(mapped, config)
        self.assertEqual(original, ["visual.blocks.0.attn.qkv_proj"])
        self.assertEqual(
            mapped.ignored_layers,
            [
                "visual.blocks.0.attn.qkv_proj",
                "visual.blocks.0.attn.q_proj",
                "visual.blocks.0.attn.k_proj",
                "visual.blocks.0.attn.v_proj",
            ],
        )


class TestSeparateVisionWeights(unittest.TestCase):
    def test_incomplete_or_unknown_visual_weights_fail_before_success(self):
        names = {"visual.patch_embed.proj.weight", "visual.blocks.0.norm1.weight"}
        with self.assertRaisesRegex(ValueError, "missing visual tensors"):
            list(
                policy.replace_vision_weights(
                    [],
                    [("patch_embed.proj.weight", 1)],
                    expected_names=names,
                    required_names=names,
                )
            )
        with self.assertRaisesRegex(ValueError, "Unknown visual tensor"):
            list(
                policy.replace_vision_weights(
                    [],
                    [("visual.unexpected.weight", 1)],
                    expected_names=names,
                    required_names=names,
                )
            )

    def test_duplicate_visual_aliases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate visual tensor"):
            list(
                policy.replace_vision_weights(
                    [],
                    [
                        ("visual.blocks.0.attn.qkv.weight", 1),
                        ("model.visual.blocks.0.attn.qkv.weight", 2),
                    ],
                )
            )

    def test_supported_bare_and_prefixed_names_use_actual_qwen_namespace(self):
        for source in (
            "blocks.0.attn.qkv.weight",
            "visual.blocks.0.attn.qkv.weight",
            "model.visual.blocks.0.attn.qkv.weight",
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    list(policy.replace_vision_weights([], [(source, 1)])),
                    [("visual.blocks.0.attn.qkv.weight", 1)],
                )

    def test_empty_visual_source_is_not_a_successful_load(self):
        with self.assertRaisesRegex(ValueError, "no visual tensors"):
            list(policy.replace_vision_weights([], [("model.layers.0.weight", 1)]))

    def test_replacement_keeps_language_weights_once_and_filters_vision(self):
        primary = iter(
            [
                ("model.layers.0.weight", "language"),
                ("model.visual.patch_embed.proj.weight", "old-vision"),
            ]
        )
        replacement = iter(
            [
                ("model.layers.0.weight", "wrong-language"),
                ("visual.patch_embed.proj.weight", "new-vision"),
            ]
        )
        self.assertEqual(
            list(policy.replace_vision_weights(primary, replacement)),
            [
                ("model.layers.0.weight", "language"),
                ("visual.patch_embed.proj.weight", "new-vision"),
            ],
        )


class TestOnlineVisionSource(unittest.TestCase):
    def test_online_guard_preserves_unquantized_vision_and_quantized_language(self):
        vision = SimpleNamespace(dtype="torch.bfloat16")
        language = SimpleNamespace(dtype="torch.float8_e4m3fn")
        inputs = [
            ("visual.blocks.0.attn.qkv.weight", vision),
            ("model.layers.0.weight", language),
        ]
        self.assertEqual(list(policy.validate_online_vision_weights(inputs)), inputs)

    def test_replacement_bare_visual_prefix_is_checked(self):
        tensor = SimpleNamespace(dtype="torch.float8_e5m2")
        with self.assertRaisesRegex(ValueError, "unquantized visual"):
            list(
                policy.validate_online_vision_weights(
                    [("blocks.0.attn.qkv.weight", tensor)], allow_bare=True
                )
            )

    def test_online_quantization_rejects_serialized_scale_metadata(self):
        tensor = SimpleNamespace(dtype="torch.float32")
        for suffix in ("weight_scale", "weight_scale_inv", "input_scale", "k_scale"):
            with (
                self.subTest(suffix=suffix),
                self.assertRaisesRegex(ValueError, "scale"),
            ):
                list(
                    policy.validate_online_vision_weights(
                        [("visual.blocks.0.attn.qkv." + suffix, tensor)]
                    )
                )

    def test_online_quantization_rejects_serialized_float8_visual_weights(self):
        tensor = SimpleNamespace(dtype="torch.float8_e4m3fn")
        with self.assertRaisesRegex(ValueError, "unquantized visual"):
            list(
                policy.validate_online_vision_weights(
                    [("model.visual.blocks.0.attn.qkv.weight", tensor)]
                )
            )


if __name__ == "__main__":
    unittest.main()
