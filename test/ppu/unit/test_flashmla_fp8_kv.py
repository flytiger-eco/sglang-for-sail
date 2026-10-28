"""CPU regressions for the PPU FlashMLA 2.1 FP8 integration.

Kernel numerics and full inference are covered separately on PPU hardware.
"""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_function(path, name, globals_):
    """Load a real method/pass without importing unrelated device backends."""
    tree = ast.parse((ROOT / path).read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    scope = dict(globals_)
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


class TestPpuFlashMlaFp8(unittest.TestCase):
    def setUp(self):
        lib = ModuleType("flash_mla")
        lib.FlashMLASchedMeta = type("FlashMLASchedMeta", (), {})
        lib.get_mla_metadata = Mock(
            side_effect=lambda *a, **k: (lib.FlashMLASchedMeta(), None)
        )
        lib.flash_mla_with_kvcache = Mock(return_value=("out", "lse"))
        lib.flash_mla_sparse_fwd = Mock()
        environ = ModuleType("sglang.srt.environ")
        environ.envs = SimpleNamespace(
            SGLANG_PROFILE_NVTX=SimpleNamespace(get=lambda: False),
            SGLANG_PROFILE_NVTX_PRINT_SEQLEN=SimpleNamespace(get=lambda: False),
        )
        with patch.dict(sys.modules, {"flash_mla": lib, "sglang.srt.environ": environ}):
            spec = importlib.util.spec_from_file_location(
                "ppu_flashmla_under_test",
                ROOT / "python/sglang/srt/hardware_backend/ppu/attention/flash_mla.py",
            )
            self.ops = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.ops)
        self.lib = lib
        self.capability = patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 9)
        )
        self.capability.start()
        self.addCleanup(self.capability.stop)

    def call(self, q=None, kv=None, **kwargs):
        return self.ops.flash_mla_with_kvcache(
            q=q if q is not None else torch.zeros(1, 1, 8, 576).to(torch.float8_e4m3fn),
            k_cache=(
                kv
                if kv is not None
                else torch.zeros(1, 64, 1, 576).to(torch.float8_e4m3fn)
            ),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            tile_scheduler_metadata=self.lib.FlashMLASchedMeta(),
            **kwargs,
        )

    def test_dense_fp8_reaches_latest_library(self):
        self.assertEqual(self.call(), ("out", "lse"))
        kwargs = self.lib.flash_mla_with_kvcache.call_args.kwargs
        self.assertNotIn("descale_q", kwargs)
        self.assertNotIn("descale_k", kwargs)

    def test_fp8_metadata_is_lazy(self):
        meta, splits = self.ops.get_mla_metadata(
            torch.tensor([64]), 8, 1, is_fp8_kvcache=True
        )
        self.assertIsInstance(meta, self.lib.FlashMLASchedMeta)
        self.assertIsNone(splits)

    def test_dense_unsupported_contracts_fail_before_library(self):
        for capability in ((8, 0), (9, 0)):
            with self.subTest(capability=capability), patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ), self.assertRaisesRegex(ValueError, "SM89"):
                self.call()
        with self.assertRaisesRegex(ValueError, "matching"):
            self.call(q=torch.zeros(1, 1, 8, 576).to(torch.float8_e5m2))
        with self.assertRaisesRegex(ValueError, "page size"):
            self.call(kv=torch.zeros(1, 16, 1, 576).to(torch.float8_e4m3fn))
        with self.assertRaisesRegex(ValueError, "descale"):
            self.call(descale_q=torch.tensor([2.0]))
        with self.assertRaisesRegex(ValueError, "DCP"):
            self.ops.validate_dense_fp8_config(
                torch.float8_e4m3fn, 576, 512, 64, "cuda", 2
            )
        self.lib.flash_mla_with_kvcache.assert_not_called()

    def test_sparse_bf16_query_and_packed_cache_pass_through(self):
        self.call(
            q=torch.zeros(1, 1, 8, 576, dtype=torch.bfloat16),
            kv=torch.zeros(1, 64, 1, 656, dtype=torch.uint8),
            indices=torch.zeros(1, 1, 64, dtype=torch.int32),
            is_fp8_kvcache=True,
        )
        self.lib.flash_mla_with_kvcache.assert_called_once()

    def test_ppu_query_matches_raw_cache_even_with_nonunit_checkpoint_scale(self):
        quant = Mock(
            side_effect=AssertionError("PPU must not use scaled FP8 quantization")
        )
        prepare = load_function(
            "python/sglang/srt/layers/attention/flashmla_backend.py",
            "_prepare_fp8_query",
            {"torch": torch, "is_ppu": lambda: True, "scaled_fp8_quant": quant},
        )
        q = torch.tensor([0.5, 1.0, -2.0, 4.0]).reshape(1, 1, 1, 4)
        result, dq, dk = prepare(None, q, SimpleNamespace(k_scale=torch.tensor([4.0])))
        torch.testing.assert_close(result.float(), q)
        self.assertIsNone(dq)
        self.assertIsNone(dk)
        quant.assert_not_called()

    def test_non_ppu_keeps_checkpoint_scale(self):
        scale = torch.tensor([4.0])
        quant = Mock(side_effect=lambda q, s: ((q / s).to(torch.float8_e4m3fn), s))
        prepare = load_function(
            "python/sglang/srt/layers/attention/flashmla_backend.py",
            "_prepare_fp8_query",
            {"torch": torch, "is_ppu": lambda: False, "scaled_fp8_quant": quant},
        )
        q = torch.tensor([0.5, 1.0, -2.0, 4.0]).reshape(1, 1, 1, 4)
        result, dq, dk = prepare(None, q, SimpleNamespace(k_scale=scale))
        torch.testing.assert_close(result.float() * dq, q)
        torch.testing.assert_close(dk, scale)
        quant.assert_called_once()

    def test_fp8_graph_metadata_uses_lazy_objects_without_cpu_seq_lens(self):
        hook = load_function(
            "python/sglang/srt/hardware_backend/ppu/attention/ppu_flashmla_hooks.py",
            "_ppu_flashmla_apply_decode_target_verify_metadata",
            {"torch": torch},
        )
        launch = Mock()

        # A minimal launchable kernel boundary; all metadata tensor work stays real.
        class Kernel:
            def __getitem__(self, grid):
                return launch

        modules = {}
        for name, attrs in {
            "triton": {"cdiv": lambda a, b: (a + b - 1) // b},
            "sglang.kernels.ops.attention.utils": {
                "create_flashmla_kv_indices_triton": Kernel(),
                "get_num_kv_index_blocks_flashmla": lambda width, page: 1,
            },
            "sglang.srt.hardware_backend.ppu.attention.flash_mla": {
                "get_mla_metadata": self.ops.get_mla_metadata,
            },
            "sglang.srt.layers.attention.flashmla_backend": {
                "PAGE_SIZE": 64,
                "FlashMLADecodeMetadata": lambda meta, splits, table, lens: SimpleNamespace(
                    flashmla_metadata=meta,
                    num_splits=splits,
                    block_kv_indices=table,
                    seq_lens_k=lens,
                ),
            },
        }.items():
            modules[name] = ModuleType(name)
            modules[name].__dict__.update(attrs)
        backend = SimpleNamespace(
            req_to_token=torch.zeros(2, 512, dtype=torch.int32),
            cuda_graph_kv_indices=torch.zeros(2, 8, dtype=torch.int32),
            cuda_graph_draft_extend_seq_lens_k=torch.zeros(2, dtype=torch.int32),
            num_draft_tokens=3,
            num_q_heads=8,
            is_fp8_kvcache=True,
        )
        for verify, draft, expected in (
            (False, False, [65, 70]),
            (True, False, [68, 73]),
            (False, True, [65, 70]),
        ):
            mode = SimpleNamespace(
                is_target_verify=lambda: verify, is_draft_extend_v2=lambda: draft
            )
            with self.subTest(verify=verify, draft=draft), patch.dict(
                sys.modules, modules
            ):
                hook(
                    backend, 2, torch.tensor([0, 1]), torch.tensor([65, 70]), None, mode
                )
            metadata = backend.forward_metadata
            self.assertIsInstance(
                metadata.flashmla_metadata, self.lib.FlashMLASchedMeta
            )
            self.assertIsNone(metadata.num_splits)
            self.assertEqual(metadata.block_kv_indices.shape, (2, 8))
            self.assertEqual(metadata.seq_lens_k.tolist(), expected)
        self.assertEqual(launch.call_count, 3)

    def test_dense_geometry_and_e5m2_are_rejected(self):
        for dtype, dim, vdim, error in (
            (torch.float8_e5m2, 576, 512, "E4M3"),
            (torch.float8_e4m3fn, 512, 512, "dim 576"),
            (torch.float8_e4m3fn, 576, 256, "V dim 512"),
        ):
            with self.subTest(dtype=dtype, dim=dim, vdim=vdim), self.assertRaisesRegex(
                ValueError, error
            ):
                self.ops.validate_dense_fp8_config(dtype, dim, vdim, 64, "cuda")

    def test_bf16_decode_remains_available(self):
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
            self.assertEqual(
                self.call(
                    q=torch.zeros(1, 1, 8, 576, dtype=torch.bfloat16),
                    kv=torch.zeros(1, 64, 1, 576, dtype=torch.bfloat16),
                ),
                ("out", "lse"),
            )

    def test_sparse_extra_cache_and_lengths_are_forwarded(self):
        extra = torch.zeros(1, 64, 1, 584, dtype=torch.uint8)
        indices = torch.zeros(1, 1, 64, dtype=torch.int32)
        lengths = torch.tensor([32], dtype=torch.int32)
        sink = torch.zeros(8)
        self.call(
            q=torch.zeros(1, 1, 8, 512, dtype=torch.bfloat16),
            kv=torch.zeros(1, 64, 1, 584, dtype=torch.uint8),
            indices=indices,
            extra_k_cache=extra,
            extra_indices_in_kvcache=indices,
            topk_length=lengths,
            extra_topk_length=lengths,
            attn_sink=sink,
        )
        kwargs = self.lib.flash_mla_with_kvcache.call_args.kwargs
        for key, value in (
            ("extra_k_cache", extra),
            ("extra_indices_in_kvcache", indices),
            ("topk_length", lengths),
            ("extra_topk_length", lengths),
            ("attn_sink", sink),
        ):
            self.assertIs(kwargs[key], value)

    def test_resolved_split_backends_and_non_ppu_are_not_overrestricted(self):
        path = "python/sglang/srt/arg_groups/overrides.py"
        backends = load_function(path, "attention_backends_of", {"Any": object})
        config = SimpleNamespace(
            kv_cache_dtype="fp8_e5m2",
            attention_backend="flashmla",
            prefill_attention_backend="fa3",
            decode_attention_backend="fa3",
            dsa_prefill_backend=None,
            dsa_decode_backend=None,
            dcp_size=1,
        )
        check = load_function(
            path,
            "_ppu_flashmla_kv_cache_checks",
            {
                "Any": object,
                "is_ppu": lambda: True,
                "attention_backends_of": backends,
            },
        )
        self.assertEqual(check(config), {})
        config.decode_attention_backend = "flashmla"
        with self.assertRaisesRegex(ValueError, "fp8_e4m3"):
            check(config)
        config.kv_cache_dtype = "fp8_e4m3"
        config.dcp_size = 2
        with self.assertRaisesRegex(ValueError, "DCP"):
            check(config)
        non_ppu_check = load_function(
            path,
            "_ppu_flashmla_kv_cache_checks",
            {
                "Any": object,
                "is_ppu": lambda: False,
                "attention_backends_of": backends,
            },
        )
        self.assertEqual(non_ppu_check(config), {})
        config.kv_cache_dtype = "bfloat16"
        self.assertEqual(check(config), {})

    def test_serial_weight_loading_is_opt_in(self):
        for disabled in (False, True):
            with self.subTest(disabled=disabled):
                should_async = load_function(
                    "python/sglang/srt/model_loader/utils.py",
                    "should_async_load",
                    {
                        "torch": torch,
                        "envs": SimpleNamespace(
                            SGLANG_DISABLE_ASYNC_WEIGHT_LOADING=SimpleNamespace(
                                get=lambda: disabled
                            )
                        ),
                    },
                )
                self.assertEqual(should_async(torch.ones(2)), not disabled)

    def test_serial_loading_keeps_device_and_streamer_weights_synchronous(self):
        should_async = load_function(
            "python/sglang/srt/model_loader/utils.py",
            "should_async_load",
            {
                "torch": torch,
                "envs": SimpleNamespace(
                    SGLANG_DISABLE_ASYNC_WEIGHT_LOADING=SimpleNamespace(
                        get=lambda: False
                    )
                ),
            },
        )
        self.assertFalse(
            should_async(SimpleNamespace(device=SimpleNamespace(type="cuda")))
        )
        self.assertFalse(
            should_async(
                SimpleNamespace(
                    device=SimpleNamespace(type="cpu"),
                    _sglang_runai_streamer_tensor=True,
                )
            )
        )

    def test_ppu_bf16_ragged_topk_widens_only_at_kernel_boundary(self):
        for ppu, dtype in (
            (True, torch.bfloat16),
            (True, torch.float32),
            (False, torch.bfloat16),
        ):
            with self.subTest(ppu=ppu, dtype=dtype):
                env = ModuleType("sglang.srt.environ")
                env.envs = SimpleNamespace(
                    SGLANG_DSA_FUSE_TOPK=SimpleNamespace(get=lambda: True),
                    SGLANG_OPT_USE_TOPK_V2=SimpleNamespace(get=lambda: False),
                )
                utils = ModuleType("sglang.srt.utils")
                utils.is_ppu = lambda: ppu
                kernel = ModuleType("sgl_kernel")
                kernel.fast_topk_transform_fused = Mock()
                kernel.fast_topk_transform_ragged_fused = Mock(return_value="indices")
                spec = importlib.util.spec_from_file_location(
                    "ragged_topk_under_test",
                    ROOT / "python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py",
                )
                mod = importlib.util.module_from_spec(spec)
                with patch.dict(
                    sys.modules,
                    {
                        "sglang.srt.environ": env,
                        "sglang.srt.utils": utils,
                        "sgl_kernel": kernel,
                    },
                ):
                    spec.loader.exec_module(mod)
                    scores = torch.tensor([[1.0, 3.0, 2.0, 0.0]], dtype=dtype)
                    offsets = torch.tensor([7], dtype=torch.int32)
                    result = mod.DSATopKBackend.SGL_KERNEL.topk_transform(
                        scores,
                        torch.tensor([4], dtype=torch.int32),
                        2,
                        mod.TopkTransformMethod.RAGGED,
                        SimpleNamespace(page_table_1=object()),
                        topk_indices_offset=offsets,
                    )
                self.assertEqual(result, "indices")
                actual = kernel.fast_topk_transform_ragged_fused.call_args.kwargs[
                    "score"
                ]
                if ppu and dtype == torch.bfloat16:
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, scores.float())
                else:
                    self.assertIs(actual, scores)
                self.assertIs(
                    kernel.fast_topk_transform_ragged_fused.call_args.kwargs[
                        "topk_indices_offset"
                    ],
                    offsets,
                )

    def test_resolved_config_checks_preserve_supported_fp8(self):
        def backends(v):
            return (
                v.prefill_attention_backend or v.attention_backend,
                v.decode_attention_backend or v.attention_backend,
            )

        check = load_function(
            "python/sglang/srt/arg_groups/overrides.py",
            "_ppu_flashmla_kv_cache_checks",
            {"Any": object, "is_ppu": lambda: True, "attention_backends_of": backends},
        )
        config = SimpleNamespace(
            kv_cache_dtype="fp8_e4m3",
            attention_backend="flashmla",
            prefill_attention_backend=None,
            decode_attention_backend=None,
            dsa_prefill_backend=None,
            dsa_decode_backend=None,
            dcp_size=1,
        )
        self.assertEqual(check(config), {})
        config.attention_backend = "dsa"
        config.dsa_prefill_backend = "flashmla_sparse"
        config.dsa_decode_backend = "flashmla_kv"
        self.assertEqual(check(config), {})
        self.assertEqual(config.kv_cache_dtype, "fp8_e4m3")
        config.dsa_decode_backend = "flashmla_sparse"
        with self.assertRaisesRegex(ValueError, "flashmla_kv"):
            check(config)
        config.dsa_decode_backend = None
        config.attention_backend = "flashmla"
        config.kv_cache_dtype = "fp8_e5m2"
        with self.assertRaisesRegex(ValueError, "fp8_e4m3"):
            check(config)


if __name__ == "__main__":
    unittest.main()
