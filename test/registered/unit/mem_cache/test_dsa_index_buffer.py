"""CPU regression tests for DSA index-cache buffer allocation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import kv_cache_configurator
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSAIndexBuffer(unittest.TestCase):
    @staticmethod
    def _make_pool(use_fp4_indexer: bool, use_bf16_indexer: bool, dtype):
        pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
        pool.custom_mem_pool = None
        pool.device = "cpu"
        pool.index_buf_size = 65
        pool.page_size = 64
        pool.index_head_dim = 128
        pool.quant_block_size = 128
        pool.mxfp4_block_size = 32
        pool.layer_num = 1
        pool.use_fp4_indexer = use_fp4_indexer
        pool.use_bf16_indexer = use_bf16_indexer
        pool.index_k_with_scale_buffer_dtype = dtype
        return pool

    def test_allocates_each_supported_index_layout(self):
        cases = (
            ("fp8", False, False, torch.uint8, (2, 64 * (128 + 4))),
            ("bf16", False, True, torch.bfloat16, (2, 64 * 128)),
            ("fp4", True, False, torch.uint8, (2, 64 * (64 + 4))),
        )

        for name, use_fp4, use_bf16, dtype, expected_shape in cases:
            with self.subTest(name=name):
                pool = self._make_pool(use_fp4, use_bf16, dtype)
                pool._create_index_buffers()

                self.assertEqual(len(pool.index_k_with_scale_buffer), 1)
                buffer = pool.index_k_with_scale_buffer[0]
                self.assertEqual(tuple(buffer.shape), expected_shape)
                self.assertEqual(buffer.dtype, dtype)

    def test_configurator_limits_fp4_layout_to_ppu(self):
        configurator = KVCacheConfigurator.__new__(KVCacheConfigurator)
        configurator.server_args = SimpleNamespace(
            enable_hisparse=False,
            enable_memory_saver=False,
            page_size=64,
        )
        configurator.model_config = SimpleNamespace(
            hf_config=object(),
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        configurator.kv_cache_dtype = torch.bfloat16
        configurator.layer_info = SimpleNamespace(
            end_layer=2,
            num_effective_layers=2,
            start_layer=0,
        )
        configurator.device = "cpu"

        class FakePool:
            def __init__(self, size, **kwargs):
                self.size = size
                self.kwargs = kwargs

        for is_ppu, expected in ((True, True), (False, False)):
            with (
                self.subTest(is_ppu=is_ppu),
                patch.object(kv_cache_configurator, "DSATokenToKVPool", FakePool),
                patch.object(
                    kv_cache_configurator,
                    "calculate_mla_kv_cache_dim",
                    return_value=576,
                ),
                patch.object(
                    kv_cache_configurator,
                    "get_dsa_index_head_dim",
                    return_value=128,
                ),
                patch.object(
                    kv_cache_configurator.current_platform,
                    "is_ppu",
                    return_value=is_ppu,
                ),
                patch(
                    "sglang.kernels.ops.attention.dsa.triton_kernel.is_fp4_indexer_cache_enabled",
                    return_value=True,
                ),
                patch(
                    "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
                    return_value=(None, None),
                ),
            ):
                pool = configurator._build_dsa_kv_pool(max_total_num_tokens=1024)

            self.assertEqual(pool.size, 1024)
            self.assertIs(pool.kwargs["use_fp4_indexer"], expected)


if __name__ == "__main__":
    unittest.main()
