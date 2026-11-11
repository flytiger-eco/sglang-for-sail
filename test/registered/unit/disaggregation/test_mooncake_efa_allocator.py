import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from sglang.srt.arg_groups.overrides import post_capture_kv_sizing_planned
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.mooncake.utils import (
    _validate_mooncake_allocator_compatibility,
    check_mooncake_custom_mem_pool_enabled,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeEfaAllocator(unittest.TestCase):
    def _check(self, protocol, custom_mem_pool=None, allocator_env=None):
        allocator_env = allocator_env or {}
        with (
            patch.object(envs.MOONCAKE_PROTOCOL, "get", return_value=protocol),
            patch.object(
                envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL,
                "get",
                return_value=custom_mem_pool,
            ),
            patch(
                "sglang.srt.disaggregation.mooncake.utils.os.environ.get",
                side_effect=lambda key, default="": allocator_env.get(key, default),
            ),
        ):
            result = check_mooncake_custom_mem_pool_enabled()
            _validate_mooncake_allocator_compatibility(*result)
            return result

    def test_efa_uses_default_allocator(self):
        self.assertEqual(self._check("efa"), (False, None))

    def test_efa_rejects_custom_memory_pools(self):
        for pool_type in ("true", "NVLINK", "BAREX"):
            with self.subTest(pool_type=pool_type):
                with self.assertRaisesRegex(
                    ValueError, "incompatible with MOONCAKE_PROTOCOL=efa"
                ):
                    self._check("efa", custom_mem_pool=pool_type)

    def test_efa_allows_intra_node_nvlink(self):
        self.assertEqual(
            self._check("efa", custom_mem_pool="INTRA_NODE_NVLINK"),
            (True, "INTRA_NODE_NVLINK"),
        )

    def test_efa_rejects_expandable_segments(self):
        for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            with self.subTest(var=var):
                with self.assertRaisesRegex(ValueError, var):
                    self._check(
                        "efa",
                        allocator_env={
                            var: "garbage_collection_threshold:0.8,"
                            "expandable_segments:True"
                        },
                    )

    def test_efa_allows_disabled_expandable_segments(self):
        self.assertEqual(
            self._check(
                "EFA",
                allocator_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"},
            ),
            (False, None),
        )

    def test_non_efa_keeps_custom_memory_pool_behavior(self):
        self.assertEqual(self._check("rdma", custom_mem_pool="true"), (True, "NVLINK"))

    def test_rdma_rejects_expandable_segments(self):
        for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            with self.subTest(var=var):
                with self.assertRaisesRegex(
                    ValueError, f"{var}.*MOONCAKE_PROTOCOL=rdma"
                ):
                    self._check(
                        "RDMA",
                        allocator_env={
                            var: "garbage_collection_threshold:0.8,"
                            "expandable_segments:True"
                        },
                    )

    def test_rdma_allows_disabled_expandable_segments(self):
        self.assertEqual(
            self._check(
                "rdma",
                allocator_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"},
            ),
            (False, None),
        )

    def test_tcp_allows_expandable_segments(self):
        self.assertEqual(
            self._check(
                "tcp",
                allocator_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            ),
            (False, None),
        )

    def test_efa_disables_post_capture_kv_sizing(self):
        cfg = SimpleNamespace(
            enable_unified_memory=False,
            device="cuda",
            dcp_size=1,
            kv_cache_dtype="auto",
            prefill_only_disable_kv_cache=False,
            enable_memory_saver=False,
            disaggregation_transfer_backend="mooncake",
        )
        with (
            patch("sglang.srt.arg_groups.overrides.resolving_view", return_value=cfg),
            patch(
                "sglang.srt.arg_groups.overrides.use_mla_backend", return_value=False
            ),
            patch.object(
                envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING, "get", return_value=True
            ),
            patch.object(
                envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL, "get", return_value=None
            ),
            patch.object(envs.MOONCAKE_PROTOCOL, "get", return_value="efa"),
        ):
            self.assertFalse(post_capture_kv_sizing_planned(object()))


class TestMooncakeRegistrationFailsLoud(unittest.TestCase):
    def _manager(self, engine):
        manager = object.__new__(MooncakeKVManager)
        manager.engine = engine
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[0x1000, 0x2000],
            kv_data_lens=[64, 64],
            aux_data_ptrs=[0x3000],
            aux_data_lens=[32],
            state_data_ptrs=[[0x1000]],
            state_data_lens=[[64]],
        )
        return manager

    def test_registers_each_deduplicated_region(self):
        engine = MagicMock()
        self._manager(engine).register_buffer_to_engine()
        self.assertEqual(
            engine.register.call_args_list,
            [call(0x1000, 64), call(0x2000, 64), call(0x3000, 32)],
        )
        engine.batch_register.assert_not_called()

    def test_region_registration_failure_raises(self):
        engine = MagicMock()
        engine.register.side_effect = [
            None,
            RuntimeError("Mooncake memory registration failed"),
        ]
        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            self._manager(engine).register_buffer_to_engine()

    def test_staging_registration_uses_raising_register(self):
        engine = MagicMock()
        self._manager(engine)._register_staging_memory(0x4000, 128)
        engine.register.assert_called_once_with(0x4000, 128)
        engine.batch_register.assert_not_called()


if __name__ == "__main__":
    unittest.main()
