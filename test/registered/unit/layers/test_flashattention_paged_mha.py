import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

with patch.dict(
    sys.modules,
    {
        module: MagicMock()
        for module in (
            "sgl_kernel",
            "sgl_kernel.quantization",
            "sgl_kernel.scalar_type",
        )
    },
):
    from sglang.srt.layers.attention import flashattention_backend

FlashAttentionBackend = flashattention_backend.FlashAttentionBackend

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _backend():
    backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
    backend.page_size = 1
    backend.kv_cache_dtype = torch.float16
    backend.kv_cache_dtype_str = "float8_e4m3fn"
    backend.kv_cache_is_mxfp8 = False
    backend.fa_impl_ver = 3
    backend.num_splits = 4
    return backend


class TestFlashAttentionPagedMHA(unittest.TestCase):
    def _decode_kernel_kwargs(self, attn_type, window):
        backend = _backend()
        backend.use_mla = False
        backend.has_local_attention = False
        backend.topk = 1
        backend.use_sliding_window_kv_pool = False
        backend.is_prefill_aware_swa = False
        backend._decode_uses_static_max_seqlen_k = False
        backend.decode_num_splits = 4
        backend.forward_metadata = SimpleNamespace(
            page_table=torch.tensor([[0]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([8], dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            max_seq_len_q=2,
            scheduler_metadata=None,
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            attn_type=attn_type,
            sliding_window_size=window,
            tp_q_head_num=2,
            tp_k_head_num=1,
            head_dim=16,
            v_head_dim=16,
            scaling=0.25,
            logit_cap=0.0,
        )
        q = torch.zeros(2, 2, 16)
        kv = torch.zeros(1, 8, 1, 16)
        backend.prepare_paged_mha_query = Mock(return_value=(q, None, None, None, None))
        backend.get_paged_mha_kv_cache = Mock(return_value=(kv, kv))
        with patch.object(
            flashattention_backend,
            "flash_attn_with_kvcache",
            return_value=q,
        ) as kernel:
            backend.forward_decode(
                q,
                None,
                None,
                layer,
                SimpleNamespace(spec_info=None, batch_size=1),
                save_kv_cache=False,
            )
        kernel.assert_called_once()
        return kernel.call_args.kwargs

    def test_encoder_only_swa_decode_uses_symmetric_window(self):
        from sglang.srt.layers.radix_attention import AttentionType

        kwargs = self._decode_kernel_kwargs(AttentionType.ENCODER_ONLY, 127)
        self.assertEqual(kwargs["window_size"], (127, 127))
        self.assertFalse(kwargs["causal"])

    def test_causal_swa_decode_retains_left_window(self):
        from sglang.srt.layers.radix_attention import AttentionType

        kwargs = self._decode_kernel_kwargs(AttentionType.DECODER, 127)
        self.assertEqual(kwargs["window_size"], (127, 0))
        self.assertTrue(kwargs["causal"])

    def test_full_attention_decode_window_is_unchanged(self):
        from sglang.srt.layers.radix_attention import AttentionType

        for attn_type in (AttentionType.ENCODER_ONLY, AttentionType.DECODER):
            for window in (-1, None):
                with self.subTest(attn_type=attn_type, window=window):
                    kwargs = self._decode_kernel_kwargs(attn_type, window)
                    self.assertEqual(kwargs["window_size"], (-1, -1))
                    self.assertEqual(
                        kwargs["causal"], attn_type == AttentionType.DECODER
                    )

    def test_get_paged_mha_kv_cache_supports_head_groups(self):
        backend = _backend()
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=Mock(
                return_value=(
                    torch.empty(8, 2, 16),
                    torch.empty(8, 2, 16),
                )
            )
        )
        layer = SimpleNamespace(
            layer_id=3,
            tp_k_head_num=2,
            tp_v_head_num=2,
            head_dim=16,
            v_head_dim=16,
        )

        key_cache, value_cache = backend.get_paged_mha_kv_cache(
            layer,
            head_group_num=2,
        )

        self.assertEqual(key_cache.shape, (16, 1, 1, 16))
        self.assertEqual(value_cache.shape, (16, 1, 1, 16))

    def test_prepare_paged_mha_query_reuses_fa_scaling_policy(self):
        backend = _backend()
        layer = SimpleNamespace(
            head_dim=16,
            k_scale=torch.tensor(2.0),
            v_scale=torch.tensor(4.0),
        )
        q = torch.ones(2, 16, dtype=torch.bfloat16)

        q, _, _, k_descale, v_descale = backend.prepare_paged_mha_query(
            q,
            None,
            None,
            layer,
            logical_batch_size=2,
            kv_head_num=1,
            is_prefill=True,
        )

        self.assertEqual(q.dtype, torch.float16)
        self.assertEqual(k_descale.shape, (2, 1))
        self.assertEqual(v_descale.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
