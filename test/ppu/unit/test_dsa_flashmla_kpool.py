from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sglang.srt.layers.attention.dsa_backend as dsa_backend
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend


def _metadata(rows):
    return SimpleNamespace(
        dsa_cache_seqlens_int32=torch.full((rows,), 67, dtype=torch.int32),
        flashmla_metadata=SimpleNamespace(flashmla_metadata=object(), num_splits=None),
    )


def _backend(index_kpool):
    return SimpleNamespace(
        dsa_index_kpool=index_kpool,
        dsa_index_topk=64,
        flashmla_kv_num_q_heads=4,
        real_page_size=64,
        kv_cache_dim=4,
        dsa_kv_cache_store_fp8=False,
    )


def _run_flashmla(index_kpool, page_table):
    calls = []

    def fake_flashmla(**kwargs):
        calls.append(kwargs)
        q = kwargs["q"]
        return q.new_zeros((*q.shape[:-1], 3)), None

    with (
        patch.object(dsa_backend, "_is_ppu", True),
        patch(
            "sglang.srt.environ.envs.SGLANG_DSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8.get",
            return_value=False,
        ),
        patch(
            "sgl_kernel.flash_mla.flash_mla_with_kvcache",
            side_effect=fake_flashmla,
        ),
    ):
        output = DeepseekSparseAttnBackend._forward_flashmla_kv(
            _backend(index_kpool),
            torch.randn((2, 2, 4), dtype=torch.bfloat16),
            torch.randn((128, 4), dtype=torch.bfloat16),
            3,
            0.125,
            SimpleNamespace(tp_q_head_num=2, head_dim=4),
            _metadata(2),
            page_table,
        )
    return output, calls[0]


def test_ppu_kpool_guard_allows_flashmla_kv_only_on_ppu():
    backend = SimpleNamespace(dsa_index_kpool=4)
    topk_indices = torch.zeros((1, 67), dtype=torch.int32)

    with patch.object(dsa_backend, "_is_ppu", True):
        DeepseekSparseAttnBackend._check_kpool_tail_backend(
            backend, topk_indices, "flashmla_kv", "decode"
        )

    with (
        patch.object(dsa_backend, "_is_ppu", False),
        pytest.raises(NotImplementedError),
    ):
        DeepseekSparseAttnBackend._check_kpool_tail_backend(
            backend, topk_indices, "flashmla_kv", "decode"
        )


def test_ppu_flashmla_kv_pads_kpool_tail_indices():
    page_table = torch.arange(134, dtype=torch.int32).view(2, 67)
    output, call = _run_flashmla(4, page_table)

    assert output.shape == (2, 1, 2, 3)
    assert call["indices"].shape == (2, 1, 128)
    torch.testing.assert_close(call["indices"][:, 0, :67], page_table)
    assert torch.all(call["indices"][:, :, 67:] == -1)
    torch.testing.assert_close(
        call["topk_length"], torch.tensor([67, 67], dtype=torch.int32)
    )


def test_ppu_flashmla_kv_keeps_non_kpool_width():
    page_table = torch.arange(128, dtype=torch.int32).view(2, 64)
    _, call = _run_flashmla(1, page_table)

    assert call["indices"].shape == (2, 1, 64)
    torch.testing.assert_close(call["indices"][:, 0], page_table)
    assert call["topk_length"] is None
