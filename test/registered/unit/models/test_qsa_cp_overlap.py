"""Query-only CP overlap ordering, cache semantics and scope regression tests."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_module
from sglang.srt.layers.attention.qsa.cp import QSAOnlyCPContext
from sglang.srt.layers.attention.qsa.kernel import QSAPrefillPhysicalSelection
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("save_cache", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
def test_query_cp_prefetch_matches_inline(monkeypatch, save_cache, prefetch):
    events = []
    main, side = Mock(), Mock()
    main.wait_stream.side_effect = lambda stream: events.append("join")
    side.wait_stream.side_effect = lambda stream: events.append("start_after_index")

    @contextmanager
    def on_stream(stream):
        assert stream is side
        yield

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "stream", on_stream)
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda *args: events.append("retain")
    )
    monkeypatch.setattr(backend_module, "get_stream", lambda name: side)
    context = QSAOnlyCPContext(5, 3, 0, 2)
    full_k = torch.arange(40, dtype=torch.float32).reshape(5, 1, 8)
    full_v = full_k + 100
    k, v = (
        full_k[::2].reshape(3, 8).contiguous(),
        full_v[::2].reshape(3, 8).contiguous(),
    )
    q = torch.zeros(3, 2, 4)
    batch = SimpleNamespace(out_cache_loc=torch.arange(8), attn_cp_metadata=None)
    layer = SimpleNamespace(
        layer_id=7,
        tp_q_head_num=2,
        head_dim=4,
        scaling=0.5,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=8,
        v_head_dim=8,
    )
    store = Mock()
    pool = SimpleNamespace(
        set_kv_buffer=store,
        get_key_buffer=lambda _: full_k,
        get_value_buffer=lambda _: full_v,
    )
    backend = SimpleNamespace(
        _qsa_only_cp_pending=None,
        _store_kv=store,
        _kv_descales=lambda *args: (1.0, 1.0),
        token_to_kv_pool=pool,
        _tilelang_kv_head_base=lambda *args: -1,
        _pad_extend_output=QwenSparseAttnBackend._pad_extend_output,
    )

    def gather(packed, ctx):
        assert ctx is context
        events.append("gather")
        torch.testing.assert_close(
            packed, torch.cat((k.view(3, 1, 8), v.view(3, 1, 8)), -1)
        )
        return torch.cat((full_k, full_v), -1)

    monkeypatch.setattr(backend_module, "qsa_only_cp_gather_rows", gather)
    monkeypatch.setattr(backend_module.qsa_tilelang, "is_available", lambda: True)
    monkeypatch.setattr(backend_module.qsa_tilelang, "is_supported", lambda *args: True)

    def attend(actual_q, kc, vc, slots, *args):
        events.append("attention")
        if save_cache:
            torch.testing.assert_close(store.call_args.args[2], full_k)
            torch.testing.assert_close(store.call_args.args[3], full_v)
            torch.testing.assert_close(store.call_args.args[1], torch.arange(5))
        else:
            store.assert_not_called()
        return actual_q + 1

    monkeypatch.setattr(backend_module, "qsa_sparse_attention", attend)
    if prefetch:
        QwenSparseAttnBackend.prefetch_qsa_only_cp_kv(
            backend, layer, k, v, batch, context
        )
        assert backend._qsa_only_cp_pending[-2] is k
        assert backend._qsa_only_cp_pending[-1] is v
    output = QwenSparseAttnBackend.forward_extend(
        backend,
        q,
        k.view(3, 1, 8),
        v.view(3, 1, 8),
        layer,
        batch,
        save_cache,
        QSAPrefillPhysicalSelection(torch.zeros(3, 2, dtype=torch.int32)),
        qsa_cp_context=context,
    )
    torch.testing.assert_close(output, torch.ones(3, 8))
    assert events.count("gather") == int(prefetch or save_cache)
    assert backend._qsa_only_cp_pending is None
    if prefetch:
        assert events == [
            "start_after_index",
            "gather",
            "join",
            "retain",
            "retain",
            "attention",
        ]


def test_query_cp_metadata_cleans_failed_prefetch(monkeypatch):
    main, side = Mock(), Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(backend_module, "get_stream", lambda _: side)
    backend = SimpleNamespace(_qsa_only_cp_pending=object())
    QwenSparseAttnBackend.init_forward_metadata(
        backend, SimpleNamespace(forward_mode=ForwardMode.IDLE)
    )
    main.wait_stream.assert_called_once_with(side)
    assert backend._qsa_only_cp_pending is None
