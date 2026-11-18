"""Short dense attention: causal CP rows, native KV bits, and state continuity."""

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.cp import QSAOnlyCPContext
from sglang.srt.layers.attention.qsa.fa3 import fa3_sparse_attention, quantize_q_fp8
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qsa.short_dense import pack_short_dense_kv
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    QwenSparseAttnMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_ppu

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_ppu(), reason="PPU required"
)


def guard_backend():
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.qsa_backend = "fa3"
    backend.qsa_profile = NS(budget=2048)
    backend._mtp_shared_sparse_indices = None
    return backend


def batch(lengths, mode=ForwardMode.EXTEND):
    return NS(
        forward_mode=mode,
        _original_forward_mode=None,
        seq_lens_cpu=torch.tensor(lengths),
        extend_seq_lens_cpu=[1] * len(lengths),
        positions=torch.arange(len(lengths)),
    )


@pytest.mark.parametrize(
    "lengths,expected",
    [
        ([1], True),
        ([2048], True),
        ([2049], False),
        ([10, 2049], False),
        ([], False),
        ([2048] * 16, True),
        ([2048] * 17, False),
    ],
)
def test_complete_context_guard(lengths, expected):
    backend = guard_backend()
    with envs.SGLANG_QSA_SHORT_CTX_DENSE.override(True):
        assert backend.should_use_short_ctx_dense(batch(lengths)) is expected


@pytest.mark.parametrize(
    "mode",
    [
        ForwardMode.DECODE,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.DRAFT_EXTEND_V2,
    ],
)
def test_speculation_keeps_sparse_selection(mode):
    assert not guard_backend().should_use_short_ctx_dense(batch([64], mode))


def test_guard_cache_invalidation_and_capture():
    backend = guard_backend()
    backend._qsa_only_cp_pending = None
    backend._metadata_from_forward_batch = lambda _: NS()
    current = batch([1024])
    with patch.object(
        backend, "_compute_short_ctx_dense", wraps=backend._compute_short_ctx_dense
    ) as compute:
        backend.init_forward_metadata(current)
        for _ in range(12):
            assert backend.should_use_short_ctx_dense(current)
        assert compute.call_count == 1
        with patch(
            "sglang.srt.model_executor.runner.get_is_capture_mode", return_value=True
        ):
            assert not backend.should_use_short_ctx_dense(current)
        current.seq_lens_cpu[0] = 4096
        backend.init_forward_metadata(current)
        assert not backend.should_use_short_ctx_dense(current)
        assert compute.call_count == 2
    with envs.SGLANG_QSA_SHORT_CTX_DENSE.override(False):
        assert not guard_backend().should_use_short_ctx_dense(batch([64]))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_native_cache_bits_and_invalid_padding(dtype):
    bits = torch.arange(65536, device="cuda")
    storage = (
        bits.short().view(dtype)
        if dtype == torch.bfloat16
        else bits.to(torch.uint8).view(dtype)
    )
    keys = storage.reshape(128, 2, 256)
    values = keys.flip(0).contiguous()
    lengths = [0, 1, 63, 65]
    slots = torch.full((4, 65), -999, device="cuda", dtype=torch.int32)
    for i, length in enumerate(lengths):
        slots[i, :length] = torch.randperm(128, device="cuda")[:length].int()
    out_k = keys.new_empty((4, 128, 2, 256))
    out_v = torch.empty_like(out_k)
    pack_short_dense_kv(
        keys,
        values,
        slots,
        torch.tensor(lengths, device="cuda", dtype=torch.int32),
        out_k,
        out_v,
    )
    for i, length in enumerate(lengths):
        assert torch.equal(
            out_k[i, :length].view(torch.uint8),
            keys[slots[i, :length].long()].view(torch.uint8),
        )
        assert torch.equal(
            out_v[i, :length].view(torch.uint8),
            values[slots[i, :length].long()].view(torch.uint8),
        )
        assert not out_k[i, length:].view(torch.uint8).any()
        assert not out_v[i, length:].view(torch.uint8).any()


@pytest.mark.parametrize("cp", [False, True])
def test_bypass_still_updates_global_indexer_state(cp):
    context = QSAOnlyCPContext(4, 2, 0, 2) if cp else None
    local = 2 if cp else 4
    updates = []
    metadata = NS(
        get_token_to_batch_idx=lambda: torch.zeros(4, dtype=torch.int32),
        pending_ring_slots=torch.arange(4),
        token_to_kv_pool=NS(
            get_qsa_compressed_k_buffer=lambda _: torch.empty(1, dtype=torch.bfloat16)
        ),
    )
    fake = NS(
        layer_id=3,
        index_n_heads=4,
        project_qk=lambda *args, **kwargs: (
            torch.zeros(local, 4, 128),
            torch.zeros(local, 128),
            False,
        ),
        update_key_state_and_compress=lambda k, p, *args, **kwargs: updates.append(
            (len(k), p.tolist())
        ),
    )
    current = NS(forward_mode=ForwardMode.EXTEND, positions=torch.arange(4))
    with patch(
        "sglang.srt.layers.attention.qsa.qsa_indexer.qsa_only_cp_gather_rows",
        return_value=torch.ones(4, 128),
    ):
        result = QSAIndexer.forward_cuda(
            fake,
            torch.zeros(local, 8),
            torch.arange(local),
            current,
            metadata,
            qsa_cp_context=context,
            skip_selection=True,
        )
    assert result is None
    assert updates == [(4, [0, 1, 2, 3])]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "lengths,extends", [([1], [1]), ([31], [31]), ([128, 65], [32, 17])]
)
@pytest.mark.parametrize("world", [1, 2])
def test_dense_causal_cache_and_cp_prefixes(dtype, lengths, extends, world):
    torch.manual_seed(20261118)
    heads, dim = 24, 256
    keys = torch.randn(512, 2, dim, device="cuda", dtype=torch.bfloat16).to(dtype)
    values = torch.randn_like(keys.to(torch.bfloat16)).to(dtype)
    keys_before, values_before = (
        keys.view(torch.uint8).clone(),
        values.view(torch.uint8).clone(),
    )
    table = torch.zeros(len(lengths), max(lengths), device="cuda", dtype=torch.int32)
    slots = torch.randperm(512, device="cuda")
    offset = 0
    for i, length in enumerate(lengths):
        table[i, :length] = slots[offset : offset + length].int()
        offset += length
    total = sum(extends)
    query = torch.randn(total, heads, dim, device="cuda", dtype=torch.bfloat16)
    ids = torch.repeat_interleave(
        torch.arange(len(lengths), device="cuda", dtype=torch.int32),
        torch.tensor(extends, device="cuda"),
    )
    positions = torch.cat(
        [torch.arange(n - e, n, device="cuda") for n, e in zip(lengths, extends)]
    )
    metadata = QwenSparseAttnMetadata(
        torch.tensor(lengths, device="cuda", dtype=torch.int32), ids, table, None
    )
    backend = guard_backend()
    backend.forward_metadata = metadata
    backend._short_dense_cache = None
    backend.token_to_kv_pool = NS(
        get_key_buffer=lambda _: keys, get_value_buffer=lambda _: values
    )
    layer = NS(layer_id=3, k_scale_float=0.7, v_scale_float=1.3, scaling=dim**-0.5)
    current = NS(positions=positions, seq_lens_cpu=torch.tensor(lengths))
    for rank in range(world):
        context = (
            None if world == 1 else QSAOnlyCPContext(total, (total + 1) // 2, rank, 2)
        )
        q, seq, pos = (
            (query, ids, positions)
            if context is None
            else tuple(context.shard(x) for x in (query, ids, positions))
        )
        capacity = len(q) if context is None else context.local_capacity
        padded = q.new_zeros((capacity, heads, dim))
        padded[: len(q)] = q
        output = backend._forward_short_ctx_dense(padded, layer, current, context)
        assert not output[len(q) :].any()
        if not len(q):
            continue
        output = output[: len(q)].reshape_as(q)
        cols = torch.arange(max(lengths), device="cuda")[None, :]
        physical = (
            torch.where(cols <= pos[:, None], table[seq.long()], -1).int().contiguous()
        )
        sparse = fa3_sparse_attention(
            q,
            keys,
            values,
            physical,
            None,
            None,
            None,
            layer.scaling,
            k_scale=0.7,
            v_scale=1.3,
            indices_are_physical=True,
        )
        qf = q.float()
        if dtype == torch.float8_e4m3fn:
            quantized, scale = quantize_q_fp8(q, 2)
            qf = quantized.float() * scale.repeat_interleave(
                heads // 2, dim=1
            ).unsqueeze(-1)
        oracle = torch.empty_like(q, dtype=torch.float32)
        for i, length in enumerate(lengths):
            selected = seq == i
            k, v = (
                keys[table[i, :length].long()].float(),
                values[table[i, :length].long()].float(),
            )
            if dtype == torch.float8_e4m3fn:
                k, v = k * 0.7, v * 1.3
            scores = (
                torch.matmul(
                    qf[selected].transpose(0, 1),
                    k.repeat_interleave(heads // 2, dim=1).permute(1, 2, 0),
                )
                * layer.scaling
            )
            scores.masked_fill_(
                torch.arange(length, device="cuda")[None, None, :]
                > pos[selected][None, :, None],
                float("-inf"),
            )
            oracle[selected] = torch.matmul(
                scores.softmax(-1),
                v.repeat_interleave(heads // 2, dim=1).transpose(0, 1),
            ).transpose(0, 1)
        dense_error = (output.float() - oracle).square().mean().sqrt()
        sparse_error = (sparse.float() - oracle).square().mean().sqrt()
        assert torch.isfinite(output).all()
        assert dense_error <= sparse_error * 1.35 + 5e-4
    assert torch.equal(keys.view(torch.uint8), keys_before)
    assert torch.equal(values.view(torch.uint8), values_before)
