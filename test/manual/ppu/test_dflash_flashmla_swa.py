"""PPU DFlash sliding MLA: real FlashMLA, cyclic DCP, and dtype boundaries.

PYTHONPATH=python python -m pytest -q test/manual/ppu/test_dflash_flashmla_swa.py
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention import flashmla_backend
from sglang.srt.mem_cache import kv_cache_dtype
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_sail

pytestmark = pytest.mark.skipif(not is_sail(), reason="requires PPU FlashMLA")


@pytest.mark.parametrize(
    "dcp_size,heads,window_left,padding,translate",
    [
        (1, 12, 1023, 0, False),
        (8, 96, 1023, 3, True),
        (8, 96, 1024, 3, True),
    ],
)
def test_sliding_attention_matches_global_reference(
    dcp_size, heads, window_left, padding, translate
):
    torch.manual_seed(31)
    requests, steps, context, page_size = 4, 5, 1536, 64
    reqs = torch.tensor([2, 1], dtype=torch.int64)
    positions = torch.tensor([0, 1, 2, 3, 4, 1100, 1101, 1102, 1103, 1104])
    rows = positions.numel()
    query = (torch.randn(rows + padding, heads, 576) * 0.1).bfloat16()
    global_kv = (torch.randn(requests, context, 576) * 0.1).bfloat16()
    # Make expired and future tokens visibly contaminate a wrong window.
    # Keep Q's latent channels zero here so these value outliers do not
    # change their attention scores.
    query[steps:, :, :512] = 0
    global_kv[1, :76, :512] = 5
    for offset in range(steps):
        global_kv[1, 1100 + offset, :512] = 2**offset
    global_kv[1, 1105:1110, :512] = -8
    expected = []
    for row, position in enumerate(positions.tolist()):
        req = reqs[row // steps]
        keys = global_kv[req, max(0, position - window_left) : position + 1].float()
        scores = query[row].float() @ keys.T * 576**-0.5
        expected.append(scores.softmax(-1) @ keys[:, :512])
    expected = torch.stack(expected)

    position = int(positions[steps])
    q_row = query[steps].float()
    for keys in (
        global_kv[1, : position + 1].float(),
        global_kv[1, position - window_left : position + steps].float(),
    ):
        wrong = (q_row @ keys.T * 576**-0.5).softmax(-1) @ keys[:, :512]
        assert (wrong - expected[steps]).abs().max() > 0.01

    mapping = torch.arange(context).repeat(requests, 1)
    mapping += (torch.arange(requests)[:, None] + 1) * context
    pool_rows = (requests + 1) * context // dcp_size
    v2p = torch.randperm(pool_rows // page_size, dtype=torch.int32)
    v2p_gpu = v2p.cuda() if translate else None
    req_to_token = mapping.to(device="cuda", dtype=torch.int32)
    query_gpu = query.cuda()
    padded_positions = torch.cat(
        [positions, torch.full((padding,), context * 2, dtype=torch.int64)]
    ).cuda()
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=window_left,
        tp_q_head_num=heads,
        v_head_dim=512,
        head_dim=576,
        scaling=576**-0.5,
    )
    batch = SimpleNamespace(
        batch_size=2,
        forward_mode=(
            ForwardMode.TARGET_VERIFY if dcp_size > 1 else ForwardMode.DRAFT_EXTEND_V2
        ),
        spec_info=SimpleNamespace(draft_token_num=steps),
        req_pool_indices=reqs.cuda(),
        positions=padded_positions,
    )
    outputs, lses = [], []
    max_lse_error = 0.0
    backends = []
    for rank in range(dcp_size):
        cache = torch.zeros(pool_rows, 1, 576, dtype=torch.bfloat16)
        owned = torch.arange(rank, context, dcp_size)
        for req in range(requests):
            slots = mapping[req, owned] // dcp_size
            if translate:
                slots = v2p[slots // page_size].long() * page_size + slots % page_size
            cache[slots, 0] = global_kv[req, owned]
        cache_gpu = cache.cuda()
        backend = flashmla_backend.FlashMLABackend.__new__(
            flashmla_backend.FlashMLABackend
        )
        backend.is_dflash_mla_draft = True
        backend.kv_lora_rank = 512
        backend.kv_cache_dim = 576
        backend.page_size = page_size
        backend.req_to_token = req_to_token
        backend.kv_index_translator = SimpleNamespace(full_v2p_table=v2p_gpu)
        backend.forward_metadata = SimpleNamespace(
            block_kv_indices=torch.empty((2, 1), device="cuda", dtype=torch.int32)
        )
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _, cache=cache_gpu: cache
        )
        parallel = SimpleNamespace(
            dcp_enabled=dcp_size > 1, dcp_size=dcp_size, dcp_rank=rank
        )
        backends.append((backend, parallel))
        with patch.object(flashmla_backend, "get_parallel", return_value=parallel):
            result = backend.forward_extend(
                query_gpu[:, :, :512],
                None,
                None,
                layer,
                batch,
                save_kv_cache=False,
                q_rope=query_gpu[:, :, 512:],
            )
        if dcp_size == 1:
            out = result
        else:
            out, lse = result
            local_lse = lse[:rows].cpu()
            for row, position in enumerate(positions.tolist()):
                owned_keys = torch.arange(max(0, position - window_left), position + 1)
                owned_keys = owned_keys[owned_keys % dcp_size == rank]
                if owned_keys.numel() == 0:
                    assert torch.isneginf(local_lse[row]).all()
                    continue
                keys = global_kv[reqs[row // steps], owned_keys].float()
                scores = query[row].float() @ keys.T * 576**-0.5
                expected_lse = scores.logsumexp(-1)
                torch.testing.assert_close(
                    local_lse[row], expected_lse, rtol=1e-4, atol=1e-3
                )
                max_lse_error = max(
                    max_lse_error, (local_lse[row] - expected_lse).abs().max().item()
                )
            lses.append(local_lse)
            if padding:
                assert torch.isneginf(lse[rows:]).all()
        assert torch.isfinite(out).all()
        if padding:
            assert torch.count_nonzero(out[rows:]) == 0
        outputs.append(out[:rows].reshape(rows, heads, 512).float().cpu())

    if dcp_size > 1:
        lses = torch.stack(lses)
        weights = lses.softmax(0)
        actual = (torch.stack(outputs) * weights[..., None]).sum(0)
        assert torch.isneginf(lses[7, :5]).all()
    else:
        actual = outputs[0]
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.002)
    error = actual - expected
    relative_l2 = (error.norm() / expected.norm()).item()
    assert relative_l2 < 0.01
    print(
        f"SWA_NUMERICS dcp={dcp_size} heads={heads} window_left={window_left} "
        f"max_abs={error.abs().max().item():.8g} "
        f"relative_l2={relative_l2:.8g} max_lse_error={max_lse_error:.8g}"
    )

    backend, parallel = backends[-1]
    with patch.object(flashmla_backend, "get_parallel", return_value=parallel):
        from sglang.kernels.ops.attention import mla_window_indices

        def run_layer():
            return backend.forward_extend(
                query_gpu[:, :, :512],
                None,
                None,
                layer,
                batch,
                save_kv_cache=False,
                q_rope=query_gpu[:, :, 512:],
            )

        def check_same(actual, reference):
            if dcp_size == 1:
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            else:
                torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
                torch.testing.assert_close(
                    actual[1], reference[1], rtol=1e-6, atol=1e-6
                )

        with patch.object(
            mla_window_indices,
            "mla_window_sparse_indices",
            wraps=mla_window_indices.mla_window_sparse_indices,
        ) as index_build:
            # Adjacent layers reuse indices, but the next forward (layer 0)
            # and a different window must rebuild them even on the same batch.
            layer.layer_id = 1
            check_same(run_layer(), result)
            assert index_build.call_count == 0
            layer.layer_id = 0
            batch.positions[:rows].add_(1)
            next_forward = run_layer()
            assert index_build.call_count == 1
            backend._ppu_dflash_window_cache = None
            check_same(next_forward, run_layer())
            layer.layer_id = 1
            layer.sliding_window_size -= 128
            next_window = run_layer()
            assert index_build.call_count == 3
            backend._ppu_dflash_window_cache = None
            check_same(next_window, run_layer())
            layer.layer_id = 0
            layer.sliding_window_size = window_left
            batch.positions[:rows].sub_(1)
            backend._ppu_dflash_window_cache = None

        # Replay the same buffers, then change positions in place to exercise
        # the captured window-index computation rather than stale indices.
        backend.forward_extend(
            query_gpu[:, :, :512],
            None,
            None,
            layer,
            batch,
            save_kv_cache=False,
            q_rope=query_gpu[:, :, 512:],
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = backend.forward_extend(
                query_gpu[:, :, :512],
                None,
                None,
                layer,
                batch,
                save_kv_cache=False,
                q_rope=query_gpu[:, :, 512:],
            )
        batch.positions[:rows].add_(1)
        eager = backend.forward_extend(
            query_gpu[:, :, :512],
            None,
            None,
            layer,
            batch,
            save_kv_cache=False,
            q_rope=query_gpu[:, :, 512:],
        )
        graph.replay()
        torch.cuda.synchronize()
        if dcp_size == 1:
            torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        else:
            torch.testing.assert_close(captured[0], eager[0], rtol=0, atol=0)
            # FP32 LSE reductions may differ by one rounding step on replay.
            torch.testing.assert_close(captured[1], eager[1], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("sail", [True, False])
def test_sliding_causality_guard_is_enabled_only_on_ppu(sail):
    from transformers import Qwen3Config

    from sglang.srt.speculative.dflash_utils import is_dflash_mla_draft_causal

    config = Qwen3Config(
        num_hidden_layers=1,
        layer_types=["sliding_attention"],
        sliding_window=1024,
        use_sliding_window=True,
        is_causal=True,
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(is_dflash_mla_draft=True, hf_config=config)
    )

    class GuardChecked(Exception):
        pass

    def parent_init(backend, *args):
        assert is_dflash_mla_draft_causal(
            config,
            allow_sliding_window=getattr(
                backend, "supports_dflash_sliding_window", False
            ),
        )
        raise GuardChecked

    with (
        patch.object(flashmla_backend, "is_sail", return_value=sail),
        patch.object(
            flashmla_backend.FlashInferMLAAttnBackend, "__init__", parent_init
        ),
        pytest.raises(GuardChecked if sail else ValueError),
    ):
        flashmla_backend.FlashMLABackend(runner)


def test_empty_dcp_batch_returns_neutral_partials():
    backend = flashmla_backend.FlashMLABackend.__new__(flashmla_backend.FlashMLABackend)
    backend.is_dflash_mla_draft = True
    backend.kv_lora_rank = 512
    backend.forward_metadata = SimpleNamespace(block_kv_indices=torch.empty(0, 1))
    q = torch.zeros(3, 96, 576, device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(tp_q_head_num=96, sliding_window_size=1023)
    batch = SimpleNamespace(spec_info=SimpleNamespace(draft_token_num=5))
    with patch.object(
        flashmla_backend, "get_parallel", return_value=SimpleNamespace(dcp_enabled=True)
    ):
        out, lse = backend.forward_extend(
            q, None, None, layer, batch, save_kv_cache=False
        )
    assert out.shape == (3, 96 * 512)
    assert torch.count_nonzero(out) == 0
    assert lse.shape == (3, 96)
    assert torch.isneginf(lse).all()


@pytest.mark.parametrize(
    "sail,draft,backend,expected",
    [
        (True, True, "flashmla", torch.bfloat16),
        (False, True, "flashmla", torch.float8_e4m3fn),
        (True, False, "flashmla", torch.float8_e4m3fn),
        (True, True, "flashinfer", torch.float8_e4m3fn),
    ],
)
def test_draft_kv_dtype_boundary(sail, draft, backend, expected):
    with patch.object(kv_cache_dtype, "is_sail", return_value=sail):
        tag, dtype = kv_cache_dtype.configure_kv_cache_dtype(
            server_args_kv_cache_dtype="fp8_e4m3",
            model=None,
            model_dtype=torch.bfloat16,
            is_draft_worker=draft,
            is_dflash=True,
            speculative_draft_attention_backend=backend,
        )
    assert dtype == expected
    assert tag == ("auto" if expected == torch.bfloat16 else None)
