"""CPU regression for the #48980 backport; accelerator numerics need E2E tests.

Execute production methods with real tensors and stub only accelerator builders.
"""

import ast
import itertools
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def load(path, name, ns, cls=None):
    tree = ast.parse(path.read_text())
    if cls:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        )
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns[name]


def mode(name):
    return SimpleNamespace(
        is_extend=lambda: name in ("extend", "verify"),
        is_target_verify=lambda: name == "verify",
        is_prefill=lambda: name == "extend",
        is_idle=lambda: name == "idle",
        is_decode=lambda: name == "decode",
        is_extend_without_speculative=lambda: name == "extend",
    )


@pytest.fixture
def ns():
    ns = dict(torch=torch, contextmanager=contextmanager, nullcontext=nullcontext)
    for name in ("real_rows_scope", "pad_with_zero_rows", "padded_extend_real_tokens"):
        load(SRT / "layers/radix_attention.py", name, ns)
    return ns


@pytest.mark.parametrize(
    "name,real,padded,expected",
    [
        ("extend", 2519, 2520, 2519),
        ("extend", 2520, 2520, None),
        ("extend", 0, 4, None),
        ("extend", None, 4, None),
        ("decode", 3, 4, None),
        ("verify", 3, 4, None),
        ("idle", 0, 4, None),
    ],
)
def test_padding_gate(ns, name, real, padded, expected):
    batch = SimpleNamespace(
        forward_mode=mode(name), global_num_token_non_padded_cpu=real
    )
    assert ns["padded_extend_real_tokens"](torch.empty(padded), batch) == expected


def test_scope_restores_after_nested_exception(ns):
    batch = SimpleNamespace(out_cache_loc=torch.arange(8), positions=torch.arange(8))
    original = vars(batch).copy()
    with pytest.raises(ValueError), ns["real_rows_scope"](batch, 5):
        with ns["real_rows_scope"](batch, 3):
            assert len(batch.positions) == len(batch.out_cache_loc) == 3
        assert len(batch.positions) == 5
        raise ValueError("test")
    assert batch.positions is original["positions"]
    assert batch.out_cache_loc is original["out_cache_loc"]


@pytest.mark.parametrize(
    "cp,name,real,padded",
    [
        (False, "extend", 2519, 2520),
        (False, "extend", 8, 8),
        (True, "extend", 7, 8),
        (False, "verify", 7, 8),
    ],
)
def test_metadata_and_tail_use_same_real_rows(ns, cp, name, real, padded):
    ns.update(
        _get_logical_forward_mode=lambda b: b.forward_mode, is_cp_active=lambda b: b.cp
    )
    init = load(
        SRT / "layers/attention/deepseek_v4_backend.py",
        "init_forward_metadata",
        ns,
        "DeepseekV4AttnBackend",
    )
    batch = SimpleNamespace(
        forward_mode=mode(name),
        cp=cp,
        encoder_swa_replay=False,
        global_num_token_non_padded_cpu=real,
        positions=torch.arange(padded),
        out_cache_loc=torch.arange(padded),
    )
    seen = []

    def build(b, *, real_num_tokens):
        seen.append(("main", len(b.positions), len(b.out_cache_loc), real_num_tokens))
        return SimpleNamespace(real_num_tokens=real_num_tokens)

    backend = SimpleNamespace(
        mtp_enabled=False,
        enable_decoder_swa_bounded_replay=True,
        _build_forward_metadata=build,
        init_forward_metadata_in_graph=lambda b: seen.append(
            ("graph", len(b.positions))
        ),
        _build_late_layer_tail_metadata=lambda b: seen.append(
            ("tail", len(b.positions))
        ),
        _activate_request_window=Mock(),
    )
    init(backend, batch)
    trimmed = real if name == "extend" and not cp and real < padded else None
    assert seen[0] == ("main", trimmed or padded, trimmed or padded, trimmed)
    assert all(item[1] == (trimmed or padded) for item in seen)
    assert len(batch.positions) == len(batch.out_cache_loc) == padded


@pytest.mark.parametrize("partial", [False, True])
def test_attention_requantizes_real_rows_and_zeroes_output(ns, partial):
    class AttnOutput:
        def __init__(self, value):
            self.partial = value

    ns["mhc"] = SimpleNamespace(AttnOutput=AttnOutput)
    forward = load(SRT / "models/deepseek_v4.py", "_forward_real_rows", ns, "MQALayer")
    batch = SimpleNamespace(positions=torch.arange(8), out_cache_loc=torch.arange(8))

    def compute(x, positions, b, *, defer_all_reduce):
        assert x.shape[0] == positions.shape[0] == len(b.out_cache_loc) == 5
        assert defer_all_reduce is partial
        # No x_quant argument: padded/swizzled scales must not be reused.
        return AttnOutput(x + 1) if partial else x + 1

    result = forward(
        SimpleNamespace(forward=compute),
        torch.ones(8, 3),
        batch.positions,
        batch,
        real_num_tokens=5,
        defer_all_reduce=partial,
    )
    out = result.partial if partial else result
    torch.testing.assert_close(out[:5], torch.full((5, 3), 2.0))
    assert torch.count_nonzero(out[5:]) == 0
    assert len(batch.positions) == len(batch.out_cache_loc) == 8


def test_contiguous_tail_excludes_mlp_padding(ns):
    take = load(SRT / "layers/attention/deepseek_v4_backend.py", "_tail_rows", ns)
    scatter = load(SRT / "models/deepseek_v4.py", "_scatter_tail_rows", ns)
    tokens = torch.tensor([4, 5, 6])
    x = torch.arange(8 * 2).view(8, 2)
    tail = take(x, token_indices=tokens, contiguous_start=4)
    torch.testing.assert_close(tail, x[tokens])
    out = scatter(SimpleNamespace(contiguous_start=4, scatter_indices=None), tail, 8)
    torch.testing.assert_close(out[4:7], tail)


def test_both_ppu_request_index_builders_receive_real_rows(ns):
    ns.update(
        itertools=itertools,
        PrefillIndexPlan=SimpleNamespace,
        async_h2d=lambda values, **kw: torch.tensor(values, **kw),
    )
    req = load(SRT / "layers/attention/dsv4/dsv41_sparse.py", "token_req_indices", ns)
    plan = load(
        SRT / "layers/attention/dsv4/v41_indexer/scoring.py", "prefill_index_plan", ns
    )
    batch = SimpleNamespace(
        forward_mode=mode("extend"),
        req_pool_indices=torch.tensor([1, 2]),
        extend_seq_lens=torch.tensor([3, 2516]),
        extend_seq_lens_cpu=[3, 2516],
        positions=torch.arange(2520),
        out_cache_loc=torch.arange(2520),
    )
    with ns["real_rows_scope"](batch, 2519):
        assert len(req(batch, num_tokens=len(batch.positions))) == 2519
        inputs = SimpleNamespace(
            compress_ratio=1,
            positions=batch.positions,
            seq_lens_cpu=[3, 2516],
            rows_per_request=[3, 2516],
            rows_per_request_device=batch.extend_seq_lens,
        )
        assert len(plan(inputs).request_starts) == 2519


def test_attention_dispatch_is_in_production_forward():
    tree = ast.parse((SRT / "models/deepseek_v4.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MQALayer"
    )
    forward = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_forward_real_rows"
        for n in ast.walk(forward)
    )


@pytest.mark.parametrize("real,padded", [(3, 4), (2519, 2520), (2520, 2520)])
@pytest.mark.parametrize("fail_store", [False, True])
def test_tail_source_kv_uses_real_rows_before_attention(ns, real, padded, fail_store):
    """Bounded replay writes source KV without going through MQALayer.forward."""
    ns.update(dsa_use_prefill_cp=lambda b: False)
    backend_path = SRT / "layers/attention/deepseek_v4_backend.py"
    methods = {
        name: load(backend_path, name, ns, "DeepseekV4AttnBackend")
        for name in (
            "write_tail_source_kv",
            "forward_low_ratio_sources",
            "_low_ratio_compress_torch",
        )
    }
    batch = SimpleNamespace(
        forward_mode=mode("extend"),
        encoder_swa_replay=False,
        positions=torch.arange(real),
        out_cache_loc=torch.arange(padded),
    )
    original = vars(batch).copy()
    x = torch.arange(padded * 2).view(padded, 2)
    meta = SimpleNamespace(
        real_num_tokens=real if real < padded else None,
        low_ratio_req_indices=torch.zeros(real, dtype=torch.int64),
        low_ratio_pos_i64=torch.arange(real),
        core_metadata=SimpleNamespace(c1_out_loc=torch.arange(real)),
    )
    stores = []

    def store(layer, kv, loc, pos, **kwargs):
        # The PPU FP4 store requires one slot for every projected row.
        assert kv.shape[0] == loc.numel() == pos.numel() == real
        assert len(batch.positions) == len(batch.out_cache_loc) == real
        torch.testing.assert_close(kv, x[:real])
        stores.append(kv)
        if fail_store:
            raise RuntimeError("store failed")

    backend = SimpleNamespace(
        forward_metadata=meta,
        tail_source_kv_layer_id=None,
        _low_ratio_in_prefill_graph=lambda: False,
        _low_ratio_write_group=store,
    )
    backend.forward_low_ratio_sources = lambda **kw: methods[
        "forward_low_ratio_sources"
    ](backend, **kw)
    backend._low_ratio_compress = lambda layer, x, req, pos, b: methods[
        "_low_ratio_compress_torch"
    ](backend, layer, x, req, pos)
    layer = SimpleNamespace(
        layer_id=17,
        compress_ratio=1,
        compressor=SimpleNamespace(project=lambda data: (data, None)),
        indexer=None,
    )
    with (
        pytest.raises(RuntimeError, match="store failed")
        if fail_store
        else nullcontext()
    ):
        methods["write_tail_source_kv"](
            backend, layer=layer, x=x, positions=batch.positions, forward_batch=batch
        )
    assert len(stores) == 1
    assert batch.positions is original["positions"]
    assert batch.out_cache_loc is original["out_cache_loc"]
    assert backend.tail_source_kv_layer_id == (None if fail_store else layer.layer_id)
