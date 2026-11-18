"""PPU CP Q preparation and replicated native-FP8 store regressions."""
from types import SimpleNamespace as NS
import pytest
import torch
from sglang.kernels.ops.attention.qsa_cp_fp8_store import store_cp_kv
from sglang.srt.layers.attention.qsa.cp import QSAOnlyCPContext
from sglang.srt.utils import is_ppu
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not is_ppu(), reason='PPU required')
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend

@pytest.mark.parametrize('cp,n,capacity', [(1, 0, 1), (2, 7, 5), (4, 3, 2), (4, 129, 36), (2, 32767, 16384)])
@pytest.mark.parametrize('scale', [1.0, 0.3, 3.14159265])
def test_cp_fp8_store(cp, n, capacity, scale):
    torch.manual_seed(123)
    h, d = (2, 256)
    packed = torch.randn(cp * capacity, h, 2 * d, device='cuda', dtype=torch.bfloat16) * 100
    k, v = packed.split(d, -1)
    loc = torch.randperm(n + 19, device='cuda')[:n].contiguous()
    a = torch.full((n + 19, h, d), 0.25, device='cuda', dtype=torch.float8_e4m3fn)
    b = a.clone()
    c = a.clone()
    e = a.clone()
    full = packed.view(cp, capacity, h, 2 * d).transpose(0, 1).reshape(cp * capacity, h, 2 * d)[:n].contiguous()
    x, y = full.split(d, -1)
    a[loc] = x.clone().div_(scale).clamp_(-448, 448).to(a.dtype)
    b[loc] = y.clone().div_(1.25).clamp_(-448, 448).to(b.dtype)
    store_cp_kv(k, v, c, e, loc, total_rows=n, cp_size=cp, local_capacity=capacity, k_scale=scale, v_scale=1.25)
    assert torch.equal(a.float(), c.float()) and torch.equal(b.float(), e.float())

@pytest.mark.parametrize('prefetch', [True, False])
@pytest.mark.parametrize('save', [True, False])
def test_backend_cp_store(monkeypatch, prefetch, save):
    import sglang.srt.layers.attention.qsa.cp as cp_mod
    import sglang.srt.layers.attention.qwen_sparse_attn_backend as backend_mod
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc
    cp, n, capacity, h, d = (2, 7, 4, 2, 256)
    context = QSAOnlyCPContext(n, capacity, 0, cp, True)
    global_k = torch.randn(n, h, d, device='cuda', dtype=torch.bfloat16) * 20
    global_v = torch.randn_like(global_k)
    gathered = torch.zeros(cp, capacity, h, 2 * d, device='cuda', dtype=torch.bfloat16)
    for rank in range(cp):
        z = torch.cat((global_k[rank::cp], global_v[rank::cp]), -1)
        gathered[rank, :z.shape[0]] = z

    class Group:

        def all_gather_into_tensor(self, output, input):
            assert torch.equal(input, gathered[0])
            output.copy_(gathered.flatten(0, 1))
    monkeypatch.setattr(cp_mod, 'get_parallel', lambda: NS(attn_tp_group=Group()))
    keys = torch.full((32, h, d), 0.25, device='cuda', dtype=torch.float8_e4m3fn)
    values = keys.clone()
    pool = NS(dtype=keys.dtype, full_kv_pool=NS(is_quantized_kv_cache=False, use_hnd=False, kv_cache_layout='NHD', page_size=64, kernel_page_blocks=1), get_key_buffer=lambda _: keys, get_value_buffer=lambda _: values)
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.qsa_backend = 'fa3'
    backend._qsa_only_cp_pending = None
    backend.token_to_kv_pool = pool
    backend._logical_to_physical = lambda *args: torch.zeros((4, 1), device='cuda', dtype=torch.int32)
    backend._resolve_metadata = lambda _: None
    monkeypatch.setattr(backend_mod, 'fa3_sparse_attention', lambda q, *a, **kw: torch.zeros_like(q))
    physical = torch.tensor([9, 12, 17, 3, 21, 27, 29], device='cuda', dtype=torch.int64)
    logical = torch.arange(1, 8, device='cuda', dtype=torch.int64)
    batch = NS(out_cache_loc=KVWriteLoc(logical, full_loc=physical))
    layer = NS(layer_id=0, tp_k_head_num=h, tp_v_head_num=h, qk_head_dim=d, v_head_dim=d, tp_q_head_num=24, head_dim=d, scaling=d ** (-0.5), k_scale_float=0.3, v_scale_float=1.25)
    k = global_k[::cp].contiguous()
    v = global_v[::cp].contiguous()
    q = torch.randn(4, 24, d, device='cuda', dtype=torch.bfloat16)
    if prefetch:
        backend.prefetch_qsa_only_cp_kv(layer, k.flatten(1), v.flatten(1), batch, context)
    before_k, before_v = (keys.clone(), values.clone())
    backend.forward_extend(q, k, v, layer, batch, save_kv_cache=save, topk_indices=torch.zeros((4, 1), device='cuda', dtype=torch.int32), qsa_cp_context=context)
    assert backend._qsa_only_cp_pending is None
    if save:
        before_k[physical] = global_k.clone().div_(0.3).clamp_(-448, 448).to(keys.dtype)
        before_v[physical] = global_v.clone().div_(1.25).clamp_(-448, 448).to(values.dtype)
    assert torch.equal(keys.float(), before_k.float()) and torch.equal(values.float(), before_v.float())

@pytest.mark.parametrize('n', [0, 127, 8202])
@pytest.mark.parametrize('axes', [1, 3])
def test_cp_qprep_deferred_state(n, axes):
    from sglang.kernels.ops.attention.qsa_indexer import qsa_index_q_norm_rope_store
    torch.manual_seed(888)
    x = torch.randn(n, 640, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(128, device='cuda', dtype=torch.bfloat16) * 0.3
    pos = torch.randint(65536, (axes, n + 13), device='cuda')[:, :n]
    if axes == 1:
        pos = pos[0]
    cache = torch.randn(65536, 64, device='cuda')
    axis = torch.randint(0, 3, (32,), device='cuda', dtype=torch.int32)
    keys = torch.full((max(n, 1), 128), 0.375, device='cuda', dtype=torch.bfloat16)
    coords = torch.full((max(n, 1), 3), -9, device='cuda', dtype=torch.int64)
    loc = torch.arange(n, device='cuda')
    args = (x, pos, cache, axis, weight)
    gold = qsa_index_q_norm_rope_store(*args, loc, keys, coords, 4, 64, 1e-06, True)
    before_keys, before_coords = (keys.clone(), coords.clone())
    out = qsa_index_q_norm_rope_store(*args, None, keys, coords, 4, 64, 1e-06, True)
    assert torch.equal(gold, out)
    assert torch.equal(keys, before_keys) and torch.equal(coords, before_coords)

@pytest.mark.parametrize('role,enabled,dtype,heads,expected', [('prefill', True, torch.bfloat16, 4, True), ('decode', True, torch.bfloat16, 4, False), ('encoder', True, torch.bfloat16, 4, False), ('prefill', False, torch.bfloat16, 4, False), ('prefill', True, torch.float32, 4, False), ('prefill', True, torch.bfloat16, 8, False)])
def test_startup_warmup_gating(monkeypatch, request, role, enabled, dtype, heads, expected):
    import sglang.srt.runtime_context as context_mod
    override = context_mod.get_context().override_server_args(attention_backend='qsa', prefill_attention_backend='qsa', decode_attention_backend='qsa')
    override.install()
    request.addfinalizer(override.restore)
    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
    import sglang.srt.layers.quantization.unquant as unquant
    import sglang.kernels.ops.attention.qsa_cp_prefill_warmup as warmup_mod
    indexer = QSAIndexer.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.index_n_heads = heads
    indexer.index_head_dim = 128
    indexer.q_layernorm = NS(weight=torch.zeros(128, dtype=dtype))
    indexer.rotary_emb = NS(rotary_dim=64, is_neox_style=True)
    root = NS(config=NS(num_key_value_heads=2, head_dim=256, hidden_size=2560), modules=lambda: [indexer])
    calls = []
    monkeypatch.setenv('SGLANG_QSA_ONLY_CP_REPLICATE', '1' if enabled else '0')
    monkeypatch.setattr(context_mod, 'get_disagg', lambda: NS(disaggregation_mode=role))
    monkeypatch.setattr(context_mod, 'get_server_args', lambda: NS(kv_cache_dtype='fp8_e4m3'))
    monkeypatch.setattr(unquant, 'precompile_splitk_tactics', lambda: False)
    monkeypatch.setattr(warmup_mod, 'warmup_qsa_cp_prefill', lambda *args: calls.append(args))
    Qwen4ExpForConditionalGeneration.precompile_kernels_after_loading(root)
    assert bool(calls) == expected

@pytest.mark.parametrize('n,width', [(0, 2560), (1, 2560), (7, 31), (128, 2560), (8192, 2560)])
def test_shared_gate_add_rounding(n, width):
    from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import sigmoid_gate_mul_add_bf16
    main = torch.randn(n, width, device='cuda', dtype=torch.bfloat16)
    shared = torch.randn_like(main)
    gate = torch.randn(n, 1, device='cuda', dtype=torch.bfloat16)
    gold = main.clone().add_(torch.sigmoid(gate) * shared)
    before_shared = shared.clone()
    before_gate = gate.clone()
    assert sigmoid_gate_mul_add_bf16(main, shared, gate) is main
    assert torch.equal(main, gold) and torch.equal(shared, before_shared) and torch.equal(gate, before_gate)
