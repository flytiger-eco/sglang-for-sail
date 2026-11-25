from types import SimpleNamespace as NS
import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

DEVICES=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
def run(runner,deltas,positions=None,explicit=False,check=True):
    n=len(deltas)
    if positions is None:positions=torch.arange(n*3,device=runner.device,dtype=torch.int64)
    mm=[None if v is None else NS(mrope_position_delta=v if isinstance(v,torch.Tensor) else torch.tensor([[v]],dtype=torch.int64)) for v in deltas]
    batch=NS(multimodal_inputs=mm,spec_info=NS(positions=positions))
    out=NS(seq_lens=torch.zeros(n,device=runner.device,dtype=torch.int32))
    ForwardBatch.compute_spec_mrope_positions(out,runner,batch,**({'seq_positions':positions} if explicit else {}))
    expected=(positions.reshape(n,-1)+torch.tensor([0 if v is None else int(v.item()) if isinstance(v,torch.Tensor) else v for v in deltas],device=runner.device).view(n,1)).flatten().unsqueeze(0).repeat(3,1)
    if check:assert torch.equal(out.mrope_positions,expected)
    out.expected=expected
    return out

@pytest.mark.parametrize('device',DEVICES)
def test_mixed_batch_cache_reorder_and_reused_rows(device):
    runner=NS(device=device)
    for ds in [[-82,None,100],[-82,None,100],[100,-82,None],[None,None,None],[17,None,-8],[-82,None,100]]:
        run(runner,ds)
    old=runner._spec_mrope_delta_cache[1]
    run(runner,[-82,None,100],torch.arange(9,device=device)+400)
    assert old is runner._spec_mrope_delta_cache[1]

@pytest.mark.parametrize('device',DEVICES)
def test_delta_mutation_and_explicit_draft_extend_positions(device):
    runner=NS(device=device);delta=torch.tensor([[-8]],dtype=torch.int64)
    run(runner,[delta,None],explicit=True);old=runner._spec_mrope_delta_cache[1]
    delta.add_(5)
    run(runner,[delta,None],explicit=True)
    assert runner._spec_mrope_delta_cache[1] is not old
    run(runner,[None,None],explicit=True)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cross_stream_read_and_cache_replacement():
    runner=NS(device='cuda');a=torch.cuda.Stream();b=torch.cuda.Stream()
    with torch.cuda.stream(a):
        busy=torch.randn((1024,1024),device="cuda");torch.mm(busy,busy)
        x=run(runner,[-128,None,77],check=False)
    with torch.cuda.stream(b):y=run(runner,[-128,None,77],check=False)
    with torch.cuda.stream(a):z=run(runner,[54,-31,None],check=False)
    with torch.cuda.stream(b):w=run(runner,[54,-31,None],explicit=True,check=False)
    torch.cuda.synchronize()
    assert torch.equal(x.mrope_positions,y.mrope_positions)
    assert torch.equal(z.mrope_positions,w.mrope_positions)

    for out in [x,y,z,w]:assert torch.equal(out.mrope_positions,out.expected)
