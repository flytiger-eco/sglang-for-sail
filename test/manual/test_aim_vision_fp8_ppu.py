
"""Real PPU regression coverage for online-FP8 vision linear shapes and bias."""
import pytest
import torch
from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear, sglang_per_token_quant_fp8
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import is_ppu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not is_ppu(), reason="requires PPU")

@pytest.mark.parametrize("shape", [(17,128), (17,1,128), (2,9,128)])
@pytest.mark.parametrize("with_bias", [False,True])
@pytest.mark.parametrize("scale_column", [False,True])
def test_ppu_online_vision_fp8_linear(shape, with_bias, scale_column):
    torch.manual_seed(123)
    with get_context().override_server_args(model_path="dummy"):
        x=torch.randn(shape,device="cuda",dtype=torch.bfloat16)
        w=torch.randn((256,128),device="cuda").to(torch.float8_e4m3fn).t()
        ws=torch.full((256,1) if scale_column else (256,),0.1,device="cuda")
        qx,xs=sglang_per_token_quant_fp8(x.reshape(-1,128))
        bias=torch.randn((256,),device="cuda",dtype=torch.bfloat16) if with_bias else None
        y=apply_fp8_linear(input=x,weight=w,weight_scale=ws,input_scale=None,bias=bias,cutlass_fp8_supported=True,use_per_token_if_dynamic=True)
        reference=(qx.float()*xs.reshape(-1,1)) @ (w.float()*ws.reshape(1,-1))
        if bias is not None: reference+=bias.float()
        reference=reference.reshape(*shape[:-1],256).to(x.dtype)
        torch.testing.assert_close(y,reference,rtol=0.04,atol=0.04)


@pytest.mark.parametrize("width", [17, 1152, 3456, 4304])
def test_channel_bias_strided_and_compiled(width):
    from sglang.kernels.ops.elementwise.channel_bias import add_channel_bias_
    x = torch.randn(33, width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(width * 2, device="cuda", dtype=torch.float32)[::2]
    expected = (x.float() + bias).to(x.dtype)
    actual = x.clone()
    add_channel_bias_(actual, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_channel_bias_survives_compile_and_cuda_graph():
    from sglang.kernels.ops.elementwise.channel_bias import add_channel_bias_
    def add(output, bias):
        add_channel_bias_(output, bias)
        return output
    compiled = torch.compile(add, fullgraph=True)
    x = torch.ones((17, 1152), device="cuda", dtype=torch.bfloat16)
    bias = torch.full((1152,), .5, device="cuda", dtype=torch.bfloat16)
    compiled(x, bias)
    torch.testing.assert_close(x, torch.full_like(x, 1.5))
    x.fill_(1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled(x, bias)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(x, bias)
    x.fill_(2)
    graph.replay()
    torch.testing.assert_close(x, torch.full_like(x, 2.5))
