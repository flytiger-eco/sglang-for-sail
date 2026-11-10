"""Exercise actual cache/encoder/receiver methods without initializing a model."""
import ast
import asyncio
import enum
import itertools
import pathlib
import types
from collections import defaultdict
from typing import List, Optional

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3] / 'python/sglang'

class Modality(enum.Enum):
    IMAGE = 1
    VIDEO = 2
    AUDIO = 3

def load_nodes(path, names, ns):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if getattr(n, 'name', None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns

def encoder():
    tree = ast.parse((ROOT / 'srt/disaggregation/encoder/server.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MMEncoder')
    methods = [n for n in cls.body if getattr(n,'name','') in ('_collect_image_pruning_indices','_validate_image_pruning_indices','_apply_pruning_metadata')]
    cls.body = methods
    ns = dict(torch=torch, Modality=Modality, List=List, Optional=Optional, time=__import__('time'), encoder_metrics_collector=None)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'encoder_methods', 'exec'), ns)
    obj = ns['MMEncoder']()
    obj.image_pruning_rate = .4
    obj.preprocessor = types.SimpleNamespace(get_num_tokens=lambda grid, modality: int(grid[0]*grid[1]*grid[2]//4))
    return obj

def test_indices_validation_and_feature_split():
    e = encoder()
    grids = [[1,4,4],[1,4,6]]  # full 4 and 6; keep 2 and 3
    kept = [[0,3],[1,3,5]]
    features = torch.arange(35).reshape(5,7)  # includes any DeepStack columns
    item = types.SimpleNamespace(image_pruning_indices=kept)
    assert e._collect_image_pruning_indices([item],grids,features) == kept
    inputs={}
    ctx=types.SimpleNamespace(
        preprocess_result=types.SimpleNamespace(grid_thw=grids, mm_inputs=inputs, token_counts=[4,6]),
        aux_data={},
    )
    e._apply_pruning_metadata(ctx,kept,features)
    assert inputs['image_pruning_indices'] == kept
    assert ctx.aux_data['image_pruning_indices'] == kept
    assert ctx.preprocess_result.token_counts == [2,3]
    assert torch.equal(torch.cat(features.split(ctx.preprocess_result.token_counts)),features)

@pytest.mark.parametrize('kept', [None, [[0,3]], [[3,0],[1,3,5]], [[0,0],[1,3,5]], [[0,4],[1,3,5]], [[0,True],[1,3,5]]])
def test_invalid_metadata_rejected(kept):
    with pytest.raises(ValueError):
        encoder()._validate_image_pruning_indices(kept,[[1,4,4],[1,4,6]],torch.zeros(5,2))

def test_embedding_cache_roundtrip_and_eviction():
    path=ROOT/'srt/mem_cache/multimodal_cache.py'
    source=path.read_text().replace('from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator','BaseTokenToKVPoolAllocator = object')
    ns={'__name__':__name__}
    exec(compile(source,str(path),'exec'),ns)
    cache=ns['MultiModalStaticCache'](40)
    indices=[[0,3]]
    value=ns['EmbeddingResult'](embedding=torch.zeros(2,4),image_pruning_indices=indices)
    key=cache.combine_hashes([123,456])
    assert cache.set(key,value)
    assert cache.get([123,456]).image_pruning_indices == indices
    assert cache.get([123,789]) is None
    cache.set(cache.combine_hashes([789]),ns['EmbeddingResult'](embedding=torch.zeros(2,4),image_pruning_indices=[[1,2]]))
    assert cache.get([123,456]) is None
    assert cache.get([789]).image_pruning_indices == [[1,2]]

def test_out_of_order_parts_and_transport_copy():
    ns=dict(torch=torch,Modality=Modality,Optional=Optional,defaultdict=defaultdict,itertools=itertools,logger=__import__('logging').getLogger())
    tree=ast.parse((ROOT/'srt/disaggregation/encoder/receiver.py').read_text())
    names={'EmbeddingData','MultiModalEmbeddingData','_cat_grid','video_meta_attrs_for','_validate_embedding_part'}
    consts={'_MODALITY_GRID_ATTRS','_GENERAL_VIDEO_META_ATTRS','_GENERAL_IMAGE_META_ATTRS','_MIMO_VIDEO_AUDIO_META_ATTRS','_VIDEO_META_TENSOR_ATTRS'}
    nodes=[n for n in tree.body if getattr(n,'name',None) in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in consts for t in n.targets)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'receiver_nodes','exec'),ns)
    Data=ns['EmbeddingData'];Multi=ns['MultiModalEmbeddingData']
    second=Data('r',2,1,torch.tensor([[1,4,6]]),Modality.IMAGE,torch.ones(3,2),image_pruning_indices=[[1,3,5]])
    first=Data('r',2,0,torch.tensor([[1,4,4]]),Modality.IMAGE,torch.zeros(2,2),image_pruning_indices=[[0,3]])
    assert first.copy_without_embedding().image_pruning_indices == [[0,3]]
    merged=Multi.from_embedding_data(second,model_type='qwen4_exp')
    assert not merged.ready
    merged.add(first)
    assert merged.ready
    assert merged.get_mm_extra_meta()['image_pruning_indices'] == [[0,3],[1,3,5]]
    assert torch.equal(merged.get_embedding(True)[Modality.IMAGE],torch.cat([first.embedding,second.embedding]))
