import asyncio
import os
from unittest.mock import patch
from types import SimpleNamespace
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import GenerateReqInput, ExpandedInputLenReq, msgpack_encode, msgpack_decode
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs


@patch.dict(os.environ, {"SGLANG_DISAGG_EXPANDED_INPUT_LEN": "1"})
def test_decode_preserves_placeholders_until_prefill_metadata():
    publish(ServerArgs(model_path="dummy", language_only=True,
                       disaggregation_mode="decode", image_pruning_rate=.5), role="tokenizer")
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.disaggregation_mode = DisaggregationMode.DECODE
    tm.mm_processor = SimpleNamespace(generates_input_ids_from_raw_prompt=True)
    tm.tokenizer = object()
    tm.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"]))
    async def tokenize(*args):
        return [11, 12, 13], None
    tm._tokenize_texts = tokenize
    tm._validate_one_request = lambda obj, ids: None
    tm._create_tokenized_object = lambda obj, text, ids, embeds, mm, types: (ids, mm)
    obj = GenerateReqInput(text="image prompt", image_data=["unused"], sampling_params={})
    obj._needs_expanded_input_len = True
    assert asyncio.run(tm._tokenize_one_request(obj)) == ([11, 12, 13], None)


def test_expanded_metadata_uses_native_scheduler_ipc():
    tm = TokenizerManager.__new__(TokenizerManager)
    sent = []
    tm._dispatch_to_scheduler = sent.append
    tm.notify_expanded_input_len("room", 556, 512, "generation-1", 2)
    decoded = msgpack_decode(msgpack_encode(sent[0]))
    assert isinstance(decoded, ExpandedInputLenReq)
    assert (decoded.rid, decoded.expanded_input_len, decoded.mrope_position_delta, decoded.metadata_version) == ("room", 556, 512, 2)
    assert decoded.request_generation == "generation-1"
