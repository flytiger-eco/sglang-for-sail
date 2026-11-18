"""Wire, prompt identity and stale-owner checks for independent-E readiness."""
import os
import types
import unittest
from unittest.mock import patch

from sglang.srt.disaggregation.encoder.receiver import _select_mm_processor_prompt
from sglang.srt.disaggregation.expanded_input import validate_expanded_input_config
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import PrefillMMReadyOutput, msgpack_decode, msgpack_encode
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class TestIndependentEncoderReady(unittest.TestCase):
    def test_wire_roundtrip_and_raw_prompt_identity(self):
        m = PrefillMMReadyOutput(rid='r', bootstrap_room=42, bootstrap_pair_key='k',
                                expanded_input_len=45624, mrope_position_delta=-32)
        self.assertEqual(msgpack_decode(msgpack_encode(m)), m)
        proc = types.SimpleNamespace(prefer_tokenized_input=False)
        r = types.SimpleNamespace(input_ids=[11, 12], input_text='raw',
                                  epd_prompt_pretokenized=True)
        self.assertEqual(_select_mm_processor_prompt(r, proc), [11, 12])
        r.epd_prompt_pretokenized = False
        self.assertEqual(_select_mm_processor_prompt(r, proc), 'raw')
        r.input_ids = None
        r.epd_prompt_pretokenized = True
        self.assertEqual(_select_mm_processor_prompt(r, proc), 'raw')

    @patch.dict(os.environ, {'SGLANG_DISAGG_EXPANDED_INPUT_LEN': '1'})
    def test_canceled_and_reused_owner_ignores_late_metadata(self):
        seen = []
        obj = types.SimpleNamespace(bootstrap_room=42, bootstrap_pair_key='k')
        state = types.SimpleNamespace(obj=obj, finished=False)
        manager = types.SimpleNamespace(
            rid_to_state={'r': state}, disaggregation_mode=DisaggregationMode.PREFILL,
            _notify_decode_expanded_len=lambda *a: a,
            _track_expanded_len_task=lambda coroutine, rid: seen.append((coroutine, rid)),
        )
        m = PrefillMMReadyOutput(rid='r', bootstrap_room=42, bootstrap_pair_key='k',
                                expanded_input_len=45624, mrope_position_delta=-32)
        TokenizerManager._handle_prefill_mm_ready(manager, m)
        self.assertEqual(seen, [(('42', 45624, 'r', -32, 'k'), 'r')])
        state.finished = True
        TokenizerManager._handle_prefill_mm_ready(manager, m)
        state.finished = False
        obj.bootstrap_pair_key = 'new-owner'
        TokenizerManager._handle_prefill_mm_ready(manager, m)
        manager.rid_to_state = {}
        TokenizerManager._handle_prefill_mm_ready(manager, m)
        self.assertEqual(len(seen), 1)

    @patch.dict(os.environ, {'SGLANG_DISAGG_EXPANDED_INPUT_LEN': '1', 'SGLANG_RUST_SERVER': '0'})
    def test_scheduler_backend_remains_gated_to_qualified_layout(self):
        cfg = types.SimpleNamespace(disaggregation_mode='prefill', language_only=True,
                                    encoder_transfer_backend='zmq_to_scheduler',
                                    pp_size=1, attn_cp_size=1,
                                    disagg_decode_urls=['http://127.0.0.1:29850'])
        validate_expanded_input_config(cfg, 'Qwen4ExpForConditionalGeneration')
        for field, value in [('pp_size', 2), ('attn_cp_size', 2),
                             ('encoder_transfer_backend', 'mooncake')]:
            old = getattr(cfg, field)
            setattr(cfg, field, value)
            with self.assertRaises(ValueError):
                validate_expanded_input_config(cfg, 'Qwen4ExpForConditionalGeneration')
            setattr(cfg, field, old)
        with self.assertRaises(ValueError):
            validate_expanded_input_config(cfg, 'Qwen3VLForConditionalGeneration')


if __name__ == '__main__':
    unittest.main()
