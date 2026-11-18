import unittest
from unittest.mock import patch
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.base.conn import KVPoll

class EarlyRejectReplay(unittest.TestCase):
    def setUp(self):
        self.m = MooncakeKVManager.__new__(MooncakeKVManager)
        self.m._init_prealloc_abort_state()
        self.m.request_status = {}
        self.m._staging_outstanding = {}
        self.acks = []
        self.m._send_prealloc_abort_ack = lambda *args: self.acks.append(args)

    def abort(self, room=123, pair='pair-a'):
        self.m._handle_prealloc_abort([b'ABORT', str(room).encode(), b'127.0.0.1', b'12345', b'WAIT_FOR_ROOM', pair.encode()])

    def test_early_reject_ack_and_retries(self):
        self.assertTrue(self.m.reject_before_sender(123, 'pair-a'))
        self.abort(); self.abort()
        self.assertEqual(len(self.acks), 2)
        self.assertFalse(self.m.request_status)

    def test_unknown_and_wrong_pair_are_not_acked(self):
        self.abort()
        self.m.reject_before_sender(123, 'pair-a')
        self.abort(pair='pair-b')
        self.assertFalse(self.acks)

    def test_live_sender_not_overwritten(self):
        self.m.request_status[123] = KVPoll.WaitingForInput
        self.assertFalse(self.m.reject_before_sender(123, 'pair-a'))
        self.assertFalse(self.m._prealloc_abort_replays)

    def test_outstanding_write_not_bypassed(self):
        self.m._staging_outstanding[123] = 1
        self.assertFalse(self.m.reject_before_sender(123, 'pair-a'))

    def test_reused_room_old_ack_does_not_abort_new_sender(self):
        self.m.reject_before_sender(123, 'pair-a')
        self.m.request_status[123] = KVPoll.WaitingForInput
        self.m._prealloc_abort_attempts[123] = 'pair-b'
        self.abort()
        self.assertEqual(len(self.acks), 1)
        self.assertEqual(self.m.request_status[123], KVPoll.WaitingForInput)
        self.assertEqual(self.m._prealloc_abort_attempts[123], 'pair-b')

    def test_capacity_is_bounded(self):
        with patch('sglang.srt.disaggregation.mooncake.conn.EXPANDED_INPUT_MAX_INFLIGHT', 1):
            self.assertTrue(self.m.reject_before_sender(123, 'pair-a'))
            self.assertFalse(self.m.reject_before_sender(124, 'pair-b'))

class ReceiverFailureIdentity(unittest.TestCase):
    def test_error_req_preserves_dispatch_pair_key(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from sglang.srt.disaggregation.encoder.receiver import MMReceiverBase
        receiver = SimpleNamespace(scheduler=MagicMock())
        receiver.scheduler.disaggregation_mode = 'prefill'
        tokenized = MagicMock()
        tokenized.bootstrap_room = 123
        tokenized.bootstrap_pair_key = 'dispatcher-attempt-a'
        tokenized.needs_expanded_input_len = False
        tokenized.need_wait_for_mm_inputs = True
        tokenized.mm_inputs = None
        with patch('sglang.srt.disaggregation.encoder.receiver.Req') as constructor:
            req = MMReceiverBase.create_req(receiver, tokenized)
        self.assertEqual(req._expanded_input_pair_key, 'dispatcher-attempt-a')
        self.assertEqual(constructor.call_args.kwargs['bootstrap_room'], 123)

if __name__ == '__main__': unittest.main(verbosity=2)
