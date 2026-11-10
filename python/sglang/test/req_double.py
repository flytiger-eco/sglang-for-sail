"""Small request fixtures that implement the scheduler's prompt-length contract."""

from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import Req


class ReqDouble(SimpleNamespace):
    needs_expanded_input_len = False
    expanded_input_len = None
    origin_input_ids = ()
    skip_radix_cache_insert = False
    effective_input_len = Req.effective_input_len
