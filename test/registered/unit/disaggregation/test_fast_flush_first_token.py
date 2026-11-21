import sys
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin


def _req(*, offset=0, output_ids=(1,), return_logprob=False):
    return SimpleNamespace(
        send_token_offset=offset,
        output_ids_through_stop=output_ids,
        return_logprob=return_logprob,
    )


def test_fast_flush_transferred_first_tokens_groups_by_logprob():
    calls = []
    scheduler = SimpleNamespace(
        output_streamer=SimpleNamespace(
            stream_output=lambda reqs, return_logprob: calls.append(
                (reqs, return_logprob)
            )
        )
    )
    plain_a = _req()
    plain_b = _req(output_ids=(2,))
    with_logprob = _req(return_logprob=True)
    already_sent = _req(offset=1)
    no_output = _req(output_ids=())

    SchedulerDisaggregationDecodeMixin.fast_flush_transferred_first_tokens(
        scheduler, [plain_a, with_logprob, already_sent, no_output, plain_b]
    )

    assert calls == [([plain_a, plain_b], False), ([with_logprob], True)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
