"""PP Prefill relays authoritative expanded IDs/MRoPE to its next stage."""

from array import array
import pickle
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.disaggregation.expanded_input import (
    optimistic_prefill_supported,
    validate_expanded_input_config,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def context():
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    yield
    reset_context()


@pytest.mark.parametrize("pp_size", [2, 4])
def test_expanded_input_accepts_pp_prefill_without_optimistic_bootstrap(pp_size):
    args = ServerArgs(
        model_path="dummy",
        disaggregation_mode="prefill",
        pp_size=pp_size,
        language_only=True,
        encoder_transfer_backend="zmq_to_tokenizer",
        disagg_decode_urls=["http://decode"],
    )
    cfg = resolving_view(args)
    with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True):
        validate_expanded_input_config(cfg, "Qwen4ExpForConditionalGeneration")
    assert not optimistic_prefill_supported(cfg)


def test_expanded_input_decode_pipeline_guard_remains_feature_scoped():
    cfg = resolving_view(
        ServerArgs(model_path="dummy", disaggregation_mode="decode", pp_size=2)
    )
    with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(False):
        validate_expanded_input_config(cfg, "Qwen4ExpForConditionalGeneration")
    with envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True):
        with pytest.raises(ValueError, match="Decode.*pipeline"):
            validate_expanded_input_config(cfg, "Qwen4ExpForConditionalGeneration")


def test_pp_relay_preserves_expanded_ids_pair_identity_and_mrope():
    request = TokenizedGenerateReqInput(
        rid="prefill",
        input_text=None,
        input_ids=array("q", range(262)),
        input_embeds=None,
        mm_inputs=MultimodalProcessorOutput(
            mm_items=[],
            input_ids=list(range(262)),
            mrope_position_delta=torch.tensor(-42),
        ),
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=True,
        bootstrap_room=7,
        bootstrap_pair_key="original-pair",
    )
    group = NS(cpu_group=object())
    scheduler = NS(world_group=group)
    first = NS(
        pp_rank=0, pp_size=2, tp_size=1, attn_dp_rank=0,
        attn_cp_rank=0, attn_tp_rank=0, attn_cp_size=1, attn_tp_size=1,
    )
    last = NS(**dict(vars(first), pp_rank=1))
    wire = {}

    def send(data, rank, group, src, dst, **kwargs):
        assert (rank, src, dst) == (0, 0, 1)
        wire["request"] = pickle.dumps(data)
        return []

    def receive(data, rank, group, src, dst):
        assert (rank, src, dst) == (1, 0, 1)
        return pickle.loads(wire["request"])

    with (
        patch("sglang.srt.managers.scheduler_pp_mixin.get_parallel", return_value=first),
        patch("sglang.srt.managers.scheduler_pp_mixin.point_to_point_pyobj", side_effect=send),
    ):
        SchedulerPPMixin._pp_send_pyobj_to_next_stage(scheduler, [request])
    with (
        patch("sglang.srt.managers.scheduler_components.request_receiver.get_parallel", return_value=last),
        patch("sglang.srt.managers.scheduler_components.request_receiver.point_to_point_pyobj", side_effect=receive),
    ):
        [received] = SchedulerRequestReceiver._pull_raw_reqs(scheduler)
    assert list(received.input_ids) == list(range(262))
    assert received.mm_inputs.input_ids == list(range(262))
    assert received.mm_inputs.mrope_position_delta.item() == -42
    assert (received.bootstrap_room, received.bootstrap_pair_key) == (7, "original-pair")
