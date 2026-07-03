import logging
import time
from typing import TYPE_CHECKING, List

import torch.cuda

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ExpertLocationMetadata

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self._model_runner = model_runner
        self._server_args = model_runner.server_args
        self._rebalance_layers_per_chunk = (
            self._server_args.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_num_iterations = self._server_args.eplb_rebalance_num_iterations

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            self._server_args.eplb_rebalance_num_iterations
            >= self._server_args.expert_distribution_recorder_buffer_size
        ), "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )

        self._prepared_rebalance_metadata = None
        self._prepared_rebalance_apply_event = None
        self._prepared_update_layer_ids_chunks = None
        self._pending_logical_count = None
        self._pending_logical_count_ready_event = None
        self._prepare_stream = (
            torch.cuda.Stream()
            if self._server_args.enable_eplb_async and torch.cuda.is_available()
            else None
        )
        self._async_main_generator = self._async_entrypoint()
        self._main_generator = self._entrypoint()

    def on_forward_pass_end(self):
        if self._server_args.enable_eplb_async:
            next(self._async_main_generator)
            return
        next(self._main_generator)

    def reset_generator(self):
        self._prepared_rebalance_metadata = None
        self._prepared_rebalance_apply_event = None
        self._prepared_update_layer_ids_chunks = None
        self._pending_logical_count = None
        self._pending_logical_count_ready_event = None
        self._async_main_generator = self._async_entrypoint()
        self._main_generator = self._entrypoint()

    # can be more complex if needed
    def _async_entrypoint(self):
        while True:
            forward_pass_id = self._model_runner.forward_pass_id
            if self._prepared_rebalance_metadata is not None:
                self._apply_prepared_async_rebalance()
                logger.info("[EPLBManager] async rebalance end")
            if self._pending_logical_count is not None:
                self._prepare_async_rebalance()
            if (
                forward_pass_id % self._rebalance_num_iterations == 0
                and self._pending_logical_count is None
                and self._prepared_rebalance_metadata is None
            ):
                logger.info("[EPLBManager] async rebalance start")
                self._start_async_rebalance_logical_count_fetch()
            yield

    def _entrypoint(self):
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield

            yield from self.rebalance()

    def rebalance(self):
        mode = "async" if self._server_args.enable_eplb_async else "sync"
        logger.info(f"[EPLBManager] rebalance start mode={mode}")

        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.get_device_module().synchronize()
            time_start = time.time()

        dump_record_output = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )
        logical_count = dump_record_output["logical_count"]
        average_utilization_rate_over_window = dump_record_output[
            "average_utilization_rate_over_window"
        ]

        # Check whether rebalancing is needed
        if not self._check_rebalance_needed(average_utilization_rate_over_window):
            return

        expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.get_device_module().synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def _prepare_async_rebalance(self):
        if self._pending_logical_count is None:
            return
        if (
            self._pending_logical_count_ready_event is not None
            and not self._pending_logical_count_ready_event.query()
        ):
            return

        logical_count = self._pending_logical_count
        logical_count_ready_event = self._pending_logical_count_ready_event
        self._pending_logical_count = None
        self._pending_logical_count_ready_event = None

        (
            self._prepared_rebalance_metadata,
            self._prepared_rebalance_apply_event,
        ) = self._init_async_prepare_expert_location_metadata(
            logical_count,
            logical_count_ready_event=logical_count_ready_event,
        )
        self._prepared_update_layer_ids_chunks = self._compute_update_layer_ids_chunks()

    def _start_async_rebalance_logical_count_fetch(self):
        dump_record_output = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )
        logical_count = dump_record_output["logical_count"]
        average_utilization_rate_over_window = dump_record_output[
            "average_utilization_rate_over_window"
        ]

        if not self._check_rebalance_needed(average_utilization_rate_over_window):
            self._pending_logical_count = None
            self._pending_logical_count_ready_event = None
            self._prepared_rebalance_metadata = None
            self._prepared_rebalance_apply_event = None
            self._prepared_update_layer_ids_chunks = None
            return

        self._pending_logical_count = logical_count
        self._pending_logical_count_ready_event = (
            self._record_logical_count_ready_event(logical_count)
        )

    def _record_logical_count_ready_event(self, logical_count: torch.Tensor):
        if logical_count.device.type != "cuda":
            return None
        ready_event = torch.cuda.Event()
        torch.cuda.current_stream(device=logical_count.device).record_event(ready_event)
        return ready_event

    def _init_async_prepare_expert_location_metadata(
        self,
        logical_count: torch.Tensor,
        logical_count_ready_event=None,
    ):
        if self._prepare_stream is None:
            return (
                ExpertLocationMetadata.init_by_eplb(
                    self._server_args, self._model_runner.model_config, logical_count
                ),
                None,
            )
        with torch.cuda.stream(self._prepare_stream):
            if logical_count_ready_event is not None:
                self._prepare_stream.wait_event(logical_count_ready_event)
            metadata = ExpertLocationMetadata.init_by_eplb(
                self._server_args, self._model_runner.model_config, logical_count
            )
            apply_event = torch.cuda.Event()
            apply_event.record(self._prepare_stream)
        return metadata, apply_event

    def _apply_prepared_async_rebalance(self):
        if self._prepared_rebalance_metadata is None:
            return
        if self._prepared_rebalance_apply_event is not None:
            self._prepared_rebalance_apply_event.synchronize()
        for update_layer_ids in self._prepared_update_layer_ids_chunks:
            self._model_runner.update_expert_location(
                self._prepared_rebalance_metadata,
                update_layer_ids=update_layer_ids,
            )
        self._prepared_rebalance_metadata = None
        self._prepared_rebalance_apply_event = None
        self._prepared_update_layer_ids_chunks = None

    def _check_rebalance_needed(self, average_utilization_rate_over_window):
        if average_utilization_rate_over_window is None:
            return True

        if (
            average_utilization_rate_over_window
            > self._server_args.eplb_min_rebalancing_utilization_threshold
        ):
            logger.info(
                f"[EPLBManager] Skipped ep rebalancing: current GPU utilization {average_utilization_rate_over_window:.2f} > minimum rebalance threshold {self._server_args.eplb_min_rebalancing_utilization_threshold:.2f}"
            )
            return False

        return True

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        routed_experts_weights_of_layer = getattr(
            self._model_runner.model, "routed_experts_weights_of_layer", None
        )
        if not routed_experts_weights_of_layer:
            return []
        all_layer_ids = sorted(list(routed_experts_weights_of_layer.keys()))
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
