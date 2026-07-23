# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ModelRunner runs the forward passes of the models."""

from __future__ import annotations

import contextlib
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed as dist

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    ModelImpl,
)
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.distributed import bootstrap
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    maybe_init_shared_mooncake_transfer_engine,
)
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    get_healthy_expert_location_src_rank,
    get_scale_cohort_target,
    join_process_groups,
    join_scale_process_group,
    maybe_rebalance_after_rank_fault,
    maybe_recover_ep_ranks,
    register_scale_cohort,
    try_admit_scale_ranks,
)
from sglang.srt.elastic_ep.expert_backup_client import ExpertBackupClient
from sglang.srt.environ import envs
from sglang.srt.eplb import eplb_algorithms
from sglang.srt.eplb.cpp_async_runtime import warmup_eplb_async_runtime_cpp
from sglang.srt.eplb.cpp_deepseek import warmup_eplb_deepseek_cpp
from sglang.srt.eplb.cpp_expert_location import warmup_eplb_expert_location_cpp
from sglang.srt.eplb.eplb_async_host_mirror import (
    EPLBAsyncHostMirrorManager,
    set_global_eplb_async_host_mirror_manager,
)
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionMetrics,
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    append_trivial_expert_slots,
    ModelConfigForExpertLocation,
    broadcast_global_expert_location_metadata,
    compute_initial_expert_location_metadata,
    format_expert_location_layout,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import (
    ExpertLocationUpdater,
    set_global_expert_location_updater,
)
from sglang.srt.eplb.lplb_solver import (
    LPLBSolver,
    assert_lplb_supported_model,
    clear_global_lplb_solvers,
    set_global_lplb_solver,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.hardware_backend.xpu.graph_runner.xpu_graph_runner import XPUGraphRunner
from sglang.srt.kv_canary.api import install_canary
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.kv_canary.token_oracle.install import install_token_oracle_from_env
from sglang.srt.layers import deep_gemm_wrapper, model_parallel
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.cp.utils import (
    get_cp_strategy,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.layers.utils.cp_utils import is_mla_prefill_cp_enabled
from sglang.srt.lora.lora_manager import LoRAManager, init_lora_cuda_graph_moe_buffers
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import sanity_check_mm_pad_shift_value
from sglang.srt.mem_cache import kv_cache_dtype
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import (
    KVCacheConfigurator,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import (
    cuda_graph_fully_disabled,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.model_runner_components import misc_utils
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    build_attention_backends,
    configure_aux_hidden_state_capture,
    get_attention_backend,
)
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    capture_cuda_graphs,
    capture_decode_graph,
    capture_prefill_graph,
)
from sglang.srt.model_executor.model_runner_components.kv_pool_runtime import (
    compute_post_capture_kv_resize,
    is_post_capture_kv_active,
)
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    ModelLayerInfo,
    adjust_hybrid_swa_layer_ids,
    resolve_layer_indices,
)
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    build_load_config,
    dist_barrier_after_load,
    load_kv_cache_scales,
    load_model_with_memory_saver,
    maybe_downgrade_dtype_for_legacy_gpu,
    maybe_register_debug_tensor_dump_hook,
    maybe_trigger_remote_instance_nccl_send_group,
    report_online_quantization,
    resolve_sliding_window_size,
)
from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
    check_quantized_moe_compatibility,
    init_lplb_solvers,
    prepare_moe_topk,
)
from sglang.srt.model_executor.model_runner_components.ngram_embedding_manager import (
    NgramEmbeddingManager,
)
from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
    RemoteInstanceWeightTransporter,
)
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
    resolve_spec_aux_hidden_state_config,
)
from sglang.srt.model_executor.model_runner_components.weight_exporter import (
    WeightExporter,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.model_executor.runner import (
    EagerRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.model_executor.runner_utils import (
    get_is_capture_mode,
)
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    register_memory_region,
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_server_args
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import (  # noqa: F401  (re-export)
    CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS,
    ServerArgs,
    add_chunked_prefix_cache_attention_backend,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import resolve_num_tokens_per_req
from sglang.srt.state_capturer.base import TopkCaptureOutput
from sglang.srt.state_capturer.indexer_topk import (
    create_indexer_capturer,
    get_global_indexer_capturer,
    set_global_indexer_capturer,
)
from sglang.srt.state_capturer.routed_experts import (
    RoutedExpertsCapturer,
    disable_routed_experts_capture_for_draft,
    get_global_experts_capturer,
    set_global_experts_capturer,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    enable_show_time_cost,
    get_available_gpu_memory,
    is_host_cpu_arm64,
    is_npu,
    numa_utils,
    require_gathered_buffer,
    reserve_rope_cache_for_long_sequences,
    set_cuda_arch,
    slow_rank_detector,
)
from sglang.srt.utils.nvtx_pytorch_hooks import PytHooks
from sglang.srt.utils.nvtx_utils import profile_range
from sglang.srt.utils.offloader import (
    create_offloader_from_server_args,
    get_offloader,
    set_offloader,
)
from sglang.srt.utils.profile_utils import build_step_span_name
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_checker import WeightChecker

_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    init_npu_backend()
elif current_platform.is_out_of_tree():
    current_platform.init_backend()

MLA_ATTENTION_BACKENDS = [
    "aiter",
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "flashmla",
    "cutedsl_mla",
    "cutlass_mla",
    "trtllm_mla",
    "tokenspeed_mla",
    "ascend",
    "dsa",
    "nsa",  # Deprecated alias for "dsa"
    "intel_xpu",
]


def add_mla_attention_backend(backend_name):
    if backend_name not in MLA_ATTENTION_BACKENDS:
        MLA_ATTENTION_BACKENDS.append(backend_name)
        logger.info(f"Added {backend_name} to MLA_ATTENTION_BACKENDS.")


SGLANG_PROFILE_NVTX = envs.SGLANG_PROFILE_NVTX.get()
if SGLANG_PROFILE_NVTX:
    try:
        from model_prof import prof_iter

        use_model_prof = True

    except ImportError as e:
        use_model_prof = False

    try:
        from torch.cuda.nvtx import range_pop as th_nvtx_range_pop
        from torch.cuda.nvtx import range_push as th_nvtx_range_push

    except ImportError as e:
        SGLANG_PROFILE_NVTX = False


# Detect stragger ranks in model loading


logger = logging.getLogger(__name__)


@dataclass
class ModelRunnerOutput:
    logits_output: Union[LogitsProcessorOutput, PPProxyTensors]
    can_run_graph: bool
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None
    routed_experts_output: Optional[TopkCaptureOutput] = None
    indexer_topk_output: Optional[TopkCaptureOutput] = None


class ModelRunner:
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        server_args: ServerArgs,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        draft_model_idx: Optional[int] = None,
    ):
        # Parse args
        self.mem_fraction_static = mem_fraction_static
        # Set on target by `_resolve_memory_pool_config`; passed in for draft
        # workers so they reuse target's resolved sizes (replaces legacy
        # `server_args._draft_pool_config` mutation hack).
        self.memory_pool_config = memory_pool_config
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.dcp_size = server_args.dcp_size
        self.dcp_rank = ps.tp_rank % self.dcp_size
        self.ps = ps
        self.model_config = model_config
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.device_timer = None
        self.is_multimodal = model_config.is_multimodal
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.capture_tail_hooks = []
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid_swa = model_config.is_hybrid_swa
        self.is_hybrid_swa_compress = model_config.is_hybrid_swa_compress
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size
        self.enable_elastic_ep = server_args.elastic_ep_backend is not None
        self.forward_pass_id = 0
        self._pending_elastic_scale_update = None
        self.init_new_workspace = False
        self.draft_model_idx = draft_model_idx
        self.enable_hisparse = server_args.enable_hisparse

        self.init_remote_instance_weight_transporter()

        self.init_msprobe()

        # auxiliary hidden capture mode. TODO: expose this to server args?
        self.init_spec_aux_hidden_state()
        if SGLANG_PROFILE_NVTX:
            self.iteration = 0

        # Apply the rank zero filter to logger
        if server_args.show_time_cost:
            enable_show_time_cost()

        misc_utils.maybe_disable_chunked_prefix_cache(
            server_args=server_args,
            use_mla_backend=self.use_mla_backend,
            is_draft_worker=self.is_draft_worker,
        )

        # Set the global server_args in the scheduler process (target worker
        # only, so a draft init cannot clobber target-derived global state).
        if not self.is_draft_worker:
            set_global_server_args_for_scheduler(server_args)

        # Init OpenMP threads binding for CPU
        if self.device == "cpu":
            self.init_threads_binding()

        # Set float32 matmul precision
        if get_server_args().enable_tf32_matmul:
            torch.set_float32_matmul_precision("high")

        # Get available memory before model loading.
        # Stored for later use by alloc_memory_pool().
        self.init_torch_distributed()

        # Initialize MooncakeTransferEngine
        self.init_shared_mooncake_transfer_engine()

        # Init forward stream for overlap schedule
        self.forward_stream = torch.get_device_module(self.device).Stream()

        # WAR fast-path: a decode-graph forward publishes a fresh event here after
        # load_batch; the scheduler's WAR barrier waits on it (then clears it)
        # instead of the whole-forward wait_stream. None -> whole-forward fallback.
        self.war_fastpath_read_done_event: Optional[torch.cuda.Event] = None

        # CPU offload
        set_offloader(
            create_offloader_from_server_args(server_args, dp_rank=self.ps.dp_rank)
        )

        self._weight_checker = WeightChecker(get_model=lambda: self.model, ps=self.ps)

        if envs.SGLANG_DETECT_SLOW_RANK.get():
            slow_rank_detector.execute()

        # Init mindspore running environment when model impl is "mindspore"
        self.init_mindspore_runner()

        # Update deep gemm configure
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id, server_args)

        # For hisparse (must be set before initialize() so CUDA graph capture can see it)
        self.hisparse_coordinator = None

        # Load model weights and configure
        self.initialize()
        self.check_quantized_moe_compatibility()

        self._initialize_elastic_ep_joiner()

        if self.is_multimodal:
            sanity_check_mm_pad_shift_value(self.model_config.vocab_size)

        # Temporary cached values
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )

        if self.ps.pp_size > 1:
            assert (
                self.support_pp
            ), "Pipeline Parallel is not compatible with this model."

        # For weight updates
        self.init_weight_updater()
        self.init_weight_exporter()

    def _initialize_elastic_ep_joiner(self) -> None:
        if not (
            self.server_args.elastic_ep_backend is not None
            and self.server_args.is_ep_joiner
        ):
            return

        is_scale_join = self.server_args.ep_join_mode == "scale"
        if is_scale_join:
            join_effective_ep_size = (
                self.server_args.ep_join_rank_offset + self.ps.tp_size
            )
            dist.barrier(group=self.tp_group.cpu_group)
            if self.ps.tp_rank == 0:
                register_scale_cohort(
                    self.server_args.ep_join_rank_offset,
                    join_effective_ep_size,
                )
            join_scale_process_group()
            self.server_args.override(
                "elastic_ep.scale_join", ep_size=join_effective_ep_size
            )
        else:
            join_process_groups()

        global_ep_rank = self.ps.tp_rank + self.server_args.ep_join_rank_offset
        broadcast_global_expert_location_metadata(
            model_config=self.model_config,
            moe_ep_rank=global_ep_rank,
            src_rank=(
                0
                if is_scale_join
                else get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=True
                )
            ),
        )
        set_global_expert_distribution_recorder(
            ExpertDistributionRecorder.init_new(
                self.server_args,
                get_global_expert_location_metadata(),
                rank=global_ep_rank,
            )
        )

        if not is_scale_join:
            ElasticEPStateManager.instance().reset()
            return

        from sglang.srt.layers.dp_attention import (
            enable_joiner_all_gather,
            update_dp_attention_post_scale,
        )

        enable_joiner_all_gather()
        update_dp_attention_post_scale(
            new_dp_size=join_effective_ep_size,
            new_dp_rank=global_ep_rank,
        )
        self.server_args.override(
            "elastic_ep.scale_join", dp_size=join_effective_ep_size
        )
        if self.eplb_manager is not None:
            self.eplb_manager.disable_rebalance(
                "EPLB rebalance is disabled after elastic EP scale-up"
            )

        state = ElasticEPStateManager.instance()
        if state is not None:
            state.active_ranks.zero_()
            state.active_ranks[:join_effective_ep_size] = 1
            state.snapshot_active_to_last()
            state.sync_active_to_cpu()
            state.scale_phase = "syncing_new_world"
        self._elastic_scale_ready_barrier(
            target_size=join_effective_ep_size,
            log_tag="JOINER",
        )
        if state is not None:
            state.scale_phase = "serving_expanded"

    def init_msprobe(self):
        self.msprobe_debugger = misc_utils.create_msprobe_debugger(self.server_args)

    def init_weight_updater(self):
        self.weight_updater = WeightUpdater(
            tp_rank=self.ps.tp_rank,
            device=self.device,
            gpu_id=self.gpu_id,
            model_config=self.model_config,
            custom_weight_loaders=self.server_args.custom_weight_loader,
            get_model=lambda: self.model,
            update_model_fields=self.update_model_fields,
            recapture_cuda_graph=self.init_decode_cuda_graph,
            get_model_runner=lambda: self,
        )

    def init_spec_aux_hidden_state(self):
        self.spec_aux_config: SpecAuxHiddenStateConfig = (
            resolve_spec_aux_hidden_state_config(
                server_args=self.server_args,
                model_config=self.model_config,
                spec_algorithm=self.spec_algorithm,
                is_draft_worker=self.is_draft_worker,
            )
        )

    def init_weight_exporter(self):
        self.weight_exporter = WeightExporter(
            tp_rank=self.ps.tp_rank,
            tp_size=self.ps.tp_size,
            gpu_id=self.gpu_id,
            get_model_path=lambda: self.model_config.model_path,
            get_model=lambda: self.model,
        )

    def init_remote_instance_weight_transporter(self):
        self.remote_instance_weight_transporter = RemoteInstanceWeightTransporter(
            server_args=self.server_args,
            get_model=lambda: self.model,
            tp_rank=self.ps.tp_rank,
            gpu_id=self.gpu_id,
        )

    def init_ngram_embedding_manager(self):
        self.ngram_embedding_manager = NgramEmbeddingManager.from_model(
            model=self.model,
            model_config=self.model_config,
            req_to_token_pool=self.req_to_token_pool,
            server_args=self.server_args,
            max_running_requests=self.max_running_requests,
            device=self.device,
        )

    def init_kv_cache_configurator(self):
        self.kv_cache_configurator = KVCacheConfigurator(
            device=self.device,
            gpu_id=self.gpu_id,
            ps=self.ps,
            pp_group=self.pp_group,
            model=self.model,
            model_config=self.model_config,
            server_args=self.server_args,
            kv_cache_dtype=self.kv_cache_dtype,
            model_dtype=self.dtype,
            page_size=self.page_size,
            sliding_window_size=self.sliding_window_size,
            spec_algorithm=self.spec_algorithm,
            is_draft_worker=self.is_draft_worker,
            post_capture_kv_active=is_post_capture_kv_active(
                server_args=self.server_args, is_draft_worker=self.is_draft_worker
            ),
            spec_aux_config=self.spec_aux_config,
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_swa_compress=self.is_hybrid_swa_compress,
            use_mla_backend=self.use_mla_backend,
            layer_info=self.layer_info,
            forward_stream=self.forward_stream,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            draft_model_idx=self.draft_model_idx,
        )

    def init_mindspore_runner(self):
        # Init the mindspore runner
        # for now, there is only some communication initialization work
        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE and _is_npu:
            from sglang.srt.model_executor.mindspore_runner import init_ms_distributed

            init_ms_distributed(
                world_size=self.ps.tp_size * self.ps.pp_size,
                rank=self.ps.tp_size * self.ps.pp_rank + self.ps.tp_rank,
                local_rank=self.gpu_id,
                server_args=self.server_args,
                port=self.dist_port,
            )

    def _warmup_eplb_cpp_extensions(self):
        if not self.server_args.enable_eplb or self.is_draft_worker:
            return

        model_config_for_expert_location = (
            ModelConfigForExpertLocation.from_model_config(self.model_config)
        )
        if model_config_for_expert_location is None:
            return

        logger.info("Warming up EPLB C++ extensions.")
        warmup_eplb_expert_location_cpp()

        algorithm = eplb_algorithms.compute_algorithm(
            raw_algorithm=self.server_args.eplb_algorithm,
            num_groups=model_config_for_expert_location.num_groups,
            num_nodes=self.server_args.nnodes,
        )
        if algorithm in [
            eplb_algorithms.EplbAlgorithm.deepseek,
            eplb_algorithms.EplbAlgorithm.deepseek_hierarchical,
        ]:
            warmup_eplb_deepseek_cpp()

        if self.server_args.enable_eplb_async:
            warmup_eplb_async_runtime_cpp()

    def initialize(self):
        self.init_memory_saver_adapter()
        self.maybe_init_remote_instance_transfer_engine()
        self.maybe_init_expert_location_metadata()
        self.maybe_init_lplb_solvers()
        self.maybe_init_eplb_manager()
        self.maybe_init_elastic_ep()
        self.init_token_oracle()

        self.eplb_async_host_mirror_manager = (
            EPLBAsyncHostMirrorManager(self.server_args, self.model_config)
            if self.server_args.enable_eplb_async and (not self.is_draft_worker)
            else None
        )
        if not self.is_draft_worker:
            set_global_eplb_async_host_mirror_manager(
                self.eplb_async_host_mirror_manager
            )
        self.expert_location_updater = ExpertLocationUpdater(
            enable_async=self.server_args.enable_eplb_async
            and not self.is_draft_worker,
        )

        if not self.is_draft_worker:
            set_global_expert_location_updater(self.expert_location_updater)

        self._warmup_eplb_cpp_extensions()
        # Load the model
        self.sampler = create_sampler()
        self.load_model()
        prepare_moe_topk(
            model=self.model,
            model_config=self.model_config,
            server_args=self.server_args,
            moe_ep_size=self.ps.moe_ep_size,
            moe_ep_rank=self.ps.moe_ep_rank,
        )
        # Must run before backend/graph init so no draft graph records a
        # routed-experts capture-write kernel.
        if self.is_draft_worker:
            disable_routed_experts_capture_for_draft(self.model)
        self.maybe_init_expert_backup_client()
        self.remote_instance_weight_transporter.maybe_register_and_publish_weight_info()
        self.layer_info: ModelLayerInfo = resolve_layer_indices(
            model=self.model,
            model_config=self.model_config,
            is_draft_worker=self.is_draft_worker,
            spec_algorithm=self.spec_algorithm,
        )
        adjust_hybrid_swa_layer_ids(
            model_config=self.model_config,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            is_hybrid_swa=self.is_hybrid_swa,
        )
        self.maybe_apply_post_load_model_transforms()
        self.maybe_init_lora_manager()
        self.maybe_enable_batch_invariant_mode()
        self.configure_kv_cache_dtype()

    def init_memory_saver_adapter(self):
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

    def maybe_init_remote_instance_transfer_engine(self):
        if self.server_args.remote_instance_weight_loader_use_transfer_engine():
            self.remote_instance_weight_transporter.init_engine()

    def maybe_init_expert_location_metadata(self):
        if self.is_draft_worker:
            return
        expert_rank = self.ps.moe_ep_rank + (
            self.server_args.ep_join_rank_offset
            if self.server_args.is_ep_scale_joiner
            else 0
        )
        set_global_expert_location_metadata(
            compute_initial_expert_location_metadata(
                server_args=self.server_args,
                model_config=self.model_config,
                moe_ep_rank=expert_rank,
            )
        )
        if self.ps.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get():
            logger.info(
                "Initial expert_location_metadata:\n%s",
                format_expert_location_layout(get_global_expert_location_metadata()),
            )
        set_global_expert_distribution_recorder(
            ExpertDistributionRecorder.init_new(
                self.server_args,
                get_global_expert_location_metadata(),
                rank=expert_rank,
            )
        )

    def maybe_init_lplb_solvers(self):
        if self.server_args.ep_dispatch_algorithm == "lp" and not self.is_draft_worker:
            init_lplb_solvers(model_config=self.model_config)

    def maybe_init_eplb_manager(self):
        self.eplb_manager = (
            EPLBManager(
                server_args=self.server_args,
                model_config=self.model_config,
                ps=self.ps,
                get_model=lambda: self.model,
                get_expert_location_updater=lambda: self.expert_location_updater,
                get_expert_backup_client=lambda: self.expert_backup_client,
                get_weight_updater=lambda: self.weight_updater,
            )
            if self.server_args.enable_eplb and (not self.is_draft_worker)
            else None
        )

    def maybe_init_elastic_ep(self):
        if self.server_args.elastic_ep_backend:
            ElasticEPStateManager.init(self.server_args)

    def init_token_oracle(self):
        self._token_oracle_manager = install_token_oracle_from_env(
            server_args=self.server_args,
            vocab_size=self.model_config.vocab_size,
        )

    def maybe_init_expert_backup_client(self):
        self.expert_backup_client = (
            ExpertBackupClient(
                server_args=self.server_args,
                model_config=self.model_config,
                moe_ep_size=self.ps.moe_ep_size,
                moe_ep_rank=self.ps.moe_ep_rank,
                get_model=lambda: self.model,
            )
            if (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
            else None
        )
        # Register MoE layers for EPLB async runtime (creates GPU-CPU sync signals)
        self.expert_location_updater.prepare_async_layers(
            getattr(self.model, "routed_experts_weights_of_layer", None)
        )

    def maybe_apply_post_load_model_transforms(self):
        # In layered loading, torchao may have been applied
        torchao_applied = getattr(self.model, "torchao_applied", False)
        if not torchao_applied:
            apply_torchao_config_to_model(self.model, get_server_args().torchao_config)
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.ps.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()

    def maybe_init_lora_manager(self):
        if self.server_args.enable_lora:
            self.init_lora_manager()

    def maybe_enable_batch_invariant_mode(self):
        if self.server_args.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

    def get_pp_proxy_topk_size(self) -> Optional[int]:
        return misc_utils.resolve_pp_proxy_topk_size(
            model_config=self.model_config,
            pp_size=self.ps.pp_size,
            pp_rank=self.ps.pp_rank,
            start_layer=self.layer_info.start_layer,
        )

    def decode_num_tokens_per_req(
        self, *, num_draft_tokens: Optional[int] = None
    ) -> int:
        """Logits rows per decode batch slot."""
        if self.spec_algorithm.is_speculative():
            return resolve_num_tokens_per_req(
                phase="target_verify",
                server_args=self.server_args,
                spec_algorithm=self.spec_algorithm,
                is_draft_worker=self.is_draft_worker,
                num_draft_tokens=num_draft_tokens,
            )
        dllm_config = DllmConfig.from_server_args(self.server_args)
        return dllm_config.block_size if dllm_config is not None else 1

    def max_decode_logits_rows(self) -> int:
        """Rows the shared logits buffer needs."""
        num_tokens_per_req = self.decode_num_tokens_per_req()
        capture_bs, _ = get_batch_sizes_to_capture(self, num_tokens_per_req)
        return max(capture_bs) * num_tokens_per_req

    def alloc_memory_pool(self, memory_pool_config: Optional[MemoryPoolConfig] = None):
        """Allocate KV cache memory pools only (no backends or cuda graphs)."""
        if memory_pool_config is not None:
            self.memory_pool_config = memory_pool_config

        self.init_kv_cache_configurator()
        result = self.kv_cache_configurator.configure(
            pre_model_load_memory=self.pre_model_load_memory
        )
        self.max_total_num_tokens = result.max_total_num_tokens
        self.max_running_requests = result.max_running_requests
        self.req_to_token_pool = result.req_to_token_pool
        self.token_to_kv_pool = result.token_to_kv_pool
        self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
        self.memory_pool_config = result.memory_pool_config
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = result.full_max_total_num_tokens
            self.swa_max_total_num_tokens = result.swa_max_total_num_tokens
        # Keep a reference so the shared byte buffer is not GC'd.
        self._unified_memory_pool = result.unified_memory_pool

        self._init_post_memory_pool_components()

    def _init_post_memory_pool_components(self):
        """Post-pool component wiring, split out of alloc_memory_pool so forks
        that build bespoke memory pools can reuse it after allocating them."""
        # Must be called AFTER init_memory_pool so the pool object exists for
        # canary to monkey-patch, and BEFORE init_decode_cuda_graph so warmup
        # forwards captured into the graph see the patched pool methods.
        self.canary_manager = install_canary(
            server_args=self.server_args,
            model_runner=self,
            token_oracle_manager=self._token_oracle_manager,
        )

        # Init ngram embedding token table
        self.init_ngram_embedding_manager()

        self.maybe_init_hisparse_coordinator()

        self.init_routed_experts_capturer()
        self.init_indexer_capturer()

        self.graph_shared_output = None

    def maybe_init_hisparse_coordinator(self):
        if not self.enable_hisparse:
            return
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
        from sglang.srt.mem_cache.sparsity import parse_hisparse_config

        hisparse_cfg = parse_hisparse_config(self.server_args)
        hisparse_top_k = getattr(
            self.model_config.hf_text_config, "index_topk", hisparse_cfg.top_k
        )
        self.hisparse_coordinator = HiSparseCoordinator(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            top_k=hisparse_top_k,
            device_buffer_size=hisparse_cfg.device_buffer_size,
            device=self.device,
            tp_group=(
                self.attention_tp_group.cpu_group
                if self.server_args.enable_dp_attention
                else self.tp_group.cpu_group
            ),
            host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
            swap_in_block_size=hisparse_cfg.swap_in_block_size,
        )

    def post_capture_resize_kv_pool(self):
        resize = compute_post_capture_kv_resize(self)
        self.max_total_num_tokens = resize.max_total_num_tokens
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = resize.full_max_total_num_tokens
            self.swa_max_total_num_tokens = resize.swa_max_total_num_tokens
        if self.memory_pool_config is not None:
            self.memory_pool_config.max_total_num_tokens = resize.max_total_num_tokens
            self.memory_pool_config.full_max_total_num_tokens = (
                resize.full_max_total_num_tokens
            )
            self.memory_pool_config.swa_max_total_num_tokens = (
                resize.swa_max_total_num_tokens
            )
        if resize.capped_max_running_requests is not None:
            self.max_running_requests = resize.capped_max_running_requests
            if self.memory_pool_config is not None:
                self.memory_pool_config.max_running_requests = (
                    resize.capped_max_running_requests
                )

    def init_attention_backends(self):
        """Initialize attention backends only (no cuda graph capture)."""
        # Must be called BEFORE init_decode_cuda_graph() so CUDA graph capture
        # runs with aux hidden state capture enabled.
        configure_aux_hidden_state_capture(
            model=self.model,
            eagle_use_aux_hidden_state=self.spec_aux_config.eagle_use_aux_hidden_state,
            eagle_aux_hidden_state_layer_ids=self.spec_aux_config.eagle_aux_hidden_state_layer_ids,
            dflash_use_aux_hidden_state=self.spec_aux_config.dflash_use_aux_hidden_state,
            dflash_target_layer_ids=self.spec_aux_config.dflash_target_layer_ids,
            is_dspark=self.spec_algorithm.is_dspark(),
        )
        backends = build_attention_backends(model_runner=self)
        self.attn_backend = backends.attn_backend
        self.decode_attn_backend = backends.decode_attn_backend
        self.decode_attn_backend_group = backends.decode_attn_backend_group
        self.prefill_attention_backend_str = backends.prefill_attention_backend_str
        self.decode_attention_backend_str = backends.decode_attention_backend_str

    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True):
        capture = capture_cuda_graphs(
            model_runner=self, capture_decode_cuda_graph=capture_decode_cuda_graph
        )
        self.eager_runner = capture.eager_runner
        self.prefill_cuda_graph_runner = capture.prefill_runner
        self.decode_cuda_graph_runner = capture.decode.runner
        self.graph_mem_usage = capture.decode.graph_mem_usage

    def build_eplb_async_host_mirror(self):
        if self.eplb_async_host_mirror_manager is None:
            return
        routed_experts_weights_of_layer = getattr(
            self.model, "routed_experts_weights_of_layer", None
        )
        if not routed_experts_weights_of_layer:
            return
        self.eplb_async_host_mirror_manager.build_from_loaded_model(
            routed_experts_weights_of_layer
        )

    def init_routed_experts_capturer(self):
        if self.is_draft_worker:
            # Capture is target-only. The draft worker runs in the same process
            # as its target and inits after it, so installing a capturer here
            # would overwrite the target's process-global one.
            return

        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                model=self.model,
                model_config=self.model_config,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_indexer_capturer(self):
        set_global_indexer_capturer(
            create_indexer_capturer(
                model_config=self.model_config,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def check_quantized_moe_compatibility(self):
        check_quantized_moe_compatibility(
            model_config=self.model_config,
            tp_size=self.ps.tp_size,
            moe_ep_size=self.ps.moe_ep_size,
            moe_dp_size=self.ps.moe_dp_size,
        )

    def init_torch_distributed(self):
        result = bootstrap.init_torch_distributed(
            server_args=self.server_args,
            model_config=self.model_config,
            device=self.device,
            ps=self.ps,
            dist_port=self.dist_port,
            is_draft_worker=self.is_draft_worker,
            local_omp_cpuid=self.local_omp_cpuid if self.device == "cpu" else None,
        )
        self.tp_group = result.tp_group
        self.pp_group = result.pp_group
        self.attention_tp_group = result.attention_tp_group
        self.pre_model_load_memory = result.pre_model_load_memory

    def init_shared_mooncake_transfer_engine(self):
        maybe_init_shared_mooncake_transfer_engine(
            server_args=self.server_args, gpu_id=self.gpu_id
        )

    def load_model(self):
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            maybe_downgrade_dtype_for_legacy_gpu(
                server_args=self.server_args, model_config=self.model_config
            )

        set_cuda_arch()

        self.load_config = build_load_config(
            server_args=self.server_args,
            tp_rank=self.ps.tp_rank,
            remote_instance_weight_transporter_engine=self.remote_instance_weight_transporter.engine,
            remote_instance_weight_transporter_session_id=self.remote_instance_weight_transporter.session_id,
            draft_model_idx=self.draft_model_idx,
        )
        if self.device == "cpu":
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.ps.tp_size
            )

        maybe_trigger_remote_instance_nccl_send_group(
            server_args=self.server_args, tp_rank=self.ps.tp_rank
        )

        loaded = load_model_with_memory_saver(
            server_args=self.server_args,
            model_config=self.model_config,
            load_config=self.load_config,
            device=self.device,
            gpu_id=self.gpu_id,
            memory_saver_adapter=self.memory_saver_adapter,
            is_draft_worker=self.is_draft_worker,
        )
        self.loader = loaded.loader
        self.model = loaded.model
        if loaded.remote_instance_weight_info is not None:
            self.remote_instance_weight_transporter.weight_info = (
                loaded.remote_instance_weight_info
            )

        if not self.is_draft_worker:
            get_offloader().post_init()

        # Register model for layerwise NVTX profiling if enabled
        if self.server_args.enable_layerwise_nvtx_marker:
            pyt_hooks = PytHooks()
            pyt_hooks.register_hooks(self.model, module_prefix="model")

        load_kv_cache_scales(model=self.model, server_args=self.server_args)

        self.sliding_window_size = resolve_sliding_window_size(
            self.model, self.model_config
        )

        self.prefill_aware_swa = (
            hasattr(self.model, "is_prefill_aware_swa")
            and self.model.is_prefill_aware_swa()
        )

        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory
        # Get quantization config from ModelConfig
        # This handles both config.json (standard) and hf_quant_config.json (ModelOpt)
        quant_str = self.model_config.get_quantization_config_log_str()

        logger.info(
            f"Load weight end. "
            f"elapsed={time.perf_counter() - tic_total:.2f} s, "
            f"type={type(self.model).__name__}, "
            f"{quant_str + ', ' if quant_str else ''}"
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={self.weight_load_mem_usage:.2f} GB."
        )

        report_online_quantization(model=self.model, server_args=self.server_args)

        maybe_register_debug_tensor_dump_hook(
            model=self.model,
            server_args=self.server_args,
            spec_algorithm=self.spec_algorithm,
            is_draft_worker=self.is_draft_worker,
            tp_size=self.ps.tp_size,
            tp_rank=self.ps.tp_rank,
            pp_rank=self.ps.pp_rank,
        )

        if dumper.may_enable:
            dumper.apply_source_patches()
            dumper.register_non_intrusive_dumper(self.model)

        # Pre-expand RoPE cache before CUDA Graph capture
        reserve_rope_cache_for_long_sequences(
            self.model,
            self.server_args,
            self.model_config,
            logger,
        )

        dist_barrier_after_load(
            elastic_ep_backend=self.server_args.elastic_ep_backend,
            tp_rank=self.ps.tp_rank,
            is_ep_scale_joiner=self.server_args.is_ep_scale_joiner,
        )

    def _prepare_moe_topk(self):
        balancer_cls = None
        num_prepared = 0
        num_routed_experts = None
        for module in self.model.modules():
            if not isinstance(module, (TopK, HashTopK)):
                continue
            if (
                not module.enable_deepep_waterfill
                or module.deepep_waterfill_balancer is not None
            ):
                continue
            if num_routed_experts is None:
                num_routed_experts = getattr(
                    self.model_config.hf_config, "n_routed_experts", None
                )
                if num_routed_experts is None:
                    raise ValueError(
                        "DeepEP waterfill requires model config n_routed_experts."
                    )
            if balancer_cls is None:
                from sglang.srt.layers.moe.deepep_waterfill import (
                    DeepEPWaterfillBalancer,
                )

                balancer_cls = DeepEPWaterfillBalancer
            # Static EPLB remaps TopK ids to physical expert ids before Waterfill.
            # Redundant experts therefore need to be included in the per-rank
            # expert count used for Waterfill's shared-expert slot remapping.
            num_physical_routed_experts = (
                num_routed_experts + self.server_args.ep_num_redundant_experts
            )
            if isinstance(module, TopK):
                routed_scaling_factor = module.topk_config.routed_scaling_factor
            else:
                routed_scaling_factor = module.routed_scaling_factor
            module.deepep_waterfill_balancer = balancer_cls(
                num_routed_experts=num_physical_routed_experts,
                world_size=self.moe_ep_size,
                rank=self.moe_ep_rank,
                layer_id=module.layer_id,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor is not None else 1.0
                ),
            )
            num_prepared += 1
        if num_prepared:
            log_info_on_rank0(
                logger, f"Prepared {num_prepared} DeepEP waterfill TopK modules."
            )

    def _init_lplb_solvers(self):
        """Initialize per-layer LPLB solvers from current expert location metadata."""
        from sglang.srt.distributed import get_moe_ep_group

        # Gate: refuse LP for non-DeepSeek MoE families whose empty-token paths
        # don't participate in the EP all-reduce (would deadlock under DP-
        # attention). Failure here happens before any forward pass.
        architectures = getattr(self.model_config.hf_config, "architectures", None)
        if architectures:
            assert_lplb_supported_model(architectures[0])

        metadata = get_global_expert_location_metadata()
        if metadata is None:
            return
        clear_global_lplb_solvers()
        ep_group = get_moe_ep_group()
        for lid in range(metadata.num_layers):
            solver = LPLBSolver(
                phy2log=metadata.physical_to_logical_map[lid],
                log2phy=metadata.logical_to_all_physical_map[lid],
                num_gpus=metadata.ep_size,
                ep_group=ep_group,
                logical_to_all_physical_map_num_valid=(
                    metadata.logical_to_all_physical_map_num_valid[lid]
                ),
            )
            set_global_lplb_solver(lid, solver)
        logger.info(f"Initialized LPLB solvers for {metadata.num_layers} layers")

    def update_expert_location(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        routed_experts_weights_of_layer = getattr(
            self.model, "routed_experts_weights_of_layer", None
        )
        if not routed_experts_weights_of_layer:
            return
        p2p_missing_logical_experts = self.expert_location_updater.update(
            routed_experts_weights_of_layer,
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes=self.server_args.nnodes,
            rank=self.tp_rank,
        )

        if (
            p2p_missing_logical_experts is not None
            and len(p2p_missing_logical_experts) > 0
        ):
            # Load the missing expert weights from disk
            if callable(getattr(self.model, "generate_weight_name_filter", None)):
                # Filter and load only missing expert weights
                weight_name_filter = self.model.generate_weight_name_filter(
                    p2p_missing_logical_experts
                )
            else:
                # Do a full reload from disk/DRAM
                logger.info(
                    "[Elastic EP] Model does not implement generate_weight_name_filter. "
                    "Performing full weight reload."
                )
                weight_name_filter = None

            if (
                self.expert_backup_client is not None
                and self.expert_backup_client.use_backup
            ):
                # Load the missing weights from the DRAM backup
                self.expert_backup_client.update_weights(weight_name_filter)
            else:
                # Load the missing weights from disk
                self.update_weights_from_disk(
                    get_global_server_args().model_path,
                    get_global_server_args().load_format,
                    weight_name_filter=weight_name_filter,
                )

        # Re-init LPLB solvers after expert location update
        if self.server_args.ep_dispatch_algorithm == "lp":
            self._init_lplb_solvers()

    def maybe_recover_ep_ranks(self):
        # TODO(perf): `active_ranks.all()` on a CUDA tensor triggers host-device
        # synchronization, and this function is on the forward-path.
        # This check only runs when `--elastic-ep-backend` is enabled, so the
        # synchronization overhead does not propagate to other configs.
        # Leave for future optimization of the elastic EP path.
        if self.tp_group.active_ranks.all() and self.tp_group.active_ranks_cpu.all():
            return

        tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
        tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
        tp_active_ranks &= tp_active_ranks_cpu
        # NOTE: `ranks_to_recover` uses indices in `tp_group`. For the current
        # Mooncake elastic EP implementation we assume `--pp-size=1`, so the
        # tp-group index is the same as the global rank index.
        ranks_to_recover = [
            i for i in range(len(tp_active_ranks)) if not tp_active_ranks[i]
        ]

        # try_recover_ranks polls peer state via Mooncake EP backend.
        # Mooncake's internal semantics guarantee that all ranks observe
        # consistent peer readiness state, so collective operations below
        # are safe even though polling appears local.
        if ranks_to_recover and try_recover_ranks(ranks_to_recover):
            self.forward_pass_id = 0
            self.eplb_manager.reset_generator()
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=False
                )
            )
            ElasticEPStateManager.instance().reset()

            broadcast_pyobj(
                [self.server_args.random_seed],
                get_world_group().rank,
                get_world_group().cpu_group,
                src=get_world_group().ranks[0],
            )
            logger.info(f"recover ranks {ranks_to_recover} done")

    def _get_healthy_expert_location_src_rank(
        self, invoked_in_elastic_ep_rejoin_path: bool
    ) -> int:
        world_group = get_world_group()
        # NOTE: do not key off `self.server_args.elastic_ep_rejoin` here.
        # A rank that was started as a rejoin rank may later act as a healthy
        # rank in a subsequent recovery cycle.
        local_rejoin_flag = bool(invoked_in_elastic_ep_rejoin_path)
        gathered_rejoin_flags = world_group.all_gather_object(local_rejoin_flag)

        for rank_in_group, is_rejoin_rank in enumerate(gathered_rejoin_flags):
            if not is_rejoin_rank:
                return world_group.ranks[rank_in_group]

        raise RuntimeError(
            "No healthy rank found for broadcasting expert location metadata. "
            "All ranks are marked as elastic_ep_rejoin."
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        if self.server_args.enable_eplb_async:
            raise RuntimeError(
                "update_weights_from_disk is incompatible with --enable-eplb-async."
            )
        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.model)
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        def model_load_weights(model, iter):
            loader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.load_config = load_config

        if recapture_cuda_graph and (
            self.device == "cuda"
            or self.device == "musa"
            or (
                current_platform.is_out_of_tree()
                and current_platform.support_cuda_graph()
            )
        ):
            self.init_decode_cuda_graph()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_weights_send_group_for_remote_instance(
        self,
        master_address,
        ports,
        group_rank,
        world_size,
        group_name,
        backend="nccl",
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        logger.info(
            f"init custom process group: tp_rank={self.tp_rank}, gpu_id={self.gpu_id}, master_address={master_address}, master_port={group_port}, "
            f"group_rank={group_rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        current_platform.empty_cache()
        success = False
        message = ""
        try:
            na = NetworkAddress(master_address, group_port)
            self._weights_send_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=group_rank,
                group_name=group_name,
                device_id=torch.device("cuda", self.gpu_id),
            )
            dist.barrier(group=self._weights_send_group[group_name])
            success = True
            message = f"Succeeded to init group through {na.to_host_port_str()} group."
        except Exception as e:
            message = f"Failed to init group: {e}."
            logger.error(message)

        current_platform.empty_cache()
        return success, message

    def send_weights_to_remote_instance(
        self,
        master_address,
        ports,
        group_name,
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        if self._weights_send_group[group_name] is not None:
            send_group = self._weights_send_group[group_name]
        else:
            message = f"Group {group_name} not in _weights_send_group list. Please call `init_weights_send_group_for_remote_instance` first."
            logger.error(message)
            return False, message

        current_platform.empty_cache()
        success = False
        na = NetworkAddress(master_address, group_port)
        message = ""
        try:
            for _, weights in self.model.named_parameters():
                torch.distributed.broadcast(
                    weights,
                    src=0,
                    group=send_group,
                )
            success = True
            message = f"Succeeded to send weights through {na.to_host_port_str()} {group_name}."
        except Exception as e:
            message = f"Failed to send weights: {e}."
            logger.error(message)

        # destroy the process group after sending weights
        del self._weights_send_group[group_name]
        torch.distributed.distributed_c10d.destroy_process_group(send_group)
        current_platform.empty_cache()
        return success, message

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            na = NetworkAddress(master_address, master_port)
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def destroy_weights_update_group(self, group_name):
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
    ):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.model.load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def _update_bucketed_weights_from_distributed(
        self, names, dtypes, shapes, group_name
    ):
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (name, torch.empty(shape, dtype=target_dtype, device=self.device))
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.model.load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, LocalSerializedTensor]]],
        load_format: Optional[str] = None,
    ):
        monkey_patch_torch_reductions()
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        device_module = torch.get_device_module(self.device)
        infered_device = device_module.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format in self.server_args.custom_weight_loader:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.model.load_weights(reconstructed_tensors)

        return True, "Success"

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        # TODO: (chenyang) Add support for Qwen models.
        try:
            return self.model.get_weights_by_name(
                name, truncate_size, tp_size=self.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None

    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            server_args=self.server_args,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.ps.tp_size,
            tp_rank=self.ps.tp_rank,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
        )
        if not cuda_graph_fully_disabled():
            init_lora_cuda_graph_moe_buffers(
                server_args=self.server_args,
                model=self.model,
                lora_manager=self.lora_manager,
                dtype=self.dtype,
            )

    def load_lora_adapter(self, lora_ref: LoRARef):
        """Load a new lora adapter from disk or huggingface."""
        return self.lora_manager.load_lora_adapter(lora_ref)

    def load_lora_adapter_from_tensors(
        self, lora_ref: LoRARef, tensors, config_dict, added_tokens_config=None
    ):
        return self.lora_manager.load_lora_adapter_from_tensors(
            lora_ref, tensors, config_dict, added_tokens_config
        )

    def unload_lora_adapter(self, lora_ref: LoRARef):
        """Unload a lora adapter that was previously loaded during initialization or dynamic loading."""
        return self.lora_manager.unload_lora_adapter(lora_ref)

    @property
    def effective_max_total_num_tokens(self):
        """Return the max token pool size considering hybrid swa settings."""
        if self.is_hybrid_swa:
            return self.full_max_total_num_tokens or self.swa_max_total_num_tokens
        else:
            return self.max_total_num_tokens

    def _record_kv_cache_dtype(self, resolved: str) -> None:
        # Load-time resolution transition: the weight-resolved kv-cache dtype
        # is declared into the flags tier; the dual-apply inside the helper
        # replaces the legacy in-place write. Mock runners whose server_args
        # is not the published object keep the plain write.
        from sglang.srt.runtime_context import get_context

        if get_context()._server_args is self.server_args:
            from sglang.srt.arg_groups.overrides import declare_load_time_override

            declare_load_time_override(
                "ModelRunner.configure_kv_cache_dtype",
                {"kv_cache_dtype": resolved},
            )
        else:
            self.server_args.override(
                "ModelRunner.configure_kv_cache_dtype", kv_cache_dtype=resolved
            )

    def configure_kv_cache_dtype(self):
        spec_algorithm = getattr(self, "spec_algorithm", None)
        resolved_kv_cache_dtype, self.kv_cache_dtype = (
            kv_cache_dtype.configure_kv_cache_dtype(
                server_args_kv_cache_dtype=self.server_args.kv_cache_dtype,
                model=getattr(self, "model", None),
                model_dtype=getattr(self, "dtype", torch.bfloat16),
                is_draft_worker=getattr(self, "is_draft_worker", False),
                is_dflash=(
                    spec_algorithm.is_dflash() if spec_algorithm is not None else False
                ),
                speculative_draft_attention_backend=getattr(
                    self.server_args, "speculative_draft_attention_backend", None
                ),
            )
        )
        if resolved_kv_cache_dtype is not None:
            self._record_kv_cache_dtype(resolved_kv_cache_dtype)

    def _get_attention_backend(self, init_new_workspace: bool = False):
        return get_attention_backend(
            model_runner=self, init_new_workspace=init_new_workspace
        )

    def init_decode_cuda_graph(self):
        self.decode_cuda_graph_runner = None
        self.graph_mem_usage = 0
        capture = capture_decode_graph(model_runner=self)
        self.decode_cuda_graph_runner = capture.runner
        self.graph_mem_usage = capture.graph_mem_usage

    def init_prefill_cuda_graph(self, force_for_draft_worker: bool = False):
        self.prefill_cuda_graph_runner = None
        self.prefill_cuda_graph_runner = capture_prefill_graph(
            model_runner=self,
            eager_runner=self.eager_runner,
            force_for_draft_worker=force_for_draft_worker,
        )

    def init_threads_binding(self):
        self.local_omp_cpuid = numa_utils.init_threads_binding(
            tp_rank=self.ps.tp_rank, tp_size=self.ps.tp_size
        )

    def apply_torch_tp(self):
        model_parallel.apply_torch_tp(
            model=self.model, device=self.device, tp_size=self.ps.tp_size
        )

    def update_decode_attn_backend(self, stream_idx: int):
        self.decode_attn_backend = self.decode_attn_backend_group[stream_idx]

    def prepare_dummy_forward_batch(self, forward_batch: ForwardBatch) -> ForwardBatch:
        """Customize a runner-created dummy batch before attention metadata initialization."""
        return forward_batch

    def _prepare_eager_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Pad / normalize a batch for the eager (non-cuda-graph) forward.

        Runs the DP/MLP-sync padding, the attn-tp num_token_non_padded
        normalization, and the hisparse-coordinator refresh that the eager
        forward path needs — the cuda-graph path does the equivalent inside the
        runner's capture/replay, so this is skipped there.
        """
        # For MLP sync
        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.prepare_mlp_sync_batch(self)
        else:
            forward_batch.prepare_attn_tp_scatter_input(self)

        # Normalize num_token_non_padded to be local to this attention TP rank if needed.
        # The skip is scoped to DSACPLayerCommunicator-style CP (DSA, MLA): those
        # flavors already feed a zigzag-split rank-local layout whose token count
        # should not be further divided by attn_tp_size. MHA-arch prefill CP
        # (Qwen3/Qwen2 MoE) keeps the attn_tp-replicated layout and wants the
        # adjustment to run — see docs/design/prefill-cp-mla.md §Phase 5.
        if (
            forward_batch.num_token_non_padded is not None
            and forward_batch.global_num_tokens_gpu is not None
            and require_gathered_buffer(self.server_args)
            and not is_dsa_enable_prefill_cp()
            and not is_mla_prefill_cp_enabled()
        ):
            forward_batch.adjust_num_token_non_padded_for_attn_tp(
                server_args=self.server_args,
            )

        # Hisparse coordinator — backends now read it from self.model_runner.
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

    def _pp_kwargs(self, pp_proxy_tensors) -> dict:
        """Build the pp_proxy_tensors forward kwarg, in one place.

        Pipeline-parallel proxy tensors are threaded into model.forward only
        when the model accepts them (``support_pp``).
        """
        return {"pp_proxy_tensors": pp_proxy_tensors} if self.support_pp else {}

    def _extend_forward_kwargs(
        self, forward_batch: ForwardBatch, pp_proxy_tensors
    ) -> dict:
        """Build the extend/prefill model.forward kwargs (pp_proxy_tensors +
        input_embeds / replace_embeds overrides + get_embedding), shared by the
        prefill cuda-graph path and the EagerRunner's eager extend path."""
        kwargs = self._pp_kwargs(pp_proxy_tensors)
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if (
            forward_batch.replace_embeds is not None
            and forward_batch.replace_positions is not None
        ):
            # Token embedding overrides: get base embeddings, scatter replacements
            if "input_embeds" not in kwargs:
                embed_layer = self.model.get_input_embeddings()
                kwargs["input_embeds"] = embed_layer(forward_batch.input_ids)
            kwargs["input_embeds"][forward_batch.replace_positions] = (
                forward_batch.replace_embeds.to(kwargs["input_embeds"].dtype)
            )
        if not self.is_generation:
            kwargs["get_embedding"] = True
        return kwargs

    def forward_split_prefill(
        self,
        forward_batch: ForwardBatch,
        reinit_attn_backend: bool = False,
        forward_count: int = 1,
    ) -> LogitsProcessorOutput:
        if forward_batch.split_index == 0 or reinit_attn_backend:
            self.attn_backend.init_forward_metadata(forward_batch)
        next_split_index = min(
            forward_batch.split_index + forward_count,
            self.model_config.num_hidden_layers,
        )
        ctx = (
            self.device_timer.wrap(metadata={"category": "split_prefill"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            ret = self.model.forward_split_prefill(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                (forward_batch.split_index, next_split_index),
            )
        forward_batch.split_index = next_split_index
        return ret

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: Optional[bool] = None,  # deprecated
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # Deprecated kwarg: pre-planners mark the batch themselves now.
        forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)

        self.forward_pass_id += 1
        # Transfer iteration info to eplb_async_runtime
        self.on_forward_pass_start()

        # Try msprob debugger
        if self.msprobe_debugger is not None:
            rank_id = (
                self.gpu_id
                if self.ps.attn_dp_size is not None and self.ps.attn_dp_size > 1
                else None
            )
            self.msprobe_debugger.start(model=self.model, rank_id=rank_id)

        # Step span
        step_span_ctx = profile_range(build_step_span_name(forward_batch))

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if not self.is_draft_worker and ((c := self.canary_manager) is not None)
            else contextlib.nullcontext()
        )

        with (
            canary_ctx,
            step_span_ctx,
            get_global_expert_distribution_recorder().with_forward_pass(
                self.forward_pass_id,
                forward_batch,
            ) as recorder_outputs,
        ):
            if SGLANG_PROFILE_NVTX:
                # Use forward_mode to detect real prefill; MTP decode fakes extend mode
                if (
                    forward_batch.forward_mode.is_extend_without_speculative()
                    and forward_batch.extend_seq_lens_cpu is not None
                ):
                    p_bs = len(forward_batch.extend_seq_lens_cpu)
                else:
                    p_bs = 0
                th_nvtx_range_push(
                    f"total bs={forward_batch.batch_size}, P bs={p_bs}, forward_pass_id={self.forward_pass_id}"
                )
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            if self.enable_elastic_ep:
                output = self._maybe_rebalance_after_rank_fault(
                    output,
                    forward_batch,
                    pp_proxy_tensors,
                    reinit_attn_backend,
                    split_forward_count,
                )
            if SGLANG_PROFILE_NVTX:
                th_nvtx_range_pop()
        output.expert_distribution_metrics = recorder_outputs.get("metrics")

        no_copy_to_cpu = not self.server_args.disable_overlap_schedule
        if (
            not self.is_draft_worker
            and (experts_capturer := get_global_experts_capturer()) is not None
        ):
            output.routed_experts_output = experts_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.decode_cuda_graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if (indexer_capturer := get_global_indexer_capturer()) is not None:
            output.indexer_topk_output = indexer_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.decode_cuda_graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        self.on_forward_pass_end()

        if dumper.may_enable:
            dumper.step()

        if self.msprobe_debugger is not None:
            self.msprobe_debugger.stop()
            self.msprobe_debugger.step()

        if self.server_args.elastic_ep_backend is not None:
            self.maybe_join_ep_ranks()

        return output

    def _maybe_execute_deferred_mamba_cow_and_clear(
        self, forward_batch: ForwardBatch
    ) -> None:
        """Run deferred clear/COW on the forward stream, before the mamba layers
        read the pool, so the copies don't race the scheduler copy stream.

        No-op unless this is an extend forward on a mamba model's target worker;
        COW/clear only happen at prefix match on extend.
        """
        pool = self.req_to_token_pool
        if (
            not isinstance(pool, HybridReqToTokenPool)
            or self.is_draft_worker
            or not forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            return
        if (
            forward_batch.mamba_clear_indices is not None
            and len(forward_batch.mamba_clear_indices) > 0
        ):
            # mamba_pool is a pure PHYSICAL store; translate before zeroing or
            # clear_slots zeroes the wrong physical slots.
            pool.mamba_pool.clear_slots(
                pool.translate_mamba_indices(forward_batch.mamba_clear_indices)
            )
        if (
            forward_batch.mamba_cow_src_indices is not None
            and len(forward_batch.mamba_cow_src_indices) > 0
        ):
            if pool.mamba_ckpt_pool is not None:
                # int8 checkpoints: dequantize src int8 ckpt slot into the active bf16 dst.
                pool.mamba_ckpt_pool.load_to_active(
                    pool.mamba_pool,
                    forward_batch.mamba_cow_src_indices,
                    forward_batch.mamba_cow_dst_indices,
                )
            else:
                # mamba_pool is a pure PHYSICAL store; translate both COW slot ids.
                pool.mamba_pool.copy_from(
                    pool.translate_mamba_indices(forward_batch.mamba_cow_src_indices),
                    pool.translate_mamba_indices(forward_batch.mamba_cow_dst_indices),
                )
        forward_batch.mamba_clear_indices = None
        forward_batch.mamba_cow_src_indices = None
        forward_batch.mamba_cow_dst_indices = None

    # Transfer iteration info to eplb_async_runtime
    def on_forward_pass_start(self):
        if self.expert_location_updater is not None:
            self.expert_location_updater.on_forward_pass_start()

    def on_forward_pass_end(self):
        if self.eplb_manager is not None:
            # Trigger EPLB rebalance logic at the end of forward pass
            self.eplb_manager.on_forward_pass_end()

    # Give layer owner to GPU
    def on_eplb_async_capture_start(self):
        if self.expert_location_updater is not None:
            self.expert_location_updater.on_capture_start()

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        if has_forward_context():
            ctx_mgr = contextlib.nullcontext()
        else:
            ctx_mgr = forward_context(ForwardContext(attn_backend=self.attn_backend))
        with ctx_mgr:
            mode_check = (
                forward_batch.forward_mode.is_cpu_graph
                if self.device == "cpu"
                else forward_batch.forward_mode.is_cuda_graph
            )
            can_run_graph = bool(
                mode_check()
                and self.decode_cuda_graph_runner
                and self.decode_cuda_graph_runner.can_run_graph(forward_batch)
            )

            if (
                forward_batch.forward_mode.is_decode()
                and self.hisparse_coordinator is not None
            ):
                forward_batch.hisparse_coordinator = self.hisparse_coordinator
                self.hisparse_coordinator.wait_for_pending_backup()
                self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

            # Replay cuda graph if applicable

            if SGLANG_PROFILE_NVTX:
                if use_model_prof:
                    # Skip during capture: prof_iter -> cudaDeviceSynchronize() is
                    # illegal under CUDA Graph capture (error 900).
                    # always use target_worker for prof_iter to avoid iteration mismatch
                    if not get_is_capture_mode() and not self.is_draft_worker:
                        prof_iter(self.iteration)
                        # Increment inside guard so capture forwards don't bump
                        # iteration and misalign the trace.
                        self.iteration += 1

            if can_run_graph:
                if SGLANG_PROFILE_NVTX:
                    th_nvtx_range_push("decode_cudagraph")
                ret = self.decode_cuda_graph_runner.execute(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                if SGLANG_PROFILE_NVTX:
                    th_nvtx_range_pop()
                return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

            # DP / MLP-sync padding + attn-tp normalization. Only the decode
            # cuda-graph path above pre-pads its static buffers and returns
            # early; split prefill, the prefill cuda graph, and the eager
            # forward all run the live batch and need this first — it sets
            # global_dp_buffer_len / padded token counts that graph eligibility
            # and the collectives depend on.
            self._prepare_eager_forward_batch(forward_batch)

            # Deferred mamba COW/clear on the forward stream, before the extend
            # dispatch below reads the pool.
            self._maybe_execute_deferred_mamba_cow_and_clear(forward_batch)

            if forward_batch.forward_mode.is_split_prefill():
                if SGLANG_PROFILE_NVTX:
                    th_nvtx_range_push("forward_split_prefill")
                # Layer-split mode; stays on ModelRunner, not the eager runner.
                ret = self.forward_split_prefill(
                    forward_batch,
                    reinit_attn_backend=reinit_attn_backend,
                    forward_count=split_forward_count,
                )
            elif (
                forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
                and not isinstance(self.prefill_cuda_graph_runner, EagerRunner)
                and self.prefill_cuda_graph_runner is not None
                and self.prefill_cuda_graph_runner.can_run_graph(forward_batch)
                and get_cp_strategy() is None
            ):
                category = (
                    "target_verify"
                    if forward_batch.forward_mode.is_target_verify()
                    else "extend"
                )
                # Prefill cuda graph (piecewise).
                kwargs = self._extend_forward_kwargs(forward_batch, pp_proxy_tensors)
                # TODO: device_timer.wrap is too broad here — it also includes
                # load_batch time. Move timing into the prefill cuda graph runner
                # to capture only the model.forward part.
                ctx = (
                    self.device_timer.wrap(metadata={"category": category})
                    if self.device_timer
                    else contextlib.nullcontext()
                )
                if SGLANG_PROFILE_NVTX:
                    th_nvtx_range_push("forward_extend")
                with ctx:
                    ret = self.prefill_cuda_graph_runner.execute(
                        forward_batch, **kwargs
                    )
                can_run_graph = True
            else:
                # Eager: decode / extend / idle dispatched inside the runner.
                if SGLANG_PROFILE_NVTX:
                    mode = forward_batch.forward_mode
                    if mode.is_decode():
                        th_nvtx_range_push("forward_decode")
                    if mode.is_idle():
                        th_nvtx_range_push("forward_idle")
                    if mode.is_extend(include_draft_extend_v2=True):
                        th_nvtx_range_push("forward_extend")
                ret = self.eager_runner.execute(
                    forward_batch, pp_proxy_tensors=pp_proxy_tensors
                )

            if (
                forward_batch.global_num_tokens_cpu is not None
                and self.pp_group.is_last_rank
            ):
                forward_batch.post_forward_mlp_sync_batch(ret)

            if SGLANG_PROFILE_NVTX:
                th_nvtx_range_pop()

            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # NOTE: In overlap mode, the function update_regex_vocab_mask (in sample)
        #       was executed after we processed last batch's results.

        # Calculate logits bias and apply it to next_token_logits.
        sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Release the vocab_mask GPU tensor immediately after it has been applied
        # to the logits. In overlap scheduling, the sampling_info (and its
        # vocab_mask) can be kept alive by the delay_sample_func closure and
        # batch_record_buf until the next iteration, causing a steady VRAM leak
        # when structured output (grammar) is used.
        sampling_info.vocab_mask = None

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """

        if SGLANG_PROFILE_NVTX:
            logits_output_shape = getattr(
                logits_output.next_token_logits, "shape", None
            )
            th_nvtx_range_push(
                f"[FW_NATIVE] op:sample,logits_output:{logits_output_shape}"
            )
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            # For prefill, we only use the position of the last token.
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
        self.ngram_embedding_manager.update_after_decode(
            next_token_ids=next_token_ids,
            forward_batch=forward_batch,
        )
        if SGLANG_PROFILE_NVTX:
            th_nvtx_range_pop()
        return next_token_ids

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> None:
        """
        Compute token_ids_logprobs without performing sampling.

        Optimized path for prefill-only requests that need token_ids_logprobs but don't
        require next token generation. Skips expensive sampling operations
        while still providing requested probability information.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
        """
        if not forward_batch.token_ids_logprobs:
            return

        # Preprocess logits (same as in sample method)
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Delegate to sampler for logprob-only computation
        # This populates logits_output with requested token probabilities
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

    def check_weights(self, action: str, allow_quant_error: bool = False):
        return self._weight_checker.handle(
            action=action, allow_quant_error=allow_quant_error
        )

    def _expand_eplb_metadata_for_scale(
        self,
        from_ep_size: int,
        effective_size: int,
    ) -> None:
        metadata = get_global_expert_location_metadata()
        if metadata is None:
            return
        old_num_physical = metadata.num_physical_experts
        num_local = old_num_physical // from_ep_size
        added = num_local * effective_size - old_num_physical
        if added <= 0:
            return

        initial_ep_size = self.server_args.elastic_ep_initial_size
        assert initial_ep_size is not None
        self.server_args.override("elastic_ep.scale", ep_size=effective_size)

        expanded_p2l = append_trivial_expert_slots(
            metadata.physical_to_logical_map,
            added,
            metadata.num_logical_experts,
            start=old_num_physical - num_local * initial_ep_size,
        )
        new_metadata = ExpertLocationMetadata.init_by_mapping(
            self.server_args,
            self.model_config,
            physical_to_logical_map=expanded_p2l,
            moe_ep_rank=self._elastic_global_rank(),
        )
        set_global_expert_location_metadata(new_metadata, allow_overwrite=True)

    def _elastic_global_rank(self) -> int:
        return self.ps.tp_rank + self.server_args.ep_join_rank_offset

    def _report_elastic_scale_failure(self, error: str, effective_size: int) -> None:
        if self.ps.tp_rank != 0 or self.server_args.is_ep_scale_joiner:
            return
        from sglang.srt.managers.io_struct import ElasticScaleUpdateReq

        self._pending_elastic_scale_update = ElasticScaleUpdateReq(
            success=False,
            effective_ep_size=effective_size,
            error=error,
        )

    def _elastic_scale_ready_barrier(self, target_size: int, log_tag: str) -> None:
        if self.ps.tp_rank == 0:
            logger.debug(
                "[Elastic EP][scale] %s entering post-scale WORLD barrier "
                "(target_ep_size=%d)",
                log_tag,
                target_size,
            )
        dist.barrier(group=dist.group.WORLD)
        if self.ps.tp_rank == 0:
            logger.debug(
                "[Elastic EP][scale] %s passed post-scale WORLD barrier "
                "(target_ep_size=%d)",
                log_tag,
                target_size,
            )

    def _finalize_scale_up(
        self,
        ranks_to_join: list[int],
        target_size: int,
        effective_size: int,
    ) -> None:
        self.forward_pass_id = 0
        ElasticEPStateManager.mark_configuring_data_plane()

        state = ElasticEPStateManager.instance()
        for rank in ranks_to_join:
            state.active_ranks[rank] = 1
        state.snapshot_active_to_last()
        state.sync_active_to_cpu()
        if self.eplb_manager is not None:
            self.eplb_manager.reset_generator()

        self._expand_eplb_metadata_for_scale(
            from_ep_size=effective_size,
            effective_size=target_size,
        )
        broadcast_global_expert_location_metadata(
            model_config=self.model_config,
            moe_ep_rank=self._elastic_global_rank(),
            src_rank=0,
        )

        ElasticEPStateManager.on_scale(effective_size, target_size)
        set_global_expert_distribution_recorder(
            ExpertDistributionRecorder.init_new(
                self.server_args,
                get_global_expert_location_metadata(),
                rank=self._elastic_global_rank(),
            )
        )

        if self.eplb_manager is not None:
            self.eplb_manager.disable_rebalance(
                "EPLB rebalance is disabled after elastic EP scale-up"
            )

        from sglang.srt.layers.dp_attention import update_dp_attention_post_scale

        update_dp_attention_post_scale(
            new_dp_size=target_size,
            new_dp_rank=self._elastic_global_rank(),
        )
        self.server_args.override("elastic_ep.scale", dp_size=target_size)

        ElasticEPStateManager.mark_syncing_new_world()
        self._elastic_scale_ready_barrier(
            target_size=target_size,
            log_tag="JOINER" if self.server_args.is_ep_scale_joiner else "PRIMARY",
        )
        ElasticEPStateManager.commit_scale()

        if self.ps.tp_rank == 0 and not self.server_args.is_ep_scale_joiner:
            from sglang.srt.managers.io_struct import ElasticScaleUpdateReq

            self._pending_elastic_scale_update = ElasticScaleUpdateReq(
                success=True,
                effective_ep_size=target_size,
                slot_offset=effective_size,
                slot_count=target_size - effective_size,
            )
            logger.info(
                "[Elastic EP] Scale completed: old_ep_size=%d "
                "new_ep_size=%d joined_ranks=%s",
                effective_size,
                target_size,
                ranks_to_join,
            )

    def maybe_join_ep_ranks(self) -> None:
        if not ElasticEPStateManager.is_scaling():
            return

        state = ElasticEPStateManager.instance()
        effective_size = ElasticEPStateManager.get_effective_ep_size()
        pending_size = ElasticEPStateManager.get_pending_ep_size()

        if pending_size is None:
            if state is not None and state.has_scaled:
                error = (
                    "Elastic EP rank recovery is unsupported after runtime scale-up. "
                    "Restart the expanded deployment."
                )
                ElasticEPStateManager.fail_recovery(error)
                self._report_elastic_scale_failure(error, effective_size)
                if self.ps.tp_rank == 0 and not self.server_args.is_ep_scale_joiner:
                    logger.error("[Elastic EP] %s", error)
                return

            recovered = maybe_recover_ep_ranks(
                tp_group=self.tp_group,
                eplb_manager=self.eplb_manager,
                random_seed=self.server_args.random_seed,
            )
            if recovered:
                self.forward_pass_id = 0
            return

        local_timeout = (
            state.pending_since is not None
            and time.monotonic() - state.pending_since
            > self.server_args.elastic_ep_scale_timeout
        )
        timeout = state.active_ranks.new_tensor(int(local_timeout))
        dist.all_reduce(timeout, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
        if timeout.item():
            error = f"Timed out waiting for ranks to join target EP size {pending_size}"
            ElasticEPStateManager.fail_scale(error)
            self._report_elastic_scale_failure(error, effective_size)
            if self.ps.tp_rank == 0 and not self.server_args.is_ep_scale_joiner:
                logger.error("[Elastic EP] %s", error)
            return

        if state.scale_phase == "waiting_for_cohort":
            cohort_target = get_scale_cohort_target(effective_size)
            if cohort_target is None:
                return
            if cohort_target != pending_size:
                error = (
                    f"Requested target EP size {pending_size} does not match "
                    f"joining cohort target {cohort_target}"
                )
                ElasticEPStateManager.fail_scale(error)
                self._report_elastic_scale_failure(error, effective_size)
                if self.ps.tp_rank == 0 and not self.server_args.is_ep_scale_joiner:
                    logger.error("[Elastic EP] %s", error)
                return
            if not ElasticEPStateManager.begin_scale():
                return

        ranks_to_join = list(range(effective_size, pending_size))
        if not ranks_to_join:
            return

        current_platform.synchronize()
        ElasticEPStateManager.mark_joining()
        if try_admit_scale_ranks(ranks_to_join):
            self._finalize_scale_up(
                ranks_to_join=ranks_to_join,
                target_size=pending_size,
                effective_size=effective_size,
            )

    def _maybe_rebalance_after_rank_fault(
        self,
        output: ModelRunnerOutput,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool,
        split_forward_count: int,
    ) -> ModelRunnerOutput:
        if maybe_rebalance_after_rank_fault(eplb_manager=self.eplb_manager):
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
        return output

    def update_model_fields(
        self,
        new_model: torch.nn.Module,
        *,
        model_path: str,
        load_format: str,
        load_config: LoadConfig,
    ) -> None:
        self.model = new_model
        self.server_args.override(
            "model_runner.update_model_fields",
            model_path=model_path,
            load_format=load_format,
        )
        self.load_config = load_config
