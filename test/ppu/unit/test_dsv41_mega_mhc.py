"""AST-isolated CPU dispatch/state regressions for shifted MegaMHC.

Execute the actual HcState and pre/post orchestration, with tensor/kernel and
msgspec dependencies stubbed. This is not numerical or accelerator acceptance.
Hardware acceptance tests live in test_dsv41_mega_mhc_ppu.py.
"""

import ast
import functools
import logging
import runpy
import sys
import unittest
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import NamedTuple, Tuple
from unittest.mock import Mock, patch

SRC = Path(__file__).resolve().parents[3] / "python/sglang"
ADAPTER = SRC / "kernels/ops/layernorm/mhc_mega.py"
MODEL = SRC / "srt/models/deepseek_v4.py"
MHC = SRC / "srt/models/deepseek_v4_mhc.py"


def load_nodes(nodes, namespace, filename):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)
    return namespace


def load_definitions(path, names, namespace):
    nodes = [
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if getattr(n, "name", None) in names
    ]
    assert {n.name for n in nodes} == set(names)
    return load_nodes(nodes, namespace, path)


def load_method(class_name, name, namespace):
    cls = next(
        n
        for n in ast.parse(MODEL.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return load_nodes([method], namespace, MODEL)[name]


# Load the production test base without importing torch/serving dependencies.
# Only its CI/retry dependencies are replaced; unittest still runs every assertion.
ENV_NS = runpy.run_path(str(SRC / "srt/environ.py"))
BASE_NS = load_definitions(
    SRC / "test/test_utils.py",
    ["CustomTestCase"],
    {
        "unittest": unittest,
        "envs": ENV_NS["envs"],
        "is_in_ci": lambda: False,
        "_wait_for_gpu_idle_in_ci": lambda: None,
        "retry": lambda fn, **kwargs: fn(),
    },
)
CustomTestCase = BASE_NS["CustomTestCase"]


class Tensor:
    def __init__(self, shape, dtype="bf16", device="cuda:0", contiguous=True):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.is_cuda = device.startswith("cuda")
        self.ndim = len(shape)
        self.contiguous = contiguous
        self.source = self

    def is_contiguous(self):
        return self.contiguous

    def unsqueeze(self, dim):
        shape = list(self.shape)
        shape.insert(len(shape) if dim == -1 else dim, 1)
        out = Tensor(shape, self.dtype, self.device, self.contiguous)
        out.source = self
        return out

    def squeeze(self, dim):
        shape = list(self.shape)
        if shape[dim] == 1:
            shape.pop(dim)
        out = Tensor(shape, self.dtype, self.device)
        out.source = self
        return out

    def flatten(self, start):
        return Tensor(
            (self.shape[0], self.shape[1] * self.shape[2]), self.dtype, self.device
        )

    def stride(self, dim):
        return 1


def adapter_namespace():
    namespace = {
        "torch": SimpleNamespace(
            Tensor=Tensor,
            bfloat16="bf16",
            float32="fp32",
            empty_like=lambda t: Tensor(t.shape, t.dtype, t.device),
            empty=lambda shape, **kw: Tensor(shape, kw["dtype"], kw["device"]),
        ),
        "functools": functools,
        "logger": logging.getLogger("test.mega_mhc"),
        "envs": ENV_NS["envs"],
        "ENABLE_JIT_DEEPGEMM": True,
        "get_device_sm": lambda: 89,
        "_is_sail": True,
        "is_cuda": lambda: True,
    }
    nodes = [
        n
        for n in ast.parse(ADAPTER.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef)
    ]
    return load_nodes(nodes, namespace, ADAPTER)


def inputs(rows=65, hidden=5120):
    return [
        Tensor((rows, hidden)),
        Tensor((rows, 4, hidden)),
        Tensor((rows, 4), "fp32"),
        Tensor((rows, 4), "fp32"),
        Tensor((rows, 4, 4), "fp32"),
        Tensor((24, 4 * hidden), "fp32"),
        Tensor((3,), "fp32"),
        Tensor((24,), "fp32"),
        Tensor((hidden,)),
    ]


class MegaMHCFixture:
    def setUp(self):
        self.ns = adapter_namespace()
        self.backend = ModuleType("deep_gemm")
        self.backend.mega_mhc = Mock()
        self.invariant = ModuleType("sglang.srt.batch_invariant_ops")
        self.invariant.is_batch_invariant_mode_enabled = lambda: False
        modules = patch.dict(
            sys.modules,
            {
                "deep_gemm": self.backend,
                self.invariant.__name__: self.invariant,
            },
        )
        modules.start()
        self.addCleanup(modules.stop)
        self.batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend_without_speculative=lambda: True,
                is_decode=lambda: False,
                is_target_verify=lambda: False,
            )
        )
        self.env = self.ns["envs"].SGLANG_OPT_DSV41_MEGA_MHC_PREFILL


class TestMegaMHCDispatch(MegaMHCFixture, CustomTestCase):
    def test_sail_sm89_and_newer_and_nvidia_sm10x(self):
        with self.env.override(True):
            for sail, sm, expected in [
                (True, 80, False),
                (True, 88, False),
                (True, 89, True),
                (True, 90, True),
                (True, 100, True),
                (True, 120, True),
                (False, 89, False),
                (False, 90, False),
                (False, 100, True),
                (False, 103, True),
                (False, 110, False),
                (False, 120, False),
            ]:
                with self.subTest(sail=sail, sm=sm):
                    self.ns["_is_sail"] = sail
                    self.ns["get_device_sm"] = lambda: sm
                    self.assertEqual(
                        self.ns["can_use_mega_mhc_prefill"](self.batch), expected
                    )

    def test_env_default_is_sail_only_and_overridable(self):
        env = ENV_NS["envs"].SGLANG_OPT_DSV41_MEGA_MHC_PREFILL
        default = ENV_NS["_default_sail"]
        utils = ModuleType("sglang.srt.utils")
        with patch.dict(sys.modules, {utils.__name__: utils}):
            for sail in (False, True):
                utils.is_sail = lambda: sail
                default.cache_clear()
                self.assertEqual(default(), sail)
                with env.override(not sail):
                    self.assertEqual(env.get(), not sail)
            default.cache_clear()
        definitions = next(
            n
            for n in ast.parse(
                (SRC / "srt/environ.py").read_text(encoding="utf-8")
            ).body
            if isinstance(n, ast.ClassDef) and n.name == "Envs"
        )
        self.assertEqual(
            sum(
                isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name)
                    and t.id == "SGLANG_OPT_DSV41_MEGA_MHC_PREFILL"
                    for t in n.targets
                )
                for n in definitions.body
            ),
            1,
        )

    def test_disabled_mode_and_missing_api_fall_back(self):
        with self.env.override(False):
            self.assertFalse(self.ns["can_use_mega_mhc_prefill"](self.batch))
        with self.env.override(True):
            self.ns["ENABLE_JIT_DEEPGEMM"] = False
            self.assertFalse(self.ns["can_use_mega_mhc_prefill"](self.batch))
            self.ns["ENABLE_JIT_DEEPGEMM"] = True
            self.ns["_is_sail"] = False
            self.ns["is_cuda"] = lambda: False
            self.assertFalse(self.ns["can_use_mega_mhc_prefill"](self.batch))
            self.ns["_is_sail"] = True
            self.batch.forward_mode.is_extend_without_speculative = lambda: False
            self.assertFalse(self.ns["can_use_mega_mhc_prefill"](self.batch))
            self.batch.forward_mode.is_extend_without_speculative = lambda: True
            self.invariant.is_batch_invariant_mode_enabled = lambda: True
            self.assertFalse(self.ns["can_use_mega_mhc_prefill"](self.batch))
            self.invariant.is_batch_invariant_mode_enabled = lambda: False
            self.backend.mega_mhc = None
            self.ns["_mega_mhc_kernel"].cache_clear()
            with self.assertLogs("test.mega_mhc", level="WARNING") as logs:
                for _ in range(2):
                    self.assertFalse(self.ns["can_use_mega_mhc_prefill"](self.batch))
                    self.assertIsNone(
                        self.ns["try_mhc_mega_boundary"](
                            *inputs(), 1e-6, 1e-6, 1e-6, 10
                        )
                    )
            self.assertEqual(len(logs.output), 1)

    def test_shifted_api_contract_and_old_signature(self):
        args = inputs()
        updated, normalized, stats = self.ns["try_mhc_mega_boundary"](
            *args, 1e-6, 2e-6, 3e-6, 10
        )
        kw = self.backend.mega_mhc.call_args.kwargs
        self.assertIs(kw["shifted_prev_mix"].source, args[2])
        self.assertIs(kw["post_mix"].source, args[3])
        self.assertIs(kw["fn"], args[5])
        self.assertIs(kw["new_residual"], updated)
        self.assertIs(kw["y_bf16"], normalized)
        self.assertIs(stats[0].source, kw["new_prev_mix"])
        self.assertIs(stats[1].source, kw["new_post_mix"])
        self.assertIs(stats[2], kw["new_comb_res_mix"])
        self.assertIsNot(updated, args[1])
        self.assertEqual(
            (kw["hc_norm_eps"], kw["hc_pre_eps"], kw["rmsnorm_eps"]), (1e-6, 2e-6, 3e-6)
        )
        self.assertEqual(
            (kw["hc_post_scale"], kw["rmsnorm_scale"], kw["num_sinkhorn_iters"]),
            (2.0, 1.0, 10),
        )
        self.assertNotIn("y_fp8", kw)
        old = self.ns["mhc_mega_boundary"](*args, 1e-6, 1e-6, 1e-6, 10)
        self.assertEqual(len(old), 3)
        self.assertEqual(old[1].shape, normalized.shape)
        self.assertEqual([t.shape for t in old[2]], [t.shape for t in stats])

    def test_invalid_inputs_decline_before_launch(self):
        cases = [inputs(0), inputs(hidden=5121), inputs(1 << 20 | 1)]
        for index, replacement in [
            (0, Tensor((65, 5120), device="cpu")),
            (1, Tensor((65, 3, 5120))),
            (2, Tensor((65, 4), "bf16")),
            (4, Tensor((65, 4, 4), "fp32", "cuda:1")),
            (5, Tensor((24, 20480), "fp32", contiguous=False)),
            (8, Tensor((5120,), "fp32")),
        ]:
            args = inputs()
            args[index] = replacement
            cases.append(args)
        for args in cases:
            with self.subTest(shapes=[t.shape for t in args]):
                self.assertIsNone(
                    self.ns["try_mhc_mega_boundary"](*args, 1e-6, 1e-6, 1e-6, 10)
                )
        self.assertIsNone(
            self.ns["try_mhc_mega_boundary"](*inputs(), 1e-6, 1e-6, 1e-6, 0)
        )
        self.backend.mega_mhc.assert_not_called()

    def test_backend_failure_is_not_swallowed(self):
        self.backend.mega_mhc.side_effect = RuntimeError("backend failed")
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            self.ns["try_mhc_mega_boundary"](*inputs(), 1e-6, 1e-6, 1e-6, 10)
        self.assertEqual(self.backend.mega_mhc.call_count, 1)


class StructStub:
    """Supply msgspec's constructor only; HcState methods/fields come from production."""

    def __init_subclass__(cls):
        dataclass(cls)


class Norm:
    cast_x_before_out_mul = False
    variance_size_override = None
    variance_epsilon = 1e-6

    def __init__(self):
        self.weight = Tensor((5120,))

    def __call__(self, x):
        return x


def mhc_namespace(namespace):
    runtime = SRC / "srt/runtime_context.py"
    flags_ns = load_definitions(
        runtime,
        ["LoRABatchLayout", "ForwardFlags"],
        {
            "contextmanager": contextmanager,
            "Enum": Enum,
            "auto": auto,
        },
    )
    flags = flags_ns["ForwardFlags"]()
    namespace.update(
        NamedTuple=NamedTuple,
        Tuple=Tuple,
        msgspec=SimpleNamespace(Struct=StructStub),
        get_forward=lambda: flags,
        get_parallel=lambda: SimpleNamespace(attn_dp_size=1),
        get_platform=lambda: SimpleNamespace(is_blackwell=False, is_sm90=False),
        _is_hip=False,
        _is_npu=False,
        _is_xpu=False,
        MoEOutput=type("MoEOutputStub", (), {}),
    )
    return load_definitions(
        MHC,
        [
            "HcConfig",
            "HcSubLayer",
            "HcNormed",
            "HcQuantized",
            "AttnOutput",
            "HcPending",
            "HcState",
            "HcNextBoundary",
            "make_boundary",
            "combine",
            "_combine",
            "post",
            "mix_stats",
            "fork_stats_stream",
            "use_stats_stream",
            "_compute_triplet",
            "_can_use_mega_mhc",
            "_post_fusion",
            "run_attn_post",
            "run_moe_post",
        ],
        namespace,
    )


class TestMegaMHCState(MegaMHCFixture, CustomTestCase):
    def setUp(self):
        super().setUp()
        self.ns = mhc_namespace(self.ns)
        self.cfg = self.ns["HcConfig"](4, 10, 1e-6, 1e-6, 5120, True, False)
        args = inputs()
        self.hc = self.ns["HcSubLayer"](self.cfg, *args[5:8], Norm())
        self.next = self.ns["make_boundary"](
            self.hc.norm, accepts_mxfp8=False, hc=self.hc
        )
        self.mix = Mock(
            side_effect=lambda hc, residual: tuple(inputs(residual.shape[0])[2:5])
        )
        self.ns["_mix_stats_impl"] = self.mix
        kernels = ModuleType("sglang.kernels.ops.layernorm.mhc")
        kernels.hc_combine = Mock(
            side_effect=lambda x, *args: Tensor((x.shape[0], 5120))
        )
        kernels.mhc_post = Mock(
            side_effect=lambda x, residual, *args: Tensor(residual.shape)
        )
        finalize = ModuleType(
            "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe"
        )
        finalize.can_fuse_all_reduce = lambda *args: False
        modules = patch.dict(
            sys.modules, {kernels.__name__: kernels, finalize.__name__: finalize}
        )
        modules.start()
        self.addCleanup(modules.stop)
        self.kernels = kernels

    def boundary(self, args=None, next_boundary=True):
        args = inputs() if args is None else args
        return self.ns["_post_fusion"](
            self.hc,
            *args[:2],
            tuple(args[2:5]),
            self.next if next_boundary else None,
            self.batch,
        )

    def test_two_boundaries_use_hcstate_stats_once(self):
        namespace = self.ns
        layer = SimpleNamespace(
            hc_cfg=self.cfg,
            attn_hc=self.hc,
            ffn_hc=self.hc,
            local_boundary=self.next,
            next_boundary=self.next,
            _init_boundaries=lambda: None,
            _can_fuse_attn_mhc=False,
            _can_fuse_ffn_mhc=False,
            self_attn=SimpleNamespace(
                attn_tp_size=1,
                accepts_mxfp8_swizzled_input=lambda: False,
                maybe_use_decode_attn_tp=lambda batch: nullcontext(),
            ),
            mlp=SimpleNamespace(tp_size=1),
        )

        class Attention:
            attn_tp_size = 1
            accepts_mxfp8_swizzled_input = staticmethod(lambda: False)
            maybe_use_decode_attn_tp = staticmethod(lambda batch: nullcontext())

            def __call__(self, **kwargs):
                return Tensor(kwargs["x"].shape)

        layer.self_attn = Attention()
        layer._run_moe_ffn_dp_sync = lambda x, *args, **kwargs: Tensor(x.shape)
        namespace["mhc"] = SimpleNamespace(**namespace)
        namespace["can_fuse_all_reduce"] = lambda *args: False
        forward = load_method(
            "DeepseekV4DecoderLayer", "forward_hc_pre_from_prev", namespace
        )
        initial = namespace["HcState"](Tensor((65, 4, 5120)), Tensor((65, 4), "fp32"))
        with self.env.override(True):
            state = forward(layer, None, initial, None, self.batch, None)
            self.assertIsNone(initial.streams)  # release empties aliases
            self.assertEqual(state.input.rows.shape, (65, 5120))
            self.assertIs(
                state.pre.source,
                self.backend.mega_mhc.call_args_list[0].kwargs["new_prev_mix"],
            )
            self.assertIs(
                state.stats[0].source,
                self.backend.mega_mhc.call_args_list[1].kwargs["new_prev_mix"],
            )
            self.assertEqual(self.mix.call_count, 1)
            self.assertEqual(self.kernels.hc_combine.call_count, 1)
            carried = state.stats
            layer.next_boundary = None
            final = forward(layer, None, state, None, self.batch, None)
        self.assertIsNone(state.streams)
        self.assertIsNone(final.input)
        self.assertIsNone(final.stats)
        self.assertEqual(final.residual.shape, (65, 4, 5120))
        self.assertIs(
            self.backend.mega_mhc.call_args.kwargs["shifted_prev_mix"].source,
            carried[0],
        )
        self.assertEqual(self.mix.call_count, 1)
        self.assertEqual(self.kernels.hc_combine.call_count, 1)
        self.assertEqual(self.kernels.mhc_post.call_count, 1)

    def test_state_gate_closes_cp_sp_unshifted_and_norm_seams(self):
        with self.env.override(True):
            for field, value in (
                ("cp_prefill", True),
                ("pre_from_prev", False),
                ("mult", 3),
            ):
                cfg = self.cfg._replace(**{field: value})
                self.assertFalse(self.ns["_can_use_mega_mhc"](cfg, self.batch))
            with self.ns["get_forward"]().scoped(sp_active=True):
                state = self.boundary()
                self.assertIsNone(state.input)
                self.assertIsNone(state.stats)
            for boundary in (
                None,
                self.next._replace(norm_fusable=False),
                self.next._replace(hc=None),
            ):
                self.next = boundary
                state = self.boundary()
                self.assertIsNone(state.input)
                self.assertIsNone(state.stats)
        self.backend.mega_mhc.assert_not_called()

    def test_decline_preserves_pure_post_and_nvidia_combined_fallback(self):
        args = inputs()
        args[5] = Tensor((24, 20480), "fp32", contiguous=False)
        self.hc = self.hc._replace(fn=args[5])
        self.next = self.next._replace(hc=self.hc)
        with self.env.override(True):
            state = self.boundary(args)
            self.assertIs(state.pre, args[2])
            self.assertIsNone(state.input)
            self.assertIsNone(state.stats)
            self.assertEqual(state.residual.shape, args[1].shape)
            combined = ModuleType(
                "sglang.kernels.ops.layernorm.mhc_post_combine_norm_prefill"
            )
            updated, normalized = Tensor((4096, 4, 5120)), Tensor((4096, 5120))
            combined.mhc_post_combine_norm_prefill = lambda *args: (updated, normalized)
            self.ns["get_platform"] = lambda: SimpleNamespace(
                is_blackwell=True, is_sm90=False
            )
            with patch.dict(sys.modules, {combined.__name__: combined}):
                state = self.boundary(inputs(4096))
            self.assertIs(state.residual, updated)
            self.assertIs(state.input.rows, normalized)
            self.assertIsNone(state.stats)
        self.backend.mega_mhc.assert_not_called()

    def test_missing_api_and_decode_keep_original_state_chain(self):
        args = inputs()
        with self.env.override(False):
            disabled = self.boundary(args)
        self.assertIs(disabled.pre, args[2])
        self.assertIsNone(disabled.input)
        self.assertIsNone(disabled.stats)
        with self.env.override(True):
            self.batch.forward_mode.is_extend_without_speculative = lambda: False
            decode = self.boundary(args)
            self.assertIs(decode.pre, args[2])
            self.assertIsNone(decode.input)
            self.assertIsNone(decode.stats)
            self.batch.forward_mode.is_extend_without_speculative = lambda: True
            self.backend.mega_mhc = None
            self.ns["_mega_mhc_kernel"].cache_clear()
            with self.assertLogs("test.mega_mhc", level="WARNING"):
                missing = self.boundary(args)
            self.assertIs(missing.pre, args[2])
            self.assertIsNone(missing.input)
            self.assertIsNone(missing.stats)
            self.assertEqual(missing.residual.shape, args[1].shape)
        self.assertEqual(self.kernels.mhc_post.call_count, 3)

    def test_tail_narrows_norm_and_stats_engram_drops_both(self):
        args = inputs()
        with self.env.override(True):
            state = self.boundary(args)

        def take(tensor):
            selected = Tensor((3, *tensor.shape[1:]), tensor.dtype, tensor.device)
            selected.source = tensor
            return selected

        tail = state.take_rows(take)
        self.assertEqual(tail.residual.shape[0], 3)
        self.assertIs(tail.pre.source, state.pre)
        self.assertIs(tail.input.rows.source, state.input.rows)
        for selected, original in zip(tail.stats, state.stats):
            self.assertEqual(selected.shape[0], 3)
            self.assertIs(selected.source, original)
        rewritten = tail.with_residual(Tensor((3, 4, 5120)))
        self.assertIs(rewritten.pre, tail.pre)
        self.assertIsNone(rewritten.input)
        self.assertIsNone(rewritten.stats)
        quantized = self.ns["HcState"](
            state.residual,
            state.pre,
            self.ns["HcQuantized"](state.input.rows, object()),
            state.stats,
        )
        narrowed = quantized.take_rows(take)
        self.assertIsNone(narrowed.input)
        self.assertEqual(narrowed.stats[0].shape[0], 3)

    def test_actual_model_loop_preserves_tail_and_engram_state_lifecycle(self):
        for tail_start, engram_index in ((None, None), (2, None), (None, 1)):
            with self.subTest(tail_start=tail_start, engram_index=engram_index):
                ns = self.ns.copy()
                ns.update(
                    mhc=SimpleNamespace(**self.ns),
                    nullcontext=nullcontext,
                    is_cp_active=lambda batch: False,
                    is_in_breakable_cuda_graph=lambda: False,
                    check_cuda_graph_backend=lambda *args: True,
                    Phase=SimpleNamespace(PREFILL=0),
                    Backend=SimpleNamespace(TC_PIECEWISE=0),
                )
                take = lambda t: Tensor((3, *t.shape[1:]), t.dtype, t.device)
                tail = SimpleNamespace(
                    rows=take, cp_metadata=None, positions=None, scatter_indices=None
                )
                backend = SimpleNamespace(
                    tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
                    enter_late_layer_tail=lambda batch: object(),
                    exit_late_layer_tail=lambda *args: None,
                )
                ns["get_attn_backend"] = lambda: backend
                seen = []

                class Engram:
                    layer_hash_index = 0

                    def __call__(self, residual, *args, **kwargs):
                        return Tensor(residual.shape)

                def forward(layer, **kwargs):
                    state = kwargs["state"]
                    seen.append((state.residual.shape[0], state.input, state.stats))
                    rows = state.residual.shape[0]
                    return self.ns["HcState"](
                        state.residual,
                        Tensor((rows, 4), "fp32"),
                        self.ns["HcNormed"](Tensor((rows, 5120))),
                        tuple(inputs(rows)[2:5]),
                    )

                layers = [
                    SimpleNamespace(
                        hc_cfg=self.cfg,
                        engram=Engram() if i == engram_index else None,
                        forward_hc_pre_from_prev=MethodType(forward, object()),
                    )
                    for i in range(3)
                ]
                model = SimpleNamespace(
                    pp_group=SimpleNamespace(world_size=1),
                    engram_hasher=(lambda *args: object())
                    if engram_index is not None
                    else None,
                    late_layer_start=tail_start,
                    start_layer=0,
                    end_layer=3,
                    layers=layers,
                    config=SimpleNamespace(
                        model_type="deepseek_v41", vision_n_layers=0
                    ),
                    _check_late_layer_tail_readers=lambda batch: None,
                )

                class HashIDs:
                    def __getitem__(self, key):
                        return None

                if engram_index is not None:
                    model.engram_hasher = lambda *args: HashIDs()
                loop = load_method(
                    "DeepseekV4Model", "_forward_layers_hc_pre_from_prev", ns
                )
                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(
                        is_extend=lambda: True,
                        is_extend_without_speculative=lambda: True,
                    ),
                    input_ids=None,
                )
                result = loop(
                    model,
                    None,
                    Tensor((65, 4, 5120)),
                    batch,
                    Tensor((65,)),
                    Tensor((65,)),
                    False,
                    [],
                )
                self.assertIsNone(seen[0][1])
                if tail_start is not None:
                    self.assertEqual(result[0].shape[0], 3)
                    self.assertIs(result[2], tail)
                    self.assertEqual(seen[2][1].rows.shape[0], 3)
                    self.assertEqual(seen[2][2][0].shape[0], 3)
                elif engram_index is not None:
                    self.assertIsNone(seen[1][1])
                    self.assertIsNone(seen[1][2])
                    self.assertIsNotNone(seen[2][1])
                else:
                    self.assertIsNotNone(seen[1][1])
                    self.assertIsNotNone(seen[2][2])


if __name__ == "__main__":
    unittest.main()
