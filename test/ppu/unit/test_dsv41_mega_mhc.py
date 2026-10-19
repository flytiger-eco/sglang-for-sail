"""CPU dispatch/state regressions for shifted MegaMHC.

Run directly with Python; CPU tests need neither torch nor the serving stack.
Hardware acceptance tests live in test_dsv41_mega_mhc_ppu.py.
"""

import ast
import functools
import logging
import sys
import unittest
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock, patch

SRC = Path(__file__).resolve().parents[3] / "python/sglang"
ADAPTER = SRC / "kernels/ops/layernorm/mhc_mega.py"
MODEL = SRC / "srt/models/deepseek_v4.py"


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


def load_method(class_name, name, namespace):
    cls = next(
        n
        for n in ast.parse(MODEL.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return load_nodes([method], namespace, MODEL)[name]


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


def adapter_namespace():
    torch = SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        empty_like=lambda t: Tensor(t.shape, t.dtype, t.device),
        empty=lambda shape, **kw: Tensor(shape, kw["dtype"], kw["device"]),
    )
    namespace = {
        "torch": torch,
        "NamedTuple": NamedTuple,
        "functools": functools,
        "logger": logging.getLogger("test.mega_mhc"),
        "envs": SimpleNamespace(
            SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=SimpleNamespace(get=lambda: True)
        ),
        "ENABLE_JIT_DEEPGEMM": True,
        "get_device_sm": lambda: 89,
        "is_ppu": lambda: True,
        "is_cuda": lambda: True,
    }
    nodes = [
        n
        for n in ast.parse(ADAPTER.read_text()).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
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


class TestMegaMHCDispatch(unittest.TestCase):
    def setUp(self):
        self.ns = adapter_namespace()
        self.backend = ModuleType("deep_gemm")
        self.backend.mega_mhc = Mock()
        invariant = ModuleType("sglang.srt.batch_invariant_ops")
        invariant.is_batch_invariant_mode_enabled = lambda: False
        self.modules = patch.dict(
            sys.modules,
            {"deep_gemm": self.backend, "sglang.srt.batch_invariant_ops": invariant},
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.invariant = invariant
        self.batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True)
        )

    def test_ppu_sm89_and_newer_and_nvidia_sm10x(self):
        for ppu, sm, expected in [
            (True, 80, False),
            (True, 88, False),
            (True, 89, True),
            (True, 90, True),
            (True, 100, True),
            (False, 89, False),
            (False, 90, False),
            (False, 100, True),
            (False, 103, True),
            (False, 120, False),
        ]:
            with self.subTest(ppu=ppu, sm=sm):
                self.ns["is_ppu"] = lambda: ppu
                self.ns["get_device_sm"] = lambda: sm
                self.assertEqual(
                    self.ns["can_use_mega_mhc_prefill"](self.batch), expected
                )

    def test_disabled_mode_and_missing_api_fall_back(self):
        for key, value in [("ENABLE_JIT_DEEPGEMM", False), ("is_cuda", lambda: False)]:
            ns = adapter_namespace()
            ns[key] = value
            if key == "is_cuda":
                ns["is_ppu"] = lambda: False
            self.assertFalse(ns["can_use_mega_mhc_prefill"](self.batch))
        self.ns["envs"].SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.get = lambda: False
        self.assertFalse(self.ns["can_use_mega_mhc_prefill"](self.batch))
        self.ns["envs"].SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.get = lambda: True
        self.batch.forward_mode.is_extend_without_speculative = lambda: False
        self.assertFalse(self.ns["can_use_mega_mhc_prefill"](self.batch))
        self.batch.forward_mode.is_extend_without_speculative = lambda: True
        self.invariant.is_batch_invariant_mode_enabled = lambda: True
        self.assertFalse(self.ns["can_use_mega_mhc_prefill"](self.batch))
        self.invariant.is_batch_invariant_mode_enabled = lambda: False
        self.backend.mega_mhc = None
        self.ns["_mega_mhc_kernel"].cache_clear()
        with self.assertLogs("test.mega_mhc", level="WARNING"):
            self.assertFalse(self.ns["can_use_mega_mhc_prefill"](self.batch))

    def test_shifted_api_contract_and_output_order(self):
        args = inputs()
        updated, carried = self.ns["try_mhc_mega_boundary"](*args, 1e-6, 1e-6, 1e-6, 10)
        kw = self.backend.mega_mhc.call_args.kwargs
        self.assertIs(kw["shifted_prev_mix"].source, args[2])
        self.assertIs(kw["post_mix"].source, args[3])
        self.assertIs(kw["fn"], args[5])
        self.assertIs(kw["new_residual"], updated)
        self.assertIs(kw["y_bf16"], carried.normalized)
        self.assertIs(carried.stats[0].source, kw["new_prev_mix"])
        self.assertIs(carried.stats[1].source, kw["new_post_mix"])
        self.assertIs(carried.stats[2], kw["new_comb_res_mix"])
        self.assertIsNot(updated, args[1])
        self.assertEqual(kw["hc_post_scale"], 2.0)
        self.assertNotIn("y_fp8", kw)

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
        self.backend.mega_mhc.assert_not_called()

    def test_backend_failure_is_not_swallowed(self):
        self.backend.mega_mhc.side_effect = RuntimeError("backend failed")
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            self.ns["try_mhc_mega_boundary"](*inputs(), 1e-6, 1e-6, 1e-6, 10)
        self.assertEqual(self.backend.mega_mhc.call_count, 1)

    def test_model_boundary_with_actual_release_forward_flags(self):
        runtime = SRC / "srt/runtime_context.py"
        flags_node = next(
            n
            for n in ast.parse(runtime.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "ForwardFlags"
        )
        layout_node = next(
            n
            for n in ast.parse(runtime.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "LoRABatchLayout"
        )
        flags_ns = load_nodes(
            [layout_node, flags_node],
            {"contextmanager": contextmanager, "Enum": Enum, "auto": auto},
            runtime,
        )
        flags = flags_ns["ForwardFlags"]()
        # Exercise the actual helper as well as the actual adapter. Previously
        # the state tests mocked this helper and missed its invalid flag read.
        self.ns["get_forward"] = lambda: flags
        boundary = load_method(
            "DeepseekV4DecoderLayer", "_try_mega_mhc_boundary", self.ns
        )
        layer = SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v41"),
            hc_pre_from_prev_sublayer=True,
        )
        args = inputs()
        norm = SimpleNamespace(
            cast_x_before_out_mul=False,
            variance_size_override=None,
            weight=args[8],
            variance_epsilon=3e-6,
        )
        next_layer = SimpleNamespace(
            input_layernorm=norm,
            post_attention_layernorm=norm,
            hc_attn_fn=args[5],
            hc_attn_scale=args[6],
            hc_attn_base=args[7],
            hc_ffn_fn=Tensor((24, 20480), "fp32"),
            hc_ffn_scale=Tensor((3,), "fp32"),
            hc_ffn_base=Tensor((24,), "fp32"),
            rms_norm_eps=1e-6,
            hc_eps=2e-6,
            hc_sinkhorn_iters=10,
        )
        for for_ffn in (True, False):
            with self.subTest(for_ffn=for_ffn):
                result = boundary(
                    layer, *args[:5], next_layer, self.batch, for_ffn=for_ffn
                )
                self.assertIsNotNone(result)
                kw = self.backend.mega_mhc.call_args.kwargs
                self.assertIs(
                    kw["fn"], next_layer.hc_ffn_fn if for_ffn else next_layer.hc_attn_fn
                )
                self.assertIs(kw["rmsnorm_weight"], norm.weight)
                self.assertEqual(kw["hc_norm_eps"], 1e-6)
                self.assertEqual(kw["hc_pre_eps"], 2e-6)
                self.assertEqual(kw["rmsnorm_eps"], 3e-6)
        self.assertEqual(self.backend.mega_mhc.call_count, 2)
        self.backend.mega_mhc.reset_mock()
        for attr, value in (
            ("cast_x_before_out_mul", True),
            ("variance_size_override", 5120),
        ):
            setattr(norm, attr, value)
            self.assertIsNone(
                boundary(layer, *args[:5], next_layer, self.batch, for_ffn=True)
            )
            setattr(norm, attr, False if attr == "cast_x_before_out_mul" else None)
        flags.set("sp_active", True)
        self.assertIsNone(
            boundary(layer, *args[:5], next_layer, self.batch, for_ffn=True)
        )
        self.backend.mega_mhc.assert_not_called()


class Attention:
    attn_tp_size = 1

    def accepts_mxfp8_swizzled_input(self):
        return False

    def maybe_use_decode_attn_tp(self, batch):
        return nullcontext()

    def __call__(self, **kwargs):
        return Tensor((65, 5120))


class TestMegaMHCState(unittest.TestCase):
    def setUp(self):
        self.ns = adapter_namespace()
        self.ns.update(
            nullcontext=nullcontext,
            get_platform=lambda: SimpleNamespace(is_blackwell=False),
        )
        invariant = ModuleType("sglang.srt.batch_invariant_ops")
        invariant.is_batch_invariant_mode_enabled = lambda: False
        fusion = ModuleType("sglang.srt.layers.moe.mhc_post_fusion")
        fusion.MhcPostFusion = Mock()
        fusion.use_mhc_post_fusion = lambda state: nullcontext()
        modules = patch.dict(
            sys.modules,
            {
                invariant.__name__: invariant,
                fusion.__name__: fusion,
            },
        )
        modules.start()
        self.addCleanup(modules.stop)
        self.forward = load_method(
            "DeepseekV4DecoderLayer", "forward_hc_pre_from_prev", self.ns
        )
        self.layer = SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v41"),
            self_attn=Attention(),
            input_layernorm="attn_norm",
            post_attention_layernorm="ffn_norm",
            hc_attn_fn="attn_fn",
            hc_attn_scale="attn_scale",
            hc_attn_base="attn_base",
            hc_ffn_fn="ffn_fn",
            hc_ffn_scale="ffn_scale",
            hc_ffn_base="ffn_base",
            _get_hc_stats_stream=lambda *a: None,
            _hc_combine=Mock(return_value=Tensor((65, 5120))),
            _hc_mix_stats=Mock(
                side_effect=lambda x, fn, *a: (fn + "_pre", fn + "_post", fn + "_comb")
            ),
            hc_post=Mock(return_value=Tensor((65, 4, 5120))),
            _run_moe_ffn_dp_sync=Mock(return_value=Tensor((65, 5120))),
        )

        def boundary(x, residual, pre, post, comb, next_layer, batch, *, for_ffn):
            stats = (
                ("fused_ffn_pre", "fused_ffn_post", "fused_ffn_comb")
                if for_ffn
                else ("next_attn_pre", "next_attn_post", "next_attn_comb")
            )
            return Tensor((65, 4, 5120)), self.ns["MegaMHCInput"](
                Tensor((65, 5120)), stats
            )

        self.batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend_without_speculative=lambda: True,
                is_decode=lambda: False,
                is_target_verify=lambda: False,
            )
        )
        self.layer._try_mega_mhc_boundary = Mock(side_effect=boundary)
        self.layer._hc_post_with_combine = Mock(
            side_effect=lambda x, residual, post, comb, *args, **kwargs: (
                self.layer.hc_post(x, residual, post, comb),
                None,
                None,
            )
        )

    def run_forward(self, **kwargs):
        return self.forward(
            self.layer,
            positions=None,
            hidden_states=Tensor((65, 4, 5120)),
            input_ids=None,
            forward_batch=self.batch,
            input_ids_global=None,
            prev_pre="previous_pre",
            **kwargs,
        )

    def test_two_boundaries_and_carried_stats_are_consumed_once(self):
        next_input = []
        residual, pre = self.run_forward(
            mega_mhc_next_layer=object(), next_input=next_input
        )
        self.assertEqual(pre, "fused_ffn_pre")
        self.assertEqual(len(next_input), 1)
        self.layer._hc_combine.assert_called_once()
        self.layer._hc_mix_stats.assert_called_once()
        self.layer.hc_post.assert_not_called()
        self.assertEqual(self.layer._try_mega_mhc_boundary.call_count, 2)
        self.layer._hc_combine.reset_mock()
        self.layer._hc_mix_stats.reset_mock()
        self.run_forward(precomputed_attn=next_input[0])
        self.layer._hc_combine.assert_not_called()
        self.layer._hc_mix_stats.assert_not_called()
        call = self.layer._try_mega_mhc_boundary.call_args
        self.assertEqual(call.args[2:5], next_input[0].stats)
        self.layer.hc_post.assert_called_once()  # last FFN has no next layer

    def test_tail_seam_closes_both_fusions(self):
        next_input = []
        _, pre = self.run_forward(
            mega_mhc_seam_open=False,
            mega_mhc_next_layer=object(),
            next_input=next_input,
        )
        self.assertEqual(pre, "ffn_fn_pre")
        self.assertEqual(next_input, [])
        self.layer._try_mega_mhc_boundary.assert_not_called()
        self.assertEqual(self.layer._hc_combine.call_count, 2)
        self.assertEqual(self.layer._hc_mix_stats.call_count, 2)
        self.assertEqual(self.layer.hc_post.call_count, 2)

    def test_shape_or_backend_decline_preserves_original_chain(self):
        self.layer._try_mega_mhc_boundary.side_effect = None
        self.layer._try_mega_mhc_boundary.return_value = None
        _, pre = self.run_forward(mega_mhc_next_layer=object(), next_input=[])
        self.assertEqual(pre, "ffn_fn_pre")
        self.assertEqual(self.layer._hc_combine.call_count, 2)
        self.assertEqual(self.layer._hc_mix_stats.call_count, 2)
        self.assertEqual(self.layer.hc_post.call_count, 2)

    def test_v0521_combined_fallback_is_preserved(self):
        self.layer._try_mega_mhc_boundary.side_effect = None
        self.layer._try_mega_mhc_boundary.return_value = None
        combined = Tensor((65, 5120))
        self.layer._hc_post_with_combine.side_effect = lambda *args, **kwargs: (
            Tensor((65, 4, 5120)),
            combined,
            None,
        )
        next_combined = []
        self.run_forward(
            mega_mhc_next_layer=object(),
            next_input=[],
            next_combined=next_combined,
        )
        self.assertEqual(self.layer._hc_post_with_combine.call_count, 2)
        self.assertIs(self.layer._hc_combine.call_args.kwargs["combined"], combined)
        self.assertEqual(next_combined, [(combined, None)])
        self.layer.hc_post.assert_not_called()

    def test_v0521_stats_overlap_and_megamhc_do_not_duplicate_stats(self):
        source = SRC / "srt/layers/moe/mhc_post_fusion.py"
        node = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "MhcPostFusion"
        )
        self.ns["dataclass"] = dataclass
        self.ns["__name__"] = __name__
        load_nodes([node], self.ns, source)
        sys.modules["sglang.srt.layers.moe.mhc_post_fusion"].MhcPostFusion = self.ns[
            "MhcPostFusion"
        ]
        stream = Mock()
        self.ns["torch"].cuda = SimpleNamespace(current_stream=lambda: stream)
        self.ns["get_parallel"] = lambda: SimpleNamespace(tp_size=4)
        self.ns["get_moe_a2a_backend"] = lambda: SimpleNamespace(is_none=lambda: True)
        self.layer._get_hc_stats_stream = lambda *args: stream
        self.layer.self_attn.attn_tp_size = 4
        self.layer.self_attn.wo_b = SimpleNamespace(reduce_results=True)
        self.layer.mlp = SimpleNamespace(tp_size=4)
        next_input = []
        self.run_forward(mega_mhc_next_layer=object(), next_input=next_input)
        self.assertEqual(self.layer._try_mega_mhc_boundary.call_count, 2)
        self.layer._hc_mix_stats.assert_called_once()
        self.layer._hc_post_with_combine.assert_not_called()
        self.layer.hc_post.assert_not_called()
        self.assertEqual(len(next_input), 1)


class TestMegaMHCLayerLoop(unittest.TestCase):
    def test_tail_and_engram_invalidate_cached_input(self):
        # Execute the actual model loop with row selection and Engram mocked.
        for tail_start, engram_index in ((None, None), (2, None), (None, 1)):
            with self.subTest(tail_start=tail_start, engram_index=engram_index):
                calls = []
                marker = object()
                ns = adapter_namespace()
                ns.update(
                    nullcontext=nullcontext,
                    is_cp_active=lambda batch: False,
                    is_in_breakable_cuda_graph=lambda: False,
                    check_cuda_graph_backend=lambda *args: True,
                    Phase=SimpleNamespace(PREFILL=0),
                    Backend=SimpleNamespace(TC_PIECEWISE=0),
                    get_platform=lambda: SimpleNamespace(is_blackwell=False),
                    can_use_mega_mhc_prefill=lambda batch: True,
                )
                tail = SimpleNamespace(
                    rows=lambda value: value, cp_metadata=None, positions=None
                )
                backend = SimpleNamespace(
                    tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
                    enter_late_layer_tail=Mock(return_value=marker),
                    exit_late_layer_tail=Mock(),
                )
                ns["get_attn_backend"] = lambda: backend

                class HashIDs:
                    def __getitem__(self, index):
                        return None

                class Engram:
                    layer_hash_index = 0

                    def __call__(self, hidden, *args, **kwargs):
                        return hidden

                layers = []
                for index in range(3):

                    def forward(index=index, **kwargs):
                        calls.append((index, kwargs))
                        # Deliberately hand over a cache even at a closed seam:
                        # the consumer must invalidate it when selecting rows.
                        kwargs["next_input"].append(marker)
                        return kwargs["hidden_states"], "pre"

                    layers.append(
                        SimpleNamespace(
                            engram=Engram() if index == engram_index else None,
                            forward_hc_pre_from_prev=forward,
                        )
                    )
                model = SimpleNamespace(
                    pp_group=SimpleNamespace(world_size=1),
                    engram_hasher=(
                        (lambda *args: HashIDs()) if engram_index is not None else None
                    ),
                    late_layer_start=tail_start,
                    start_layer=0,
                    end_layer=3,
                    layers=layers,
                    config=SimpleNamespace(
                        model_type="deepseek_v41", vision_n_layers=0
                    ),
                    _check_late_layer_tail_readers=Mock(),
                )
                mode = SimpleNamespace(
                    is_extend=lambda: True, is_extend_without_speculative=lambda: True
                )
                batch = SimpleNamespace(forward_mode=mode, input_ids=None)
                forward_loop = load_method(
                    "DeepseekV4Model", "_forward_layers_hc_pre_from_prev", ns
                )
                result = forward_loop(
                    model, None, Tensor((65, 4, 5120)), batch, None, None, False, []
                )
                self.assertIsNone(calls[0][1]["precomputed_attn"])
                self.assertIsNone(calls[2][1]["mega_mhc_next_layer"])
                if tail_start is not None:
                    self.assertFalse(calls[1][1]["mega_mhc_seam_open"])
                    self.assertIsNone(calls[1][1]["mega_mhc_next_layer"])
                    self.assertIsNone(calls[2][1]["precomputed_attn"])
                    backend.exit_late_layer_tail.assert_called_once_with(marker, batch)
                    self.assertIs(result[2], tail)
                elif engram_index is not None:
                    self.assertIsNone(calls[0][1]["mega_mhc_next_layer"])
                    self.assertIsNone(calls[1][1]["precomputed_attn"])
                    self.assertIs(calls[2][1]["precomputed_attn"], marker)
                else:
                    self.assertIs(calls[1][1]["precomputed_attn"], marker)
                    self.assertIs(calls[2][1]["precomputed_attn"], marker)
                    self.assertIs(calls[0][1]["mega_mhc_next_layer"], layers[1])


if __name__ == "__main__":
    unittest.main()
