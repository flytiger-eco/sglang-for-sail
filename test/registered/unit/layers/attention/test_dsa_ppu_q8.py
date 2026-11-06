"""Native PPU q8 dispatch contracts without loading a model or vendor extension."""

import ast
import importlib.util
import logging
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[5]
DSA = ROOT / "python/sglang/srt/layers/attention/dsa_backend.py"
WRAPPER = ROOT / "python/sglang/srt/hardware_backend/ppu/attention/flash_mla.py"
VENDOR_MODULE = "sglang.srt.hardware_backend.ppu.attention.flash_mla"

# Load CI markers without importing hardware-specific runtime dependencies.
spec = importlib.util.spec_from_file_location(
    "ppu_q8_ci_registry", ROOT / "python/sglang/test/ci/ci_register.py"
)
registry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = registry
spec.loader.exec_module(registry)
register_cpu_ci = registry.register_cpu_ci
register_ppu_ci = registry.register_ppu_ci
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
# backend-specific: PPU q8 dispatch and real query/indices graph-buffer isolation.
register_ppu_ci(est_time=30, suite="stage-b-test-1-gpu-ppu")


def load_functions(path, names, namespace, class_name=None):
    """Execute the real source methods in isolation, stubbing only dependencies."""
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        body = next(
            n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    functions = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def build_backend(device="cpu", capacity=8):
    names = (
        "_acquire_ppu_q8_buffer",
        "_build_ppu_q8_query",
        "_pad_ppu_q8_topk_indices",
        "_normalize_ppu_q8_topk_lengths",
        "_forward_flashmla_sparse_q8",
        "_forward_flashmla_sparse_decode_q8",
        "_resolve_ppu_q8_backends",
        "_validate_flashmla_sparse_q8_config",
    )
    namespace = {
        "torch": torch,
        "_is_ppu": True,
        "logger": logging.getLogger(__name__),
        "get_disagg": lambda: SimpleNamespace(disaggregation_mode="null"),
        "print_info_once": lambda *args: None,
    }
    # Use source ABI constants so changing a limit changes these tests too.
    for node in ast.parse(DSA.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("_PPU_Q8_"):
                    namespace[target.id] = ast.literal_eval(node.value)
    if torch.device(device).type == "cuda":
        from sglang.kernels.ops.attention.utils import q8kv8_topk_length_from_indices
        from sglang.kernels.ops.kvcache.cache_ops import concat_and_cast_q_fp8_pad

        namespace.update(
            concat_and_cast_q_fp8_pad=concat_and_cast_q_fp8_pad,
            q8kv8_topk_length_from_indices=q8kv8_topk_length_from_indices,
        )
    else:

        def write_query(out, q_nope, q_rope, heads):
            out[:, :heads].copy_(torch.cat((q_nope, q_rope), dim=-1).to(out.dtype))

        def lengths_from_indices(indices):
            positions = torch.arange(indices.shape[-1], device=indices.device) + 1
            return (
                torch.where(indices >= 0, positions, 0)
                .amax(-1)
                .clamp_min(1)
                .to(torch.int32)
            )

        namespace.update(
            concat_and_cast_q_fp8_pad=write_query,
            q8kv8_topk_length_from_indices=lengths_from_indices,
        )
    methods = load_functions(DSA, names, namespace, "DeepseekSparseAttnBackend")
    backend = type("Backend", (), methods)()
    backend._ppu_q8_buffers = {}
    backend._ppu_q8_max_num_tokens = capacity
    backend._ppu_q8_prefill_num_heads = backend.flashmla_padding = 64
    backend._ppu_q8_topk_capacity = 2176
    backend.dsa_index_kpool = 1
    backend.dsa_prefill_impl = backend.dsa_decode_impl = "flashmla_sparse_q8"
    backend.enable_auto_select_prefill_impl = False
    return backend, namespace


def vendor_recorders():
    module = ModuleType(VENDOR_MODULE)

    def result(**kwargs):
        q = kwargs["q"]
        out = torch.ones(
            (*q.shape[:2], kwargs["d_v"]), device=q.device, dtype=torch.bfloat16
        )
        return out, torch.empty(0, device=q.device), torch.empty(0, device=q.device)

    module.flash_mla_sparse_quantized_fwd = Mock(side_effect=result)
    module.flash_mla_sparse_quantized_decode_fwd = Mock(
        side_effect=lambda **kw: result(**kw)[:2]
    )
    return module


class TestPPUQ8Dispatch(unittest.TestCase):
    def test_decode_geometry_and_kpool_tail_preservation(self):
        for rope, width, pool, route in (
            (0, 2048, 1, "prefill"),
            (0, 2051, 4, "prefill"),
            (64, 2051, 4, "prefill"),
            (64, 2048, 1, "decode"),
        ):
            with self.subTest(rope=rope, width=width):
                backend, _ = build_backend()
                backend.dsa_index_kpool = pool
                q_nope, q_rope = torch.zeros(3, 8, 512), torch.zeros(3, 8, rope)
                kv = torch.zeros(64, 1, 512 + rope).to(torch.float8_e4m3fn)
                # Strided tables must become contiguous without losing any tail.
                indices = torch.full((3, width * 2), -1, dtype=torch.int32)[:, ::2]
                indices[:, 0] = 1
                indices[:, width - 1] = 47
                vendor = vendor_recorders()
                with patch.dict(sys.modules, {VENDOR_MODULE: vendor}):
                    output = backend._forward_flashmla_sparse_decode_q8(
                        q_nope, q_rope, kv, 512, indices, 0.125, reuse_buffers=True
                    )
                selected = getattr(
                    vendor,
                    "flash_mla_sparse_quantized_"
                    + ("fwd" if route == "prefill" else "decode_fwd"),
                )
                other = (
                    vendor.flash_mla_sparse_quantized_decode_fwd
                    if route == "prefill"
                    else vendor.flash_mla_sparse_quantized_fwd
                )
                selected.assert_called_once()
                other.assert_not_called()
                args = selected.call_args.kwargs
                actual = args["indices"][:, 0]
                self.assertTrue(actual.is_contiguous())
                self.assertTrue(torch.equal(actual[:, :width], indices))
                self.assertTrue((actual[:, width:] == -1).all())
                self.assertEqual(args["q"].dtype, torch.float8_e4m3fn)
                self.assertIs(args["kv"], kv)
                self.assertEqual(output.shape, (3, 8, 512))
                self.assertNotIn("num_cta", args)
                self.assertNotIn("q_scale", args)
                self.assertNotIn("kv_scale", args)
                if pool > 1:
                    self.assertTrue(
                        torch.equal(
                            args["topk_length"],
                            torch.full((3,), width, dtype=torch.int32),
                        )
                    )
                if route == "prefill":
                    self.assertIs(args["compute_aux"], False)

    def test_decode_lengths_follow_query_rows(self):
        backend, _ = build_backend()
        indices = torch.full((3, 2048), -1, dtype=torch.int32)
        indices[:, :9] = 1
        vendor = vendor_recorders()
        with patch.dict(sys.modules, {VENDOR_MODULE: vendor}):
            backend._forward_flashmla_sparse_decode_q8(
                torch.zeros(3, 8, 512),
                torch.zeros(3, 8, 64),
                torch.zeros(64, 1, 576).to(torch.float8_e4m3fn),
                512,
                indices,
                0.125,
                topk_length=torch.tensor([9, 9], dtype=torch.int32),
            )
        self.assertTrue(
            torch.equal(
                vendor.flash_mla_sparse_quantized_decode_fwd.call_args.kwargs[
                    "topk_length"
                ],
                torch.tensor([9, 9, 9], dtype=torch.int32),
            )
        )

    def test_prefill_fixed_buffers_across_sizes_and_capacity(self):
        backend, _ = build_backend(capacity=8)
        address = None
        for rows in (8, 1, 5, 8, 3):
            query = backend._build_ppu_q8_query(
                torch.ones(rows, 8, 512), torch.empty(rows, 8, 0), 64, True
            )
            indices = backend._pad_ppu_q8_topk_indices(
                torch.ones(rows, 2051, dtype=torch.int32), 128, True
            )
            pair = query.data_ptr(), indices.data_ptr()
            address = address or pair
            self.assertEqual(pair, address)
            self.assertTrue(query.is_contiguous() and indices.is_contiguous())
            self.assertTrue((query[:, 8:].float() == 0).all())
            self.assertTrue((indices[:, 2051:] == -1).all())
        cached = dict(backend._ppu_q8_buffers)
        oversized = backend._acquire_ppu_q8_buffer(
            "indices", (9, 2176), torch.int32, torch.device("cpu")
        )
        self.assertEqual(oversized.shape[0], 9)
        self.assertEqual(set(cached), set(backend._ppu_q8_buffers))
        for key, value in cached.items():
            self.assertIs(value, backend._ppu_q8_buffers[key])

    def test_stream_keys_are_isolated(self):
        backend, _ = build_backend()
        device = torch.device("cuda:0")
        with (
            patch.object(
                torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=11),
            ),
            patch.object(
                torch,
                "empty",
                side_effect=lambda shape, **kw: torch.zeros(shape, dtype=kw["dtype"]),
            ),
        ):
            first = backend._acquire_ppu_q8_buffer(
                "query", (2, 64, 512), torch.float8_e4m3fn, device
            )
            with patch.object(
                torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=22),
            ):
                second = backend._acquire_ppu_q8_buffer(
                    "query", (2, 64, 512), torch.float8_e4m3fn, device
                )
            again = backend._acquire_ppu_q8_buffer(
                "query", (1, 64, 512), torch.float8_e4m3fn, device
            )
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(first.data_ptr(), again.data_ptr())

    def test_wrappers_use_pr17_keywords_without_legacy_api(self):
        # Deliberately expose only PR17 names and signatures; legacy kwargs fail.
        def prefill(
            *, q, kv, indices, sm_scale, d_v, attn_sink, topk_length, compute_aux
        ):
            return q, kv, indices

        def decode(*, q, kv, indices, sm_scale, d_v, topk_length):
            return q, indices

        vendor = SimpleNamespace(
            flash_mla_sparse_quantized_fwd=prefill,
            flash_mla_sparse_quantized_decode_fwd=decode,
        )
        functions = load_functions(
            WRAPPER,
            ["flash_mla_sparse_quantized_fwd", "flash_mla_sparse_quantized_decode_fwd"],
            {"_import_error": None, "_flashmla": vendor},
        )
        marker = object()
        self.assertEqual(
            functions["flash_mla_sparse_quantized_fwd"](
                marker, marker, marker, 0.125, compute_aux=False
            ),
            (marker, marker, marker),
        )
        self.assertEqual(
            functions["flash_mla_sparse_quantized_decode_fwd"](
                marker, marker, marker, 0.125, topk_length=marker
            ),
            (marker, marker),
        )

    def test_startup_rejects_q8_trtllm_before_pd_rebinding(self):
        for role in ("null", "prefill", "decode"):
            for prefill, decode in (
                ("flashmla_sparse_q8", "trtllm"),
                ("trtllm", "flashmla_sparse_q8"),
            ):
                with self.subTest(role=role, prefill=prefill, decode=decode):
                    backend, namespace = build_backend()
                    namespace["get_disagg"] = lambda: SimpleNamespace(
                        disaggregation_mode=role
                    )
                    backend.dsa_prefill_impl = prefill
                    backend.dsa_decode_impl = decode
                    backend._configure_ppu_q8 = Mock(
                        side_effect=backend._resolve_ppu_q8_backends
                    )
                    with self.assertRaisesRegex(
                        ValueError, "cannot be mixed with trtllm"
                    ):
                        backend._validate_flashmla_sparse_q8_config()
                    backend._configure_ppu_q8.assert_not_called()
                    self.assertEqual(
                        (backend.dsa_prefill_impl, backend.dsa_decode_impl),
                        (prefill, decode),
                    )

    def test_startup_accepts_pure_q8_for_all_roles(self):
        for role in ("null", "prefill", "decode"):
            with self.subTest(role=role):
                backend, namespace = build_backend()
                namespace["get_disagg"] = lambda: SimpleNamespace(
                    disaggregation_mode=role
                )
                backend._configure_ppu_q8 = Mock(
                    side_effect=backend._resolve_ppu_q8_backends
                )
                backend._validate_flashmla_sparse_q8_config()
                backend._configure_ppu_q8.assert_called_once()

    def test_non_ppu_prefill_contract_is_unchanged(self):
        backend, namespace = build_backend()
        namespace["_is_ppu"] = False
        backend.kv_cache_dtype = torch.float8_e4m3fn
        backend.device_sm_major = 9
        backend.dsa_decode_impl = "trtllm"
        backend._configure_ppu_q8 = Mock()
        backend._validate_flashmla_sparse_q8_config()
        backend._configure_ppu_q8.assert_not_called()


@unittest.skipUnless(
    torch.cuda.is_available()
    and "ZW" in torch.cuda.get_device_name()
    and torch.cuda.get_device_capability() == (8, 9),
    "Requires PPU0015",
)
class TestPPUQ8GraphBuffers(unittest.TestCase):
    def test_graph_sizes_keep_query_and_tail_addresses(self):
        backend, _ = build_backend("cuda", capacity=8)
        addresses = None
        records = []
        for rows in (8, 1, 5):
            q_nope = torch.ones(rows, 8, 512, device="cuda", dtype=torch.bfloat16)
            q_rope = torch.empty(rows, 8, 0, device="cuda", dtype=torch.bfloat16)
            source = torch.full((rows, 2051), -1, device="cuda", dtype=torch.int32)
            source[:, 2048:2051] = torch.tensor(
                [11, 12, 13], device="cuda", dtype=torch.int32
            )
            stream = torch.cuda.Stream()
            # Use one dedicated stream for every size; different streams are tested below.
            if records:
                stream = records[0][0]
            with torch.cuda.stream(stream):
                stream.wait_stream(torch.cuda.default_stream())
                for _ in range(3):
                    backend._build_ppu_q8_query(q_nope, q_rope, 64, True)
                    backend._pad_ppu_q8_topk_indices(source, 128, True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    query = backend._build_ppu_q8_query(q_nope, q_rope, 64, True)
                    indices = backend._pad_ppu_q8_topk_indices(source, 128, True)
                pair = query.data_ptr(), indices.data_ptr()
                addresses = addresses or pair
                self.assertEqual(pair, addresses)
                graph.replay()
            stream.synchronize()
            self.assertTrue(torch.equal(indices[:, :2051], source))
            self.assertTrue((indices[:, 2051:] == -1).all())
            expected = q_nope.to(torch.float8_e4m3fn).view(torch.uint8)
            self.assertTrue(
                torch.equal(query[:, :8].contiguous().view(torch.uint8), expected)
            )
            records.append((stream, graph, query, indices, q_nope, q_rope, source))
        for stream, graph, query, indices, q_nope, _, source in reversed(records):
            q_nope.fill_(2)
            with torch.cuda.stream(stream):
                stream.wait_stream(torch.cuda.default_stream())
                graph.replay()
            stream.synchronize()
            self.assertTrue(
                torch.equal(
                    query[:, :8].contiguous().view(torch.uint8),
                    q_nope.to(torch.float8_e4m3fn).view(torch.uint8),
                )
            )
            self.assertTrue(torch.equal(indices[:, :2051], source))

    def test_real_stream_isolation(self):
        backend, _ = build_backend("cuda")
        records = []
        for value in (1, 2):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                q_nope = torch.full(
                    (3, 8, 512), value, device="cuda", dtype=torch.bfloat16
                )
                q_rope = torch.empty(3, 8, 0, device="cuda", dtype=torch.bfloat16)
                source = torch.full((3, 2051), value, device="cuda", dtype=torch.int32)
                query = backend._build_ppu_q8_query(q_nope, q_rope, 64, True)
                indices = backend._pad_ppu_q8_topk_indices(source, 128, True)
            records.append((stream, query, indices, value))
        for stream, query, indices, value in records:
            stream.synchronize()
            self.assertTrue((query[:, :8].float() == value).all())
            self.assertTrue((indices[:, :2051] == value).all())
        self.assertNotEqual(records[0][1].data_ptr(), records[1][1].data_ptr())
        self.assertNotEqual(records[0][2].data_ptr(), records[1][2].data_ptr())


if __name__ == "__main__":
    unittest.main()
