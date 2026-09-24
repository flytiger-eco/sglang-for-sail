"""Unit tests for MiniMax-M3's direct-NHD SAIL MSA path."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.test_utils import CustomTestCase

try:
    from sglang.srt.layers.attention.minimax_sparse_ops.msa_ppu import (
        _allocate_eager_plan_workspace,
        _decode_layer,
        _extend_build,
        _extend_reuse_key,
        _force_init_local_scores_into,
        _is_nhd_layout_unambiguous,
        _LockedJITEntry,
        _pack_len_meta_into,
        _paged_nhd_view,
        _PpuMsaDecodeState,
        _prefill_chunk_token_cap,
        compute_msa_ppu_gate,
        msa_ppu_forward_extend,
        ppu_msa_available,
        ppu_msa_indexer_fp8_available,
    )

    _HAS_MODULE = True
    _IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    _HAS_MODULE = False
    _IMPORT_ERROR = e

HAS_GPU = torch.cuda.is_available()

PAGE = 128
HEAD_DIM = 128


class TestModuleImport(CustomTestCase):
    def test_module_importable(self):
        self.assertTrue(_HAS_MODULE, f"msa_ppu import failed: {_IMPORT_ERROR}")

    @unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
    def test_ppu_msa_available_returns_bool(self):
        self.assertIsInstance(ppu_msa_available(), bool)

    @unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
    def test_ppu_msa_available_is_cached(self):
        self.assertIs(ppu_msa_available(), ppu_msa_available())


@unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
class TestInputValidation(CustomTestCase):
    def test_prefill_chunk_cap_requires_positive_dimensions(self):
        for args in ((0, 128, 4), (4, 0, 4), (4, 128, 0)):
            with (
                self.subTest(args=args),
                self.assertRaisesRegex(ValueError, "must all be positive"),
            ):
                _prefill_chunk_token_cap(*args)

    def test_extend_build_rejects_an_empty_batch_before_importing_fmha(self):
        with self.assertRaisesRegex(ValueError, "non-empty batch"):
            _extend_build(
                SimpleNamespace(),
                torch.empty(1),
                None,
                None,
                torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.int32),
                [],
            )


@unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
class TestLockedJITEntry(CustomTestCase):
    def test_kwargs_order_does_not_duplicate_memo_entries(self):
        result = object()
        fn = mock.Mock(return_value=result)
        with tempfile.TemporaryDirectory() as cache_dir:
            lock_fd = os.open(str(Path(cache_dir) / "lock"), os.O_CREAT | os.O_RDWR)
            try:
                entry = _LockedJITEntry("test", fn, lock_fd, Path(cache_dir), "entry")
                self.assertIs(entry(first=1, second=2), result)
                self.assertIs(entry(second=2, first=1), result)
            finally:
                os.close(lock_fd)
        fn.assert_called_once()

    def test_unhashable_arguments_skip_memoization(self):
        fn = mock.Mock(return_value=object())
        with tempfile.TemporaryDirectory() as cache_dir:
            lock_fd = os.open(str(Path(cache_dir) / "lock"), os.O_CREAT | os.O_RDWR)
            try:
                entry = _LockedJITEntry("test", fn, lock_fd, Path(cache_dir), "entry")
                entry([1])
                entry([1])
            finally:
                os.close(lock_fd)
        self.assertEqual(fn.call_count, 2)


class _MsaGateFixture(CustomTestCase):
    _SPARSE_CFG = {
        "sparse_num_index_heads": 4,
        "sparse_block_size": PAGE,
        "sparse_topk_blocks": 16,
        "sparse_index_dim": HEAD_DIM,
        "sparse_attention_freq": [0, 1],
        "sparse_disable_index_value": [0, 1],
        "sparse_init_block": 1,
        "sparse_local_block": 1,
    }

    def _evaluate(
        self,
        *,
        page_size=PAGE,
        sm=89,
        msa_enabled=True,
        api_available=True,
        indexer_fp8="1",
        attend=True,
        speculative=None,
        main_dtype=torch.bfloat16,
        model_dtype=torch.bfloat16,
        head_dim=HEAD_DIM,
        kv_heads=4,
        context_len=4096,
        tp_size=1,
        dcp_size=1,
        sparse_overrides=None,
        fp8_attn_gemm=False,
        verify_pool=False,
    ):
        from sglang.srt.configs.model_config import (
            get_minimax_sparse_window_blocks,
        )
        from sglang.srt.mem_cache.memory_pool import get_minimax_sparse_index_dtype
        from sglang.srt.runtime_context import get_context, get_parallel, get_spec

        sparse_cfg = {**self._SPARSE_CFG, **(sparse_overrides or {})}
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                sparse_attention_config=sparse_cfg,
                architectures=["MiniMaxM3SparseForCausalLM"],
            ),
            head_dim=head_dim,
            context_len=context_len,
            num_attention_heads=8,
            get_num_kv_heads=lambda tp, dcp=1: max(1, kv_heads // (tp // dcp)),
        )
        runner = SimpleNamespace(
            model_config=model_config,
            dtype=model_dtype,
            max_running_requests=16,
        )
        mod = "sglang.srt.layers.attention.minimax_sparse_ops.msa_ppu"
        with (
            get_context().override_server_args(
                tp_size=tp_size, dcp_size=dcp_size, page_size=page_size
            ),
            # The raw seed still has no speculative algorithm: both gate readers
            # must see this post-publish change, not a runner.server_args double.
            get_spec().override(speculative_algorithm=speculative),
            mock.patch("sglang.srt.utils.common.get_device_sm", return_value=sm),
            mock.patch(f"{mod}.ppu_msa_available", return_value=api_available),
            mock.patch(
                "sglang.srt.mem_cache.memory_pool.current_platform.is_sail",
                return_value=True,
            ),
            mock.patch.dict(
                os.environ,
                {
                    "SGLANG_SAIL_MINIMAX_M3_MSA": str(int(msa_enabled)),
                    "SGLANG_SAIL_MINIMAX_M3_MSA_INDEXER_FP8": indexer_fp8,
                    "SGLANG_SAIL_MINIMAX_M3_MSA_INDEXER_SCORE_BF16": "1",
                    "SGLANG_SAIL_MINIMAX_M3_MSA_ATTEND": str(int(attend)),
                },
            ),
        ):
            if indexer_fp8 == "":
                os.environ.pop("SGLANG_SAIL_MINIMAX_M3_MSA_INDEXER_FP8")
                os.environ.pop("SGLANG_SAIL_MINIMAX_M3_MSA_INDEXER_SCORE_BF16")
            fp8 = ppu_msa_indexer_fp8_available(
                model_config,
                page_size=page_size,
                main_dtype=main_dtype,
                model_dtype=model_dtype,
            )
            index_dtype = get_minimax_sparse_index_dtype(
                fp8_attn_gemm=fp8_attn_gemm,
                kv_cache_dtype=main_dtype,
                model_dtype=model_dtype,
                model_config=model_config,
                page_size=page_size,
            )
            init_blocks, local_blocks = get_minimax_sparse_window_blocks(sparse_cfg)
            backend = SimpleNamespace(
                kv_pool=SimpleNamespace(
                    main_pool=SimpleNamespace(
                        head_num=model_config.get_num_kv_heads(
                            get_parallel().attn_tp_size, get_parallel().attn_dcp_size
                        ),
                        head_dim=head_dim,
                        dtype=main_dtype,
                    ),
                    index_kv_pool=None,
                    index_k_pool=SimpleNamespace(dtype=index_dtype),
                ),
                block_size_k=sparse_cfg["sparse_block_size"],
                page_size=page_size,
                topk_blocks=sparse_cfg["sparse_topk_blocks"],
                idx_head_dim=sparse_cfg["sparse_index_dim"],
                sparse_layer_ids={1},
                disable_value_layer_ids={
                    i
                    for i, flag in enumerate(sparse_cfg["sparse_disable_index_value"])
                    if flag
                },
                score_type=sparse_cfg.get("sparse_score_type", "max"),
                max_context_len=context_len,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
            )
            if verify_pool:
                from functools import partial

                from sglang.srt.mem_cache.kv_cache_configurator import (
                    KVCacheConfigurator,
                )
                from sglang.srt.mem_cache.memory_pool import (
                    MHATokenToKVPool,
                    MiniMaxSparseKVPool,
                )
                from sglang.srt.model_executor.pool_configurator import (
                    DefaultPoolConfigurator,
                )
                from sglang.srt.runtime_context import get_schedule

                # Real CPU buffers, only the accelerator stream boundary is disabled.
                kvc = SimpleNamespace(
                    model_config=model_config,
                    kv_cache_dtype=main_dtype,
                    kv_cache_dtype_str="auto",
                    model_dtype=model_dtype,
                    pool_page_size=get_schedule().page_size,
                    layer_info=SimpleNamespace(start_layer=0, end_layer=2),
                    device="cpu",
                    use_mla_backend=False,
                    is_draft_worker=False,
                )
                pool_mod = "sglang.srt.mem_cache.kv_cache_configurator"
                with (
                    mock.patch(
                        f"{pool_mod}.MiniMaxSparseKVPool",
                        new=partial(
                            MiniMaxSparseKVPool,
                            main_pool_cls=partial(
                                MHATokenToKVPool, enable_alt_stream=False
                            ),
                        ),
                    ),
                    mock.patch.dict(os.environ, {"SGLANG_USE_HND_KVCACHE": "0"}),
                ):
                    pool = KVCacheConfigurator._build_minimax_sparse_kv_pool(
                        kvc, max_total_num_tokens=PAGE
                    )
                estimator = object.__new__(DefaultPoolConfigurator)
                cell_bytes = estimator._compute_cell_size(kvc, num_layers=2)
                self.assertEqual(
                    sum(pool.get_kv_size_bytes()),
                    (PAGE + pool.page_size) * cell_bytes,
                )
                self.assertEqual(pool.index_k_pool.dtype, index_dtype)
                backend.kv_pool = pool
            gate = compute_msa_ppu_gate(backend, runner)
        return fp8, index_dtype, backend, gate


@unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
class TestPpuMsaGate(_MsaGateFixture):
    def test_arch_defaults_and_main_kv_dtype(self):
        for sm in (80, 89):
            with self.subTest(sm=sm):
                fp8, dtype, backend, (ok, use_attend, reasons) = self._evaluate(
                    sm=sm, indexer_fp8="", main_dtype=torch.float8_e4m3fn
                )
                self.assertTrue(ok, reasons)
                self.assertTrue(use_attend)
                self.assertEqual(fp8, sm >= 89)
                self.assertEqual(
                    dtype, torch.float8_e4m3fn if sm >= 89 else torch.bfloat16
                )
                self.assertEqual(backend._ppu_msa_use_fp8_kvcache, sm >= 89)
                self.assertEqual(backend._ppu_msa_indexer_fp8, sm >= 89)
                self.assertEqual(
                    backend._ppu_msa_score_dtype,
                    torch.bfloat16 if sm >= 89 else torch.float32,
                )

    def test_attend_disabled_still_allows_fp8_indexer(self):
        fp8, _, _, (ok, use_attend, reasons) = self._evaluate(attend=False)
        self.assertTrue(fp8)
        self.assertTrue(ok)
        self.assertFalse(use_attend)
        self.assertIn("SGLANG_SAIL_MINIMAX_M3_MSA_ATTEND=0", reasons)


@unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
class TestPpuMsaIndexerFp8Allocation(_MsaGateFixture):
    """Every fixed gate failure must keep the index cache readable by Triton."""

    def test_allocation_and_backend_share_gate(self):
        cases = [
            ({}, None),
            ({"page_size": 1}, "page_size!=128"),
            ({"msa_enabled": False}, "SGLANG_SAIL_MINIMAX_M3_MSA=0"),
            ({"sm": 79}, "SM<80"),
            ({"api_available": False}, "fmha_sm100 unavailable or API drifted"),
            ({"speculative": "EAGLE"}, "speculative decoding enabled"),
            ({"head_dim": 64}, "main head_dim!=128"),
            ({"kv_heads": 128}, "num_kv_heads==128 is layout-ambiguous"),
            ({"kv_heads": 2}, "num_idx_heads!=num_kv_heads"),
            ({"main_dtype": torch.float16}, "main KV dtype must be bf16 or fp8_e4m3"),
            (
                {"main_dtype": torch.float8_e5m2},
                "main KV dtype must be bf16 or fp8_e4m3",
            ),
            (
                {"main_dtype": torch.float8_e4m3fn, "model_dtype": torch.float16},
                "fp8 main KV requires bf16 model Q for fp8 attend staging",
            ),
            (
                {"context_len": 12288 * PAGE},
                "aligned max_k_tiles >= 12288 (context too long)",
            ),
            ({"sparse_overrides": {"sparse_block_size": 64}}, "block_size_k!=128"),
            ({"sparse_overrides": {"sparse_topk_blocks": 8}}, "topk_blocks!=16"),
            ({"sparse_overrides": {"sparse_index_dim": 64}}, "idx_head_dim!=128"),
            ({"sparse_overrides": {"sparse_score_type": "lse"}}, "score_type!=max"),
            (
                {"sparse_overrides": {"sparse_disable_index_value": [0, 0]}},
                "non-disable-value sparse layers exist (MSA produces no idx_o)",
            ),
            (
                {"sparse_overrides": {"sparse_local_block": 1025}},
                "max(init_blocks, local_blocks) > 1024",
            ),
        ]
        for options, reason in cases:
            with self.subTest(options=options):
                fp8, dtype, _, (ok, _, reasons) = self._evaluate(**options)
                self.assertEqual(fp8, reason is None)
                self.assertEqual(ok, reason is None, reasons)
                self.assertEqual(
                    dtype,
                    torch.float8_e4m3fn
                    if reason is None
                    else options.get("model_dtype", torch.bfloat16),
                )
                if reason is not None:
                    self.assertIn(reason, reasons)

    def test_indexer_disabled_keeps_bf16_and_backend_usable(self):
        fp8, dtype, _, (ok, _, reasons) = self._evaluate(indexer_fp8="0")
        self.assertFalse(fp8)
        self.assertEqual(dtype, torch.bfloat16)
        self.assertTrue(ok, reasons)

    def test_other_fp8_mode_cannot_bypass_sail_gate(self):
        fp8, dtype, _, (ok, _, _) = self._evaluate(
            page_size=1, main_dtype=torch.float8_e4m3fn, fp8_attn_gemm=True
        )
        self.assertFalse(fp8)
        self.assertFalse(ok)
        self.assertEqual(dtype, torch.bfloat16)

    def test_token_window_rounding_matches_backend(self):
        from sglang.srt.configs.model_config import get_minimax_sparse_window_blocks

        for init_tokens, local_tokens, expected in (
            (0, 0, (0, 1)),
            (1, PAGE, (1, 2)),
            (PAGE + 1, PAGE + 1, (2, 3)),
        ):
            with self.subTest(init_tokens=init_tokens, local_tokens=local_tokens):
                self.assertEqual(
                    get_minimax_sparse_window_blocks(
                        {
                            "sparse_block_size": PAGE,
                            "sparse_init_tokens": init_tokens,
                            "sparse_local_tokens": local_tokens,
                        }
                    ),
                    expected,
                )

    def test_estimate_matches_constructed_pool(self):
        for options in (
            {},
            {"page_size": 1},
            {"msa_enabled": False},
            {"indexer_fp8": "0"},
            {"main_dtype": torch.float8_e4m3fn},
            {
                "tp_size": 2,
                "dcp_size": 2,
                "sparse_overrides": {"sparse_num_index_heads": 8},
            },
        ):
            with self.subTest(options=options):
                fp8, _, _, (ok, _, reasons) = self._evaluate(
                    verify_pool=True, **options
                )
                self.assertEqual(ok, fp8 or options.get("indexer_fp8") == "0", reasons)

    def test_tp_dcp_head_counts_match_allocation(self):
        for idx_heads, expected in ((4, False), (8, True)):
            with self.subTest(idx_heads=idx_heads):
                fp8, _, backend, (ok, _, reasons) = self._evaluate(
                    tp_size=2,
                    dcp_size=2,
                    sparse_overrides={"sparse_num_index_heads": idx_heads},
                )
                self.assertEqual(backend.kv_pool.main_pool.head_num, 4)
                self.assertEqual(fp8, expected)
                self.assertEqual(ok, expected, reasons)


@unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
class TestPagedNhdView(CustomTestCase):
    def test_view_mapping_is_zero_copy(self):
        pages, heads, dim = 3, 4, 64
        cache = torch.arange(pages * PAGE * heads * dim, dtype=torch.float32).reshape(
            pages * PAGE, heads, dim
        )

        view = _paged_nhd_view(cache)

        self.assertEqual(view.shape, (pages, PAGE, heads, dim))
        self.assertEqual(view.data_ptr(), cache.data_ptr())
        self.assertEqual(view.stride(), (PAGE * heads * dim, heads * dim, dim, 1))
        for page in range(pages):
            self.assertTrue(
                torch.equal(view[page], cache[page * PAGE : (page + 1) * PAGE])
            )

    def test_view_requires_page_multiple(self):
        with self.assertRaises(AssertionError):
            _paged_nhd_view(torch.empty(PAGE + 1, 1, 8))

    def test_hkv_128_is_rejected_as_layout_ambiguous(self):
        self.assertTrue(_is_nhd_layout_unambiguous(1))
        self.assertTrue(_is_nhd_layout_unambiguous(64))
        self.assertFalse(_is_nhd_layout_unambiguous(PAGE))


@unittest.skipUnless(
    _HAS_MODULE and HAS_GPU and ppu_msa_available(),
    "msa_ppu module or fmha_sm100/CUDA unavailable",
)
class TestPlanWorkspaceIsolation(CustomTestCase):
    def test_dedicated_workspaces_preserve_plan_metadata(self):
        from fmha_sm100 import allocate_graph_workspace, fmha_sm100_plan

        indexer_workspace = allocate_graph_workspace(
            1,
            4,
            num_kv_heads=1,
            max_qo_len=186,
            page_size=PAGE,
            num_kv_splits=-1,
            device="cuda",
            output_maxscore=True,
        )
        indexer_plan = fmha_sm100_plan(
            torch.tensor([186], dtype=torch.int32),
            torch.tensor([186], dtype=torch.int32),
            4,
            num_kv_heads=1,
            page_size=PAGE,
            num_kv_splits=-1,
            output_maxscore=True,
            causal=True,
            given_workspace=indexer_workspace,
        )
        before = indexer_plan[3]["qo_segment_lens"].cpu().tolist()

        attend_workspace = allocate_graph_workspace(
            1,
            4,
            num_kv_heads=4,
            max_qo_len=16,
            page_size=PAGE,
            kv_block_num=16,
            num_kv_splits=-1,
            device="cuda",
        )
        fmha_sm100_plan(
            torch.tensor([16], dtype=torch.int32),
            torch.tensor([16], dtype=torch.int32),
            4,
            num_kv_heads=4,
            page_size=PAGE,
            kv_block_num=16,
            num_kv_splits=-1,
            causal=True,
            given_workspace=attend_workspace,
        )

        self.assertEqual(before, [186])
        self.assertEqual(indexer_plan[3]["qo_segment_lens"].cpu().tolist(), [186])

    def test_exact_workspace_handles_skewed_prefill_batch(self):
        from fmha_sm100 import fmha_sm100_plan

        qo_lens = torch.tensor([1024] + [1] * 63, dtype=torch.int32)
        kv_lens = qo_lens + 4096
        indexer_workspace = _allocate_eager_plan_workspace(
            qo_lens,
            4,
            num_kv_heads=1,
            page_size=PAGE,
            kv_block_num=-1,
            num_kv_splits=-1,
            output_maxscore=True,
            use_fp8_kvcache=False,
            device=torch.device("cuda"),
        )
        indexer_plan = fmha_sm100_plan(
            qo_lens,
            kv_lens,
            4,
            num_kv_heads=1,
            page_size=PAGE,
            num_kv_splits=-1,
            output_maxscore=True,
            causal=True,
            given_workspace=indexer_workspace,
        )
        self.assertEqual(
            indexer_plan[3]["qo_segment_lens"].cpu().tolist(), qo_lens.tolist()
        )

        attend_workspace = _allocate_eager_plan_workspace(
            qo_lens,
            8,
            num_kv_heads=8,
            page_size=PAGE,
            kv_block_num=16,
            num_kv_splits=-1,
            output_maxscore=False,
            use_fp8_kvcache=False,
            device=torch.device("cuda"),
        )
        attend_plan = fmha_sm100_plan(
            qo_lens,
            kv_lens,
            8,
            num_kv_heads=8,
            page_size=PAGE,
            kv_block_num=16,
            num_kv_splits=-1,
            causal=True,
            given_workspace=attend_workspace,
        )
        self.assertEqual(
            attend_plan[3]["qo_segment_lens"].sum().item(), qo_lens.sum().item()
        )


@unittest.skipUnless(
    _HAS_MODULE and HAS_GPU, "msa_ppu module or CUDA/PPU device unavailable"
)
class TestGpuMetadataKernels(CustomTestCase):
    def test_pack_len_meta_matches_reference(self):
        page_counts = [2, 3]
        seq_lens = torch.tensor(
            [PAGE + 1, 2 * PAGE + 1], dtype=torch.int32, device="cuda"
        )
        req_to_token = torch.zeros(2, 3 * PAGE, dtype=torch.int64, device="cuda")
        physical_pages = [[4, 1], [2, 5, 3]]
        for request, pages in enumerate(physical_pages):
            for logical_page, physical_page in enumerate(pages):
                tokens = torch.arange(
                    physical_page * PAGE,
                    (physical_page + 1) * PAGE,
                    dtype=torch.int64,
                    device="cuda",
                )
                req_to_token[
                    request, logical_page * PAGE : (logical_page + 1) * PAGE
                ] = tokens

        out = torch.full((sum(page_counts),), -1, dtype=torch.int32, device="cuda")
        kv_lens = torch.empty(2, dtype=torch.int32, device="cuda")
        tok_pages = torch.empty(2, dtype=torch.int32, device="cuda")
        page_starts = torch.empty(2, dtype=torch.int32, device="cuda")
        _pack_len_meta_into(
            out,
            req_to_token,
            torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
            seq_lens,
            kv_lens,
            tok_pages,
            page_starts,
            len(page_counts),
            max(page_counts),
            8,
        )
        torch.cuda.synchronize()

        self.assertEqual(out.cpu().tolist(), [4, 1, 2, 5, 3])
        self.assertEqual(kv_lens.cpu().tolist(), [PAGE + 1, 2 * PAGE + 1])
        self.assertEqual(tok_pages.cpu().tolist(), page_counts)
        self.assertEqual(page_starts.cpu().tolist(), [0, 2])

        # Page ids past the end of the pool clamp to the last physical page.
        out.fill_(-1)
        _pack_len_meta_into(
            out,
            req_to_token,
            torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
            seq_lens,
            kv_lens,
            tok_pages,
            page_starts,
            len(page_counts),
            max(page_counts),
            4,
        )
        torch.cuda.synchronize()
        self.assertEqual(out.cpu().tolist(), [3, 1, 2, 3, 3])

    def test_force_init_local_scores_matches_reference(self):
        heads, k_tiles, tokens = 4, 160, 7
        max_score = torch.full(
            (heads, k_tiles, tokens), -float("inf"), dtype=torch.float32, device="cuda"
        )
        tok_pages = torch.tensor(
            [1, 5, 40, 3, 100, 64, 160], dtype=torch.int32, device="cuda"
        )

        _force_init_local_scores_into(
            max_score, tok_pages, init_blocks=2, local_blocks=3
        )
        torch.cuda.synchronize()

        expected = torch.full(
            (heads, k_tiles, tokens), -float("inf"), dtype=torch.float32
        )
        for token, pages in enumerate(tok_pages.cpu().tolist()):
            expected[:, : min(2, pages), token] = 1.0e30
            expected[:, max(pages - 3, 0) : pages, token] = 1.0e29
        self.assertTrue(torch.equal(max_score.cpu(), expected))


@unittest.skipUnless(
    _HAS_MODULE and HAS_GPU and ppu_msa_available(),
    "msa_ppu module, fmha_sm100, or CUDA/PPU device unavailable",
)
class TestDecodeFp8AttendStaging(CustomTestCase):
    def test_direct_attend_stages_bf16_q_to_fixed_fp8_buffer(self):
        import fmha_sm100

        device = torch.device("cuda")
        bs, max_bs, pages, h_idx, h_q, h_kv = 2, 4, 2, 4, 8, 4
        backend = SimpleNamespace(
            _ppu_msa_max_bs=max_bs,
            max_context_len=pages * PAGE,
            kv_pool=SimpleNamespace(main_pool=SimpleNamespace(size=pages * PAGE)),
            _ppu_msa_num_idx_heads=h_idx,
            _ppu_msa_num_q_heads=h_q,
            topk_blocks=16,
            init_blocks=1,
            local_blocks=1,
            _ppu_msa_score_dtype=torch.float32,
            _ppu_msa_indexer_fp8=False,
            _ppu_msa_attend=True,
            _ppu_msa_use_fp8_kvcache=True,
            idx_head_dim=HEAD_DIM,
            _ppu_msa_idx_scale=HEAD_DIM**-0.5,
            _ppu_msa_scale=HEAD_DIM**-0.5,
        )
        state = _PpuMsaDecodeState(backend, device, torch.bfloat16)
        state.indexer_plan = object()
        state.attend_plan = object()
        state.max_score_active = torch.full(
            (h_idx, state.k_tiles, bs),
            -float("inf"),
            dtype=backend._ppu_msa_score_dtype,
            device=device,
        )
        q = torch.randn(bs, h_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
        idx_q = torch.randn(bs, h_idx, HEAD_DIM, dtype=torch.bfloat16, device=device)
        idx_k = torch.randn(
            pages * PAGE, 1, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        k_cache = torch.randn(
            pages * PAGE, h_kv, HEAD_DIM, dtype=torch.bfloat16, device=device
        ).to(torch.float8_e4m3fn)
        v_cache = torch.randn(
            pages * PAGE, h_kv, HEAD_DIM, dtype=torch.bfloat16, device=device
        ).to(torch.float8_e4m3fn)
        run_mock = mock.MagicMock(return_value=(None, None))
        with (
            mock.patch.object(fmha_sm100, "fmha_sm100", run_mock),
            mock.patch.object(fmha_sm100, "sparse_topk_select", mock.MagicMock()),
            mock.patch(
                "sglang.srt.layers.attention.minimax_sparse_ops.msa_ppu"
                "._force_init_local_scores_into"
            ),
        ):
            out = _decode_layer(backend, state, bs, q, idx_q, idx_k, k_cache, v_cache)

        self.assertEqual(run_mock.call_count, 2)
        self.assertIs(run_mock.call_args_list[0].args[0], idx_q)
        attend_call = run_mock.call_args_list[1]
        attend_q = attend_call.args[0]
        self.assertEqual(attend_q.dtype, torch.float8_e4m3fn)
        self.assertEqual(attend_q.data_ptr(), state.attend_q_fp8.data_ptr())
        self.assertTrue(torch.equal(attend_q, q.to(torch.float8_e4m3fn)))
        self.assertEqual(attend_call.args[1].dtype, torch.float8_e4m3fn)
        self.assertEqual(attend_call.args[2].dtype, torch.float8_e4m3fn)
        self.assertEqual(attend_call.kwargs["out"].dtype, torch.bfloat16)
        self.assertEqual(out.dtype, torch.bfloat16)


@unittest.skipUnless(_HAS_MODULE, "msa_ppu not importable")
class TestExtendReuseKey(CustomTestCase):
    """The prefill state key must be stable across the layers of one forward.

    ``MiniMaxSparseAttnBackend.forward_extend`` re-allocates ``cu_seqlens`` on
    every layer, so the key must not depend on any per-layer tensor address.
    """

    @staticmethod
    def _make_forward_batch(seq_lens, seq_lens_cpu):
        return SimpleNamespace(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(seq_lens_cpu, dtype=torch.int64),
        )

    def test_key_is_stable_across_layers_of_one_forward(self):
        forward_batch = self._make_forward_batch([300], [300])
        first = _extend_reuse_key(forward_batch, [300], 300)
        for _ in range(8):
            self.assertEqual(_extend_reuse_key(forward_batch, [300], 300), first)

    def test_key_discriminates_different_batches(self):
        batch_a = self._make_forward_batch([300], [300])
        batch_b = self._make_forward_batch([300], [300])
        self.assertNotEqual(
            _extend_reuse_key(batch_a, [300], 300),
            _extend_reuse_key(batch_b, [300], 300),
        )

    def test_key_tracks_extend_and_kv_lengths(self):
        forward_batch = self._make_forward_batch([300], [300])
        base = _extend_reuse_key(forward_batch, [300], 300)
        self.assertNotEqual(_extend_reuse_key(forward_batch, [200], 200), base)

        # Same object identity and seq_lens buffer, different kv lengths: the
        # content tuples must still force a rebuild.
        other = self._make_forward_batch([300], [300])
        other.seq_lens = forward_batch.seq_lens
        other.seq_lens_cpu = torch.tensor([400], dtype=torch.int64)
        self.assertNotEqual(_extend_reuse_key(other, [300], 300), base)

    def test_key_handles_missing_cpu_mirrors(self):
        forward_batch = SimpleNamespace(seq_lens=torch.tensor([300], dtype=torch.int32))
        self.assertEqual(
            _extend_reuse_key(forward_batch, None, 300),
            _extend_reuse_key(forward_batch, None, 300),
        )


@unittest.skipUnless(
    _HAS_MODULE and HAS_GPU and ppu_msa_available(),
    "msa_ppu module or fmha_sm100/CUDA unavailable",
)
class TestPrefillPlanReuse(CustomTestCase):
    """All sparse layers of one prefill forward share one set of plans.

    Simulates a multi-layer forward: each layer calls ``msa_ppu_forward_extend``
    with the same forward batch but a freshly allocated ``cu_seqlens`` (what
    ``MiniMaxSparseAttnBackend.forward_extend`` does). Only the indexer chunk
    plans and the whole-batch attend plan may run ``fmha_sm100_plan`` — once
    per forward, never per layer.
    """

    def test_fmha_sm100_plan_runs_once_per_forward_not_per_layer(self):
        import fmha_sm100

        device = torch.device("cuda")
        h_idx, h_q, h_kv, topk = 4, 8, 4, 16
        total_q, kv_len, pages = 64, 2 * PAGE, 4

        backend = SimpleNamespace(
            _ppu_msa_state=None,
            _ppu_msa_attend=True,
            _ppu_msa_use_fp8_kvcache=False,
            _ppu_msa_indexer_fp8=False,
            _ppu_msa_score_dtype=torch.float32,
            _ppu_msa_num_idx_heads=h_idx,
            _ppu_msa_num_q_heads=h_q,
            num_kv_heads=h_kv,
            topk_blocks=topk,
            max_context_len=pages * PAGE,
            init_blocks=1,
            local_blocks=2,
            req_to_token=torch.arange(
                pages * PAGE, dtype=torch.int64, device=device
            ).unsqueeze(0),
            kv_pool=SimpleNamespace(main_pool=SimpleNamespace(size=pages * PAGE)),
            _ppu_msa_idx_scale=128**-0.5,
            _ppu_msa_scale=128**-0.5,
        )
        forward_batch = SimpleNamespace(
            seq_lens=torch.tensor([kv_len], dtype=torch.int32, device=device),
            seq_lens_cpu=torch.tensor([kv_len], dtype=torch.int64),
            req_pool_indices=torch.zeros(1, dtype=torch.int64, device=device),
        )
        q = torch.randn(total_q, h_q, 128, dtype=torch.bfloat16, device=device)
        idx_q = torch.randn(total_q, h_idx, 128, dtype=torch.bfloat16, device=device)
        idx_k_cache = torch.randn(
            pages * PAGE, 1, 128, dtype=torch.bfloat16, device=device
        )
        k_cache = torch.randn(
            pages * PAGE, h_kv, 128, dtype=torch.bfloat16, device=device
        )
        v_cache = torch.randn_like(k_cache)
        extend_seq_lens = torch.tensor([total_q], dtype=torch.int32, device=device)
        prefix_lens = torch.tensor([kv_len - total_q], dtype=torch.int32, device=device)

        plan_mock = mock.MagicMock(return_value=None)
        run_mock = mock.MagicMock(return_value=(None, None))
        select_mock = mock.MagicMock(return_value=None)
        with (
            mock.patch.object(fmha_sm100, "fmha_sm100_plan", plan_mock),
            mock.patch.object(fmha_sm100, "fmha_sm100", run_mock),
            mock.patch.object(fmha_sm100, "sparse_topk_select", select_mock),
        ):
            for layer_id in range(3):
                # The backend allocates cu_seqlens inside forward_extend, so
                # every layer sees a different tensor/address.
                cu_seqlens = torch.tensor(
                    [0, total_q], dtype=torch.int32, device=device
                )
                msa_ppu_forward_extend(
                    backend,
                    q,
                    idx_q,
                    idx_k_cache,
                    k_cache,
                    v_cache,
                    forward_batch,
                    cu_seqlens,
                    forward_batch.seq_lens,
                    prefix_lens,
                    extend_seq_lens,
                    [total_q],
                    max_seqlen_q=total_q,
                    layer_id=layer_id,
                )

        # 1 indexer chunk plan + 1 whole-batch attend plan, shared by 3 layers.
        self.assertEqual(plan_mock.call_count, 2)
        self.assertNotIn("use_fp8_kvcache", plan_mock.call_args_list[0].kwargs)
        self.assertFalse(plan_mock.call_args_list[1].kwargs["use_fp8_kvcache"])
        self.assertGreaterEqual(
            run_mock.call_count, 3 * 2
        )  # indexer + attend per layer
        self.assertIsNotNone(backend._ppu_msa_state)
        self.assertNotIn("attend_q_fp8", backend._ppu_msa_state)

        # A new forward (init_forward_metadata_out_graph clears the state and a
        # new ForwardBatch arrives) rebuilds: plan count grows again.
        forward_batch2 = SimpleNamespace(
            seq_lens=torch.tensor([kv_len + PAGE], dtype=torch.int32, device=device),
            seq_lens_cpu=torch.tensor([kv_len + PAGE], dtype=torch.int64),
            req_pool_indices=torch.zeros(1, dtype=torch.int64, device=device),
        )
        backend._ppu_msa_state = None
        backend._ppu_msa_use_fp8_kvcache = True
        k_cache_fp8 = k_cache.to(torch.float8_e4m3fn)
        v_cache_fp8 = v_cache.to(torch.float8_e4m3fn)
        with (
            mock.patch.object(fmha_sm100, "fmha_sm100_plan", plan_mock),
            mock.patch.object(fmha_sm100, "fmha_sm100", run_mock),
            mock.patch.object(fmha_sm100, "sparse_topk_select", select_mock),
        ):
            cu_seqlens = torch.tensor([0, total_q], dtype=torch.int32, device=device)
            msa_ppu_forward_extend(
                backend,
                q,
                idx_q,
                idx_k_cache,
                k_cache_fp8,
                v_cache_fp8,
                forward_batch2,
                cu_seqlens,
                forward_batch2.seq_lens,
                prefix_lens,
                extend_seq_lens,
                [total_q],
                max_seqlen_q=total_q,
                layer_id=0,
            )
        self.assertEqual(plan_mock.call_count, 4)
        self.assertNotIn("use_fp8_kvcache", plan_mock.call_args_list[-2].kwargs)
        self.assertTrue(plan_mock.call_args_list[-1].kwargs["use_fp8_kvcache"])
        indexer_call, attend_call = run_mock.call_args_list[-2:]
        self.assertTrue(torch.equal(indexer_call.args[0], idx_q))
        attend_q = attend_call.args[0]
        self.assertEqual(attend_q.dtype, torch.float8_e4m3fn)
        self.assertEqual(
            attend_q.data_ptr(), backend._ppu_msa_state["attend_q_fp8"].data_ptr()
        )
        self.assertTrue(torch.equal(attend_q, q.to(torch.float8_e4m3fn)))
        self.assertEqual(attend_call.args[1].dtype, torch.float8_e4m3fn)
        self.assertEqual(attend_call.args[2].dtype, torch.float8_e4m3fn)
        self.assertEqual(attend_call.kwargs["out"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
