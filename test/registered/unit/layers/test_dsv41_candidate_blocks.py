import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    select_candidate_block_ids,
    topk_among_blocks,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
    BlockIds,
    _consume_prefill_blocks,
    _publish_prefill_blocks,
    _publish_tile_blocks,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import DeepGEMMPrefillData
from sglang.srt.layers.attention.dsv4.v41_indexer.types import get_tail_row_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def id_sets(blocks: torch.Tensor) -> list:
    return [set(row.tolist()) - {-1} for row in blocks]


def masked_block_ids(scores, lens, topk_blocks, block_size):
    """Pre-change oracle: mask tokens before block-max and force the newest
    block, retaining torch.amax's handling of valid NaNs.
    """
    width = scores.shape[-1]
    lens = torch.as_tensor(lens, device=scores.device).reshape(-1, 1)
    visible = scores.masked_fill(
        torch.arange(width, device=scores.device)[None, :] >= lens, -torch.inf
    )
    padding = -width % block_size
    if padding:
        visible = torch.nn.functional.pad(visible, (0, padding), value=-torch.inf)
    keys = visible.unflatten(-1, (-1, block_size)).amax(-1)
    keys = keys.masked_fill(
        torch.arange(keys.shape[-1], device=scores.device) == (lens - 1) // block_size,
        torch.inf,
    )
    top = keys.topk(min(topk_blocks, keys.shape[-1]), dim=-1)
    return top.indices.to(torch.int32).masked_fill_(~(top.values > -torch.inf), -1)


class RejectFullTokenMask(TorchDispatchMode):
    def __init__(self, shapes):
        super().__init__()
        self.shapes = set(shapes)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for value in tree_leaves(result):
            if (
                isinstance(value, torch.Tensor)
                and value.dtype == torch.bool
                and value.ndim >= 2
                and (value.shape[0], math.prod(value.shape[1:])) in self.shapes
            ):
                raise AssertionError("full-size token causal mask allocated")
        return result


def masked_topk(scores, lens, blocks, k, block_size):
    """The reference: -inf outside the chosen blocks and past each row's length,
    then a plain top-k; -1 where fewer than k candidates are finite."""
    rows, width = scores.shape
    keep = torch.zeros(rows, width, dtype=torch.bool)
    for r in range(rows):
        for b in blocks[r].tolist():
            if b >= 0:
                keep[r, b * block_size : (b + 1) * block_size] = True
    keep &= torch.arange(width)[None, :] < lens[:, None]
    masked = scores.masked_fill(~keep, -torch.inf)
    top = masked.topk(min(k, width), dim=-1)
    return [
        {c for c, v in zip(ci.tolist(), vi.tolist()) if v > -torch.inf}
        for ci, vi in zip(top.indices, top.values)
    ]


class TestCandidateBlockIds(CustomTestCase):
    def test_unmasked_tails_match_token_masked_reference(self):
        """Future blocks and partial newest-block tails may contain stale
        finite values, infinities or NaNs. They must not displace valid blocks,
        including zero-length rows and scalar length inputs. NaNs in visible
        older blocks must retain the original torch.amax behavior, too.
        """
        generator = torch.Generator().manual_seed(7)
        for dtype in (torch.float32, torch.bfloat16):
            for width in (19, 24):
                clean = torch.stack(
                    [torch.randperm(width, generator=generator) for _ in range(6)]
                ).to(dtype)
                clean[-1, 0] = float("nan")
                for lens in (torch.tensor([[0], [1], [7], [8], [9], [width]]), 9):
                    for poison in (float("nan"), float("inf"), 10000.0):
                        with self.subTest(
                            dtype=dtype,
                            width=width,
                            scalar=isinstance(lens, int),
                            poison=poison,
                        ):
                            tail = torch.arange(width)[None, :] >= torch.as_tensor(lens)
                            scores = clean.masked_fill(tail, poison)
                            original = scores.clone()
                            expected = masked_block_ids(scores, lens, 2, 8)
                            actual = select_candidate_block_ids(scores, lens, 2, 8)
                            torch.testing.assert_close(actual, expected)
                            torch.testing.assert_close(
                                scores, original, rtol=0, atol=0, equal_nan=True
                            )

    def test_causal_partial_blocks_and_forced_newest_block(self):
        scores = torch.tensor([[100.0] * 8 + [50.0] * 8 + [-10.0] * 3] * 4)
        lengths = torch.tensor([[0], [1], [9], [19]])
        scores.masked_fill_(torch.arange(19)[None, :] >= lengths, -torch.inf)
        original_scores = scores.clone()
        blocks = select_candidate_block_ids(
            logits=scores, compress_lens=lengths, topk_blocks=2, block_size=8
        )
        self.assertEqual(blocks.dtype, torch.int32)
        self.assertEqual(tuple(blocks.shape), (4, 2))
        # nothing visible; one partial block; two blocks; the newest (partial)
        # block is forced in ahead of the 50-score one
        self.assertEqual(id_sets(blocks), [set(), {0}, {0, 1}, {0, 2}])
        torch.testing.assert_close(scores, original_scores)

    def test_underfilled_candidates_keep_every_block(self):
        for width in (0, 1, 7, 8, 9):
            with self.subTest(width=width):
                blocks = select_candidate_block_ids(
                    logits=torch.zeros((2, width)),
                    compress_lens=width,
                    topk_blocks=2048,
                    block_size=8,
                )
                expected = set(range((width + 7) // 8))
                self.assertEqual(id_sets(blocks), [expected, expected])

    def test_publish_tile_uses_aligned_view_without_pad(self):
        """Partial final blocks must not copy the full logits tile and OOM (#244)."""
        rows, lc = 4, 19
        lens = torch.tensor([19, 18, 9, 1], dtype=torch.int32)
        original = torch.randn(rows, 24)
        expected = masked_block_ids(original[:, :lc], lens, 2, 8)
        data = SimpleNamespace(
            compress_lens=lens,
            rows_per_request=[rows],
            lens_per_request=[lc],
        )
        actual = torch.full((rows, 2), -1, dtype=torch.int32)
        working = original.clone()
        with (
            RejectFullTokenMask({(rows, 24)}),
            patch(
                "sglang.kernels.ops.attention.dsv4.candidate_blocks.F.pad",
                side_effect=AssertionError("F.pad should not be called"),
            ),
        ):
            _publish_tile_blocks(
                data=data,
                tile=slice(0, rows),
                logits=working,
                blocks=actual,
                topk_blocks=2,
                block_size=8,
            )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(working, original)

    def test_unmasked_publish_and_consume_across_request_and_tile_boundaries(self):
        """Removing the publisher's in-place token mask must preserve both
        its length-aware token top-k and later layers' selection within blocks.
        Exercise two score tiles, a split request and a tile shared by requests.
        """
        from sglang.kernels.ops.attention.dsv4 import index_logits
        from sglang.srt.layers.attention.dsv4.v41_indexer import scoring

        rows, width, topk = 8, 32, 3
        lens = torch.tensor([0, 1, 8, 9, 25, 0, 9, 19], dtype=torch.int32)
        starts = torch.tensor([0] * 5 + [25] * 3, dtype=torch.int32)
        generator = torch.Generator().manual_seed(23)
        # A wider backing tensor reproduces the producer's padded row stride.
        publisher = torch.full((rows, 40), float("nan"))
        consumer = torch.full_like(publisher, float("inf"))
        for row, length in enumerate(lens.tolist()):
            publisher[row, :length] = torch.randperm(
                length, generator=generator
            ).float()
            consumer[row, :length] = torch.randperm(length, generator=generator).float()
        original_publisher, original_consumer = publisher.clone(), consumer.clone()
        expected_blocks = torch.full((rows, 2), -1, dtype=torch.int32)
        expected_consumed = []
        for request_rows, request_width in ((slice(0, 5), 32), (slice(5, 8), 24)):
            ids = masked_block_ids(
                publisher[request_rows, :request_width], lens[request_rows], 2, 8
            )
            expected_blocks[request_rows] = ids
            expected_consumed.extend(
                masked_topk(
                    consumer[request_rows, :request_width],
                    lens[request_rows],
                    ids,
                    topk,
                    8,
                )
            )
        expected_published = [
            set(
                (
                    publisher[row, :length].topk(min(topk, length)).indices
                    + starts[row]
                ).tolist()
            )
            for row, length in enumerate(lens.tolist())
        ]
        q = torch.zeros((rows, 32, 64), dtype=torch.int8)
        q[:, 0, 0] = torch.arange(rows, dtype=torch.int8)
        data = DeepGEMMPrefillData(
            k_slots=torch.arange(44) * 7 + 5,
            request_starts=starts,
            lens_per_request=[25, 19],
            rows_per_request=[5, 3],
            compress_lens=lens,
            q_fp4=q,
            q_sf=torch.zeros((rows, 32), dtype=torch.int32),
            weights=torch.ones((rows, 32)),
        )
        calls = []

        def produce(
            query, kv, weights, ks, ke, clean_logits, max_seqlen_k, logits_dtype
        ):
            tag = int(query[0][0, 0, 0])
            layer, start = tag // 32, tag % 32
            stop = start + query[0].shape[0]
            calls.append((layer, start, stop))
            self.assertFalse(clean_logits)
            self.assertEqual(max_seqlen_k, index_logits.FLAT_LOGITS_WIDTH_ALIGN)
            torch.testing.assert_close(ks, starts[start:stop])
            torch.testing.assert_close(ke - ks, lens[start:stop])
            return (publisher if layer == 0 else consumer)[start:stop, :width]

        def ragged_topk(scores, lengths, *, out_offsets, out_indices, row_starts=None):
            # External kernel boundary: honor the real ragged API's length and
            # output-offset contract without reading or cleaning the causal tail.
            self.assertIsNone(row_starts)
            out_indices.fill_(-1)
            for row, length in enumerate(lengths.tolist()):
                indices = (
                    scores[row, :length]
                    .topk(min(out_indices.shape[1], length), sorted=False)
                    .indices
                )
                out_indices[row, : indices.numel()] = indices + out_offsets[row]

        with (
            patch.dict(
                "sys.modules",
                {"deep_gemm": SimpleNamespace(fp8_fp4_mqa_logits=produce)},
            ),
            patch.object(index_logits, "is_sail", return_value=False),
            patch.object(scoring, "_deep_gemm_score_budget_bytes", return_value=4096),
            patch(
                "sglang.kernels.ops.attention.dsv4.topk_transform_ragged_v2",
                new=ragged_topk,
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.candidate_blocks.F.pad",
                side_effect=AssertionError("no full padding"),
            ),
            RejectFullTokenMask({(4, 32), (1, 32), (3, 24)}),
        ):
            selected, blocks = _publish_prefill_blocks(
                data=data, kv=(None, None), topk=topk, topk_blocks=2, block_size=8
            )
            q[:, 0, 0] += (
                32  # Distinguish the later layer's operands at the native boundary.
            )
            consumed = _consume_prefill_blocks(
                data=data, kv=(None, None), topk=topk, blocks=blocks, block_size=8
            )
        torch.testing.assert_close(blocks, expected_blocks)
        self.assertEqual(id_sets(selected), expected_published)
        expected_global = [
            {position + int(starts[row]) for position in positions}
            for row, positions in enumerate(expected_consumed)
        ]
        self.assertEqual(id_sets(consumed), expected_global)
        self.assertEqual(calls, [(0, 0, 4), (0, 4, 8), (1, 0, 4), (1, 4, 8)])
        torch.testing.assert_close(
            publisher, original_publisher, rtol=0, atol=0, equal_nan=True
        )
        torch.testing.assert_close(
            consumer, original_consumer, rtol=0, atol=0, equal_nan=True
        )
        output = SimpleNamespace(
            out_page_indices=torch.empty((rows, topk), dtype=torch.int32),
            out_raw_indices=torch.empty((rows, topk), dtype=torch.int32),
        )
        data.write_selection(selected=consumed, inputs=output)
        self.assertEqual(
            id_sets(output.out_page_indices),
            [
                {int(data.k_slots[position]) for position in positions}
                for positions in expected_global
            ],
        )
        self.assertEqual(id_sets(output.out_raw_indices), expected_consumed)


class TestTopkAmongBlocks(CustomTestCase):
    def test_matches_masked_topk(self):
        """Gathering the chosen blocks and taking the top-k picks the same
        positions as masking everything else to -inf, including causal cuts,
        -1 padded blocks and rows with fewer than k candidates."""
        g = torch.Generator().manual_seed(0)
        for rows, width, topk_blocks, k in (
            (7, 100, 3, 16),
            (5, 1000, 20, 64),
            (3, 13, 4, 32),
            (2, 8, 1, 4),
        ):
            with self.subTest(width=width, k=k):
                scores = torch.randn(rows, width, generator=g)
                lens = torch.randint(0, width + 1, (rows,), generator=g)
                visible = scores.masked_fill(
                    torch.arange(width)[None, :] >= lens[:, None], -torch.inf
                )
                blocks = select_candidate_block_ids(
                    visible, lens[:, None], topk_blocks=topk_blocks, block_size=8
                )
                picked = topk_among_blocks(scores, lens, blocks, k, block_size=8)
                self.assertEqual(tuple(picked.shape), (rows, k))
                self.assertEqual(
                    id_sets(picked), masked_topk(scores, lens, blocks, k, 8)
                )

    def test_no_candidate_selects_nothing(self):
        scores = torch.randn(2, 16)
        blocks = torch.full((2, 2), -1, dtype=torch.int32)
        picked = topk_among_blocks(scores, torch.tensor([16, 0]), blocks, 4, 8)
        self.assertTrue((picked == -1).all())


class TestBlockIdsTail(CustomTestCase):
    def test_tail_keeps_each_requests_last_rows(self):
        blocks = torch.arange(16, dtype=torch.int32).view(8, 2)
        ids = BlockIds(blocks=blocks, rows_per_request=[5, 0, 3])
        tail = ids.tail([2, 0, 0])
        torch.testing.assert_close(tail.blocks, blocks[3:5])
        self.assertEqual(tail.rows_per_request, [2, 0, 0])
        self.assertEqual(
            get_tail_row_indices(
                full_rows_per_request=[5, 0, 3],
                tail_rows_per_request=[2, 0, 3],
                device="cpu",
            ).tolist(),
            [3, 4, 5, 6, 7],
        )


if __name__ == "__main__":
    unittest.main()
