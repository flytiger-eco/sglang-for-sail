"""CPU tests; supports direct files or an in-memory PRUNING_SOURCE for Pod checks."""
import importlib.util
import os
from pathlib import Path
import types
import unittest

from sglang.test.test_utils import CustomTestCase

import torch

if 'PRUNING_SOURCE' in globals():
    pruning = types.ModuleType('pruning')
    exec(PRUNING_SOURCE, pruning.__dict__)
else:
    path = Path(__file__).resolve().parents[3] / 'python/sglang/srt/layers/attention/visual_token_pruning.py'
    spec = importlib.util.spec_from_file_location('pruning', path)
    pruning = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pruning)


class Item:
    def __init__(self, grids):
        self.image_grid_thw = torch.tensor(grids)

    def set(self, key, value):
        setattr(self, key, value)


class TestVisionPruning(CustomTestCase):
    def test_encoder_and_language_worker_execution_policy(self):
        gate = pruning.should_extract_image_pruning_score
        self.assertTrue(gate(.4, False, False, 1, False))
        self.assertFalse(gate(.4, True, False, 4, False))
        self.assertFalse(gate(.4, True, True, 4, True))
        self.assertFalse(gate(0, False, False, 4, True))
        for dp, tp, graph in [(False, 4, False), (True, 1, False), (False, 1, True)]:
            with self.assertRaises(ValueError):
                gate(.4, False, dp, tp, graph)

    def test_raster_order_and_deepstack(self):
        rows = torch.arange(60).reshape(10, 6)
        score = torch.tensor([0., 9., 1., 8., 2., 7., 3., 6., 4., 5.])
        kept, indices = pruning.prune_visual_tokens_dominant_only(rows, score, .4)
        self.assertEqual(indices.tolist(), [1, 3, 5, 7, 8, 9])
        self.assertTrue(torch.equal(kept, rows[[1, 3, 5, 7, 8, 9]]))

    def test_multiple_items_images_count_floor_and_original_grid(self):
        items = [Item([[1, 2, 6], [1, 2, 10]]), Item([[1, 2, 2]])]
        grids = [i.image_grid_thw.clone() for i in items]
        rows = torch.arange(36).reshape(9, 4)
        kept = pruning.prune_image_items(rows, torch.arange(9.), items, 2, .4)
        self.assertEqual(kept.shape[0], 5)  # floor(3*.6)+floor(5*.6)+max(1,floor(.6))
        self.assertEqual(items[0].image_pruning_indices, [[2], [2, 3, 4]])
        self.assertEqual(items[1].image_pruning_indices, [[0]])
        self.assertTrue(torch.equal(kept, rows[[2, 5, 6, 7, 8]]))
        for item, original in zip(items, grids):
            self.assertTrue(torch.equal(item.image_grid_thw, original))

    def test_rate_zero_identity(self):
        rows = torch.randn(3, 7)
        kept, indices = pruning.prune_visual_tokens_dominant_only(rows, torch.ones(3), 0)
        self.assertIs(rows, kept)
        self.assertEqual(indices.tolist(), [0, 1, 2])

    def test_invalid_metadata_and_rate_rejected(self):
        for rate in [-.1, 1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                pruning.prune_visual_tokens_dominant_only(torch.ones(3, 4), torch.ones(3), rate)
        with self.assertRaises(ValueError):
            pruning.prune_image_items(torch.ones(3, 4), torch.ones(3), [Item([[1, 2, 2]])], 2, .4)
        with self.assertRaises(ValueError):
            pruning.prune_image_items(torch.ones(3, 4), torch.ones(2), [Item([[1, 2, 6]])], 2, .4)


@unittest.skipUnless(os.environ.get("RUN_VISION_GPU_SMOKE") == "1", "opt-in GPU model smoke")
class TestActualVisionModel(CustomTestCase):
    def test_attention_scores_match_dense_reference(self):
        from sglang.srt.layers.attention.flash_attn_with_score import flash_attn_varlen_func_with_score

        torch.manual_seed(31)
        q, k, v = [torch.randn(40, 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        cu = torch.tensor([0, 16, 40], device="cuda", dtype=torch.int32)
        output, scores = flash_attn_varlen_func_with_score(q, k, v, cu, 24, 0.125)
        expected_outputs, expected_scores = [], []
        for start, end in [(0, 16), (16, 40)]:
            qs, ks, vs = [x[start:end].float().transpose(0, 1) for x in (q, k, v)]
            probabilities = torch.softmax(qs @ ks.transpose(-2, -1) * 0.125, dim=-1)
            expected_outputs.append((probabilities @ vs).transpose(0, 1))
            expected_scores.append(probabilities.sum(dim=-2).mean(dim=0))
        torch.testing.assert_close(output.float(), torch.cat(expected_outputs), atol=.02, rtol=.02)
        torch.testing.assert_close(scores.float(), torch.cat(expected_scores), atol=.02, rtol=.02)

    def test_tiny_vision_forward_and_pruned_deepstack(self):
        import socket
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.runtime_context import publish, get_parallel
        from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
        from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel, Qwen3VLForConditionalGeneration

        args = ServerArgs(model_path="dummy", image_pruning_rate=.4, mm_attention_backend="fa3")
        publish(args, role="scheduler")
        get_parallel().override_permanently(
            tp_rank=0, pp_rank=0, attn_dp_size=1, attn_dp_rank=0,
            attn_tp_size=1, attn_tp_rank=0, attn_cp_rank=0, attn_dcp_size=1,
            moe_ep_size=1, moe_ep_rank=0, moe_dp_size=1, moe_tp_size=1,
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl", distributed_init_method=f"tcp://127.0.0.1:{port}")
        self.addCleanup(torch.distributed.destroy_process_group)
        initialize_model_parallel()
        config = Qwen3VLVisionConfig(depth=2, hidden_size=128, intermediate_size=256,
            num_heads=2, in_channels=3, patch_size=2, temporal_patch_size=2,
            spatial_merge_size=2, out_hidden_size=32, num_position_embeddings=64,
            deepstack_visual_indexes=[0])
        torch.manual_seed(7)
        with torch.device("cuda"):
            visual = Qwen3VLMoeVisionModel(config).to(dtype=torch.bfloat16)
        with torch.no_grad():
            for param in visual.parameters():
                param.uniform_(-.05, .05)
        grids = torch.tensor([[1, 4, 4], [1, 4, 6]], device="cuda")
        pixels = torch.randn(40, 24, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            full, scores = visual(pixels, grids)
        self.assertEqual(tuple(full.shape), (10, 64))  # main plus one DeepStack
        self.assertEqual(tuple(scores.shape), (10,))
        model = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.visual = visual
        model.language_model_only = False
        model.use_data_parallel = False
        model._materialize_visual_items = lambda items, indices, **kwargs: (torch.cat([items[i].feature for i in indices]), [], None)
        item = Item([[1, 4, 4], [1, 4, 6]])
        item.image_grid_thw = grids
        item.feature = pixels
        with torch.no_grad():
            kept = model.get_image_feature([item])
        self.assertEqual(tuple(kept.shape), (5, 64))
        expected_indices = item.image_pruning_indices[0] + [4 + i for i in item.image_pruning_indices[1]]
        torch.testing.assert_close(kept, full[expected_indices], atol=.002, rtol=.02)
        # Score extraction does not change the unpruned model's embedding rows.
        visual.extract_image_pruning_score = False
        with torch.no_grad():
            unpruned = visual(pixels, grids)
        torch.testing.assert_close(full, unpruned, atol=.002, rtol=.02)
        print("Tiny Qwen3VL real forward/DeepStack/pruning PASS", item.image_pruning_indices)


if __name__ == '__main__':
    unittest.main()
