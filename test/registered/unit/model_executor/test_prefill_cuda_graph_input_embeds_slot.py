"""Unit tests: `input_embeds` static slot for the prefill CUDA-graph body
(breakable / full) — text-only EAGLE/MTP draft regression.

Root cause under test (E8 verdict): the prefill CUDA-graph buffer registry
registered the static `input_embeds` slot only for multimodal models, so a
text-only draft (MTP) graph captured the `input_embeds is None` branch of the
inner layer model. On replay, the outer MTP wrapper computes the real-time
fusion input `fc(cat(embed_tokens(ids), target_hidden_states))` and passes it
as the 4th positional arg of `layer_model.forward`, but with no slot
`replay_layer_forward` silently dropped it — the draft prefill only ever
consumed token embeddings and the target hidden states never entered the
draft computation.

Fix under test (branch xujiantao/agent/draft_prefill_graph_fixes, commit
c8720fb): the slot registration is gated by `register_input_embeds`
independently of is_multimodal; the runner derives it from
`needs_input_embeds_slot = is_multimodal or (eagle draft + eager-tail
backend)`; a missing `input_embeds` source buffer is owned by the registry
instead of raising; and a live fusion input without a slot now raises
RuntimeError instead of being dropped.

Covered contracts (C1-C7 + 无畏号 T1-T22):
  C1/T1-T5  slot registration matrix + slot attributes + reset_padding,
  C2a-d/T6-T8  source adoption: input_embeds optional, others strict,
  C3/T22  capture binds forward_batch.input_embeds to the static slot
          (same storage as replay) or None,
  C4/T16/T20  replay copies the live fusion input bitwise into the slot
              before graph replay, live rows only,
  C5/T17  no fusion input ⇒ `embed()(input_ids)` fallback, fail-closed
          without get_input_embeddings(),
  C6/T18-T19/T21  guardrail: live fusion input with no slot ⇒ RuntimeError
          naming the slot and the disabled-backend escape hatch, raised
          before replay,
  C7  non-first PP rank ⇒ no slot fill, replay still runs,
  T9-T11  the `needs_input_embeds_slot` predicate matrix (target unchanged,
          draft-only fix).

No server, no model loading, no accelerator, no network. CPU-only. Small
explicit geometry (max_num_tokens=64, hidden=16/32) — no production config
is read.
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    PaddingPolicy,
    build_prefill_registry,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
    _build_layer_model_forward_kwargs,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_utils.buffers import PrefillInputBuffers

# Measured locally (aarch64 CPU, torch 2.15): 5.3-5.7s wall for the whole
# file; est_time budgets CI scheduling headroom on slower runners.
register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 16
HIDDEN_ALT = 32
MAX_NUM_TOKENS = 64
MAX_BS = 4
NUM_TOKENS = 32
EMBED_DTYPE = torch.bfloat16
DEVICE = torch.device("cpu")


def _embed_formula(input_ids: torch.Tensor) -> torch.Tensor:
    """Deterministic embedding value: depends only on the token id."""
    return ((input_ids.float() % 7.0) + 1.0).unsqueeze(-1) / 8.0


def _build_registry(
    *,
    is_multimodal=False,
    register_input_embeds=True,
    hidden_size=HIDDEN,
    max_num_tokens=MAX_NUM_TOKENS,
    source=None,
):
    return build_prefill_registry(
        device=DEVICE,
        max_bs=MAX_BS,
        max_num_token=max_num_tokens,
        cache_loc_dtype=torch.int64,
        is_multimodal=is_multimodal,
        hidden_size=hidden_size,
        embed_dtype=EMBED_DTYPE,
        share_pool=False,
        source=source,
        register_input_embeds=register_input_embeds,
    )


def _create_buffers(*, is_multimodal, hidden_size=HIDDEN):
    return PrefillInputBuffers.create(
        device=DEVICE,
        max_bs=MAX_BS,
        max_num_tokens=MAX_NUM_TOKENS,
        cache_loc_dtype=torch.int64,
        is_multimodal=is_multimodal,
        hidden_size=hidden_size,
        dtype=EMBED_DTYPE,
        enable_mamba_track=False,
    )


def _detached_input_ids(num_tokens=NUM_TOKENS):
    return torch.arange(num_tokens, dtype=torch.int64) % 1000


def _detached_fusion(num_tokens=NUM_TOKENS):
    return (
        torch.arange(num_tokens * HIDDEN, dtype=torch.float32)
        .reshape(num_tokens, HIDDEN)
        .to(EMBED_DTYPE)
    )


class _StubEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids):
        self.calls.append(input_ids.clone())
        return _embed_formula(input_ids).repeat(1, HIDDEN)


class _StubLayerModel(torch.nn.Module):
    """Mimics the inner Qwen3_5ForCausalLM signature: input_embeds is the
    4th positional parameter (detected via inspect.signature)."""

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        raise AssertionError(
            "layer_model.forward must be monkey-patched by the runner during "
            "body capture/replay; the stub must never run eagerly here."
        )


class _StubOuterModel(torch.nn.Module):
    """Mimics the MTP wrapper: computes a fusion input and passes it as the
    4th positional arg of the (possibly monkey-patched) layer model."""

    def __init__(self, layer_model, embed):
        super().__init__()
        self.layer_model = layer_model
        self._embed = embed

    def get_input_embeddings(self):
        return self._embed

    def forward(self, input_ids, positions, forward_batch, fusion=None):
        if fusion is None:
            return self.layer_model(input_ids, positions, forward_batch)
        return self.layer_model(input_ids, positions, forward_batch, fusion)


class _StubOuterModelNoEmbeddings(torch.nn.Module):
    """Same as _StubOuterModel but exposes no get_input_embeddings()."""

    def __init__(self, layer_model):
        super().__init__()
        self.layer_model = layer_model

    def forward(self, input_ids, positions, forward_batch, fusion=None):
        return self.layer_model(input_ids, positions, forward_batch)


class _StubBackend:
    """Records replay calls; snapshots the input_embeds slot at replay time
    so tests can prove the slot was filled BEFORE the graph replayed."""

    def __init__(self, registry=None):
        self.replay_calls = []
        self.replay_slot_snapshots = []
        self._registry = registry

    def replay(self, shape_key, static_forward_batch, **kwargs):
        self.replay_calls.append(shape_key)
        if self._registry is not None and self._registry.has_slot("input_embeds"):
            self.replay_slot_snapshots.append(
                self._registry.get_slot("input_embeds")
                .slice_for(1, len(static_forward_batch.input_ids))
                .clone()
            )
        return torch.zeros(NUM_TOKENS, HIDDEN, dtype=EMBED_DTYPE)


def _make_forward_batch(num_tokens=NUM_TOKENS, input_embeds=None):
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=_detached_input_ids(num_tokens),
        positions=torch.arange(num_tokens, dtype=torch.int64),
        input_embeds=input_embeds,
        out_cache_loc=torch.arange(num_tokens, dtype=torch.int64),
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        seq_lens=torch.full((1,), num_tokens, dtype=torch.int32),
        seq_lens_cpu=torch.full((1,), num_tokens, dtype=torch.int32),
        seq_lens_sum=num_tokens,
        spec_info=None,
    )


def _make_runner(
    registry,
    *,
    pp_is_first_rank=True,
    input_embeds_arg_idx=3,
    is_multimodal=False,
):
    """Build a PrefillCudaGraphRunner without __init__: only the attributes
    touched by capture_prepare / _execute_body_capture /
    _fill_input_embeds_slot / _live_input_embeds /
    _assert_no_dropped_input_embeds / _uses_eager_prefill_tail."""
    runner = object.__new__(PrefillCudaGraphRunner)
    embed = _StubEmbedding()
    layer_model = _StubLayerModel()
    outer = _StubOuterModel(layer_model, embed)
    runner.model_runner = SimpleNamespace(
        model=outer,
        is_draft_worker=True,
        pp_group=SimpleNamespace(is_first_rank=pp_is_first_rank, is_last_rank=False),
        attn_backend=object(),
        attn_tp_sequence_sharded=lambda num_tokens: False,
        model_config=SimpleNamespace(context_len=4096),
        ngram_embedding_manager=SimpleNamespace(enabled=False),
        prepare_dummy_forward_batch=lambda forward_batch: forward_batch,
        graph_shared_output=None,
    )
    runner.is_multimodal = is_multimodal
    runner.device = DEVICE
    runner.max_bs = MAX_BS
    runner.max_context_size = None
    runner.prefill_backend_name = Backend.BREAKABLE
    runner._capture_req_slots = 1
    runner._prefill_static_buffers = None
    runner.buffer_registry = registry
    runner._input_embeds_arg_idx = input_embeds_arg_idx
    runner.require_mlp_tp_gather = False
    runner.require_attn_tp_gather = False
    runner.dp_size = 1
    runner._capture_lora = False
    runner.capture_hidden_mode = CaptureHiddenMode.LAST
    runner.capture_return_pooled_hidden_states = False
    runner.static_draft_hidden_states = None
    runner.tbo_plugin = SimpleNamespace(
        capture_one_batch_size=lambda forward_batch, num_tokens: None
    )
    runner.layer_model = layer_model
    runner._is_full_backend = False
    runner.attention_layers = []
    runner.mha_companion_layers = []
    runner.has_mha_companion_layers = False
    runner.moe_layers = []
    runner.moe_fusions = []
    runner.dsa_indexers = None
    runner.quant_config = None
    runner.buffers = SimpleNamespace(pp_proxy_tensors=None)
    runner.backend = _StubBackend(registry)
    return runner


def _run_body(runner, *, fusion=None, num_tokens=NUM_TOKENS, static_num_tokens=None):
    static_num_tokens = static_num_tokens or num_tokens
    forward_batch = _make_forward_batch(num_tokens=num_tokens)
    shape_key = ShapeKey(size=static_num_tokens)
    kwargs = {} if fusion is None else {"fusion": fusion}
    output = runner._execute_body_capture(
        forward_batch,
        forward_batch,
        static_num_tokens=static_num_tokens,
        raw_num_tokens=num_tokens,
        shape_key=shape_key,
        **kwargs,
    )
    return forward_batch, shape_key, output


def _predicate_expression():
    """Extract the real `needs_input_embeds_slot` RHS from __init__ source.

    The predicate is an inline assignment in __init__ (not a callable), so
    the test executes the actual source expression against a stub instance —
    if the fix's predicate changes shape or disappears, this fails loudly.
    """
    source = textwrap.dedent(inspect.getsource(PrefillCudaGraphRunner.__init__))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "needs_input_embeds_slot"
                ):
                    return ast.get_source_segment(source, node.value)
    raise AssertionError(
        "PrefillCudaGraphRunner.__init__ no longer assigns "
        "needs_input_embeds_slot; the fix predicate is missing."
    )


class TestRegistrySlotRegistration(CustomTestCase):
    """C1/C2c (T1-T4): the input_embeds slot registration matrix."""

    def test_text_only_registry_registers_input_embeds_slot(self):
        # C1 regression: at baseline the slot only exists under
        # is_multimodal, so a text-only model whose layer_model.forward
        # accepts input_embeds got no slot and replay dropped the live
        # fusion input. Also T1: mrope stays multimodal-gated.
        reg = _build_registry(is_multimodal=False, register_input_embeds=True)
        self.assertTrue(
            reg.has_slot("input_embeds"),
            "text-only model whose layer_model.forward accepts input_embeds "
            "must have the input_embeds static slot",
        )
        self.assertFalse(reg.has_slot("mrope_positions"))
        slot = reg.get_slot("input_embeds")
        self.assertEqual(tuple(slot.buffer.shape), (MAX_NUM_TOKENS, HIDDEN))
        self.assertEqual(slot.buffer.dtype, EMBED_DTYPE)
        self.assertIs(slot.padding_policy, PaddingPolicy.ZERO)
        self.assertFalse(slot.copy_from_fb)

    def test_text_only_registry_input_embeds_slot_cross_geometry_hidden32(self):
        # C1 under a different hidden width: the slot must size from the
        # passed hidden_size, not from any production model config.
        reg = _build_registry(
            is_multimodal=False, register_input_embeds=True, hidden_size=HIDDEN_ALT
        )
        self.assertTrue(reg.has_slot("input_embeds"))
        slot = reg.get_slot("input_embeds")
        self.assertEqual(tuple(slot.buffer.shape), (MAX_NUM_TOKENS, HIDDEN_ALT))
        self.assertEqual(slot.buffer.dtype, EMBED_DTYPE)

    def test_multimodal_registry_keeps_input_embeds_and_mrope_slots(self):
        # T3: multimodal path unchanged.
        reg = _build_registry(is_multimodal=True, register_input_embeds=True)
        self.assertTrue(reg.has_slot("input_embeds"))
        self.assertTrue(reg.has_slot("mrope_positions"))

    def test_eager_extend_registry_omits_input_embeds_slot(self):
        # C2c/T2: the eager extend path passes register_input_embeds=False
        # and must keep carrying input_embeds from the batch (read input),
        # never written in-graph.
        reg = _build_registry(is_multimodal=False, register_input_embeds=False)
        self.assertFalse(reg.has_slot("input_embeds"))

    def test_input_embeds_slot_reset_padding_clears_only_tail(self):
        # T5: reset_padding(raw_n, padded_n) zeroes [raw_n:padded_n] and
        # leaves [0:raw_n] bitwise untouched.
        reg = _build_registry(is_multimodal=False, register_input_embeds=True)
        self.assertTrue(
            reg.has_slot("input_embeds"),
            "input_embeds slot missing for text-only registry",
        )
        buffer = reg.get_slot("input_embeds").buffer
        head = _detached_fusion(NUM_TOKENS)
        buffer[:NUM_TOKENS] = head
        reg.get_slot("input_embeds").reset_padding(
            raw_n=NUM_TOKENS, padded_n=NUM_TOKENS + 8
        )
        self.assertTrue(torch.equal(buffer[:NUM_TOKENS], head))
        self.assertTrue(
            torch.equal(
                buffer[NUM_TOKENS : NUM_TOKENS + 8],
                torch.zeros(8, HIDDEN, dtype=EMBED_DTYPE),
            )
        )


class TestSourceAdoption(CustomTestCase):
    """C2a/C2d (T6-T8): source buffers — input_embeds optional, others strict."""

    def test_text_only_source_build_registers_registry_owned_input_embeds_slot(self):
        # C2a/T6: a text-only PrefillInputBuffers does not allocate
        # input_embeds (方案B: buffers.py unchanged); build_prefill_registry
        # must not raise and must let the registry own the allocation.
        buffers = _create_buffers(is_multimodal=False)
        self.assertIsNone(buffers.input_embeds)
        reg = _build_registry(
            is_multimodal=False, register_input_embeds=True, source=buffers
        )
        self.assertTrue(
            reg.has_slot("input_embeds"),
            "text-only source with register_input_embeds=True must still get "
            "the input_embeds slot (registry-owned allocation)",
        )
        slot = reg.get_slot("input_embeds")
        self.assertIsNotNone(slot.buffer)
        self.assertEqual(tuple(slot.buffer.shape), (MAX_NUM_TOKENS, HIDDEN))
        self.assertEqual(slot.buffer.dtype, EMBED_DTYPE)

    def test_multimodal_source_adopts_same_input_embeds_buffer(self):
        # T7: multimodal adoption unchanged — the registry adopts the source
        # buffer (same object) instead of reallocating.
        buffers = _create_buffers(is_multimodal=True)
        self.assertIsNotNone(buffers.input_embeds)
        reg = _build_registry(
            is_multimodal=True, register_input_embeds=True, source=buffers
        )
        self.assertTrue(reg.has_slot("input_embeds"))
        self.assertIs(reg.get_slot("input_embeds").buffer, buffers.input_embeds)

    def test_source_missing_non_input_embeds_buffer_raises(self):
        # C2d/T8: the relaxation is scoped to input_embeds only; any other
        # missing source buffer must still raise ValueError.
        source = SimpleNamespace(
            input_ids=torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64),
            out_cache_loc=torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64),
        )
        with self.assertRaisesRegex(ValueError, "source is missing buffer"):
            _build_registry(
                is_multimodal=False, register_input_embeds=False, source=source
            )


class TestNeedsInputEmbedsSlotPredicate(CustomTestCase):
    """T9-T11 / 契约用例17: the __init__ predicate matrix.

    The predicate is an inline assignment in __init__; this test executes
    the real assignment expression (extracted via ast from the actual
    source) with a stub instance whose `_uses_eager_prefill_tail` is the
    real method. Expected matrix (multimodal, is_eagle, draft, backend):

        text-only target   (F, T, F, breakable)    ⇒ False  (target unchanged)
        eagle draft        (F, T, T, breakable)    ⇒ True
        eagle draft        (F, T, T, full)         ⇒ True
        eagle draft        (F, T, T, tc_piecewise) ⇒ False
        multimodal         (T, *, *, *)            ⇒ True
        non-eagle draft    (F, F, T, breakable)    ⇒ False
    """

    def _eval_predicate(self, *, is_multimodal, is_eagle, is_draft_worker, backend):
        runner = object.__new__(PrefillCudaGraphRunner)
        runner.is_multimodal = is_multimodal
        runner.prefill_backend_name = backend
        model_runner = SimpleNamespace(is_draft_worker=is_draft_worker)
        # Restricted namespace: the evaluated string is the fix's own
        # predicate expression extracted from the pinned source of this
        # repo (never external input); builtins are disabled because the
        # expression only performs attribute lookups and a method call.
        namespace = {
            "__builtins__": {},
            "self": runner,
            "is_eagle": is_eagle,
            "model_runner": model_runner,
        }
        return eval(_predicate_expression(), namespace)

    def test_predicate_matrix(self):
        cells = [
            # (is_multimodal, is_eagle, is_draft_worker, backend, expected)
            (False, True, False, Backend.BREAKABLE, False),
            (False, True, True, Backend.BREAKABLE, True),
            (False, True, True, Backend.FULL, True),
            (False, True, True, Backend.TC_PIECEWISE, False),
            (True, False, False, Backend.BREAKABLE, True),
            (True, True, True, Backend.TC_PIECEWISE, True),
            (False, False, True, Backend.BREAKABLE, False),
        ]
        for is_multimodal, is_eagle, is_draft, backend, expected in cells:
            with self.subTest(
                is_multimodal=is_multimodal,
                is_eagle=is_eagle,
                is_draft_worker=is_draft,
                backend=backend,
            ):
                actual = self._eval_predicate(
                    is_multimodal=is_multimodal,
                    is_eagle=is_eagle,
                    is_draft_worker=is_draft,
                    backend=backend,
                )
                self.assertEqual(actual, expected)

    def test_init_wires_predicate_into_registry_build(self):
        # The predicate must feed build_prefill_registry's
        # register_input_embeds argument (wiring pin; textual by design).
        source = inspect.getsource(PrefillCudaGraphRunner.__init__)
        self.assertIn(
            "register_input_embeds=self.needs_input_embeds_slot",
            source,
            "build_prefill_registry must receive the needs_input_embeds_slot "
            "predicate as register_input_embeds",
        )


class TestCapturePrepareInputEmbedsBinding(CustomTestCase):
    """C3/C2b (T22 capture half): capture binds the static slot / None."""

    def test_capture_prepare_binds_static_slot_for_text_only(self):
        runner = _make_runner(_build_registry(is_multimodal=False))
        forward_batch, _ = runner.capture_prepare(num_tokens=NUM_TOKENS)
        expected = runner.buffer_registry.get_slot("input_embeds").slice_for(
            1, NUM_TOKENS
        )
        self.assertIsNotNone(
            forward_batch.input_embeds,
            "capture must bind forward_batch.input_embeds to the static slot "
            "so the captured body takes the input_embeds branch",
        )
        self.assertEqual(forward_batch.input_embeds.data_ptr(), expected.data_ptr())
        self.assertEqual(tuple(forward_batch.input_embeds.shape), (NUM_TOKENS, HIDDEN))

    def test_capture_prepare_no_slot_passes_none(self):
        runner = _make_runner(
            _build_registry(is_multimodal=False, register_input_embeds=False)
        )
        forward_batch, _ = runner.capture_prepare(num_tokens=NUM_TOKENS)
        self.assertIsNone(forward_batch.input_embeds)

    def test_capture_prepare_multimodal_binding_unchanged(self):
        runner = _make_runner(_build_registry(is_multimodal=True))
        forward_batch, _ = runner.capture_prepare(num_tokens=NUM_TOKENS)
        expected = runner.buffer_registry.get_slot("input_embeds").slice_for(
            1, NUM_TOKENS
        )
        self.assertIsNotNone(forward_batch.input_embeds)
        self.assertEqual(forward_batch.input_embeds.data_ptr(), expected.data_ptr())

    def test_build_layer_model_forward_kwargs_binds_input_embeds_by_signature(self):
        # Capture-side branch solidification: the capture forward passes
        # forward_batch.input_embeds into layer_model.forward by name, so with
        # the slot bound the graph captures the input_embeds branch instead of
        # embed_tokens(input_ids).
        layer_model = _StubLayerModel()
        forward_batch = _make_forward_batch(input_embeds=torch.zeros(2, HIDDEN))
        kwargs = _build_layer_model_forward_kwargs(layer_model, forward_batch, None)
        self.assertEqual(kwargs, {"input_embeds": forward_batch.input_embeds})

        class _NoEmbedsLayerModel(torch.nn.Module):
            def forward(self, input_ids, positions, forward_batch):
                raise AssertionError

        self.assertEqual(
            _build_layer_model_forward_kwargs(
                _NoEmbedsLayerModel(), forward_batch, None
            ),
            {},
        )


class TestLiveInputEmbedsLookup(CustomTestCase):
    """T12-T15: _live_input_embeds lookup order."""

    def test_lookup_order(self):
        runner = _make_runner(_build_registry(is_multimodal=True))
        live = runner._live_input_embeds
        embeds_x = torch.zeros(2, HIDDEN)
        embeds_y = torch.ones(2, HIDDEN)
        embeds_z = torch.full((2, HIDDEN), 2.0)
        # T12: kwarg input_embeds wins.
        self.assertIs(live([], {"input_embeds": embeds_x}), embeds_x)
        # T13: fallback kwarg inputs_embeds.
        self.assertIs(live([], {"inputs_embeds": embeds_y}), embeds_y)
        # T14: positional slot at _input_embeds_arg_idx (3 for the MTP draft).
        self.assertIs(
            live([None, None, None, embeds_z], {}), embeds_z
        )
        # T15: nothing present ⇒ None (triggers the embed fallback).
        self.assertIsNone(live([], {}))


class TestReplayFusionInputCopy(CustomTestCase):
    """C4/C5 (T16-T17, T20): replay fills the slot from the live input."""

    def test_replay_copies_live_fusion_input_into_slot_before_replay(self):
        for is_multimodal in (True, False):
            with self.subTest(is_multimodal=is_multimodal):
                registry = _build_registry(is_multimodal=is_multimodal)
                runner = _make_runner(registry, is_multimodal=is_multimodal)
                original_forward = runner.layer_model.forward
                fusion = _detached_fusion()
                # Core regression (C4): on the baseline the text-only branch
                # fails this assertion (no slot) instead of raising from
                # helper code.
                self.assertTrue(
                    registry.has_slot("input_embeds"),
                    "text-only model whose layer_model.forward accepts "
                    "input_embeds must have the input_embeds static slot",
                )
                _, shape_key, output = _run_body(runner, fusion=fusion)
                slot = registry.get_slot("input_embeds").slice_for(1, NUM_TOKENS)
                self.assertTrue(torch.equal(slot[:NUM_TOKENS], fusion))
                self.assertEqual(runner.backend.replay_calls, [shape_key])
                # T20: the slot was already filled when the graph replayed.
                self.assertEqual(len(runner.backend.replay_slot_snapshots), 1)
                self.assertTrue(
                    torch.equal(runner.backend.replay_slot_snapshots[0], fusion)
                )
                self.assertEqual(tuple(output.shape), (NUM_TOKENS, HIDDEN))
                # The monkey-patch must be restored after the body runs.
                # (bound methods compare equal per (func, instance) pair)
                self.assertEqual(runner.layer_model.forward, original_forward)

    def test_replay_copies_exact_live_rows_when_bucket_matches(self):
        registry = _build_registry(is_multimodal=True)
        runner = _make_runner(registry)
        fusion = _detached_fusion()
        _run_body(runner, fusion=fusion)
        slot = registry.get_slot("input_embeds").slice_for(1, NUM_TOKENS)
        self.assertTrue(torch.equal(slot, fusion))

    def test_replay_copies_live_rows_when_bucket_pads_the_batch(self):
        # C4: static_num_tokens (capture bucket) > raw_num_tokens (live
        # tokens) — the copy must fill exactly the live prefix of the static
        # slot and leave the padding rows bitwise untouched (asserted against
        # a snapshot, not against any particular initial value).
        registry = _build_registry(is_multimodal=True)
        runner = _make_runner(registry)
        live_rows = NUM_TOKENS - 4
        fusion = _detached_fusion(num_tokens=live_rows)
        buffer = registry.get_slot("input_embeds").buffer
        tail_pattern = torch.arange(
            1000, 1000 + buffer[live_rows:].numel(), dtype=torch.float32
        ).reshape(buffer[live_rows:].shape)
        buffer[live_rows:] = tail_pattern.to(EMBED_DTYPE)
        before_tail = buffer[live_rows:].clone()
        _run_body(
            runner,
            fusion=fusion,
            num_tokens=live_rows,
            static_num_tokens=NUM_TOKENS,
        )
        slot = registry.get_slot("input_embeds").slice_for(1, NUM_TOKENS)
        self.assertTrue(torch.equal(slot[:live_rows], fusion))
        self.assertTrue(
            torch.equal(registry.get_slot("input_embeds").buffer[live_rows:],
                        before_tail)
        )

    def test_replay_without_fusion_input_uses_embed_fallback(self):
        # C5/T17: no live fusion input ⇒ the slot is fed from
        # get_input_embeddings()(input_ids), called exactly once.
        registry = _build_registry(is_multimodal=True)
        runner = _make_runner(registry)
        forward_batch, _, output = _run_body(runner)
        embed = runner.model_runner.model._embed
        self.assertEqual(len(embed.calls), 1)
        self.assertTrue(torch.equal(embed.calls[0], forward_batch.input_ids))
        slot = registry.get_slot("input_embeds").slice_for(1, NUM_TOKENS)
        expected = _embed_formula(forward_batch.input_ids).repeat(1, HIDDEN).to(EMBED_DTYPE)
        self.assertTrue(torch.equal(slot[:NUM_TOKENS], expected))
        self.assertEqual(len(runner.backend.replay_calls), 1)
        self.assertEqual(tuple(output.shape), (NUM_TOKENS, HIDDEN))

    def test_replay_embed_fallback_fails_closed_without_get_input_embeddings(self):
        # C5/T17: no fusion input and no get_input_embeddings ⇒ fail closed.
        registry = _build_registry(is_multimodal=True)
        runner = _make_runner(registry)
        layer_model = runner.layer_model
        runner.model_runner.model = _StubOuterModelNoEmbeddings(layer_model)
        with self.assertRaises(AssertionError):
            _run_body(runner)


class TestNoDroppedInputEmbedsGuardrail(CustomTestCase):
    """C6/T18-T21: a live fusion input without a slot must fail loudly."""

    def test_replay_fusion_input_without_slot_raises_before_replay(self):
        # C6/T21 via the full replay dispatch: the guard raises before the
        # graph replays (replay_calls stays empty).
        registry = _build_registry(is_multimodal=False, register_input_embeds=False)
        self.assertFalse(registry.has_slot("input_embeds"))
        runner = _make_runner(registry)
        with self.assertRaisesRegex(RuntimeError, r"(?s)input_embeds.*slot"):
            _run_body(runner, fusion=_detached_fusion())
        self.assertEqual(runner.backend.replay_calls, [])

    def test_guardrail_message_names_slot_and_disabled_escape_hatch(self):
        # T18: the RuntimeError names the input_embeds slot and points at
        # --cuda-graph-backend-prefill=disabled.
        registry = _build_registry(is_multimodal=False, register_input_embeds=False)
        runner = _make_runner(registry)
        args = [None, None, None, _detached_fusion()]
        with self.assertRaisesRegex(RuntimeError, r"(?s)input_embeds.*slot") as ctx:
            runner._assert_no_dropped_input_embeds(args, {})
        self.assertIn("--cuda-graph-backend-prefill=disabled", str(ctx.exception))

    def test_guardrail_is_noop_without_live_input(self):
        # T19: no slot and no live input ⇒ the guard must not raise.
        registry = _build_registry(is_multimodal=False, register_input_embeds=False)
        runner = _make_runner(registry)
        self.assertIsNone(runner._assert_no_dropped_input_embeds([None, None, None], {}))

    def test_replay_skips_slot_fill_on_non_first_pp_rank(self):
        # C7: a non-first PP rank neither fills the slot nor trips the
        # guardrail; the graph replay still runs. The replay-time snapshot
        # proves the slot was still empty when the graph replayed.
        registry = _build_registry(is_multimodal=True)
        runner = _make_runner(registry, pp_is_first_rank=False)
        _run_body(runner, fusion=_detached_fusion())
        slot = registry.get_slot("input_embeds").slice_for(1, NUM_TOKENS)
        self.assertTrue(
            torch.equal(slot, torch.zeros(NUM_TOKENS, HIDDEN, dtype=EMBED_DTYPE))
        )
        self.assertEqual(len(runner.backend.replay_calls), 1)
        self.assertEqual(len(runner.backend.replay_slot_snapshots), 1)
        self.assertTrue(
            torch.equal(
                runner.backend.replay_slot_snapshots[0],
                torch.zeros(NUM_TOKENS, HIDDEN, dtype=EMBED_DTYPE),
            )
        )


if __name__ == "__main__":
    unittest.main()
