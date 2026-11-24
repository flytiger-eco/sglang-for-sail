"""CPU tests for the prefill CUDA-graph static ``input_embeds`` slot.

The captured prefill body (breakable / full backends) reads its stack input
from a static slot. When the outer model computes that input itself -- a
multimodal merge, or an EAGLE/MTP draft fusing the target's hidden states with
the token embeddings -- the slot must exist before capture and be refreshed
before every replay. Without it, the graph keeps the capture-time
``embed_tokens(input_ids)`` branch and the live input is silently dropped.

Contracts under test:

  C1  ``register_input_embeds`` gates the slot independently of
      ``is_multimodal``; omitting the flag keeps the multimodal-only default.
  C2  ``input_embeds`` falls back to a registry-owned buffer when the source
      buffers object has none; any other missing source buffer raises.
  C3  capture binds ``forward_batch.input_embeds`` to the static slot.
  C4  replay copies the live stack input into the slot before the graph runs,
      live rows only.
  C5  with no live input the slot is filled from the embedding accessor; with
      no accessor at all the replay fails closed.
  C6  a live stack input with no slot raises instead of being dropped, unless
      it is exactly ``embed(input_ids)`` -- the same value the captured body
      computes for itself.
  C7  non-first PP ranks neither fill the slot nor trip the guard.

No server, no model loading, no accelerator, no network. CPU only. Small
explicit geometry (max_num_tokens=64, hidden=16/32).
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

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
    _needs_input_embeds_slot,
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

_OMITTED = object()


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
    kwargs = {}
    if register_input_embeds is not _OMITTED:
        kwargs["register_input_embeds"] = register_input_embeds
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
        **kwargs,
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


def _detached_fusion(num_tokens=NUM_TOKENS, width=HIDDEN):
    return (
        torch.arange(num_tokens * width, dtype=torch.float32)
        .reshape(num_tokens, width)
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
    """Mimics a transformer body whose forward takes ``input_embeds`` as its
    4th positional parameter."""

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        raise AssertionError(
            "layer_model.forward must be monkey-patched by the runner during "
            "body capture/replay; the stub must never run eagerly here."
        )


class _StubOuterModel(torch.nn.Module):
    """Mimics an MTP wrapper: computes the body's stack input and passes it
    as the 4th positional argument of the layer model."""

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


class _StubOuterModelWithEmbedInputIds(torch.nn.Module):
    """Wrapper exposing ``embed_input_ids()`` instead of
    ``get_input_embeddings()``."""

    def __init__(self, layer_model, embed):
        super().__init__()
        self.layer_model = layer_model
        self._embed = embed

    def embed_input_ids(self, input_ids):
        return self._embed(input_ids)

    def forward(self, input_ids, positions, forward_batch, fusion=None):
        if fusion is None:
            return self.layer_model(input_ids, positions, forward_batch)
        return self.layer_model(input_ids, positions, forward_batch, fusion)


class _StubOuterModelNoEmbedAccessor(torch.nn.Module):
    """Wrapper with no embedding accessor at all."""

    def __init__(self, layer_model):
        super().__init__()
        self.layer_model = layer_model

    def forward(self, input_ids, positions, forward_batch):
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
        model_config=SimpleNamespace(context_len=4096, hidden_size=HIDDEN),
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


class TestRegistrySlotRegistration(CustomTestCase):
    """C1: the input_embeds slot registration matrix."""

    def test_text_only_registry_registers_input_embeds_slot(self):
        # A text-only model whose layer_model.forward accepts input_embeds must
        # be able to opt in; mrope stays multimodal-gated.
        reg = _build_registry(is_multimodal=False, register_input_embeds=True)
        self.assertTrue(reg.has_slot("input_embeds"))
        self.assertFalse(reg.has_slot("mrope_positions"))
        slot = reg.get_slot("input_embeds")
        self.assertEqual(tuple(slot.buffer.shape), (MAX_NUM_TOKENS, HIDDEN))
        self.assertEqual(slot.buffer.dtype, EMBED_DTYPE)
        self.assertIs(slot.padding_policy, PaddingPolicy.ZERO)
        self.assertFalse(slot.copy_from_fb)

    def test_text_only_registry_input_embeds_slot_cross_geometry_hidden32(self):
        # The slot sizes from the passed hidden_size, not from any production
        # model config.
        reg = _build_registry(
            is_multimodal=False, register_input_embeds=True, hidden_size=HIDDEN_ALT
        )
        self.assertTrue(reg.has_slot("input_embeds"))
        slot = reg.get_slot("input_embeds")
        self.assertEqual(tuple(slot.buffer.shape), (MAX_NUM_TOKENS, HIDDEN_ALT))
        self.assertEqual(slot.buffer.dtype, EMBED_DTYPE)

    def test_multimodal_registry_keeps_input_embeds_and_mrope_slots(self):
        reg = _build_registry(is_multimodal=True, register_input_embeds=True)
        self.assertTrue(reg.has_slot("input_embeds"))
        self.assertTrue(reg.has_slot("mrope_positions"))

    def test_eager_extend_registry_omits_input_embeds_slot(self):
        # The eager extend path passes register_input_embeds=False and keeps
        # carrying input_embeds from the batch (a read input, never written
        # in-graph).
        reg = _build_registry(is_multimodal=False, register_input_embeds=False)
        self.assertFalse(reg.has_slot("input_embeds"))

    def test_omitted_flag_keeps_multimodal_only_default(self):
        # Callers that pass no flag keep the historical behavior: text-only
        # gets no slot, and a zero default hidden_size cannot fabricate a
        # zero-width one.
        text_reg = _build_registry(
            is_multimodal=False, register_input_embeds=_OMITTED, hidden_size=0
        )
        self.assertFalse(text_reg.has_slot("input_embeds"))
        mm_reg = _build_registry(is_multimodal=True, register_input_embeds=_OMITTED)
        self.assertTrue(mm_reg.has_slot("input_embeds"))

    def test_input_embeds_slot_reset_padding_clears_only_tail(self):
        # reset_padding(raw_n, padded_n) zeroes [raw_n:padded_n] and leaves
        # [0:raw_n] bitwise untouched.
        reg = _build_registry(is_multimodal=False, register_input_embeds=True)
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
    """C2: source buffers -- input_embeds optional, others strict."""

    def test_text_only_source_build_registers_registry_owned_input_embeds_slot(self):
        # A text-only PrefillInputBuffers does not allocate input_embeds;
        # build_prefill_registry must not raise and must own the allocation.
        buffers = _create_buffers(is_multimodal=False)
        self.assertIsNone(buffers.input_embeds)
        reg = _build_registry(
            is_multimodal=False, register_input_embeds=True, source=buffers
        )
        self.assertTrue(reg.has_slot("input_embeds"))
        slot = reg.get_slot("input_embeds")
        self.assertIsNotNone(slot.buffer)
        self.assertEqual(tuple(slot.buffer.shape), (MAX_NUM_TOKENS, HIDDEN))
        self.assertEqual(slot.buffer.dtype, EMBED_DTYPE)

    def test_multimodal_source_adopts_same_input_embeds_buffer(self):
        # Multimodal adoption is unchanged: the registry adopts the source
        # buffer (same object) instead of reallocating.
        buffers = _create_buffers(is_multimodal=True)
        self.assertIsNotNone(buffers.input_embeds)
        reg = _build_registry(
            is_multimodal=True, register_input_embeds=True, source=buffers
        )
        self.assertTrue(reg.has_slot("input_embeds"))
        self.assertIs(reg.get_slot("input_embeds").buffer, buffers.input_embeds)

    def test_source_missing_non_input_embeds_buffer_raises(self):
        # The relaxation is scoped to input_embeds only; any other missing
        # source buffer must still raise ValueError.
        source = SimpleNamespace(
            input_ids=torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64),
            out_cache_loc=torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64),
        )
        with self.assertRaisesRegex(ValueError, "source is missing buffer"):
            _build_registry(
                is_multimodal=False, register_input_embeds=False, source=source
            )


class TestSlotEligibility(CustomTestCase):
    """C1 eligibility: the predicate and its effect on the built registry."""

    CELLS = [
        # (is_multimodal, is_eagle, is_draft_worker, captures_body, expected)
        (False, True, False, True, False),  # eagle target: unchanged
        (False, True, True, True, True),  # eagle draft under a body backend
        (False, True, True, False, False),  # no body capture: no slot
        (True, False, False, True, True),  # multimodal
        (True, True, True, False, True),  # multimodal wins over the backend
        (False, False, True, True, False),  # non-eagle draft
    ]

    def test_eligibility_matrix(self):
        for is_mm, is_eagle, is_draft, captures, expected in self.CELLS:
            with self.subTest(
                is_multimodal=is_mm,
                is_eagle=is_eagle,
                is_draft_worker=is_draft,
                captures_body=captures,
            ):
                self.assertEqual(
                    _needs_input_embeds_slot(
                        is_multimodal=is_mm,
                        is_eagle=is_eagle,
                        is_draft_worker=is_draft,
                        captures_body=captures,
                    ),
                    expected,
                )

    def test_eligibility_drives_registry_slot(self):
        # The eligibility decision must be the one the registry acts on.
        for is_mm, is_eagle, is_draft, captures, expected in self.CELLS:
            eligible = _needs_input_embeds_slot(
                is_multimodal=is_mm,
                is_eagle=is_eagle,
                is_draft_worker=is_draft,
                captures_body=captures,
            )
            with self.subTest(expected=expected):
                reg = _build_registry(
                    is_multimodal=is_mm, register_input_embeds=eligible
                )
                self.assertEqual(reg.has_slot("input_embeds"), expected)


class TestCapturePrepareInputEmbedsBinding(CustomTestCase):
    """C3: capture binds the static slot, or None when there is none."""

    def test_capture_prepare_binds_static_slot_for_text_only(self):
        runner = _make_runner(_build_registry(is_multimodal=False))
        forward_batch, _ = runner.capture_prepare(num_tokens=NUM_TOKENS)
        expected = runner.buffer_registry.get_slot("input_embeds").slice_for(
            1, NUM_TOKENS
        )
        self.assertIsNotNone(forward_batch.input_embeds)
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
        # The capture forward passes forward_batch.input_embeds by name, so with
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
    """Replay-side lookup order of the live stack input."""

    def test_lookup_order(self):
        runner = _make_runner(_build_registry(is_multimodal=True))
        live = runner._live_input_embeds
        embeds_x = torch.zeros(2, HIDDEN)
        embeds_y = torch.ones(2, HIDDEN)
        embeds_z = torch.full((2, HIDDEN), 2.0)
        # kwarg input_embeds wins.
        self.assertIs(live([], {"input_embeds": embeds_x}), embeds_x)
        # fallback kwarg inputs_embeds.
        self.assertIs(live([], {"inputs_embeds": embeds_y}), embeds_y)
        # positional slot at _input_embeds_arg_idx (3 for the MTP draft).
        self.assertIs(live([None, None, None, embeds_z], {}), embeds_z)
        # nothing present: None, which triggers the embed fallback.
        self.assertIsNone(live([], {}))


class TestReplayFusionInputCopy(CustomTestCase):
    """C4/C5: replay fills the slot from the live input or the embeddings."""

    def test_replay_copies_live_fusion_input_into_slot_before_replay(self):
        for is_multimodal in (True, False):
            with self.subTest(is_multimodal=is_multimodal):
                registry = _build_registry(is_multimodal=is_multimodal)
                runner = _make_runner(registry, is_multimodal=is_multimodal)
                original_forward = runner.layer_model.forward
                fusion = _detached_fusion()
                _, shape_key, output = _run_body(runner, fusion=fusion)
                slot = registry.get_slot("input_embeds").slice_for(1, NUM_TOKENS)
                self.assertTrue(torch.equal(slot[:NUM_TOKENS], fusion))
                self.assertEqual(runner.backend.replay_calls, [shape_key])
                # The slot was already filled when the graph replayed.
                self.assertEqual(len(runner.backend.replay_slot_snapshots), 1)
                self.assertTrue(
                    torch.equal(runner.backend.replay_slot_snapshots[0], fusion)
                )
                self.assertEqual(tuple(output.shape), (NUM_TOKENS, HIDDEN))
                # The monkey-patch must be restored after the body runs.
                self.assertEqual(runner.layer_model.forward, original_forward)

    def test_replay_copies_live_rows_when_bucket_pads_the_batch(self):
        # static_num_tokens (capture bucket) > raw_num_tokens (live tokens):
        # the copy fills exactly the live prefix and leaves the padding rows
        # bitwise untouched (asserted against a snapshot, not against any
        # particular initial value).
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
            torch.equal(
                registry.get_slot("input_embeds").buffer[live_rows:], before_tail
            )
        )

    def test_replay_copies_widened_fused_input_into_widened_slot(self):
        # A wrapper that fuses several copies of the hidden state declares the
        # wider stack-input width; the slot must be sized from that declaration
        # so the per-replay copy matches.
        probe = _make_runner(_build_registry())
        probe.model_runner.model.input_embeds_hidden_size = HIDDEN_ALT
        self.assertEqual(probe._input_embeds_hidden_size(), HIDDEN_ALT)

        registry = _build_registry(
            is_multimodal=False,
            register_input_embeds=True,
            hidden_size=probe._input_embeds_hidden_size(),
        )
        runner = _make_runner(registry)
        widened = _detached_fusion(width=HIDDEN_ALT)
        _run_body(runner, fusion=widened)
        slot = registry.get_slot("input_embeds").slice_for(1, NUM_TOKENS)
        self.assertTrue(torch.equal(slot, widened))

    def test_replay_without_fusion_input_uses_embed_fallback(self):
        # No live stack input: the slot is fed from
        # get_input_embeddings()(input_ids), called exactly once.
        registry = _build_registry(is_multimodal=True)
        runner = _make_runner(registry)
        forward_batch, _, output = _run_body(runner)
        embed = runner.model_runner.model._embed
        self.assertEqual(len(embed.calls), 1)
        self.assertTrue(torch.equal(embed.calls[0], forward_batch.input_ids))
        slot = registry.get_slot("input_embeds").slice_for(1, NUM_TOKENS)
        expected = (
            _embed_formula(forward_batch.input_ids).repeat(1, HIDDEN).to(EMBED_DTYPE)
        )
        self.assertTrue(torch.equal(slot[:NUM_TOKENS], expected))
        self.assertEqual(len(runner.backend.replay_calls), 1)
        self.assertEqual(tuple(output.shape), (NUM_TOKENS, HIDDEN))

    def test_replay_without_fusion_input_falls_back_to_embed_input_ids(self):
        # Some wrappers expose embed_input_ids() instead of
        # get_input_embeddings(); the fallback must use it.
        registry = _build_registry(is_multimodal=True)
        runner = _make_runner(registry)
        embed = _StubEmbedding()
        runner.model_runner.model = _StubOuterModelWithEmbedInputIds(
            runner.layer_model, embed
        )
        forward_batch, _, _ = _run_body(runner)
        self.assertEqual(len(embed.calls), 1)
        slot = registry.get_slot("input_embeds").slice_for(1, NUM_TOKENS)
        expected = (
            _embed_formula(forward_batch.input_ids).repeat(1, HIDDEN).to(EMBED_DTYPE)
        )
        self.assertTrue(torch.equal(slot[:NUM_TOKENS], expected))

    def test_replay_embed_fallback_fails_closed_without_embed_accessor(self):
        # No live input and no embedding accessor: fail closed.
        registry = _build_registry(is_multimodal=True)
        runner = _make_runner(registry)
        runner.model_runner.model = _StubOuterModelNoEmbedAccessor(runner.layer_model)
        with self.assertRaisesRegex(RuntimeError, "get_input_embeddings"):
            _run_body(runner)


class TestNoDroppedInputEmbedsGuardrail(CustomTestCase):
    """C6/C7: a live stack input without a slot must fail loudly."""

    def test_replay_fusion_input_without_slot_raises_before_replay(self):
        # Via the full replay dispatch: the guard raises before the graph
        # replays (replay_calls stays empty).
        registry = _build_registry(is_multimodal=False, register_input_embeds=False)
        self.assertFalse(registry.has_slot("input_embeds"))
        runner = _make_runner(registry)
        with self.assertRaisesRegex(RuntimeError, r"(?s)input_embeds.*slot"):
            _run_body(runner, fusion=_detached_fusion())
        self.assertEqual(runner.backend.replay_calls, [])

    def test_guardrail_message_names_slot_and_disabled_escape_hatch(self):
        registry = _build_registry(is_multimodal=False, register_input_embeds=False)
        runner = _make_runner(registry)
        args = [None, None, None, _detached_fusion()]
        with self.assertRaisesRegex(RuntimeError, r"(?s)input_embeds.*slot") as ctx:
            runner._assert_no_dropped_input_embeds(args, {})
        self.assertIn("--cuda-graph-backend-prefill=disabled", str(ctx.exception))

    def test_guardrail_is_noop_without_live_input(self):
        registry = _build_registry(is_multimodal=False, register_input_embeds=False)
        runner = _make_runner(registry)
        self.assertIsNone(
            runner._assert_no_dropped_input_embeds([None, None, None], {})
        )

    def test_guardrail_accepts_precomputed_token_embeddings(self):
        # A wrapper that only pre-computes embed_tokens(input_ids) (a
        # language-model-only multimodal layout) hands the body exactly what
        # its captured embedding branch computes: the replay must proceed.
        registry = _build_registry(is_multimodal=False, register_input_embeds=False)
        runner = _make_runner(registry)
        input_ids = _detached_input_ids()
        token_embeds = runner.model_runner.model.get_input_embeddings()(input_ids)
        _, shape_key, _ = _run_body(runner, fusion=token_embeds)
        self.assertEqual(runner.backend.replay_calls, [shape_key])

    def test_replay_skips_slot_fill_on_non_first_pp_rank(self):
        # A non-first PP rank neither fills the slot nor trips the guardrail;
        # the replay still runs. The replay-time snapshot proves the slot was
        # still empty when the graph replayed.
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
