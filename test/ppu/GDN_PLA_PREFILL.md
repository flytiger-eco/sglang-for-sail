# GDN prefill with PLA

Use the existing Triton linear-attention backend as the dispatch entry:

```bash
SGLANG_SAIL_PLA_CUDA=1 python -m sglang.launch_server \
  --model-path <original-model-path> \
  --linear-attn-backend triton --linear-attn-prefill-backend triton \
  <original-model-options>
```

Set `SGLANG_SAIL_PLA_CUDA=0` to use the original Triton implementation.
This existing flag also controls the existing PLA decode paths. PPU server
initialization already enables it when unset; an explicit `0` is respected.
An explicitly selected FlashInfer/CuTe prefill backend keeps its own dispatch.

The prefill adapter lazily imports `pla.prefill.flashqla.chunk_gated_delta_rule_fwd`.
Install a PLA build containing this prefill extension on the target PPU.
The adapter preserves model and SSM state precision. It passes activated log-gates
and beta, enables Q/K L2 normalization, and uses the indexed `[slot,H,V,K]` pool
with INT32 slot indices. CPU sequence lengths supply trusted cumulative offsets
without a device-to-host planning read. PLA's supported head shapes and hardware
constraints still apply; errors from PLA are propagated rather than hidden.

PLA updates the final pool in place. Its returned `ht` is not SGLang's intermediate
chunk state `h`: the adapter returns `(output, None, None)`. If a batch needs
intermediate states for prefix-cache tracking, the complete batch uses Triton
and emits a warning once. Aligned-only tracking can read the updated final pool
and uses PLA. Noncontiguous pools are gathered and scattered without changing
their layout. Speculative target verification retains its existing dispatch.

Successful PLA prefill emits `USE PPU SAIL CUDA PLA GDN prefill kernel` once.

## Validation

```bash
PYTHONPATH=python python test/ppu/test_gdn_pla_prefill.py -v
```

CPU contract checks cover FP32/BF16 indexed state, noncontiguous inputs and pools,
nonsequential slots, metadata conversion, output extraction and untouched slots.
On a configured PPU host the test additionally compares PLA and Triton outputs
and final states for ragged sequences and nonzero incoming states, and verifies
the intermediate-state fallback. Device comparisons use `atol=2e-2, rtol=1e-2`.

Before declaring the adaptation validated, run the original model configuration
with both flag values on the target machine, verify generated results and the
prefill selection log, and exercise prefix reuse and chunked prefill. Record
actual hardware, SDK/driver/PCCL/PLA versions and request conditions. The local
CPU checks do not establish kernel accuracy or end-to-end correctness.
