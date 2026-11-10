# Opt-in vision policy on v0.5.21

Adapted from `18d93fac3567116812d5849e5a5a2e275683ecad`.

The default visual tower remains unquantized. The new configuration fields live
in the `mm` namespace and are validated by the normal argument-resolution
pipeline, including the dummy-model path.

- `--vision-quantization fp8` quantizes unquantized visual checkpoint tensors
  online. Serialized quantized tensors and scale metadata are rejected.
- `--vision-model-path PATH` replaces only visual tensors; language-model
  tensors still come from `--model-path`. The separate checkpoint must supply
  all required visual weights and biases. Unknown, missing and duplicate visual
  tensors are rejected.
- `SGLANG_QUANTIZE_VIT=1` explicitly inherits a supported FP8 backbone policy.
  An unsupported inherited quantization policy is rejected.

These options require a supported native Qwen3VL-derived visual tower, serial
DefaultModelLoader loading and no weight cache. Separate visual weights require
pipeline parallel size 1. ModelOpt/INC and MXFP8 loading are not supported.
Literal visual exclusion paths are remapped without mutating the language
model's quantization policy.

On PPU, channelwise FP8 linears use DeepGEMM with normalized scale layouts.
Biased visual linears and arbitrary leading input dimensions preserve the
output shape and output dtype.

## Validation

Policy and real CPU-loader integration tests:
```bash
python3 -m pytest -q \
  test/registered/unit/multimodal/test_aim_vision_policy.py \
  test/registered/unit/multimodal/test_aim_vision_policy_integration.py
```

Real PPU numerical tests compare quantized matrix multiplication against an FP32
reference, covering two-dimensional and three-dimensional inputs, optional bias,
and vector/column channel scales:
```bash
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 python3 -m pytest -q \
  test/manual/test_aim_vision_fp8_ppu.py
```

End-to-end feature activation, performance and multimodal comparisons are recorded
separately for each tested source revision and container image. Unit-test success
alone does not establish end-to-end correctness.

## Position-aware image pruning

Adapted from 6fb0d26d6e, including the required expanded-length handoff from
15bed020e9. Set --image-pruning-rate (default 0, valid range [0,1)) and
--image-pruning-strategy attention_score consistently on E/P/D. The supported
encoder path is eager TP1, without DP vision, using
SGLANG_VIT_ENABLE_CUDA_GRAPH=0 and zmq_to_tokenizer transport. Global multimodal
cache is rejected; the local embedding cache preserves retained indices.

Prefill uses --language-only and --disagg-decode-urls pointing to Decode.
Decode uses --language-only. Pruning enables the expanded-length handoff;
SGLANG_DISAGG_EXPANDED_INPUT_LEN=1 also enables it independently. Prefill sends
the final prompt length and MRoPE continuation delta in one versioned message
before Decode allocates KV. Early messages, duplicates, conflicts, and missing
metadata have bounded handling.

Attention importance selects rows after the visual merger. Selected indices
remain in original raster order and select all DeepStack columns together.
The original grid supplies MRoPE coordinates; image placeholders, embedding row
counts, and request lengths use the retained indices.

Run test/manual/epd_pruning with RUN_VISION_GPU_SMOKE=1 for the real two-layer
PPU model and dense attention-score numerical reference, in addition to CPU
cache/transport/metadata tests.


The pruning port uses the generation-aware expanded-input handoff from
`daee1ae624`. Set `SGLANG_DISAGG_EXPANDED_INPUT_LEN=1` on Prefill and Decode,
configure `--disagg-decode-urls`, and send a fresh `X-Request-ID` per paired
request when using an older gateway. Unsupported configurations fail at startup.
Image pruning still preserves retained spatial indices through the encoder cache
and transport, including caller-supplied hashes and DeepStack channels.
