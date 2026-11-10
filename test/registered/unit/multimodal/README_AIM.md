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
## Bounded Encoder CPU preparation

Adapted from 92048b9e8b to the split encoder/server.py, runtime.py and
preprocessor.py architecture. Enable SGLANG_AIM_ENCODER_PIPELINE=1 for CPU
image processing without model-specific preprocessing hooks. Current and next
preparation are bounded to two tickets; SGLANG_AIM_ENCODER_MAX_PENDING (default
32) bounds queued requests. GPU dispatch and TP status/metadata collectives
remain on the main event-loop thread and retain submission order.

Includes the source commit's necessary AIM image/RedKV loading prerequisites.
SGLANG_ENABLE_BATCH_JPEG_DECODE=1 with SGLANG_ENABLE_JPEG_GPU_DECODE=0 enables
bounded torchvision CPU JPEG decoding. SGLANG_JPEG_CPU_DECODE_WORKERS defaults
to 1. Dictionaries carrying content hashes/metadata keep the existing verified
loading contract. Remote HTTP continues through download_remote_media security
and timeout policy. RedKV MGET and 1px fallback are separately opt-in.

Twenty-nine CPU/runtime image-loader and scheduler tests cover ordering,
backpressure, cancellation, current/next overlap under blocking forward,
main-thread TP preparation consensus, shutdown, bounded I/O, cache hit/miss/error
and image fallback. Real EPD tests use PNG and JPEG ordered/repeated multi-image
requests, plus separate 64K and 16K performance measurements.

## Image and Encoder stage metrics

Adapted from bd12da416a. Existing TokenizerMetricsCollector and
EncoderMetricsCollector own AIM mm_items_total / mm_stage_seconds families.
No Prometheus objects are created on import or per request. Bounded stage/result
labels distinguish cache errors, misses, cancellation and explicit 1px fallback,
without using image URLs as labels. Stages are nested and must not be summed as
TTFT. JPEG worker time is distinct from batch decode wall time.

The split EncoderPreprocessor records through the same process recorder as
ordinary image loading, HTTP/cache fallback, tokenizer text tokenization and
CPU preparation. Target legacy prompt semantics are preserved; the source
parent's optional tokenizer-prefetch API is not introduced solely for metrics.
The bounded image fast path requires batch JPEG or MGET to be enabled.

Validation: 76 CPU/runtime tests passed; one optional real HEIF round-trip was
skipped because the image lacks pillow-heif. Real PPU PNG/JPEG workloads and
serving /metrics snapshots are tested per frozen deployment revision. External
EDS/RedKV live service is not assumed; cache outcomes use controlled fixtures.

## Fatal CUDA state at Encoder boundaries

Adapted from 4921713a2d to the split HTTP/gRPC Encoder runtime. Fatal native
CUDA diagnostics mark the process unhealthy before later CPU preparation, TP
dispatch and busy health shortcuts. Input errors and recoverable OOM remain
request failures. Exception causes are inspected without trusting URL contents.
The marker resets after fork. Optional process exit is enabled with
SGLANG_ENCODER_EXIT_ON_CUDA_FATAL=1 (or SGLANG_EXIT_ON_CUDA_FATAL=1); delay is
configured by the corresponding CUDA_FATAL_EXIT_DELAY_S variable. Defaults
leave exit disabled.

Health and admission return unavailable after a fatal error. Pending CPU
preparation is closed before new dispatch; no rank-local check is inserted
after a TP broadcast. HTTP and gRPC share the same process state.

Validation: combined image/metrics/health suite passes 100 tests with one
optional HEIF skip. Coverage includes cause chains, nonfatal URL/OOM, fork
reset, single scheduled exit, dispatch-lock races and both protocol guards.
Real serving validation records ordinary performance/correctness separately
from controlled native-error injection; injected faults do not poison hardware.
