# PPU QSA backends: compatibility and validation

The optional QSA ports provide two independent switches:

- `SGLANG_SAIL_QSA_INDEXER_FP8_AVG_LOGITS=1`: use the PPU DeepGEMM FP8 AVG
  scorer for supported **prefill** queries. Decode scoring retains the existing
  TileLang/torch path.
- `SGLANG_SAIL_QSA_FA3_SPARSE_ATTN=1`: request the separate QSA-enabled FA3
  attention interface when its schema, inputs and cache format are supported.

Both switches default to false. The measurements below enable only the first
switch. A generic `attention_backend=fa3` setting is separate from the QSA FA3
switch.

## Current DeepGEMM API

The adapter targets the compressed-output `fp8_mqa_avg_logits` API exported by
the PPU DeepGEMM version listed below:

- Omit the removed `clean_logits` argument and pass positive `max_seqlen_k`.
- Keep FP8 Q/K and their FP32 dequantization scales, with explicit FP32 logits.
- Interpret each output row as relative columns `[0, ke - ks)`, regardless of
  that query's start in the packed K tensor. Invalid columns are unspecified.
- Use bounded, strided top-k512 and retain request-local block indices for
  expansion. Do not add the packed K offset again.
- Derive capacity from the host-known batch context bound in compressed K
  units, and account for the native FP32 row-stride alignment in the workspace
  budget. No additional GPU reduction or scalar readback is needed.

The native path is restricted to PPU SM89, four query heads, logical D64/D128,
and the validated top-k512 consumer. Unsupported configurations or an
unrecognized/legacy library schema retain the original scorer. The public
`qsa_mqa_prefill` reference/baseline helper keeps its global-column, masked
output contract.

## Validation environment

Recorded on 2026-11-11 using four PPU-890P devices, 147456 MiB per device, with
ICN8 links. The tested environment was Python 3.12.3, torch 2.13.0, driver
2.4.0-d56262, SAIL 2.3.0-9ab4d9 with the CUDA-compatible SDK, and PCCL 2.2.1.
The provided October 30 image was used with
`deep-gemm==1.1.0.dev33+v1.0.0.ppu2.2.0.gb9c9bf5`. The wheel's version suffix
does not replace the observed SDK version.

The model was a Qwen4-Exp-compatible hybrid QSA checkpoint with FP8 weights
per channel and FP8 activations per token, TP4. The checkpoint is not included
here. No MXFP4 conversion or model-precision change was made.

The hardware-tested source snapshot has digest
`48a572a22d59d9d6326a4dd3849e2ab2d067e3b9684b3735d0cb56d369abcbd8`.
The validation note and the legacy reproducer's documentation were added after
these runs; runtime implementation bytes have not changed.

## Targeted tests and unresolved numerical checks

Run inside a verified PPU task container after synchronizing the source:

```bash
python3 -m pytest test/ppu/unit/test_qsa_ppu_compressed_logits.py -v -s
python3 -m pytest test/ppu/unit/test_qsa_ppu_backends.py -v -s
python3 -m pytest test/registered/kernels/ops/attention/qsa/test_qsa.py -v
```

| Suite | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Compressed AVG API / layout | 9 | 2 | 0 |
| PPU backend regression | 20 | 1 | 0 |
| Related QSA regression | 38 | 0 | 1 |
| Total | 67 | 3 | 1 |

The skip is SM121-specific. These results are **not an all-green validation**:

- `(Q=3, K=7, D=128)` still exceeds the existing BF16-reference normalized
  squared-error threshold: `0.0024725438185 > 1e-3`.
- `(17,1100,128)` and `(7,777,64)` fail the added quantized-input FP64
  pointwise comparison at unchanged `rtol=atol=1e-4`. Maximum absolute errors
  are `0.00190287552` and `0.00091542288`; normalized squared errors are
  `4.58565696e-9` and `1.43950905e-9`.
- All three native cases complete bitwise comparison of wrapper output versus
  direct native calls on identical quantized inputs. Bounded top-k, nonzero
  packed starts, empty/ragged windows and deliberately poisoned NaN padding
  checks also complete successfully. The pointwise assertions execute afterward
  and still fail; they are neither skipped nor reclassified as passed.
- Tiny native FA3 decode/extend tests pass in this image. Model-level FA3 QSA
  performance and accuracy were not tested.

`test_deep_gemm_fp8_mqa_avg_logits_d32.py` is a separate legacy-ABI crash
reproducer. Its older-library D32 failures are historical evidence, not a claim
that the current API accepts D32 or still reproduces that device fault. It is
not part of the three-suite result above.

## Serving and performance reproduction

Set `MODEL` to the checkpoint, `SERVED_MODEL` to its served name, and
`SHAREGPT_DATA` to a local ShareGPT V3 file. Both groups use identical arguments
and server/client seed 1; change only the indexer switch between 0 and 1.
The data file used SHA256
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`.

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_SAIL_QSA_FA3_SPARSE_ATTN=0
export SGLANG_SAIL_QSA_INDEXER_FP8_AVG_LOGITS=1  # 0 for the baseline

python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED_MODEL" \
  --tp 4 --max-running-requests 32 --mem-fraction-static 0.85 \
  --chunked-prefill-size 8192 \
  --linear-attn-prefill-backend triton \
  --linear-attn-decode-backend triton \
  --linear-attn-verify-backend triton \
  --mamba-ssm-dtype bfloat16 --reasoning-parser auto \
  --random-seed 1 --host 0.0.0.0 --port 18990
```

After health and minimal-inference checks, run three rounds per group, each
with one equal-length warmup followed by cache flushing and ten formal requests:

```bash
python3 -m sglang.bench_serving \
  --backend sglang --base-url http://127.0.0.1:18990 \
  --model "$MODEL" --tokenizer "$MODEL" --served-model-name "$SERVED_MODEL" \
  --dataset-name random --dataset-path "$SHAREGPT_DATA" \
  --num-prompts 10 --random-input-len 100000 --random-output-len 1 \
  --random-range-ratio 1 --max-concurrency 1 --request-rate inf --seed 1 \
  --tokenize-prompt --warmup-requests 1 --flush-cache --cache-report \
  --output-details --output-file "$ROUND_OUTPUT" \
  --extra-request-body '{"sampling_params":{"temperature":0,"max_new_tokens":1,"ignore_eos":true,"skip_special_tokens":false}}'
```

| Metric | Original TileLang scorer | DeepGEMM scorer | Change |
| --- | ---: | ---: | ---: |
| Mean TTFT | 7350.540 ms | 6965.107 ms | -5.2436% |
| Input throughput | 13600.546 tokens/s | 14352.893 tokens/s | +5.5317% |
| Formal requests | 30/30 | 30/30 | No errors |
| Actual input/output per request | 100000 / 1 | 100000 / 1 | Matched |
| Cache hits | 0 | 0 | Matched |

All six rounds completed on their first attempt. Approximately 20-second PPU
samples bracketed all client windows, with no observed foreign GPU processes.
Cold loading, compilation and warmups are excluded. This is a sequential
comparison with no alternating AB/BA or statistical-significance claim.
It measures the entire current indexer integration, not an isolated
`clean_logits` toggle.

Automatic capacities differ slightly despite the same launch arguments:
KV tokens `3876544 -> 3876032`, Mamba slots `3098 -> 3097`. Both exceed this
C1/100K load; the difference is retained rather than claiming identical runtime
states. The raw `max_concurrent_requests=2` metric uses inclusive one-second
buckets and can count adjacent serial requests twice. The enforced CLI limit
and semaphore are 1, with measured average concurrency about 0.9997.
O1 supplies no inter-token latency/TPOT sample.

## GSM8K512

The enabled indexer was also evaluated using EvalScope 1.14.1 and local GSM8K
v2 main data: 512 samples, four few-shot examples, batch 64, streaming,
`max_tokens=8192`, temperature 0.6, top_p 0.95, n=1 and seed 1. The isolated
client environment did not replace the server's framework or native libraries.

With `PPU_DEV_EVAL_GSM8K` supplied by the validation runner and the corresponding
client environment selected, the evaluation command was:

```bash
"$CLIENT_PYTHON" "$PPU_DEV_EVAL_GSM8K" \
  --evalscope-bin "$EVALSCOPE_BIN" --pid "$SERVER_PID" \
  --model-id "$SERVED_MODEL" --dataset-path "$GSM8K_DATASET" \
  --batch-size 64 --limit 512 --max-tokens 8192 \
  --temperature 0.6 --top-p 0.95 --n 1 \
  --generation-config '{"seed":1}' \
  --dataset-args '{"gsm8k":{"filters":{},"few_shot_num":4}}' \
  --work-dir "$GSM8K_OUTPUT"
```

All 512 unique predictions and reviews are present: **500/512 correct,
97.65625%**, 12 incorrect, zero API errors/missing/duplicate samples. One sample
reached 8192 tokens (index 119, empty extracted answer) and remains an incorrect
sample in the denominator. No longer generation limit or sample exclusion was
used to improve the score.

There is no complete disabled-indexer GSM8K baseline, so no accuracy delta or
acceptance claim is made. Actual prompts were 692-809 tokens: at fourfold
compression they have fewer than 512 blocks. This does not validate sparse
selection quality for 100K contexts or resolve the numerical test failures.

The evaluation CLI completed and scoring files were saved before the resource
deadline. Retaining the server until that deadline caused the final container
source check to be interrupted; the original runner state remains
`source-diverged` with exit 124 and verification command 137. A subsequent
read-only check of the exact source bind mount matched all 10201 files and Git
HEAD. That recovery is distinguished from a successful original post-run
container check. The performance run's pre/post source checks completed.

Raw predictions, reviews, benchmark JSONL files, source manifests and status
records were retained and transferred with per-file hashes. They are not
included in the source patch; this note records the reproducible configuration,
observed results and unresolved checks without publishing internal locations.
