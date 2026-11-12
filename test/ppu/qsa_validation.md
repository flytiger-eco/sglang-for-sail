# PPU QSA backends: compatibility and validation

## v0.5.23 port status

This branch ports the three v0.5.21 implementation commits onto
`ywj/v0.5.23` at `c6d7a44accc666e634140be45899c942ed0b4d88`.
It uses the target's `is_sail()` platform helper, preserves its relative
TileLang scoring and CP paths, and reuses `max_row_keys` for the native output
bound. Existing block-index expansion remains ahead of FA3 decode dispatch;
already-FP8 scorer inputs retain the original relative fallback.

The numerical, serving, GSM8K and LongBench results below are **historical
v0.5.21 measurements**, not measurements of this v0.5.23 port. Only syntax,
static integration and repository formatting checks have been performed on the
port. Runtime correctness and performance require separate validation with the
target SDK and native libraries; historical numerical failures remain recorded.


The optional QSA ports provide two independent switches:

- `SGLANG_SAIL_QSA_INDEXER_FP8_AVG_LOGITS=1`: use the PPU DeepGEMM FP8 AVG
  scorer for supported **prefill** queries. Decode scoring retains the existing
  TileLang/torch path.
- `SGLANG_SAIL_QSA_FA3_SPARSE_ATTN=1`: request the separate QSA-enabled FA3
  attention interface when its schema, inputs and cache format are supported.

Both switches default to false. The November 11 measurements enable only the
first switch; the November 12 validation below enables both. A generic
`attention_backend=fa3` setting is separate from the QSA FA3 switch.

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

## Validation environment (November 11 indexer-only runs)

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
those runs. The later FA3 integration fixes and their separately measured
source snapshot are documented below.

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
  performance and accuracy were not tested in those November 11 runs; see the
  separate November 12 results below.

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


## FA3 graph replay and model validation (November 12)

CUDA-graph metadata deliberately retains a one-column dummy token-slot table.
Using it for FA3 decode maps valid logical indices to physical slot zero, so
prefill can match a reference while generation repeats tokens. Graph dispatch
now gathers physical slots from the persistent request pool using replay-updated
request IDs, with the original row-length mask. Eager dispatch keeps its real
per-sequence/per-query table.

Two associated integration fixes resolve wrappers lazily to avoid duplicate
custom-op registration, and keep cumulative prefill query lengths in int32
because integer `cumsum` otherwise promotes to int64.

Twelve targeted cases passed with zero failures/skips: wrapper resolution,
request-pool mapping, H4/H6 native decode and production extend, padded width
2048, and actual CUDA-graph capture/replay at batches 1 and 3. Replay tests update
request IDs, lengths and selections twice after capture. Existing tolerances
remain `rtol=atol=0.02`. The command was:

```bash
python3 -m pytest test/ppu/unit/test_qsa_ppu_backends.py \
  -k 'test_fa3_sparse_matches_reference or test_fa3_resolver_does_not_import_second_valid_wrapper or test_fa3_sparse_padded_production_width or test_fa3_graph' \
  -v -s
```

The tested source snapshot is
`ad7c8ae0078660e80609440dc750d6ca8a868226a21fd111d745aeb8c724d6cb`.
Four PPU-890P devices used the original FP8 checkpoint precision and TP4, Python
3.12.3, torch 2.13.0, driver 2.4.0-d56262, **SAIL 2.2.0-663c22**, PCCL 2.2.1,
DeepGEMM dev33 as above, and QSA-enabled `flash-attn-3==3.0.0b1` with the
`flash-attn==2.8.2` Python interface. This is a different SDK environment from
the historical SAIL 2.3.0 measurements. Native FA3 execution was observed on all
four ranks for a real user prefill `[64,6,256]`. `qsa_allow_aiu=False`: arbitrary
four-token selections do not certify sixteen-token contiguous runs.

### LongBench v2: all 503 samples, maximum output 32K

EvalScope 1.14.1 evaluated both QSA switches enabled with prefix caching disabled.
The dataset was `zai-org/LongBench-v2`, revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`. Original `data.json` SHA256:
`15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`.
The native adapter used train split, zero-shot step-by-step prompting, all three
original length subsets and rule-based multiple-choice accuracy.

| Subset | Correct / total | Accuracy |
| --- | ---: | ---: |
| short | 129 / 180 | 71.6667% |
| medium | 141 / 215 | 65.5814% |
| long | 68 / 108 | 62.9630% |
| Overall | **338 / 503** | **67.1968%** |

Generation used concurrency 16, temperature 0, top_p 1, n=1, seed 1, streaming
and **max_tokens=32768**, including reasoning and the final answer. Requests
were checked against the prepared prompt hash and generation settings. All
503 predictions and reviews are unique and present: zero API failures, missing
samples or duplicates. There are 475 normal stops and 28 output-limit endings;
all 28 remain scored in the 503-sample denominator.

The model context limit is 262144 tokens. Reserving 32768 for output and a
256-token margin gives a 229120-token chat input budget. The context field of
113 samples was split into token-balanced head/tail segments; question, choices,
answer and original length labels were unchanged. Maximum prepared input was
229058 tokens. All 503 actual server input counts matched the prepared counts.
Processed JSONL SHA256:
`dc3d3b29cfcb0ccf1afd03b3314af8f80be32afe2c89be9c05518b214ce71258`.

For reproduction, start the server with the arguments above, both QSA switches
set to 1 and `--disable-radix-cache`. Use a prepared local dataset **directory**
containing `default_train.jsonl` with the stated head/tail policy. The EvalScope
`TaskConfig` settings were:

```python
eval_type = "openai_api"
api_url = "http://127.0.0.1:18990/v1"
datasets = ["longbench_v2"]
dataset_args = {"longbench_v2": {
    "local_path": PREPARED_DATA_DIRECTORY,
    "subset_list": ["short", "medium", "long"],
    "few_shot_num": 0,
}}
eval_batch_size = 16
generation_config = {
    "max_tokens": 32768, "temperature": 0.0, "top_p": 1.0,
    "n": 1, "seed": 1, "stream": True, "timeout": 1800, "retries": 1,
}
model_args = {"max_retries": 0}
seed = 1
repeats = 1
limit = None
```

Independent audits checked original targets, complete IDs, request limits,
reviews and report rounding. A post-evaluation audit inside the model container
matched all 10202 source files, Git metadata, both native-library hashes and the
SDK. Full client evidence was transferred and checked across 8153 files/entries,
then independently rechecked locally. Original SSH transport errors are retained
separately from successful remote completion and the fresh managed audit.

This is an absolute score under the stated context/output limits and truncation
policy. No matching disabled-FA3 LongBench baseline or accuracy threshold was
supplied. Historical numerical failures above remain unresolved and are not
reclassified by these results.

### Incremental FA3 serving performance

With the same new FA3 installation in both groups, identical launch arguments
and prefix caching disabled, C1/I100000/O1/N10, three rounds per group, give:

| Metric | DeepGEMM indexer only | DeepGEMM indexer + FA3 QSA | Change |
| --- | ---: | ---: | ---: |
| Mean TTFT | 6998.400 ms | 6666.216 ms | -4.7466% |
| Input throughput | 14284.853 tokens/s | 14996.166 tokens/s | +4.9795% |
| Formal successful requests | 30/30 | 30/30 | No API errors |

Actual lengths were 100000/1 with zero cache hits. Both groups derived KV capacity
7327360 tokens and Mamba capacity 32 slots. The disabled-FA3 baseline predates the
graph-mapping correction, which is not executed in its CUDA path. Its source and
post-measurement verification were retained separately; the candidate's
post-measurement container check also passed. No AB/BA or significance claim is
made. Gains from different SDK environments must not be added together.
