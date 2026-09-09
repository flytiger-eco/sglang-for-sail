# PPU public-benchmark accuracy nightly

This directory holds the registered PPU accuracy nightly tests together with the
versioned execution contracts they read. The evaluator lives in
`python/sglang/test/kits/accuracy_eval_kit.py` and the on-machine driver in
`python/sglang/test/kits/accuracy_suite_kit.py`.

```
accuracy_eval/
├── configs/<model family>/<config>.json   reviewed execution contracts
└── test_ppu_*.py                           registered entry points
```

It is a sibling of `answer_eval/` and `perf_eval/` rather than part of either.
All three serve a checkpoint on a PPU board and then ask it questions, but what
they conclude, and on whose authority, is different in each case, and they read
different environment variables on purpose so a change made for one cannot move
another's config.

| Line | Question | Scored by | Red when |
| --- | --- | --- | --- |
| `answer_eval/` | does it answer our own corpus correctly? | our judge, in this repo | an answer is wrong |
| `perf_eval/` | how fast is it? | `bench_serving` | nothing was measured |
| `accuracy_eval/` | how does it score on a published benchmark? | EvalScope | a score fell outside its baseline band |

The third row is why this line exists at all. The Answer corpus is ours, so a
score on it is comparable only to itself; GSM8K, C-Eval and IFEval are public
splits scored by a tool nobody here maintains, so a number from this line is
comparable to a published one. That is also its constraint: the report reader has
to follow EvalScope's schema rather than a shape of our choosing.

## What this line measures, and when it judges

One evaluation per config: the config names one dataset, the suite serves the
checkpoint, runs `evalscope eval` against the served OpenAI-compatible API, and
reads the primary metric out of the report EvalScope writes.

`accuracy_eval_kit.REASON_CODES` enumerates every way a run can be red, and they
divide in two. Five are *not measured* — the server never came up
(`server_start_failed`), EvalScope failed or timed out (`evalscope_failed`), it
wrote no report (`report_missing`), the report could not be read
(`report_unreadable`), or it named no primary metric
(`primary_metric_missing`). Four are *measured and judged* — the metric that
concluded was not the one the config asked for (`primary_metric_mismatch`), the
evaluation did not cover the whole split (`incomplete_samples`), or the score fell
below or above its baseline band (`below_baseline`, `above_baseline`).

Completeness is checked **before** the score is compared to anything, because a
score over 40 of 1319 prompts is not a smaller version of the measurement — it is
a different measurement that can pass a band it has no right to.

`above_baseline` exists because the source cases state a two-sided band —
`threshold: ["0.98", "2.00"]`, read here as a ratio band around a recorded
baseline. A score twice its baseline is not a triumph; it is evidence that
something changed that nobody meant to change, usually in the harness.

### No baseline yet, so nothing is judged yet

Every config here ships with `baseline: null`, and the schema then refuses a ratio
band as well: a band without a baseline judges nothing, and one that quietly
defaulted would be a rule nobody reviewed. Such a run reports `measured`, carries
a `no_baseline` warning, and says so on the run page in as many words. Filling in
a baseline is a reviewed change, made from a green run of this line on this
hardware — not from the vendor's published number for the unquantised model.

## The suites

The source set is `btv1.5/llm_infer_sglang_evalscope/P0_daily`: 39 cases, which
are 13 (model, quantisation) pairs × 3 datasets, one case per pair per dataset.
Seven of those 13 pairs are checkpoints this board has already served on the
Answer or perf lines, giving **21 portable cases**; the other six pairs are
models this repository has never stood up:

| Portable (checkpoint already served here) | Not yet portable |
| --- | --- |
| GLM-5.2 FP8-Channelwise, GLM-5.2 MXFP4-FP8 | DeepSeek-V3.2-FP8 |
| Kimi-K2.6 MXFP4-FP8 | DeepSeek-V4-Flash MXFP4-FP8 |
| MiniMax-M2.7 FP8-Channelwise, MiniMax-M2.7 MXFP4-FP8 | DeepSeek-V4-Pro MXFP4-FP8 |
| Qwen3.5-397B-A17B FP8-Channelwise, Qwen3.5-397B-A17B MXFP4-FP8 | GLM-5.1 FP8-Channelwise, GLM-5.1 MXFP4-FP8 |
| | Qwen3.7 MXFP4-FP8 |

Ported so far is one of the 21, deliberately:

| Suite | Test file | Model | Dataset | Devices |
| --- | --- | --- | --- | --- |
| `nightly-accuracy-8-glm52-ppu` | `test_ppu_glm52_accuracy.py` | GLM-5.2 FP8-Channelwise | GSM8K | 8 |

Two configs sit behind that one suite and differ only in `limit`:
`fp8-channelwise-144g-gsm8k.json` evaluates the whole 1319-prompt split, and
`fp8-channelwise-144g-gsm8k-smoke.json` evaluates 20. The workflow selects which
by naming it in `SGLANG_PPU_ACCURACY_TEST_CONFIG`; a suite is a file, because
`register_ppu_ci` registers per file.

One entry rather than twenty-one because two facts about this line could only be
established by running it, and each of them would have been wrong twenty-one
times over: whether the pip index carries EvalScope, and where these datasets
live on the NAS. Both are now measured rather than assumed, in a 1-PPU probe
(runs `34374868204` and `34376132868`) that cost minutes instead of a night:

* the index carries `evalscope` up to and including the pinned `1.11.1`, so
  `setup_evalscope.sh` needs no wheelhouse on this cluster;
* `/nas_aisw/datasets` held only `checkpoints`, `dsmManager`, `hf_cache` and
  `packages` — no dataset area existed at all — and it is writable from a pod,
  so `evalscope/<dataset_id>` was created there and GSM8K staged into it. That
  is the path the two configs name.

Neither is a fact about a model, so paying for it once was enough. The remaining
twenty cases follow once one has run green.

## The datasets

Three datasets appear across the source set, and the contracts are in
`accuracy_eval_kit.DATASET_CONTRACTS`:

| Dataset | ModelScope id | Split | Shots | Primary metric | Samples |
| --- | --- | --- | --- | --- | --- |
| `gsm8k` | `AI-ModelScope/gsm8k` | test | 4 | `accuracy` | 1319 |
| `ceval` | `evalscope/ceval` | val | 5 | `accuracy` | 1346 |
| `ifeval` | `opencompass/ifeval` | train | 0 | `prompt_level_strict` | 541 |

Every column is the adapter's own `BenchmarkMeta` in 1.11.1, not a reading of the
documentation. The primary metric column is the one worth knowing about: GSM8K and
C-Eval declare theirs as `acc`, which is a *legacy* spelling — `BenchmarkMeta`
normalizes it through `migrate_legacy_identity`, and the identity that reaches the
report is `accuracy`. A config naming `acc` would be refused by a run that
measured perfectly well. IFEval declares `prompt_level_strict` explicitly and it
is not an alias, so it arrives unchanged.

They are not repository assets, unlike the Answer corpus: they are public splits
far too large to check in. The config names an absolute directory on shared
storage, and EvalScope is pointed at it through `--dataset-args` so it reads that
directory instead of resolving a hub id.

Staged rather than fetched per run, although these pods *can* reach ModelScope —
the probe above downloaded GSM8K from one, while `huggingface.co` answered
nothing at all. Two reasons: a nightly whose input arrives over egress can be red
for a reason that has nothing to do with the model, and a score is comparable
across nights only if the data behind it did not change between them. A missing
dataset is therefore a failure the board script reports before it loads a
checkpoint, with the `modelscope download` command that fixes it, and never a
download it starts on its own.

What that directory has to be is not a free choice. EvalScope treats a
`dataset_id` that exists on disk as local and then hands it to
`datasets.load_dataset(path=<dir>, name=<subset>, split=<split>)`, so the
directory must be the **snapshot root** of the dataset repository — the level that
contains the per-subset directories, such as `main/` for GSM8K — and not one of
those subdirectories and not a single parquet file. That is exactly what
`modelscope download --local_dir` produces, which is why the message names that
command and not a file copy. What GSM8K staged as, for reference:
`main/{train,test}-00000-of-00001.parquet`, `socratic/` beside it, and
`eval.yaml` at the root.

The staged location is the one path in a reviewed config that this repository
cannot check, so `SGLANG_PPU_ACCURACY_DATASET_DIR` overrides it and the workflow
takes a `dataset_dir` input that sets it. Every report records the directory
actually read and whether it was overridden, so a score cannot be traced back to
data other than the data it was measured on.

`SGLANG_USE_MODELSCOPE`, which every source case sets, is deliberately **not** a
supported variable: the configs name absolute local checkpoint paths, and since
these pods do reach ModelScope, what honouring it can buy is a hub fetch of
weights that are already on the NAS.

## EvalScope lives in an environment of its own

`scripts/ci/ppu/setup_evalscope.sh` installs `evalscope[ifeval]==1.11.1` into
`/opt/evalscope-venv` rather than into the image's environment, because the two
dependency closures disagree where it is not negotiable: EvalScope requires
`modelscope[datasets]>=1.34`, which resolves `datasets>=4.0.0`, while the v2.1.1
image ships `datasets 3.1.0` pinned to `dill<0.3.9` — and
`ppu_install_dependency.sh` installs `dill` deliberately inside that band because
Python 3.12 unpickles `_abc._abc_data` only from 0.3.8 up. Resolving EvalScope
into the serving environment would move `datasets` and `dill` under the very
server this suite measures.

The isolation costs nothing: with `--eval-type openai_api` EvalScope is a pure
HTTP client, its core requirements carry no torch, and the two environments share
only a socket. The board script exports `SGLANG_PPU_EVALSCOPE_BIN` so the suite
scores with that EvalScope rather than with whatever else the image might carry.

The version is pinned above 1.11.0 rather than floated because report **schema
v2** — the structured metric list, `primary_metric_identity` and
`execution_summary` — first appears there, and three things about that schema are
easy to misread. Report-level `score` is a plain property and is never written to
the file; `Metric.name` is likewise a property, so a metric in the file carries
only a nullable `legacy_name` and a nested `identity` object; and the metric list
holds diagnostics (`output_tokens` and such) beside the quality metric. A reader
that took the first metric in the list, or looked for a name field, would report a
plausible wrong number. `test_ppu_accuracy_eval_unit.py` reproduces the real file
shape, diagnostics included, and pins the reading. Its fixture is not a
description of the schema from memory: it was compared field by field against
`Report.model_dump(mode="json")` from 1.11.1 itself, at report, metric, category,
subset and identity level, and invents no key the library does not write.

## Departures from the source cases

The `evaluation` block of each config is the source case faithfully: dataset,
`eval_batch_size`, `generation_config` and the threshold band all come across. The
`server` block does not, and this is the one substantive departure on the line.

The source case serves GLM-5.2 with `prefill_attention_backend: fa3`,
`decode_attention_backend: flashmla` and EAGLE speculative decoding at 2 steps.
The config here serves it with the `dsa` sparse-attention configuration that the
Answer and perf lines have actually started on this board, and no speculative
decoding. The reason is what the two configurations would tell you: a run on
parameters this board has never started measures the startup path, not the model,
and a first accuracy number should be a number about the model. The source
serving configuration is worth porting once it has been stood up somewhere, and
the schema already models the parameters it needs — `test_ppu_glm52_accuracy.py`
records the departure at its point of use.

Two smaller ones:

- **`eval_batch_size` 128 → 40.** In `openai_api` mode this is the client's
  concurrency, and the server is configured for 40 concurrent requests. The extra
  88 would queue, adding latency and no throughput.
- **`limit: 0` is refused.** The source cases spell "the whole split" that way;
  EvalScope spells it as no `--limit` at all, so a config that copied the 0 across
  would evaluate nothing and report a perfect score on it. The schema refuses 0
  and asks for `null`.

## Workflow

| Workflow | Suite | Boards | Trigger |
| --- | --- | --- | --- |
| `test-ppu-accuracy-k8s.yml` | `nightly-accuracy-8-glm52-ppu` | 1 | dispatch, `workflow_call` |

Not on a schedule: cron is honoured only from the default branch and this file
lives on a version branch, so a cron here would never fire.

It shares its shape with `test-ppu-answer-k8s.yml` for the reasons that file's
header records at length: one job per entry rather than a matrix, because
`flytiger-eco/ppu-distributed-action` derives both its NAS staging path and its
K8s job name from `$GITHUB_JOB`, which every leg of a matrix shares; the job body
in `scripts/ci/ppu/`, because a local composite action is not available to this
runner group's container-hooked jobs; and two checkouts per job, because the
github.com egress is the least reliable step in the run.

The `mode` input picks which of the two configs runs. `full` is the default and
evaluates the whole split; `smoke` evaluates 20 samples, which is not an accuracy
measurement and is not meant to be — it is proof that the environment, the staged
dataset, the server, the tool and the report reader all agree, for the price of
one weight load. `both` runs the smoke first and then the full split on the same
lane, ordered by `needs` plus `if: ${{ !cancelled() }}` so they never hold two
boards at once.

Evidence is collected by `scripts/ci/ppu/collect_accuracy_evidence.sh`, which
reads the report back off the NAS and prints the score as an annotation whether or
not it passed — a run that judged nothing because no baseline exists has still
measured the only thing anyone wanted. The annotations are the whole delivery
mechanism: this runner drives the job through a container hook that silently drops
what is written to `GITHUB_STEP_SUMMARY`.

Unlike the other two collectors it copies named files rather than the whole
directory. EvalScope's work directory sits beside the report and holds every
prompt sent, every completion received, every per-sample review and the caches
redirected in next to them — gigabytes for a full split, and generated text. The
verdict, the raw report and the tool log go into the artifact; the predictions stay
on the NAS for whoever is debugging a specific answer.

## Capacity and time budget

One ZW-M890P board (144 GiB × 8) per entry, held for the whole run.

The budget here is an order of magnitude larger than the perf line's: 1319 GSM8K
prompts against a reasoning model at `max_tokens` 32768 and a concurrency of 40 is
hours of generation, not minutes. The full entry is given
`ACCURACY_TIMEOUT_PER_FILE=30600` (8.5 h) around a 6-hour EvalScope timeout, with
the pod and job timeouts above that; the smoke entry is given 12600 s, almost all
of which is the weight load. **Every one of these numbers is an estimate a cold
run has to survive, not a measured budget** — the first green run is what should
replace them, here and in `register_ppu_ci(est_time=...)`.
