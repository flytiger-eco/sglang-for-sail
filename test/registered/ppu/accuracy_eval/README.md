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
`threshold: ["0.98", "2.00"]`. Both constants are carried over, but **the
denominator is not**: the source cases divide by a score recorded on an H20 GPU
and so ask *does this board match the GPU?*, while this line divides by its own
first green full run on this board and asks *has this board drifted from
itself?*. Same two numbers, different question — a red here says this checkpoint
scores worse than it used to, not that the board trails a GPU. A score twice its
baseline is not a triumph; it is evidence that something changed that nobody
meant to change, usually in the harness.

### Two entries are judged; the rest are measured

A config without a baseline ships `baseline: null`, and the schema then refuses a
ratio band as well: a band without a baseline judges nothing, and one that quietly
defaulted would be a rule nobody reviewed. Such a run reports `measured`, carries
a `no_baseline` warning, and says so on the run page in as many words. Filling in
a baseline is a reviewed change, made from a green *full* run of this line on this
hardware — not from the vendor's published number for the unquantised model, and
not from a smoke run's twenty samples.

Two entries have earned one, from run `34429388425` (see below), and are the only
entries on the line that a future night can be red against:

| Entry | Metric | Baseline | Floor | Ceiling (2.00) | H20 cross-check | Ratio to H20 |
| --- | --- | --- | --- | --- | --- | --- |
| `glm52-fp8chan-ceval` | `accuracy` | 0.9420 | 0.9046 | refused above 1.0 anyway | 0.9435–0.9443 | 0.9976–0.9984 |
| `glm52-fp8chan-ifeval` | `prompt_level_strict` | 0.9279 | 0.8577 | refused above 1.0 anyway | 0.9242–0.9335 | 0.9940–1.0040 |

The ceiling is inert for scores this high — twice either baseline exceeds the 1.0
the metric can reach — so in practice these two are floors. That is the expected
shape for a benchmark a checkpoint already scores well on; the ceiling earns its
keep on a dataset where a harness fault can inflate a low score, not here.

The last two columns are not part of the judgement. A baseline taken from this
board's own first run has one blind spot: if that first run was itself degraded
by a porting fault, the fault is frozen into the baseline and no later night can
see it. The cross-check closes that gap once, at the moment the baseline is
filled, by comparing against the H20 scores the source cases record for the same
model and dataset (`kpi_result.core_indic_list`, quoted as the range across the
recorded history). Both entries sit within half a percent of the GPU, so what
went into the baseline is a healthy number rather than a frozen defect. The
check is advisory and manual: it is run when a baseline is written, not on every
night. It also carries different weight per dataset — the recorded H20 history is
tight on C-Eval and GSM8K (within 1% for these checkpoints) but wide on IFEval,
where some models span 0.74–0.92, so there only the order of magnitude is worth
reading.

The other nineteen full entries still carry `baseline: null`. Filling those in is
the same reviewed change, one green full run at a time, and each fill states its
own floor and cross-check alongside the score.

### The floor is computed, not inherited

`0.98` arrived from the source cases as one constant for every dataset. Accuracy
is a binomial proportion, so its sampling noise depends on how many samples the
split holds and on how high the score already sits; one constant cannot be right
for three splits that run from 541 to 1346 samples. The floors above are computed
instead — a one-sided 99% Wilson bound, Bonferroni-corrected across the 21 full
entries a night (per-test α = 0.01/21, z = 3.30), then widened by √2 because the
baseline is itself a single measurement and not a known truth:

| Entry | n | Baseline | Computed floor | `0.98` would give | Slack: computed vs `0.98` |
| --- | --- | --- | --- | --- | --- |
| `glm52-fp8chan-ceval` | 1346 | 0.9420 | 0.9046 (r = 0.9603) | 0.9232 | 50 vs 25 samples |
| `glm52-fp8chan-ifeval` | 541 | 0.9279 | 0.8577 (r = 0.9244) | 0.9093 | 38 vs 10 samples |

`0.98` is tighter than the statistics support on both, and markedly so on IFEval,
whose 541 prompts leave only ten flipped answers between a pass and a red — less
than sampled decoding (`temperature: 1.0`) produces on its own. These two entries
therefore state `min_ratio` rather than inherit the default. The conclusion does
not rest on the family size: correcting across two judged entries instead of
twenty-one still puts the honest floors at r = 0.9705 and r = 0.9444, both below
`0.98`.

Two limits on that arithmetic, both in the direction of the floors being
conservative rather than lax. The binomial model treats each prompt as a fixed
coin and so understates the spread under sampled decoding, which makes these
floors a lower bound on the tolerance actually needed. And correcting across all
21 entries is right for the line once it is fully judged, not for today's two —
held deliberately, so that a floor does not move every time another baseline
lands.

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

All 21 are ported, across four suites — one per model family, because the
configs of a family share one registered file and one process-wide
`SGLANG_PPU_ACCURACY_TEST_CONFIG`:

| Suite | Test file | Model family | Datasets | Configs | Devices |
| --- | --- | --- | --- | --- | --- |
| `nightly-accuracy-8-glm52-ppu` | `test_ppu_glm52_accuracy.py` | GLM-5.2 (2 quantisations) | GSM8K, C-Eval, IFEval | 6 + 1 smoke | 8 |
| `nightly-accuracy-8-kimi26-ppu` | `test_ppu_kimi_k26_accuracy.py` | Kimi-K2.6 (1 quantisation) | GSM8K, C-Eval, IFEval | 3 + 1 smoke | 8 |
| `nightly-accuracy-8-minimax27-ppu` | `test_ppu_minimax_m27_accuracy.py` | MiniMax-M2.7 (2 quantisations) | GSM8K, C-Eval, IFEval | 6 + 1 smoke | 8 |
| `nightly-accuracy-8-qwen35-ppu` | `test_ppu_qwen35_accuracy.py` | Qwen3.5-397B-A17B (2 quantisations) | GSM8K, C-Eval, IFEval | 6 + 1 smoke | 8 |

Twenty-eight configs behind the four files: 21 full evaluations (7 checkpoints ×
3 datasets) and one GSM8K smoke config per checkpoint. The full and smoke configs
of a checkpoint differ only in `limit` — the smoke one evaluates 20 samples. The
workflow selects which config runs by naming it in
`SGLANG_PPU_ACCURACY_TEST_CONFIG`; a suite is a file, because `register_ppu_ci`
registers per file, and the number in each suite name is 8 because every config
declares eight devices.

GLM-5.2 GSM8K went first and alone, deliberately, because two facts about this
line could only be established by running it and each would have been wrong
twenty-one times over: whether the pip index carries EvalScope, and where these
datasets live on the NAS. Both were measured rather than assumed, in a 1-PPU
probe (runs `34374868204` and `34376132868`) that cost minutes instead of a
night:

* the index carries `evalscope` up to and including the pinned `1.11.1`, so
  `setup_evalscope.sh` needs no wheelhouse on this cluster;
* `/nas_aisw/datasets` held only `checkpoints`, `dsmManager`, `hf_cache` and
  `packages` — no dataset area existed at all — and it is writable from a pod,
  so `evalscope/<dataset_id>` was created there and GSM8K staged into it. That
  is the path the configs name.

Neither is a fact about a model, so paying for it once was enough. That entry
then ran green (run `34376918707`), so the remaining twenty were ported behind
the same shape. C-Eval and IFEval were then staged the same way, each in its own
1-PPU probe (runs `34427746656` and `34427755288`), which is why all three
`dataset_dir` paths in the configs now point at data that exists: C-Eval as 52
subject directories of parquet (3.9 MB), IFEval as a single jsonl beside its
`dataset_infos.json` (220 KB).

Nineteen of the twenty-one full entries are still code rather than runs. Each is
one dispatch away from a number — nothing about them is unproven except the
checkpoint's score itself.

### What has actually run

Three runs, in the order they answered something:

| Run | Entry | Result | Registered file |
| --- | --- | --- | --- |
| `34376918707` | `glm52-fp8chan-gsm8k-smoke` | `accuracy=0.9500`, 20/20 samples, 4 shots | 1454s |
| `34429388425` | `glm52-fp8chan-ceval` | `accuracy=0.9420`, 1346/1346 samples, 5 shots | 9956s |
| `34429388425` | `glm52-fp8chan-ifeval` | `prompt_level_strict=0.9279`, 541/541 samples, 0 shots | 4740s |

The two full entries ran on the same dispatch, one board each, and both reported
the whole split — 1346 of 1346 and 541 of 541, which is what earns them a
baseline. Their scores are the two rows in the baseline table above. The other
twenty-six jobs of that dispatch were skipped by the `entries` gate, which is the
selection behaviour working rather than a fault.

Both came in under the `est_time=12000` the registered file declares, which is
now a measurement rather than a guess for these two datasets. GSM8K's full split
is still unmeasured and is the one to watch: 1319 prompts of arithmetic reasoning
at 4 shots generate far more tokens per prompt than C-Eval's multiple choice.

### Where a smoke run's 35 minutes went

From run `34376918707`, and the reason the timeouts and `est_time` are drawn to a
measurement rather than guessed:

| Phase | Measured |
| --- | --- |
| Building the EvalScope venv (`setup_evalscope.sh`, no wheelhouse) | 305s |
| Installing this repository editable beside it | ~100s |
| Server up to `ready to roll`, of which CUDA graph capture was 1050s | 1160s |
| `evalscope eval`, twenty samples in one batch | 278s |
| The registered file, end to end | 1454s |

Two things carry forward from that table, and the full runs since have inverted
one of them. The evaluation is the cheap part of a *smoke* run — bringing the
server up cost four times what scoring twenty samples did — but on a full split
the scoring dominates: C-Eval spent 8787s of its 9956s inside `evalscope eval`
and IFEval 3530s of 4740s, against a CUDA graph capture that stayed near 1050s in
both. Setup is a fixed cost the split amortises. And the venv is rebuilt per run,
which is 305s that a wheelhouse on the NAS would remove if a night ever needs it
back.

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
`generation` and the threshold band all come across, and `eval_batch_size` is set
to the server's own concurrency (below). The `server` block does not come across,
and that is the one substantive departure on the line: each config serves the
checkpoint with the parameters the Answer line has already stood up on this board,
not the parameters the source case names. The reason is what the two
configurations would tell you — a run on parameters this board has never started
measures the startup path, not the model, and an accuracy number should be a
number about the model. What that drops, per checkpoint:

- **GLM-5.2 (both quantisations).** The source serves it with
  `prefill_attention_backend: fa3`, `decode_attention_backend: flashmla` and
  EAGLE speculative decoding at 2 steps; the config serves it with the `dsa`
  sparse-attention configuration the Answer and perf lines run, and no
  speculative decoding.
- **MiniMax-M2.7 (both quantisations).** Served at tp 8, not the source's tp 4
  (FP8) / tp 2 (MXFP4). `reasoning_parser: minimax` rather than the source's
  `minimax-append-think`, matching the Answer line. `top_k: 40` from the source
  travels into `generation` unchanged.
- **Qwen3.5-397B-A17B (both quantisations).** Served at tp 8, not the source's
  tp 4. `mamba_scheduler_strategy`, and the MXFP4 case's `page_size: 64` and
  `disable_shared_experts_fusion`, are left off — the Answer line starts these
  checkpoints without them. Its source cases pin the widest sampling of the
  seven (`top_k` 20 with `min_p`, `presence_penalty` and `repetition_penalty`),
  which is why the schema models the penalty keys; all of it travels into
  `generation`.
- **Kimi-K2.6 MXFP4-FP8.** One quantisation only, the one this board has served.

tp 8 throughout, including the two models the source and perf lines serve at tp 2
and tp 4: a job here holds the whole board for the night either way, a full split
is generation-bound, and a narrower split would leave six cards idle. The suite
name's number is 8 for the same reason. Each test file records its own departure
at its point of use.

Two smaller ones, on every config:

- **`eval_batch_size` set to the server's concurrency.** In `openai_api` mode
  this is the client's concurrency: GLM-5.2, MiniMax-M2.7 and Qwen3.5 serve 40
  concurrent requests (the source's own `max_running_requests`), Kimi-K2.6
  serves 32 (its source's `eval_batch_size`). The source cases' 128 would only
  queue behind these, adding latency and no throughput; the Answer configs
  declare no concurrency at all, which for a 32768-token budget is worse.
- **`limit: 0` is refused.** The source cases spell "the whole split" that way;
  EvalScope spells it as no `--limit` at all, so a config that copied the 0 across
  would evaluate nothing and report a perfect score on it. The schema refuses 0
  and asks for `null`.

## Workflow

| Workflow | Suites | Jobs | Boards each | Trigger |
| --- | --- | --- | --- | --- |
| `test-ppu-accuracy-k8s.yml` | the four above | 28 (21 full + 7 smoke) | 1 | dispatch, `workflow_call` |

Not on a schedule: cron is honoured only from the default branch and this file
lives on a version branch, so a cron here would never fire.

It shares its shape with `test-ppu-answer-k8s.yml` for the reasons that file's
header records at length: one job per entry rather than a matrix, because
`flytiger-eco/ppu-distributed-action` derives both its NAS staging path and its
K8s job name from `$GITHUB_JOB`, which every leg of a matrix shares; the job body
in `scripts/ci/ppu/`, because a local composite action is not available to this
runner group's container-hooked jobs; and two checkouts per job, because the
github.com egress is the least reliable step in the run.

Two inputs choose what runs. `mode` picks the kind: `full` evaluates the whole
split, `smoke` evaluates 20 samples — not an accuracy measurement and not meant
to be, but proof that the environment, the staged dataset, the server, the tool
and the report reader all agree, for the price of one weight load — and `both`
runs the smoke first and then the full split on the same lane, ordered by `needs`
plus `if: ${{ !cancelled() }}` so they never hold two boards at once.

`entries` picks which. A job here holds a whole board for most of a night, so it
is a selection rather than a filter: empty runs `glm52-fp8chan-gsm8k` alone, a
comma-separated list runs exactly those, and the literal `all` is what a caller
writes to mean all twenty-one. That default is why a stray dispatch cannot ask the
farm for twenty-one boards until morning. Run `34429388425` is what that gate
looks like when it works: two named entries ran, twenty-six were skipped.

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
the pod and job timeouts above that. The smoke entry is given 7200 s (2 h), drawn
to the one measured smoke run — 35 minutes of board time, almost all of it the
weight load — with room to spare.

The two measured full runs put numbers under the full budget for the first time:
C-Eval held its board for 9956 s and IFEval for 4740 s, against the 30600 s
ceiling. Both are comfortably inside it, and the ceiling is deliberately left
where it is rather than drawn down to them. GSM8K is the reason — its full split
is the generation-heaviest of the three and still unmeasured, so tightening the
fence to a C-Eval-shaped run would risk killing the first GSM8K night on a budget
that was never measured against it. A run that is merely slower than an
extrapolation costs the same night that letting a hung one sit there does. What
replaces 30600 is a green GSM8K full run, here and in
`register_ppu_ci(est_time=...)` — where 12000 s already covers both measured
datasets.
