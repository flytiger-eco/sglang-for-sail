# PPU Answer MVP

This directory holds the registered PPU Answer nightly tests together with the
public, versioned inputs they read. The evaluator lives in
`python/sglang/test/kits/answer_eval_kit.py` and the on-machine driver in
`python/sglang/test/kits/answer_suite_kit.py`.

```
answer_eval/
├── configs/<model family>/<model config>.json   reviewed execution contracts
├── dataset/                                     corpus, quality profile, schema
└── test_ppu_*.py                                registered entry points
```

The test files sit here rather than in `test/registered/ppu/` so that the suites,
the contracts they execute, and the corpus they are judged against are one
reviewable unit. Nothing about discovery changes: `run_suite.py` and the
`check-registered-tests` hook both walk `test/registered/**/*.py`, and a suite is
owned by the `register_ppu_ci` call inside the file, not by its directory.

Two models are covered, one registered file and one suite each because
`register_ppu_ci` registers per file and the two claim different device counts:

| Suite | Test file | Model | Devices |
| --- | --- | --- | --- |
| `nightly-answer-1-ppu` | `test_ppu_qwen38_answer.py` | Qwen3.8-27B, BF16 | 1 |
| `nightly-answer-8-ppu` | `test_ppu_qwen35_answer.py` | Qwen3.5-397B-A17B-W8A8-INT8 | 8 |

Each suite is executed on two boards. That is a matter of configuration rather
than of registration: the model, the corpus, and the judging standard are the
same, so a board is chosen by handing the suite a different reviewed config.

| Board | Config suffix | Workflow | Trigger |
| --- | --- | --- | --- |
| ZW810E, 96GiB | none | `nightly-test-ppu-answer.yml` | cron 05:00 Beijing, dispatch |
| ZW-M890P, 144GiB | `-144g` | `test-ppu-answer-k8s.yml` | dispatch only |

The dedicated `.github/workflows/nightly-test-ppu-answer.yml` workflow runs both
ZW810E entries as a `max-parallel: 1` matrix. It has its own workflow instead of
being dispatched by the general `nightly-test-ppu.yml` workflow, so Answer
failures, timeouts, scheduling, and artifacts remain isolated; its cron is
offset to 05:00 Beijing so the two workflows do not contend for the same runner.
Both entries are executed through `run_suite.py`, so the executed set is exactly
what the registry declares. `fail-fast` is off: one model's verdict must not
suppress the other's evidence, and the single-card entry runs first so a break in
the shared serving path appears hours before the 8-card entry would report it.
The ZW-M890P workflow keeps all of that and differs only where the cluster forces
it to; see [The ZW-M890P line](#the-zw-m890p-line).

The current phase enforces only request integrity and deterministic facts and
quality checks. LLM-as-Judge is intentionally deferred. Open-ended cases are
therefore marked `hard_constraints_only` in the result rather than being
presented as fully semantic evaluations.

## Runner configuration

`configs/<model family>/<model config>.json` are the reviewed sources of truth
for hardware selection, checkpoint identity and path, SGLang server parameters,
request generation parameters, timeouts, dataset, and quality profile. The file
name spells the identity its `test_id` declares, with the ZW810E baseline
carrying no board suffix: `configs/qwen3.8/27b-bf16.json` is
`qwen3.8-27b-bf16-answer-96g` and its ZW-M890P sibling
`configs/qwen3.8/27b-bf16-144g.json` is `qwen3.8-27b-bf16-answer-144g`. The
suffix is the per-device capacity rather than the board name, which is how the
internal plans separate these same two boards (`answer_96g`, `answer_144g`) and
what `answer_expected_hardware` renders into provenance (`zw810e-8x96g`,
`zw-m890p-8x144g`). The workflow reads the same file the test will read to select
the visible devices and to warm the checkpoint page cache, then exports its path
as `SGLANG_PPU_ANSWER_TEST_CONFIG`. This keeps the Python tests generic and makes
the exact execution contract visible in the workflow log.

Only the `hardware` section and the checkpoint path differ between a board pair.
Every server and generation parameter is held identical on purpose, so that a
difference between two verdicts is attributable to the board alone.

The two models share `dataset/answer_cases_zh_v1.json` and
`dataset/quality_profile.json`: the ten prompts are public general knowledge with
reviewed reference facts, so they identify a judgement standard rather than a
model, and a shared file keeps the two suites comparable by construction.

`evaluation.dataset` and `evaluation.quality_profile` are resolved against this
directory, which each test file declares as `data_root`, rather than against the
directory holding the config. Resolving from the config would put the shared
corpus above it, reachable only through `..`, which `validate_test_config`
rejects so that no config can name inputs outside the reviewed tree. `data_root`
is a class attribute with no environment override, unlike the config path: the
corpus a verdict was produced against stays pinned to the checkout.

CI runs each suite the same way every other registered test runs:

```bash
cd test
python3 run_suite.py --hw ppu --suite nightly-answer-8-ppu --nightly \
  --timeout-per-file 14400
```

`run_suite.py` invokes each file as `python3 <file> -f`, so the configuration
arrives through the environment variable. A local run can select the contract
the same way:

```bash
SGLANG_PPU_ANSWER_TEST_CONFIG=test/registered/ppu/answer_eval/configs/qwen3.5/397b-a17b-w8a8-int8.json \
  python3 test/registered/ppu/answer_eval/test_ppu_qwen35_answer.py
```

Each test file also names its own config as the default, so the variable is only
needed to depart from it. A config handed to the wrong file still fails loudly
rather than silently testing something else: validation ties `tp_size` to the
declared devices and the preflight ties the checkpoint to the devices actually
visible.

For local pytest users, `test/registered/ppu/conftest.py` exposes the same
selection as `--answer-test-config <path>`; that option is not part of the CI
path.

The tests validate the configuration, checkpoint directory, and `config.json`
before starting SGLang. The test configuration digest and checkpoint
configuration digest are included in provenance; the first on-machine run must
establish the reviewed checkpoint digest baseline.

Candidate generation is deterministic for both models: temperature 0, top-p 1,
and at most 2048 output tokens. The 397B server configuration is TP=8, FA3
attention, static memory fraction 0.8, and `w8a8_int8` quantization; the 27B is
TP=1, FA3, static memory fraction 0.85, and `unquant`, which is how
`server_args` spells an explicit opt-out for a BF16 checkpoint.

## The ZW-M890P line

`.github/workflows/test-ppu-answer-k8s.yml` runs the same two suites against the
144GiB board, which is reachable only through the K8s cluster. It is a sibling
workflow rather than a second matrix dimension of the bare-metal one because
everything around the test differs, while the test itself does not:

- **Selection.** The board is chosen by `node_selector: board-type=ZW-M890P`, and
  the work runs in a worker pod submitted by `ppu-distributed-action` from a
  CPU-only orchestration shell that holds no device.
- **Devices.** `nproc_per_node` is a resource request, not isolation. A pod that
  asks for one PPU still sees all eight `/dev/alixpu_ppu*` nodes, and
  `torch.cuda.device_count()` still returns 8, so the preflight — which requires
  the visible device count to equal the configured one — would fail the
  single-card entry outright. `CUDA_VISIBLE_DEVICES` is therefore exported inside
  the pod from the same config the test reads, and both entries request all eight
  PPUs so that no second pod can land on the board and contend for a device with
  a server that has already claimed most of its memory. The internal btv1.5 plan
  schedules both answer cases as `1node8ppu` for the same reason.
- **Checkpoint path.** The 397B weights live under `T-HEAD/v3.5/` on this NAS
  rather than under `qwen/v3.5/` as on the ZW810E line; the 27B path is the same
  on both. The path is a per-config field, so this costs nothing beyond the two
  new configs — `checkpoint_name` is unchanged, because only the parent directory
  differs.
- **Evidence channel.** The action returns pod logs but not pod files, so the
  report travels over the NAS that both sides mount: the pod writes
  `SGLANG_PPU_ANSWER_RESULTS_DIR` under `/mnt/wl_nas/devops/<run>/<job>/`, and the
  orchestration shell reads the same bytes under `/wl_nas/...`, publishes the
  summary, and uploads the artifact. The suite name is part of the path because
  `github.job` is identical for every matrix entry. The NAS copy is read and left
  in place rather than moved: the pod writes as root and the orchestration shell
  is a different, non-root uid, so it can read those bytes but not unlink them.
- **Provenance.** `base_image_digest` is null on this path. The orchestration
  shell has no Docker daemon to inspect the image with and the pod cannot see its
  own digest, so only the image tag is recorded; on the bare-metal line the digest
  is resolved by `docker image inspect`.
- **Trigger.** No cron. GitHub honours `schedule` only on the default branch, so a
  cron on a version branch would never fire while claiming the workflow is
  scheduled. The line is dispatched by hand or through `workflow_call`.
- **Warm.** The page-cache warm runs inside the pod, immediately before
  `run_suite.py`. It has to: the cache it warms belongs to the node that will then
  load the weights, which no orchestration-shell step can reach.

The board's identity and capacity are measured rather than assumed. An inventory
probe submitted to this cluster on 2026-09-02 reported eight devices named
`ZW-M890P` with compute capability 8.9, 39 multiprocessors, 32MB of L2, and
`total_memory` 147456MB — 144.0GiB exactly — under torch 2.11.0, with both
checkpoints present (94 and 18 shards) and the JIT and Hugging Face caches in
place. `hardware.generation` is a free-form string that nothing validates against
a list, so a guessed value would be indistinguishable from a measured one in the
provenance record; `zw-m890p` and `memory_gib_per_device: 144` are what the
hardware reported. Board type alone would not have settled the capacity in any
case: the internal plans schedule this same board in both a 96GiB and a 144GiB
configuration.

Both entries have since been run on this board; see
[Measured baseline (ZW-M890P)](#measured-baseline-zw-m890p).

## Relation to the internal test cases

Both suites are ports of internal `llm_infer_sglang_evalscope` answer cases
(`qwen3.5-397b-a17b-w8a8-int8_3001` and `qwen3.8-27b-bf16_3001`). The port keeps
the checkpoint, device count, attention backend, memory fraction, and
quantization intent, and deliberately departs in two places:

- generation parameters are pinned to the deterministic house line (temperature
  0, top-p 1) instead of the sampling values the internal cases carry, because a
  rule-based verdict must be reproducible;
- serving parameters that the reviewed schema does not model — `page_size`,
  `watchdog_timeout`, `stream_interval`, `max_running_requests`,
  `cuda_graph_max_bs_decode`, `dist_timeout`, and the `disable_*` flags — are
  left at SGLang defaults. `validate_test_config` accepts exactly the reviewed
  parameter set, so adding one of them is a schema change with its own review
  rather than a silent config edit.

The btv1.5 plan for the 144GiB board carries `qwen3.8-27b-fp8_3001` where this
repository keeps BF16, and no BF16 entry of its own. The port stays on BF16
deliberately: the two `-144g` configs are the `-96g` ones with a different board,
and nothing else, so a difference in verdicts has one candidate explanation. An
FP8 entry is a separate case with its own baseline, not a substitution.

## Measured baseline (ZW810E)

The first on-machine run was executed on `ptg-ppu-02` (16 × PPU-ZW810E, 96GiB
per device, driver 1.6.1, SDK 2.1.1) on 2026-09-01 using eight devices, and it
establishes the reference cost of the 397B test:

| Phase | Cost | Observation |
| --- | --- | --- |
| Checkpoint page-cache warm | 26m17s | ~250MB/s across eight parallel readers |
| Weight load | 66s | 47.35 GB per device, 47.64 GB free afterwards |
| CUDA graph capture | ~3m | 18.66 GB free afterwards |
| Ten candidate generations | ~4m | 98.6 decode tokens/s |
| Test body total | 7m31s | well inside the 280-minute step budget |

Two properties of this host are load-bearing and are the reason the warm step
carries the comment it does: the checkpoint is only reachable over NFS at about
57MB/s per stream, and the page cache is reclaimed back to its baseline within
30 minutes regardless of free memory. Neither is a property of the test, and a
run that skips or defers the warm will exhaust `startup_timeout_seconds`
instead of producing a verdict.

During capture each of the eight TP ranks emits one
`Scheduler watchdog timeout (soft=True)` record with a py-spy dump. The
watchdog is soft, no process is killed, and the run proceeds normally; the only
effect is several hundred extra log lines.

The run reported nine of ten cases passing. `deepseek-letter-count` answered
`3` where the reviewed fact is `4`, with `finish_reason=stop` and complete
usage accounting, so the failure is a model-capability result rather than a
serving or evaluation defect. `fact_rule_failed` is classified `hard_fail`
by the kit, and neither the quality profile nor the case schema carries a
severity field, so this case fails the suite until the judgement contract is
revised upstream.

The 27B entry was measured on the same host on 2026-09-02 using one device:

| Phase | Cost | Observation |
| --- | --- | --- |
| Checkpoint page-cache warm | 3m34s | 51.7GiB in 18 shards, ~247MB/s in parallel |
| Weight load | 11.4s | 51.05 GB used, 44.93 GB free afterwards |
| CUDA graph capture | 87.5s | batch sizes up to 33, 13.96 GB free afterwards |
| Server startup, launch to ready | 2m43s | against the 1800s `startup_timeout_seconds` |
| Ten candidate generations | ~27s | ~37 decode tokens/s |
| Test body total | 3m10s | reported elapsed 199s against `est_time` 1200 |
| Whole job | 8m41s | against the 60-minute job budget |

The generous budgets are kept deliberately: they cover a cold checkpoint. At the
measured single-stream 57MB/s, 51.7GiB needs about 15m30s, which together with
capture still fits inside `startup_timeout_seconds`, so this entry produces a
verdict even if the warm step is skipped — unlike the 397B entry, for which the
warm is a hard dependency. `max_total_num_tokens` came out at 264724 with
`context_len` 262144, so `mem_fraction_static` 0.85 leaves the KV pool ample
room on a 96GiB device.

That run reported seven of ten cases passing, all three failures being
`fact_rule_failed` on objective cases:

| Case | Answer | Reviewed fact |
| --- | --- | --- |
| `deepseek-letter-count` | `2` | 4 |
| `henan-bordering-provinces` | includes 湖南, omits 河北 | the six bordering provinces |
| `red-ball-probability` | `12.5` | 1/7, about 14.29% |

The first two are model-capability results: the request path is clean and the
rules are correct — the second even reproduces the error that the reviewed rule
was written to correct in the internal golden. The third is a wrong answer as
well, but it also exposed a gap in the `probability` rule, which accepted `a/b`,
`X%`, or a bare number already in `[0, 1]`. The prompt asks for a percentage, so
a bare `14.29` was discarded exactly as `12.5` was, and no correctly formed
bare-percentage answer could pass at all. A case now declares how a bare number
may be read, and this one admits either reading:

```json
{"type": "probability", "target": 0.142857142857, "tolerance": 0.0006,
 "bare_number_units": ["percent", "probability"], "description": "..."}
```

The declaration is the only way in: the default stays the probability reading
alone and an unknown unit fails the run, because inferring `percent` from a value
above 1 would admit a rounded `14` against a target of 1/7 under a loose enough
tolerance. The two readings of one bare number are alternatives for a single
claim, so the matching one is kept — otherwise `答案是0.1429` would contradict
itself, its percentage reading being both asserted and wrong. Neither the
tolerance nor the severity classification is changed, so `12.5` and a rounded
`14` still fail and the verdict for the run above stands. Changing what a rule
accepts changes the judging standard, so `dataset/answer_cases_zh_v1.json` is now
`revision` 2: an annotation keyed on revision 1 must not be read as though it had
been produced under the current standard.

## Measured baseline (ZW-M890P)

Both entries were run on the btv1.5 cluster on 2026-09-02 (run `33645062958`,
eight ZW-M890P devices of 144GiB, torch 2.11.0). The verdicts are the ZW810E ones,
case for case:

| Suite | Passed | Failing cases | Job | Test body |
| --- | --- | --- | --- | --- |
| `nightly-answer-1-ppu` (27B, 1 device) | 7/10 | `deepseek-letter-count`, `henan-bordering-provinces`, `red-ball-probability` | 19m06s | 399.8s |
| `nightly-answer-8-ppu` (397B, 8 devices) | 9/10 | `deepseek-letter-count` | 26m39s | 770.6s |

The failing answers are the ZW810E ones character for character: `2`, the same
Henan sentence that includes 湖南 and omits 河北, and `12.5` from the 27B; `3` from
the 397B. At temperature 0 that is the expected result, and it is what makes the
board a controlled variable — the `-144g` configs differ from the `-96g` ones only
in `hardware` and, for the 397B, in the checkpoint's parent directory, and no
verdict moves.

Where the board does show is in capacity and in the cost of using it:

| Phase | 27B, 1 device | 397B, 8 devices | ZW810E counterpart |
| --- | --- | --- | --- |
| Free device memory before load | 143.98 GB | 143.07 GB per rank | 96GiB board |
| Weight load | 3.8s, 51.05 GB | 21.9–24.5s, 47.35 GB per rank | 11.4s / 66s |
| CUDA graph capture | 260.8s, batch sizes to 78 | 621.7s | 87.5s (to 33) / ~3m |
| Launch to ready | 6m08s | 12m23s | 2m43s / — |
| `max_total_num_tokens` | 616103 | 2467089 | 264724 / — |
| Free memory after capture | 20.88 GB | 26.33 GB | 13.96 GB / 18.66 GB |

Capture is what grew. `mem_fraction_static` is a fraction, so the same 0.85 on a
144GiB device yields a KV pool 2.3× the 96GiB one; the captured batch-size list is
capped by `max_running_requests`, which the pool raises from 33 to 78 for the 27B,
and every additional batch size is another graph to record. Both entries still
reach ready far inside the 1800s `startup_timeout_seconds` and finish well inside
their step budgets. The 397B also reproduces the soft `Scheduler watchdog timeout`
record on each of the eight ranks during capture that the ZW810E line reports, with
the same absence of consequence.

Dependency install and the page-cache warm cannot be separated in these logs — the
pod's stdout reaches the runner in batches, so both land on a single flush
timestamp — but together they account for roughly 12 minutes for the 27B and 13.5
for the 397B. The install dominates: a worker pod starts from the base image and
re-resolves the wheels on every run, where the bare-metal host reuses what is
already installed. That is a property of the K8s path rather than of the test, and
it is why this line's pod budgets are 120 and 330 minutes.

One defect surfaced in that run and is fixed in the workflow as it now stands: the
evidence step also tried to delete the NAS copy after taking it, which fails with
`EPERM` on every file and turned the step red on both entries even though the copy
and the upload had succeeded. The step now only reads.

## Results and annotations

The workflow uploads `result.json` (rule findings and provenance), `summary.md`,
`junit.xml`, and — because `SGLANG_PPU_ANSWER_INCLUDE_RAW_OUTPUTS` is set to
`1` — `result.raw.json` and `label_candidates.jsonl`, which carry the candidate
answers verbatim. Publishing the candidates is deliberate: the prompts are
public general knowledge that already lives in this directory, the answers are
this project's own model output, and a red nightly is otherwise not diagnosable
without occupying eight devices for a second run. `result.json` stays redacted
so the schema-stable report keeps one shape whether or not raw collection is
enabled.

The same switch also prints the candidates into the job log, so a failure can be
read without downloading anything: after the summary table the test emits one
block per case with the prompt, the answer, and the observed value that tripped
each rule. `ci_utils.run_unittest_files` runs the test file as a plain
subprocess with an inherited stdout, so the block reaches the log whether the
run ends green or red. Because both the artifact and the log carry candidate
text, set the switch back to `0` for any dataset whose prompts or answers cannot
be published — one switch covers both surfaces.

Both raw files are written with escaped non-ASCII (`\uXXXX`) so that an unpaired
surrogate in a candidate answer cannot fail the write. Read them with a JSON
tool, which decodes the escapes: `jq '.cases[] | select(.case_id=="...")'`, or
`jq -r '.cases[] | "[\(.verdict)] \(.case_id)\n  \(.final_answer)\n"'` for every
answer at once. The log block, in contrast, is already plain text: only
characters that UTF-8 cannot encode stay escaped there.

In provenance, `source_revision` is the authoritative identifier of what ran: it
is the full commit SHA of the checkout, injected by the workflow.
`package_versions.sglang` reads `0.0.0`, which is a property of the CI install
path rather than a collection defect. `ppu_install_dependency.sh` swaps in
`pyproject_other.toml`, whose version is dynamic, and installs with
`--no-build-isolation`, so `setuptools-scm` is never present to supply one; had
it run and found no tag, the configured `fallback_version` would have produced
`0.0.0.dev0` instead. The checkout is `--no-tags --depth=2` in any case, so no
tag is reachable on the machine to describe against. Recover the version number
from the SHA in a full clone:

```bash
git checkout <source_revision>
python3 python/tools/get_version_tag.py   # e.g. 0.5.13+v0.1.0-121-g5c9c6bce54
```

Public data belongs here:

- prompts, reviewed reference facts, and quality profiles;
- synthetic mutations that are safe to disclose;
- explicitly released calibration snapshots.

Pending blind reviews, adjudication records, and unrevealed holdout samples
belong in a private annotation repository. Records should conform to
`dataset/annotation_record.schema.json`. GitHub artifacts are evidence for a run,
not the long-term annotation system of record, so promote a candidate into that
repository rather than relying on the 30-day artifact retention.
