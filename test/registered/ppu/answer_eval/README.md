# PPU Answer MVP

This directory contains the public, versioned inputs for the PPU Answer nightly
tests. The evaluator lives in `python/sglang/test/kits/answer_eval_kit.py` and
the on-machine driver in `python/sglang/test/kits/answer_suite_kit.py`.

Two models are covered, one registered file and one suite each because
`register_ppu_ci` registers per file and the two claim different device counts:

| Suite | Test file | Model | Devices |
| --- | --- | --- | --- |
| `nightly-answer-1-ppu` | `test_ppu_qwen38_answer.py` | Qwen3.8-27B, BF16 | 1 |
| `nightly-answer-8-ppu` | `test_ppu_qwen35_answer.py` | Qwen3.5-397B-A17B-W8A8-INT8 | 8 |

The dedicated `.github/workflows/nightly-test-ppu-answer.yml` workflow runs both
as a `max-parallel: 1` matrix. It has its own workflow instead of being
dispatched by the general `nightly-test-ppu.yml` workflow, so Answer failures,
timeouts, scheduling, and artifacts remain isolated; its cron is offset to
05:00 Beijing so the two workflows do not contend for the same runner. Both
entries are executed through `run_suite.py`, so the executed set is exactly what
the registry declares. `fail-fast` is off: one model's verdict must not suppress
the other's evidence, and the single-card entry runs first so a break in the
shared serving path appears hours before the 8-card entry would report it.

The current phase enforces only request integrity and deterministic facts and
quality checks. LLM-as-Judge is intentionally deferred. Open-ended cases are
therefore marked `hard_constraints_only` in the result rather than being
presented as fully semantic evaluations.

## Runner configuration

`data/*_test_config.json` are the reviewed sources of truth for hardware
selection, checkpoint identity and path, SGLang server parameters, request
generation parameters, timeouts, dataset, and quality profile. The workflow
reads the same file the test will read to select the visible devices and to warm
the checkpoint page cache, then exports its path as
`SGLANG_PPU_ANSWER_TEST_CONFIG`. This keeps the Python tests generic and makes
the exact execution contract visible in the workflow log.

The two models share `data/answer_cases_zh_v1.json` and
`data/quality_profile.json`: the ten prompts are public general knowledge with
reviewed reference facts, so they identify a judgement standard rather than a
model, and a shared file keeps the two suites comparable by construction.

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
SGLANG_PPU_ANSWER_TEST_CONFIG=test/registered/ppu/answer_eval/data/qwen3_5_397b_a17b_w8a8_int8_test_config.json \
  python3 test/registered/ppu/test_ppu_qwen35_answer.py
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

## Measured baseline

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

The 27B entry has no measured baseline yet. Its budgets are derived rather than
observed: the checkpoint is 52GB in 18 shards, which even at the measured
single-stream 57MB/s loads inside the 1800s `startup_timeout_seconds`, so the
step is given 50 minutes and the job 60. Replace this paragraph with a measured
table once the suite has run on the machine.

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
`data/annotation_record.schema.json`. GitHub artifacts are evidence for a run,
not the long-term annotation system of record, so promote a candidate into that
repository rather than relying on the 30-day artifact retention.
