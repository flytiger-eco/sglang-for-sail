# PPU Answer MVP

This directory contains the public, versioned inputs for the Qwen3.5 Answer
nightly test. The evaluator lives in
`python/sglang/test/kits/answer_eval_kit.py`.

The dedicated `.github/workflows/nightly-test-ppu-answer.yml` workflow runs
this test on eight PPU devices. It has its own workflow instead of being
dispatched by the general `nightly-test-ppu.yml` workflow, so Answer failures,
timeouts, scheduling, and artifacts remain isolated; its cron is offset to
05:00 Beijing so the two workflows do not contend for the same runner. The
test is registered as `nightly-answer-8-ppu` and executed through
`run_suite.py`, so the executed set is exactly what the registry declares.

The current phase enforces only request integrity and deterministic facts and
quality checks. LLM-as-Judge is intentionally deferred. Open-ended cases are
therefore marked `hard_constraints_only` in the result rather than being
presented as fully semantic evaluations.

## Runner configuration

`data/qwen3_5_397b_a17b_w8a8_int8_test_config.json` is the reviewed source of
truth for hardware selection, checkpoint identity and path, SGLang server
parameters, request generation parameters, timeouts, dataset, and quality
profile. The workflow reads the same file to select the visible devices and to
warm the checkpoint page cache, then exports its path as
`SGLANG_PPU_ANSWER_TEST_CONFIG`. This keeps the Python test generic and makes
the exact execution contract visible in the workflow log.

CI runs the suite the same way every other registered test runs:

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

For local pytest users, `test/registered/ppu/conftest.py` exposes the same
selection as `--answer-test-config <path>`; that option is not part of the CI
path.

The test validates the configuration, checkpoint directory, and `config.json`
before starting SGLang. The test configuration digest and checkpoint
configuration digest are included in provenance; the first on-machine run must
establish the reviewed checkpoint digest baseline.

The server configuration is TP=8, FA3 attention, static memory fraction 0.8,
and `w8a8_int8` quantization. Candidate generation is deterministic:
temperature 0, top-p 1, and at most 2048 output tokens.

## Measured baseline

The first on-machine run was executed on `ptg-ppu-02` (16 × PPU-ZW810E, 96GiB
per device, driver 1.6.1, SDK 2.1.1) on 2026-09-01 using eight devices, and it
establishes the reference cost of this test:

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

## Results and annotations

The workflow uploads `result.json` (rule findings and provenance), `summary.md`,
`junit.xml`, and — because `SGLANG_PPU_ANSWER_INCLUDE_RAW_OUTPUTS` is set to
`1` — `result.raw.json` and `label_candidates.jsonl`, which carry the candidate
answers verbatim. Publishing the candidates is deliberate: the prompts are
public general knowledge that already lives in this directory, the answers are
this project's own model output, and a red nightly is otherwise not diagnosable
without occupying eight devices for a second run. `result.json` stays redacted
so the schema-stable report keeps one shape whether or not raw collection is
enabled; the raw files are the only place candidate text appears, so set the
switch back to `0` for any dataset whose prompts or answers cannot be published.

Both raw files are written with escaped non-ASCII (`\uXXXX`) so that an unpaired
surrogate in a candidate answer cannot fail the write. Read them with a JSON
tool, which decodes the escapes: `jq '.cases[] | select(.case_id=="...")'`.

Public data belongs here:

- prompts, reviewed reference facts, and quality profiles;
- synthetic mutations that are safe to disclose;
- explicitly released calibration snapshots.

Pending blind reviews, adjudication records, and unrevealed holdout samples
belong in a private annotation repository. Records should conform to
`data/annotation_record.schema.json`. GitHub artifacts are evidence for a run,
not the long-term annotation system of record, so promote a candidate into that
repository rather than relying on the 30-day artifact retention.
