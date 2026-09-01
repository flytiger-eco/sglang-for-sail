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

## Results and annotations

The public workflow uploads only redacted rule findings, provenance, a
Markdown summary, and JUnit XML. It removes candidate text and all stable
answer-derived hashes because short answers can be recovered by enumeration.
`SGLANG_PPU_ANSWER_INCLUDE_RAW_OUTPUTS=1` may only be set in a separate,
access-controlled collection job whose output is written to the private
annotation system of record. The public workflow pins this switch to `0`.

Public data belongs here:

- prompts, reviewed reference facts, and quality profiles;
- synthetic mutations that are safe to disclose;
- explicitly released calibration snapshots.

Pending blind reviews, adjudication records, raw nightly answers, and unrevealed
holdout samples belong in a private annotation repository. Records should
conform to `data/annotation_record.schema.json`. GitHub artifacts are evidence
for a run, not the long-term annotation system of record.
