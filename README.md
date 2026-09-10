# nightly-test-data

This branch carries no source code. It accumulates what the nightly test lines
measure, so that a series outlives the thirty-day artifact retention that would
otherwise erase it.

It is an orphan branch: it shares no history with any code branch and must never
be merged into one. Nothing here is built, imported or tested.

## Layout

    data/<test_id>/<measured date>-<run id>-<run attempt>.jsonl

One file per (test_id, run, attempt), never appended to once written. Several
performance workflows finish on the same night and each pushes its own commit, so
the branch has more than one writer; giving every writer paths no other writer
uses means a push that loses the race only has to rebase, never to merge two
writers' content.

Rows are filed under the `test_id` they carry rather than under a workflow or
suite name, because `test_id` is part of the series key: partitioning by it is the
same grouping a reader of the series has to do anyway. The date is the one in the
row's own `generated_at`, not the day the publishing job ran, so a run that
straddles midnight UTC still files each row under the day it was measured.

Each line is one measurement, in the `ppu-perf-trend-point/v1` shape that
`perf_eval_kit.render_trend_jsonl` writes, copied here byte for byte — nothing
between the benchmark and this branch reserialises a number. A row is one point
in a series keyed on `(test_id, measurement_id, config_digest)`. The digest
belongs in the key because editing a config changes what is being measured; the
series has to end there and a new one begin, rather than the edit reading as a
step change.

A measurement that failed is kept, with null metrics and the reason code that
explains it. A series has to be able to tell a night that could not measure from
a night on which nothing ran.

## Who writes it

The `publish-trend-rows` job of each `test-ppu-*perf*-k8s.yml` workflow, using
that run's own `GITHUB_TOKEN` under a job-level `permissions: contents: write`.
The workflow-level grant in those files stays read-only, so nothing that launches
a server or runs a benchmark can push here. The rows come from the artifact each
measuring job already uploaded; this branch is never written by hand.

## What is not here

Nothing in this branch compares two rows. The nightly performance line enforces
no threshold — what turns it red is an inability to measure, not a slow result —
and the run-to-run spread has not been measured yet, so a comparison added now
would fire on noise. Accumulating rows is the whole purpose of this branch for
the time being.
