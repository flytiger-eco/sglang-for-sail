# nightly-test-data

This branch carries no source code. It accumulates what the nightly test lines
measure, so that a series outlives the thirty-day artifact retention that would
otherwise erase it.

It is an orphan branch: it shares no history with any code branch and must never
be merged into one. Nothing here is built, imported or tested. It is written only
by nightly workflows, using the run's own `GITHUB_TOKEN` under a job-level
`permissions: contents: write`.

## Layout

    data/<suite>/<date>-<run_id>.jsonl

One file per workflow run, never appended to once written. Several performance
workflows finish on the same night and each pushes its own commit, so a file per
run means two of them cannot contend for one path — the conflict is removed by
the layout rather than handled by retry.

Each line is one measurement, in the `ppu-perf-trend-point/v1` shape that
`perf_eval_kit.render_trend_jsonl` writes. A row is one point in a series keyed
on `(test_id, measurement_id, config_digest)`. The digest belongs in the key
because editing a config changes what is being measured; the series has to end
there and a new one begin, rather than the edit reading as a step change.

A measurement that failed is kept, with null metrics and the reason code that
explains it. A series has to be able to tell a night that could not measure from
a night on which nothing ran.

## What is not here

Nothing in this branch compares two rows. The nightly performance line enforces
no threshold — what turns it red is an inability to measure, not a slow result —
and the run-to-run spread has not been measured yet, so a comparison added now
would fire on noise. Accumulating rows is the whole purpose of this branch for
the time being.
