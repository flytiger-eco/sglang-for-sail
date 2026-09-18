#!/usr/bin/env python3
"""Stage this run's trend rows into the data branch's working tree.

Reads every ``trend.jsonl`` under ``INCOMING_DIR`` -- one per measuring job's
artifact -- and writes the rows under ``data/<test_id>/``, one file per
(test_id, run, attempt).

The line text is copied through unchanged rather than re-serialised. The kit
already emitted each row with sorted keys and no spare whitespace, so passing the
bytes along keeps what lands on the branch identical to what the run measured,
and leaves this script with no way to alter a number.

A row that does not parse, or whose ``test_id`` would not be a plain directory
name, stops the run before anything is filed: a series is only as good as the
worst line anyone ever appended to it, and a bad line is far cheaper to reject
here than to find months later.

Reads:
  INCOMING_DIR         directory the run's artifacts were downloaded into
  GITHUB_RUN_ID        \\
  GITHUB_RUN_ATTEMPT   /  what makes a filename unique within a series

Usage: python3 scripts/ci/ppu/stage_trend_rows.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

SCHEMA_PREFIX = "ppu-perf-trend-point/"
# A test_id becomes a directory name, so it has to be one. Every id this line
# produces is already of this shape; the check is here so that a row can never
# name a path outside data/.
SAFE_TEST_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def fail(message: str) -> None:
    print(f"::error::{message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    incoming = Path(os.environ.get("INCOMING_DIR", "incoming"))
    run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")

    sources = sorted(incoming.rglob("trend.jsonl"))
    if not sources:
        print("::warning::no trend.jsonl in this run's artifacts, nothing to file")
        return

    # (test_id, date) -> list of raw lines, so that a run straddling midnight UTC
    # files each row under the day it was measured rather than the day the
    # publishing job happened to start.
    staged: dict[tuple[str, str], list[str]] = {}
    for source in sources:
        for number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                fail(f"{source}:{number} is not JSON: {error}")
            schema = row.get("schema_version", "")
            if not schema.startswith(SCHEMA_PREFIX):
                fail(f"{source}:{number} is not a trend row but {schema!r}")
            test_id = row.get("test_id")
            if not isinstance(test_id, str) or not SAFE_TEST_ID.match(test_id):
                fail(f"{source}:{number} has an unusable test_id {test_id!r}")
            generated_at = row.get("generated_at")
            if not isinstance(generated_at, str) or len(generated_at) < 10:
                fail(f"{source}:{number} has no usable generated_at")
            staged.setdefault((test_id, generated_at[:10]), []).append(line)

    filed = 0
    for (test_id, date), lines in sorted(staged.items()):
        target = Path("data") / test_id / f"{date}-{run_id}-{attempt}.jsonl"
        content = "".join(f"{line}\n" for line in lines)
        if target.exists():
            # Re-running the publishing job of an attempt already filed is
            # harmless; the same attempt producing *different* rows is not, and
            # silently overwriting would erase the earlier ones.
            if target.read_text(encoding="utf-8") == content:
                print(f"already filed: {target}")
                continue
            fail(f"{target} exists and differs from what this run measured")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"filed {len(lines)} row(s): {target}")
        filed += len(lines)

    print(f"{filed} row(s) across {len(staged)} series from {len(sources)} report(s)")


if __name__ == "__main__":
    main()
