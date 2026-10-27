#!/usr/bin/env python3
"""Name the nightly files that failed, from the per-partition artifacts.

Failures come from ``timings.jsonl`` (one object per executed file with its
``passed`` flag), so a bad exit and a timeout are both caught; the reason text
comes from the log, which is also the only source when a pod died before the
TIMINGS block was printed.

Reads: argv[1] artifact directory, plus NEEDS_JSON (toJSON(needs)) to tell a
  level with no failures from one whose artifact never arrived.
Writes: markdown on stdout, for the caller to append to $GITHUB_STEP_SUMMARY
  (a runner-side write; the level jobs' container hook drops it).

Usage: python3 scripts/ci/ppu/nightly_failures.py nightly-artifacts
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ARTIFACT = re.compile(r"ppu-nightly-k8s-(?P<job>.+?)-\d+-\d+\Z")
FAILED = re.compile(r"✗ FAILED: (\S+) returned exit code (-?\d+)")
TIMEOUT = re.compile(r"✗ TIMEOUT: (\S+) after (\d+) seconds")
COUNTS = re.compile(r"Test Summary: (\d+)/(\d+) passed")
# A job that never started is no gap; one that concluded was able to report.
CONCLUDED = frozenset({"success", "failure"})


def cell(value):
    """Escape a value for a markdown table cell."""
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "&#124;").replace("\n", " ")


def _files(count):
    """Count with the noun the sentence around it needs."""
    return f"{count} {'file' if count == 1 else 'files'}"


def _suite_lines(path):
    """Read the collector's key=value manifest, tolerating a missing one."""
    values = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if value:
            values[key.strip()] = value.strip()
    return values


def _reasons(log):
    """Map each failing file to the reason the log recorded for it."""
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    reasons = {}
    for match in FAILED.finditer(text):
        reasons[match.group(1)] = "exit code " + match.group(2)
    # A timeout also has a non-zero exit code; the cap is the real reason.
    for match in TIMEOUT.finditer(text):
        reasons[match.group(1)] = "timeout after " + match.group(2) + "s"
    return reasons


def _counts(log):
    """Last 'Test Summary: N/M passed' the log printed, as (passed, total)."""
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found = COUNTS.findall(text)
    if not found:
        return None
    passed, total = found[-1]
    return int(passed), int(total)


def _timings(path):
    """Per-file {'file', 'passed', 'elapsed'} objects, in the order written."""
    rows = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("file"), str):
            rows.append(row)
    return rows


def read_partition(directory):
    """Everything one level job's artifact says about its partition."""
    match = ARTIFACT.match(directory.name)
    job = match.group("job") if match else directory.name
    meta = _suite_lines(directory / "suite.txt")
    log = directory / "run.log"
    rows = _timings(directory / "timings.jsonl")
    reasons = _reasons(log)
    # Path first, then basename: the log and timings.jsonl differ by a prefix.
    by_name = {Path(name).name: reason for name, reason in reasons.items()}
    failed = []
    for row in rows:
        if row.get("passed") is True:
            continue
        name = row["file"]
        failed.append(
            {
                "file": name,
                "reason": reasons.get(name) or by_name.get(Path(name).name) or "failed",
                "elapsed": row.get("elapsed"),
            }
        )
    # A pod that died before run_suite.py returned has no TIMINGS block, so fall
    # back to the log's summary line for a coarser count.
    counts = _counts(log)
    if rows:
        total = len(rows)
        passed = sum(1 for row in rows if row.get("passed") is True)
        listed = True
    elif counts is not None:
        passed, total = counts
        # An empty suite prints a TIMINGS block with no entries, so rows is empty
        # there without anything having been lost.
        listed = total == 0
    else:
        passed = total = None
        listed = False
    partition = meta.get("partition", "")
    if partition in {"none/none", "none"}:
        partition = ""
    return {
        "job": job,
        "suite": meta.get("suite", ""),
        "partition": partition,
        "reported": total is not None,
        "listed": listed,
        "total": total,
        "passed": passed,
        "failed": failed,
    }


def render(partitions, needs):
    """Render the markdown section the run page carries."""
    ran = [p for p in partitions if p["reported"]]
    failed_files = [(p, f) for p in ran for f in p["failed"]]
    total = sum(p["total"] for p in ran)
    passed = sum(p["passed"] for p in ran)
    lines = ["## Nightly test files that failed (PPU)", ""]
    if not ran:
        lines += [
            "> No partition artifact arrived, so no file list can be built. "
            "The per-level job logs still name each failing file in their "
            "`✗ FAILED:` annotations.",
            "",
        ]
        return lines

    failed_count = total - passed
    if failed_count:
        lines.append(f"**{failed_count} of {total} files did not pass.**")
    else:
        lines.append(f"**All {_files(total)} passed.**")
    lines.append("")

    lines += [
        "| Level | Suite / partition | Files | Passed | Failed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for partition in ran:
        label = partition["suite"] or partition["job"]
        if partition["partition"]:
            label += " " + partition["partition"]
        lines.append(
            "| "
            + " | ".join(
                cell(value)
                for value in (
                    partition["job"],
                    label,
                    partition["total"],
                    partition["passed"],
                    partition["total"] - partition["passed"],
                )
            )
            + " |"
        )

    if failed_files:
        lines += [
            "",
            "### Failed files",
            "",
            "Each row is one file `run_suite.py` ran and did not mark passed.",
            "",
            "| Level | File | Reason | Seconds |",
            "| --- | --- | --- | --- |",
        ]
        for partition, failure in failed_files:
            lines.append(
                "| "
                + " | ".join(
                    cell(value)
                    for value in (
                        partition["job"],
                        "`" + failure["file"] + "`",
                        failure["reason"],
                        "" if failure["elapsed"] is None else failure["elapsed"],
                    )
                )
                + " |"
            )

    coarse = [p["job"] for p in ran if not p["listed"]]
    if coarse:
        lines += [
            "",
            "> " + ", ".join("`" + job + "`" for job in coarse) + " produced no "
            "per-file list (its pod stopped before `run_suite.py` printed the "
            "TIMINGS block), so the counts above come from the log's "
            "`Test Summary:` line and its failing files are only named there.",
        ]

    # A level that concluded but left nothing readable is a gap, not a pass.
    missing = [
        job
        for job, info in sorted((needs or {}).items())
        if (info or {}).get("result") in CONCLUDED
        and job not in {p["job"] for p in ran}
    ]
    if missing:
        lines += [
            "",
            "> No result for: " + ", ".join("`" + job + "`" for job in missing) + ". "
            "Those levels ran but left nothing this table can read; their "
            "`✗ FAILED:` annotations are on the job page, and the NAS copy under "
            "`/wl_nas/devops/<run>-<attempt>/<job>/nightly-results/` is what a "
            "back-fill reads.",
        ]
    lines.append("")
    return lines


def main(argv):
    root = Path(argv[1] if len(argv) > 1 else "nightly-artifacts")
    directories = []
    if root.is_dir():
        directories = sorted(p for p in root.glob("*") if p.is_dir())
    partitions = [read_partition(directory) for directory in directories]
    try:
        needs = json.loads(os.environ.get("NEEDS_JSON") or "{}")
    except ValueError:
        needs = {}
    if not isinstance(needs, dict):
        needs = {}
    sys.stdout.write("\n".join(render(partitions, needs)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
