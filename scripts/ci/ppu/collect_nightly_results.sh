#!/bin/bash
# Carry one nightly level's suite log off the NAS onto the run page and into
# the artifact directory.
#
# Run by every job of nightly-test-ppu.yml in the orchestration shell. The
# action carries no files out of the worker pod, only its logs, so the log
# travels over the NAS both sides mount: the pod writes under /mnt/wl_nas and
# this shell reads the same bytes under /wl_nas.
#
# Reads:
#   NIGHTLY_JOB                job id, and what keeps the levels'
#                              annotations apart on one run page
#   NIGHTLY_RESULTS_ON_RUNNER  this job's NAS result directory
#
# Usage: bash scripts/ci/ppu/collect_nightly_results.sh
set -u

: "${NIGHTLY_RESULTS_ON_RUNNER:?the NAS result directory is required}"
job="${NIGHTLY_JOB:-nightly}"

# The NAS copy is read, not moved. The pod writes it as root and this shell is a
# different, non-root uid, so it can read those bytes but cannot unlink them: an
# rm here fails with EPERM on every file and would turn a passing suite red,
# which is the trap collect_answer_evidence.sh records.
destination="${GITHUB_WORKSPACE:-.}/nightly-artifacts"
mkdir -p "$destination"
if [ -d "$NIGHTLY_RESULTS_ON_RUNNER" ]; then
  cp -a "${NIGHTLY_RESULTS_ON_RUNNER}/." "${destination}/"
else
  echo "::warning::${job}: the pod produced no result directory at ${NIGHTLY_RESULTS_ON_RUNNER}"
  exit 0
fi

log="${destination}/run.log"
if [ ! -f "$log" ]; then
  echo "::warning::${job}: no run.log was written; the suite did not reach run_suite.py"
  exit 0
fi

# The suite's machine-readable block, one JSON object per executed file, pulled
# out so a consumer does not have to strip the runner's log prefix.
sed -n '/========== TIMINGS BEGIN ==========/,/========== TIMINGS END ==========/p' "$log" |
  sed -n 's/.*\({"file".*}\)$/\1/p' > "${destination}/timings.jsonl"

# Annotations travel over this step's stdout, which does reach the runner;
# GITHUB_STEP_SUMMARY written from a container-hook job does not
# (collect_answer_evidence.sh measured that on run 34132812028). So the counts
# and each failing file are surfaced here instead of in the step summary.
counts=$(grep -o 'Test Summary: [0-9]*/[0-9]* passed' "$log" | tail -1 || true)
if [ -n "$counts" ]; then
  echo "::notice::${job}: ${counts#Test Summary: }"
fi
# Only the per-file lines carry an argument after the marker: the summary block's
# own header is the bare string "✗ FAILED:" with nothing after it.
grep -E '✗ (FAILED|TIMEOUT): [^[:space:]]' "$log" | while IFS= read -r line; do
  echo "::error::${job}: ${line##*✗ }"
done

exit 0
