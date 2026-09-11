#!/bin/bash
# Say that this entry measured something the run will not be able to read back.
#
# Run by an entry of the test-ppu-*perf*-k8s.yml workflows only when every upload
# attempt was reset, and held in a script rather than a composite action for the
# reason collect_perf_evidence.sh records: these jobs cannot resolve `uses: ./`
# against the checkout at all.
#
# The uploads are best effort by design. The numbers reached the run page as
# notices before the first of them ran, and the NAS copy outlives the thirty-day
# artifact retention, so a reset on the github egress must not turn a passing
# measurement red. What it must not do either is pass in silence: run 34548547824
# lost three of its ten reports with every step green, and nothing anywhere in it
# named them or said where their copies still were.
#
# The path is not merely printed. Whether the copy is readable is the difference
# between a gap that can be back-filled and one that cannot, and this shell is the
# last thing in the run standing next to that mount.
#
# Reads:
#   ENTRY                   entry name, matching the annotations the collector
#                           already printed for this job
#   PERF_RESULTS_ON_RUNNER  the run's own report directory on the NAS
#
# Usage: bash scripts/ci/ppu/report_lost_perf_artifact.sh
set -euo pipefail

: "${ENTRY:?the entry name is required}"
: "${PERF_RESULTS_ON_RUNNER:?the NAS report directory is required}"

# One line, because an annotation is one line: echo joins these with spaces.
echo "::warning::${ENTRY}: every artifact upload attempt was reset." \
  "The machine-readable report did not leave this job, so the trend series" \
  "will have a hole where this entry should be."

if [ -d "${PERF_RESULTS_ON_RUNNER}" ]; then
  echo "the copy a back-fill reads from, readable from here as of now:"
  echo "  ${PERF_RESULTS_ON_RUNNER}"
  # Listed rather than asserted. What a back-fill needs is trend.jsonl, and
  # whether it is among these bytes is worth learning here rather than weeks
  # later, when the artifact retention has expired and this is all there is.
  find "${PERF_RESULTS_ON_RUNNER}" -type f | sed 's/^/  /'
else
  echo "::warning::${ENTRY}: no copy is readable at ${PERF_RESULTS_ON_RUNNER}" \
    "either, so these numbers survive only as the notices this job printed."
fi
