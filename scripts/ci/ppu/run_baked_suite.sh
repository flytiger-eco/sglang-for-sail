#!/bin/bash
# Preserve run_suite's unittest policy and require positive execution evidence.
set -euo pipefail

# shellcheck source=scripts/ci/ppu/use_nightly_image.sh
source /sgl-workspace/sglang/scripts/ci/ppu/use_nightly_image.sh
cd /sgl-workspace/sglang/test
log=$(mktemp "${TMPDIR:-/tmp}/ppu-baked-suite.XXXXXX")
trap 'rm -f "$log"' EXIT

set +e
python3 run_suite.py "$@" --require-nonempty 2>&1 | tee "$log"
statuses=("${PIPESTATUS[@]}")
set -e
if [[ "${statuses[0]}" != 0 ]]; then
  exit "${statuses[0]}"
fi
if [[ "${statuses[1]}" != 0 ]]; then
  exit "${statuses[1]}"
fi
python3 /sgl-workspace/sglang/scripts/ci/ppu/verify_nightly_result.py "$log"
