#!/bin/bash
# Run one node's share of an Answer suite, keeping both its log and its status.
#
# The K8s action streams the log of worker-0 and of no other pod, so a worker
# node's stdout would otherwise never leave it -- and a worker is exactly the
# node that would report a device shortfall, a rendezvous it never joined, or a
# server it lost. Each node therefore writes its own log into the directory the
# group shares, which is also what the workflow collects as the artifact.
#
# Invoked with bash on purpose: the exit status of the suite has to survive
# being piped into tee, and PIPESTATUS is what makes that exact rather than a
# reconstruction. Handing back tee's status instead would turn every failed
# suite green.
#
# Usage: run_answer_suite_node.sh <suite> <timeout_per_file> <rank_dir>
# Run from the directory holding run_suite.py.
set -u

suite="${1:-}"
timeout_per_file="${2:-}"
rank_dir="${3:-}"
if [ -z "$suite" ] || [ -z "$timeout_per_file" ] || [ -z "$rank_dir" ]; then
  echo "usage: run_answer_suite_node.sh <suite> <timeout_per_file> <rank_dir>" >&2
  exit 2
fi

node_rank="${SGLANG_PPU_ANSWER_NODE_RANK:-${NODE_RANK:-0}}"
mkdir -p "$rank_dir"
log="${rank_dir}/rank-${node_rank}.log"
status_file="${rank_dir}/rank-${node_rank}.status"

set -o pipefail
python3 run_suite.py \
  --hw ppu \
  --suite "$suite" \
  --nightly \
  --timeout-per-file "$timeout_per_file" 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
set +o pipefail

# Written after the run, and read back by the workflow: rank 0 publishes its
# verdict before it releases the workers, so a worker that failed afterwards is
# visible in nothing else.
printf '%s\n' "$status" > "$status_file"
exit "$status"
