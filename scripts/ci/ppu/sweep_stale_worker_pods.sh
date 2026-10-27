#!/bin/bash
# Remove the worker pods earlier runs of this level left behind.
#
# flytiger-eco/ppu-distributed-action deletes its pod in an `always()` step, but
# a job can be torn down before that step runs: a cancelled run whose runner is
# reclaimed, a runner that loses the API connection, a workflow cancelled hard
# enough that no post step survives. The pod then outlives its run and the
# server inside it keeps holding the cards it was handed, so the next run of the
# same level finds its slice occupied and fails every file in setUpClass for a
# reason that exists nowhere in the source -- run 40482998667 saw a 20.84 GiB
# process on device 4 before the first test of nightly-4-ppu.
#
# The sweep is deliberately narrow. It touches only pods carrying this level's
# job id, only ones older than the level's own watchdog could still be alive,
# and only through a normal delete, so kubelet stops the container: `--force`
# would leave the container running on the node, which is the state this script
# exists to end. Anything else on the board -- a neighbour pod, another tenant,
# another lane's long run -- is none of its business and is never touched.
#
# It never decides a level either. No kubectl, no kubeconfig, no permission: say
# so and exit 0, and the suite runs exactly as it would have without this step.
set -uo pipefail

namespace="${PPU_SWEEP_NAMESPACE:-ppu-sched}"
job="${PPU_SWEEP_JOB:?PPU_SWEEP_JOB is required}"
max_age_hours="${PPU_SWEEP_MAX_AGE_HOURS:-6}"

# The action installs kubectl the same three ways; a hosted mount is the common
# case on these runners, and PATH already works when the image ships it.
kubectl_bin=""
for candidate in /opt/bin/kubectl "$(command -v kubectl 2>/dev/null || true)"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    kubectl_bin="$candidate"
    break
  fi
done
if [ -z "$kubectl_bin" ]; then
  echo "::warning::kubectl not found; skipped the sweep of stale $job worker pods"
  exit 0
fi

if ! pods="$("$kubectl_bin" get pods -n "$namespace" -l ppu-job -o \
  jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.creationTimestamp}{"\n"}{end}' 2>&1)"; then
  echo "::warning::cannot list pods in $namespace; skipped the sweep of stale $job worker pods: $pods"
  exit 0
fi

now="$(date -u +%s)"
swept=0
kept=0

# The pod name carries the job id, ppu-<owner>-<run>-<attempt>-<job>-<hash>, so
# a substring match on the job id is what separates this level's leftovers from
# the other levels' live pods on the same board.
while IFS=$'\t' read -r name created; do
  [ -n "$name" ] || continue
  case "$name" in
  *"$job"*) ;;
  *) continue ;;
  esac
  # An unparsable timestamp is left alone: a pod this script cannot date is a
  # pod it cannot prove stale.
  created_epoch="$(date -u -d "$created" +%s 2>/dev/null || true)"
  [ -n "$created_epoch" ] || continue
  age_hours=$(((now - created_epoch) / 3600))
  if ((age_hours < max_age_hours)); then
    kept=$((kept + 1))
    continue
  fi
  echo "Removing worker pod $name (age ${age_hours}h >= ${max_age_hours}h)"
  "$kubectl_bin" delete pod "$name" -n "$namespace" \
    --ignore-not-found=true --wait=false ||
    echo "::warning::could not delete worker pod $name"
  swept=$((swept + 1))
done <<<"$pods"

echo "Swept $swept stale $job worker pod(s); $kept left alone (younger than ${max_age_hours}h)."
