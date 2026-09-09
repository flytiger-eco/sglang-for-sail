#!/bin/bash
# Install EvalScope into a virtual environment of its own, inside the worker pod.
#
# A separate environment rather than the image's, because EvalScope's dependency
# closure and the image's serving stack disagree where it is not negotiable:
# EvalScope requires modelscope[datasets]>=1.34, which resolves datasets>=4.0.0,
# while the v2.1.1 image ships datasets 3.1.0 pinned to dill<0.3.9 -- and
# ppu_install_dependency.sh installs dill deliberately inside that band because
# Python 3.12 unpickles _abc._abc_data only from 0.3.8 up.  Letting pip resolve
# EvalScope into the serving environment would move datasets and dill under the
# server this suite is here to measure.
#
# The isolation costs nothing here.  With --eval-type openai_api EvalScope is a
# pure HTTP client: it reads a staged dataset, calls /v1/chat/completions and
# scores the text that comes back.  Its core requirements carry no torch and it
# never imports the serving stack, so the two environments share only the socket.
#
# The version is pinned rather than floated, and above 1.11.0 rather than at any
# release: report schema v2 -- structured metric list, primary_metric_identity,
# execution_summary -- first appears there, and accuracy_eval_kit reads that
# schema and refuses v1 rather than misreading it (a v1 report has a report-level
# `score` whose value is a different thing).  The import check at the end is that
# contract, asserted before a checkpoint is loaded rather than after.
#
# Reads:
#   EVALSCOPE_VENV        where to build the environment (default /opt/evalscope-venv)
#   EVALSCOPE_SPEC        the requirement to install
#   EVALSCOPE_WHEELHOUSE  optional local wheel directory, tried before the index
#
# Usage: bash /workspace/source/scripts/ci/ppu/setup_evalscope.sh
set -euo pipefail

EVALSCOPE_VENV="${EVALSCOPE_VENV:-/opt/evalscope-venv}"
EVALSCOPE_SPEC="${EVALSCOPE_SPEC:-evalscope[ifeval]==1.11.1}"
EVALSCOPE_WHEELHOUSE="${EVALSCOPE_WHEELHOUSE:-/nas_aisw/datasets/packages/evalscope}"

evalscope_bin="${EVALSCOPE_VENV}/bin/evalscope"
venv_python="${EVALSCOPE_VENV}/bin/python"

# Idempotent, because a pod that retries a step should not pay for the install
# twice and because a human debugging inside the pod runs this by hand.
if [ -x "${evalscope_bin}" ] && "${venv_python}" -c "import evalscope" >/dev/null 2>&1; then
  echo "EvalScope environment already present at ${EVALSCOPE_VENV}"
else
  echo "Building the EvalScope environment at ${EVALSCOPE_VENV}"
  python3 -m venv "${EVALSCOPE_VENV}"
  "${venv_python}" -m pip install --no-cache-dir --upgrade pip wheel

  # The wheel directory first when the NAS carries one: it makes the install
  # independent of egress on a cluster whose network is the least reliable thing
  # about it. Not --no-index by itself, because a wheelhouse assembled for one
  # release is not guaranteed to close over a later pin -- the index stays as the
  # fallback rather than as the plan.
  installed=0
  if [ -d "${EVALSCOPE_WHEELHOUSE}" ]; then
    echo "Trying the local wheel directory ${EVALSCOPE_WHEELHOUSE}"
    if "${venv_python}" -m pip install --no-cache-dir --no-index \
      --find-links "${EVALSCOPE_WHEELHOUSE}" "${EVALSCOPE_SPEC}"; then
      installed=1
    else
      echo "the local wheels do not close over ${EVALSCOPE_SPEC}, using the index"
    fi
  fi
  if [ "${installed}" != 1 ]; then
    "${venv_python}" -m pip install --no-cache-dir "${EVALSCOPE_SPEC}"
  fi
fi

# What the evaluator actually depends on, checked here so a wrong version costs
# seconds rather than the hour a weight load and an evaluation would spend before
# the report turned out to be unreadable.
"${venv_python}" - <<'PYEOF'
import evalscope
from evalscope.api.metric.semantics import MetricIdentity  # noqa: F401
from evalscope.report import Report

version = getattr(evalscope, "__version__", "unknown")
fields = Report.model_fields
for name in ("schema_version", "metrics", "primary_metric_identity"):
    if name not in fields:
        raise SystemExit(
            f"evalscope {version} has no Report.{name}: this is not the report v2 "
            "schema accuracy_eval_kit reads"
        )
print(f"evalscope {version}, report schema v2 present")
PYEOF

echo "evalscope binary: ${evalscope_bin}"
"${evalscope_bin}" --help >/dev/null
