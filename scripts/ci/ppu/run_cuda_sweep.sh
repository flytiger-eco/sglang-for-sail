#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PPU_EVIDENCE_DIR="${PPU_EVIDENCE_DIR:?}"
export PYTHONPATH="$ROOT/python:$ROOT/test${PYTHONPATH:+:$PYTHONPATH}"
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$ROOT"
export SGLANG_IS_IN_CI=true SGLANG_TEST_MAX_RETRY=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export GITHUB_STEP_SUMMARY="$PPU_EVIDENCE_DIR/test-summary.log"
export TVM_FFI_CACHE_DIR=/tmp/ppu-sweep-tvm-ffi
export SGLANG_JIT_CACHE_DIR=/tmp/ppu-sweep-jit
export DG_JIT_CACHE_DIR=/tmp/ppu-sweep-deep-gemm
mkdir -p "$PPU_EVIDENCE_DIR"
start=$SECONDS
finish() {
    status=$?
    printf '{"exit_code":%s,"elapsed":%s}\n' "$status" "$((SECONDS-start))" > "$PPU_EVIDENCE_DIR/worker.json"
    python3 scripts/ci/ppu/pr_report.py sanitize --directory "$PPU_EVIDENCE_DIR"
    exit "$status"
}
trap finish EXIT
python3 - <<'PY' > "$PPU_EVIDENCE_DIR/preflight.log" 2>&1
import json
import os
import re
import subprocess
from pathlib import Path

sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if not re.fullmatch(r'[a-f0-9]{40}', os.environ['PPU_SWEEP_SHA']) or sha != os.environ['PPU_SWEEP_SHA']:
    raise ValueError('Worker checkout differs from requested SHA')
digest = os.environ['PPU_BASE_IMAGE_DIGEST']
if not re.fullmatch(r'sha256:[a-f0-9]{64}', digest) or not os.environ['PPU_BASE_IMAGE_REF'].endswith('@' + digest):
    raise ValueError('Worker image is not pinned to its digest')
plan = json.loads(Path('sweep-plan/sweep_plan.json').read_text())
if plan['sha'] != sha or plan['image_digest'] != digest:
    raise ValueError('Worker provenance differs from preflight plan')
attempt = int(os.environ['PPU_SWEEP_ATTEMPT'])
if plan['run_id'] != os.environ['PPU_SWEEP_RUN_ID'] or not 1 <= plan['attempt'] <= attempt:
    raise ValueError('Worker run or attempt differs from preflight plan')
entry = next(row for row in plan['matrix']['include'] if row['key'] == os.environ['PPU_SWEEP_KEY'])
for field, variable in (
    ('suite', 'PPU_SWEEP_SUITE'), ('n_devices', 'PPU_SWEEP_N_DEVICES'),
    ('partition_id', 'PPU_SWEEP_PARTITION_ID'), ('partition_size', 'PPU_SWEEP_PARTITION_SIZE'),
    ('timeout_per_file', 'PPU_SWEEP_TIMEOUT'), ('board', 'PPU_SWEEP_BOARD'),
    ('ppu_supports_fp8', 'PPU_SUPPORTS_FP8'),
):
    if str(entry[field]) != os.environ[variable]:
        raise ValueError('Worker execution differs from preflight plan: ' + field)
fp8 = int(os.environ['PPU_SUPPORTS_FP8'])
if fp8 != {'890P': 1, '810E': 0}[entry['board']]:
    raise ValueError('FP8 configuration differs from board capability')
metadata = dict(entry, sha=sha, image_digest=digest, mode='pytest', ppu_supports_fp8=fp8,
                run_id=os.environ['PPU_SWEEP_RUN_ID'], attempt=attempt, plan_attempt=plan['attempt'],
                source_root=str(Path.cwd()))
(Path(os.environ['PPU_EVIDENCE_DIR']) / 'sweep.json').write_text(json.dumps(metadata, indent=2))
PY
python3 - <<'PY' > "$PPU_EVIDENCE_DIR/environment.json" 2> "$PPU_EVIDENCE_DIR/environment.log"
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import torch
import sglang

if not Path(sglang.__file__).resolve().is_relative_to(Path.cwd() / 'python'):
    raise ValueError('SGLang import is outside the frozen checkout')
if not torch.cuda.is_available():
    raise ValueError('CUDA-compatible PPU device is unavailable')
requested = int(os.environ['PPU_SWEEP_N_DEVICES'])
visible = torch.cuda.device_count()
print(json.dumps({
    'python': platform.python_version(), 'torch': torch.__version__,
    'checkout_import': sglang.__file__,
    'board': os.environ['PPU_SWEEP_BOARD'],
    'ppu_supports_fp8': int(os.environ['PPU_SUPPORTS_FP8']),
    'requested_devices': requested, 'visible_devices': visible,
    'devices': [torch.cuda.get_device_name(i) for i in range(visible)],
    'packages': sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name']),
}, indent=2))
if visible != requested:
    raise ValueError(f'PPU allocation differs from request: requested={requested}, visible={visible}')
PY
python3 test/run_suite.py --hw cuda --suite "$PPU_SWEEP_SUITE" \
    --pytest --junit-dir "$PPU_EVIDENCE_DIR/junit" --continue-on-error \
    --timeout-per-file "$PPU_SWEEP_TIMEOUT" \
    --auto-partition-id "$PPU_SWEEP_PARTITION_ID" --auto-partition-size "$PPU_SWEEP_PARTITION_SIZE"
