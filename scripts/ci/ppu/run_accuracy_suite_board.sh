#!/bin/bash
# Run one single-board accuracy entry inside the worker pod.
#
# This is the body every entry of test-ppu-accuracy-k8s.yml executes on its
# board, held here rather than in the workflow for the reason the Answer and perf
# board scripts record: that workflow needs one job id per concurrent entry,
# because flytiger-eco/ppu-distributed-action derives both its NAS staging path
# and its K8s job name from $GITHUB_JOB, and a local composite action is not
# available to those jobs -- their runner group drives the job through a
# container hook, so the workspace the checkout populates is inside the container
# while the runner resolves `uses: ./` against its own filesystem (measured, run
# 34085102675).
#
# A third board script rather than a flag on one of the other two, for the reason
# there is a second: the three lines read different environment variables on
# purpose, so a change made for one cannot move another's config, and they can
# share a board without reading each other's settings.
#
# What is specific to this line is the order of the preflights.  An accuracy
# entry spends an hour on a weight load and then hours more on the split, so
# every fact that can be checked cheaply is checked before anything expensive
# starts: the staged dataset directory (a stat), then the EvalScope environment
# (a pip install and an import), then the serving stack, then the page cache
# warm.  Each of the first two would otherwise surface as a failure after the
# load, having spent a board-hour to report a missing directory.
#
# Reads:
#   SGLANG_PPU_ACCURACY_TEST_CONFIG  absolute path of the reviewed config, and
#                                    the one source for the visible devices, the
#                                    checkpoint and the dataset directory -- a
#                                    second copy in the workflow could drift
#                                    from what the test reads
#   ACCURACY_SUITE                   registered suite name for run_suite.py
#   ACCURACY_TIMEOUT_PER_FILE        seconds run_suite.py gives the registered file
#
# Usage: bash /workspace/source/scripts/ci/ppu/run_accuracy_suite_board.sh
set -e

mkdir -p ~/.pip && cat > ~/.pip/pip.conf << 'PIPEOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
PIPEOF

: "${SGLANG_PPU_ACCURACY_TEST_CONFIG:?the reviewed config path is required}"
: "${ACCURACY_SUITE:?the registered suite name is required}"
: "${ACCURACY_TIMEOUT_PER_FILE:?the per-file timeout is required}"

# Constant for every entry, so they are set here rather than repeated in each
# job. HF_HUB_OFFLINE keeps the tokenizer load off the network on a cluster that
# has none; HF_HUB_CACHE is the shared read-only cache the checkpoints' companion
# repositories were fetched into. PPU_SUPPORTS_FP8=0 reflects the board, not a
# preference.
export SGLANG_IS_IN_CI=true
export CPLUS_INCLUDE_PATH=/usr/local/PPU_SDK/targets/x86_64-linux/include
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub
export PPU_SUPPORTS_FP8=0

# Gloo carries the CPU side of every process group SGLang creates, and left to
# itself it picks its address by resolving the pod's own hostname, which on
# several of these nodes has no address: ranks either fall back to loopback or
# raise out of the resolver at `new_group` before a weight is read (measured on
# the Answer line, run 34085800820, nodes swu10/swu12/swu15). `lo` is correct
# here: this line serves single-pod entries only, whose ranks are processes in one
# network namespace.
export GLOO_SOCKET_IFNAME=lo

cd /workspace/source
git config --global --add safe.directory /workspace/source
SGLANG_PPU_SOURCE_REVISION=$(git rev-parse HEAD)
export SGLANG_PPU_SOURCE_REVISION

# The device list, the checkpoint and the dataset come out of the config the test
# will read. nproc_per_node is a resource request and not device isolation -- a
# pod that asks for eight PPUs sees all eight and so would a pod that asked for
# two -- so the visible set has to be narrowed here for the preflight, which
# requires the visible count to equal the configured one, to mean anything.
CUDA_VISIBLE_DEVICES=$(python3 -c "import json, os; print(','.join(str(int(device)) for device in json.load(open(os.environ['SGLANG_PPU_ACCURACY_TEST_CONFIG']))['hardware']['visible_devices']))")
export CUDA_VISIBLE_DEVICES
ACCURACY_MODEL_PATH=$(python3 -c "import json, os; print(json.load(open(os.environ['SGLANG_PPU_ACCURACY_TEST_CONFIG']))['model']['path'])")
ACCURACY_DATASET=$(python3 -c "import json, os; print(json.load(open(os.environ['SGLANG_PPU_ACCURACY_TEST_CONFIG']))['evaluation']['dataset'])")
# The one path the test reads that the config does not have the last word on.
# accuracy_suite_kit records why the override exists; the preflight below has to
# honour it, or it would refuse a run the test would have completed.
ACCURACY_DATASET_DIR="${SGLANG_PPU_ACCURACY_DATASET_DIR:-$(python3 -c "import json, os; print(json.load(open(os.environ['SGLANG_PPU_ACCURACY_TEST_CONFIG']))['evaluation']['dataset_dir'])")}"
echo "source revision: $SGLANG_PPU_SOURCE_REVISION"
echo "test config:     $SGLANG_PPU_ACCURACY_TEST_CONFIG"
echo "suite:           $ACCURACY_SUITE"
echo "visible devices: $CUDA_VISIBLE_DEVICES"
echo "checkpoint:      $ACCURACY_MODEL_PATH"
echo "dataset:         $ACCURACY_DATASET at $ACCURACY_DATASET_DIR"

# The cheapest preflight, and the one whose failure a reader is least likely to
# diagnose from a traceback: the pod has no network to a hub, so a dataset that
# was never staged onto the NAS cannot be fetched at evaluation time and the
# message has to say what to stage and where.
if [ ! -d "$ACCURACY_DATASET_DIR" ]; then
  echo "ERROR: the ${ACCURACY_DATASET} dataset is not staged at ${ACCURACY_DATASET_DIR}."
  echo "       This pod has no route to a dataset hub. Stage it once from a host"
  echo "       that does, then re-dispatch:"
  echo "         modelscope download --dataset <dataset_id> --local_dir ${ACCURACY_DATASET_DIR}"
  echo "       The dataset_id for each supported dataset is in DATASET_CONTRACTS,"
  echo "       python/sglang/test/kits/accuracy_eval_kit.py."
  echo "       The directory has to be the snapshot root that holds the per-subset"
  echo "       directories, because EvalScope hands a local path to"
  echo "       datasets.load_dataset(path=<dir>, name=<subset>, split=<split>)."
  echo "       If it is already staged elsewhere, set SGLANG_PPU_ACCURACY_DATASET_DIR"
  echo "       to that directory instead; the report records that it was overridden."
  # Which is the likely case the first time this line runs anywhere, so this
  # dispatch answers "where is it then?" rather than only "not there". Bounded on
  # both depth and wall clock: this is a shared filesystem and a broad walk of it
  # would cost more than the answer is worth.
  parent=$(dirname "$ACCURACY_DATASET_DIR")
  while [ "$parent" != "/" ] && [ ! -d "$parent" ]; do
    parent=$(dirname "$parent")
  done
  echo "       The deepest existing ancestor is ${parent}, which holds:"
  ls -1 "$parent" 2>/dev/null | head -40 | sed 's/^/         /'
  echo "       Anything named after the dataset under /nas_aisw/datasets:"
  timeout 120 find /nas_aisw/datasets -maxdepth 4 -iname "*${ACCURACY_DATASET}*" 2>/dev/null |
    head -20 | sed 's/^/         /' || echo "         (search timed out)"
  exit 1
fi

# EvalScope lives in an environment of its own; setup_evalscope.sh records why.
# The test reads the binary out of this variable rather than off PATH, so the
# suite cannot silently score against some other EvalScope the image happened to
# carry.
bash scripts/ci/ppu/setup_evalscope.sh
export SGLANG_PPU_EVALSCOPE_BIN="${EVALSCOPE_VENV:-/opt/evalscope-venv}/bin/evalscope"

bash scripts/ci/ppu/ppu_install_dependency.sh

# The JIT cache is keyed on source mtime, and a fresh checkout stamps every file
# with the checkout time, so without this every run recompiles.
JIT_MTIME=$(git log -1 --format=%ct 2>/dev/null || echo 0)
if [ "$JIT_MTIME" != 0 ]; then
  find python/sglang/jit_kernel -type f -exec touch -d @"$JIT_MTIME" {} +
  echo 'Pinned JIT source mtimes'
fi

# The warm sits immediately before the load, on the node that will do the
# loading, which no step of the orchestration shell could do; the measured host
# reclaims the cache within thirty minutes, so a warm separated from its load is
# wasted machine time. It is the startup budget this protects, not the score:
# the evaluation runs against an already-loaded server.
bash scripts/ci/warm_page_cache.sh "$ACCURACY_MODEL_PATH"

# Routed through run_suite.py so the executed set is exactly what
# register_ppu_ci declares. No --continue-on-error: one registered file runs
# here, and a suite that cannot evaluate has to turn the run red. Whether a score
# it did measure is a pass is the evaluator's judgement, not this script's.
cd test
python3 run_suite.py \
  --hw ppu \
  --suite "$ACCURACY_SUITE" \
  --nightly \
  --timeout-per-file "$ACCURACY_TIMEOUT_PER_FILE"
