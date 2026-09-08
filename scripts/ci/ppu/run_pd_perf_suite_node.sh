#!/bin/bash
# Run one node's share of a prefill/decode-disaggregated performance entry.
#
# A PD group is two pods of one job: each runs this script, is told by the action
# which rank it is, and serves the role the reviewed config assigns to that rank.
# Rank 0 also runs the router the benchmark posts to and owns the report; the
# other node serves its half of the KV path and holds its board until rank 0 is
# done.  Neither of those decisions is made here -- they are made in
# `pd_perf_suite_kit` off the config -- so this script is only the setup a board
# needs before the suite can run, plus the per-node log every pod but worker-0
# would otherwise lose.
#
# A separate script from run_perf_suite_board.sh rather than a flag on it, for the
# reason that one is separate from the Answer board script: the two lines read
# different environment variables on purpose, so a change made for one cannot move
# the other's config.  What it does share is the tail -- run_perf_suite_node.sh --
# because keeping a node's log and its exact exit status is the same problem here.
#
# Two things the multi-node colocated entry does and this one deliberately does
# not:
#
#   * No rendezvous address.  The two servers of a PD group do not join one
#     process group -- each is a rank 0 of its own eight devices -- so there is
#     nothing to meet at.  What the nodes exchange instead is an HTTP endpoint,
#     published through the results directory by the suite itself.
#
#   * No RoCE GID index.  The cross-node path here is the Mooncake transfer
#     engine's, over the devices the config names in `disaggregation.ib_devices`,
#     not a collective's, so the index the collective line has to be told does not
#     apply.  Should the KV handshake turn out to need its own hint, it belongs in
#     the config next to those devices rather than here.
#
# Reads:
#   SGLANG_PPU_PD_PERF_TEST_CONFIG   absolute path of the reviewed PD config, and
#                                    the one source for the visible devices and
#                                    the checkpoint
#   SGLANG_PPU_PD_PERF_RESULTS_DIR   absolute path every node of the group shares
#   PERF_SUITE                       registered suite name for run_suite.py
#   PERF_TIMEOUT_PER_FILE            seconds run_suite.py gives the registered file
#
# Usage: bash /workspace/source/scripts/ci/ppu/run_pd_perf_suite_node.sh
set -e

mkdir -p ~/.pip && cat > ~/.pip/pip.conf << 'PIPEOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
PIPEOF

: "${SGLANG_PPU_PD_PERF_TEST_CONFIG:?the reviewed PD config path is required}"
: "${SGLANG_PPU_PD_PERF_RESULTS_DIR:?the shared results directory is required}"
: "${PERF_SUITE:?the registered suite name is required}"
: "${PERF_TIMEOUT_PER_FILE:?the per-file timeout is required}"

# Constant for every entry of this line, as on the single-board one:
# HF_HUB_OFFLINE keeps the tokenizer load off a cluster that has no network, and
# a PD run loads a tokenizer three times over -- once in each server and once in
# bench_serving; PPU_SUPPORTS_FP8=0 reflects the board, not a preference.
export SGLANG_IS_IN_CI=true
export CPLUS_INCLUDE_PATH=/usr/local/PPU_SDK/targets/x86_64-linux/include
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub
export PPU_SUPPORTS_FP8=0

# `lo`, exactly as on the single-board entries and for the same reason: gloo
# carries the CPU side of every process group SGLang creates, and each server here
# is one node's worth of ranks in a single network namespace.  The peer of this
# pod is reached over HTTP and over RDMA, never over gloo, so the interface
# derivation the multi-node colocated entry performs has nothing to derive here.
export GLOO_SOCKET_IFNAME=lo

cd /workspace/source
git config --global --add safe.directory /workspace/source
SGLANG_PPU_SOURCE_REVISION=$(git rev-parse HEAD)
export SGLANG_PPU_SOURCE_REVISION

# The device list and the checkpoint come out of the config the test will read: a
# second copy in the workflow could drift from what the suite validates against.
# nproc_per_node is a resource request and not device isolation, so narrowing the
# visible set here is what makes this node's tp 8 server a whole-board server
# rather than one sharing a board the test never checked it had to itself.
CUDA_VISIBLE_DEVICES=$(python3 -c "import json, os; print(','.join(str(int(device)) for device in json.load(open(os.environ['SGLANG_PPU_PD_PERF_TEST_CONFIG']))['hardware']['visible_devices']))")
export CUDA_VISIBLE_DEVICES
PD_MODEL_PATH=$(python3 -c "import json, os; print(json.load(open(os.environ['SGLANG_PPU_PD_PERF_TEST_CONFIG']))['model']['path'])")
# Which role this node serves is the config's answer to the rank it was handed,
# and it is echoed rather than acted on: the suite resolves it again, and a log
# that names it is what makes a two-pod run readable. Read out of the JSON rather
# than through `pd_perf_eval_kit`, because this runs before the install.
PD_ROLE=$(python3 -c "import json, os; c = json.load(open(os.environ['SGLANG_PPU_PD_PERF_TEST_CONFIG'])); print('prefill' if int(os.environ.get('NODE_RANK', '0')) < c['disaggregation']['prefill_nodes'] else 'decode')")
echo "source revision: $SGLANG_PPU_SOURCE_REVISION"
echo "test config:     $SGLANG_PPU_PD_PERF_TEST_CONFIG"
echo "suite:           $PERF_SUITE"
echo "node rank:       ${NODE_RANK:-0} of ${NNODES:-1} on ${NODE_NAME:-an unnamed node}"
echo "role:            $PD_ROLE"
echo "results dir:     $SGLANG_PPU_PD_PERF_RESULTS_DIR"
echo "gloo interface:  $GLOO_SOCKET_IFNAME"
echo "visible devices: $CUDA_VISIBLE_DEVICES"
echo "checkpoint:      $PD_MODEL_PATH"

bash scripts/ci/ppu/ppu_install_dependency.sh

# The JIT cache is keyed on source mtime, and a fresh checkout stamps every file
# with the checkout time, so without this every run recompiles.
JIT_MTIME=$(git log -1 --format=%ct 2>/dev/null || echo 0)
if [ "$JIT_MTIME" != 0 ]; then
  find python/sglang/jit_kernel -type f -exec touch -d @"$JIT_MTIME" {} +
  echo 'Pinned JIT source mtimes'
fi

# The JIT build directory moves off the shared NAS cache before anything can
# compile; the script it comes from records why.
# shellcheck source=scripts/ci/ppu/use_local_jit_cache.sh
. scripts/ci/ppu/use_local_jit_cache.sh

# Warmed on both nodes, unlike the pp-2 colocated entry that skips it: each role
# here loads the whole checkpoint into its own eight devices, so each node reads
# the whole tree and neither warm is wasted. It is the startup budget this
# protects, not the numbers.
bash scripts/ci/warm_page_cache.sh "$PD_MODEL_PATH"

# The tail is the colocated multi-node line's, because a PD node has the same two
# things to preserve: its own log, which the action streams for worker-0 only, and
# its exact exit status through the tee. It reads NODE_RANK for the file names and
# nothing else, so it needs no PD-specific variant.
cd test
bash ../scripts/ci/ppu/run_perf_suite_node.sh \
  "$PERF_SUITE" "$PERF_TIMEOUT_PER_FILE" "$SGLANG_PPU_PD_PERF_RESULTS_DIR/ranks"
