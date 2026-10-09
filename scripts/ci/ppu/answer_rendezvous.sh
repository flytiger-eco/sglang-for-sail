#!/bin/bash
# Print the address the nodes of a multi-node Answer group meet at, as
# "<host>:<port>", for SGLANG_PPU_ANSWER_DIST_INIT_ADDR.
#
# Why this exists: the K8s action injects MASTER_ADDR as the headless Service's
# <job>-worker-0.<job>.<ns>.svc.cluster.local, and these pods run with
# hostNetwork, which hands them the host's resolver. Without dnsPolicy
# ClusterFirstWithHostNet that name does not resolve, so every node's rendezvous
# fails on a name lookup before it ever reaches rank 0. The fix belongs in the
# action and was offered there; until it lands, the group exchanges the address
# itself, over the results directory all four nodes already mount.
#
# Rank 0 publishes; the others read. Nothing here talks to the cluster API, so
# it works the same on bare metal, and it runs before the editable install on
# purpose -- publishing early means a worker never waits on rank 0's install or
# its page-cache warm, only on its own.
#
# Usage: answer_rendezvous.sh <exchange_dir>
set -u

exchange_dir="${1:-}"
if [ -z "$exchange_dir" ]; then
  echo "usage: answer_rendezvous.sh <exchange_dir>" >&2
  exit 2
fi

node_rank="${SGLANG_PPU_ANSWER_NODE_RANK:-${NODE_RANK:-0}}"
port="${MASTER_PORT:-29500}"
budget="${ANSWER_RENDEZVOUS_TIMEOUT_SECONDS:-600}"
address_file="${exchange_dir}/rendezvous"

if [ "$node_rank" -eq 0 ]; then
  # The same detection order SGLang's own get_local_ip_auto uses -- explicit
  # host IP, then the source address the kernel would pick for an outbound
  # route, then the hostname -- so the address the group is told to meet at is
  # the one the server would have chosen for itself. Written inline against the
  # standard library rather than imported from sglang: this runs before the
  # editable install, and the rendezvous of a test must not depend on the tree
  # it is about to test being importable.
  host=$(python3 - <<'PYEOF'
import os
import socket

host = os.environ.get("SGLANG_HOST_IP") or os.environ.get("HOST_IP")
if not host:
    for probe in (("8.8.8.8", 80), ("10.255.255.255", 1)):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                # A UDP connect sends nothing; it only selects a source address.
                sock.connect(probe)
                host = sock.getsockname()[0]
            break
        except OSError:
            continue
if not host:
    try:
        candidate = socket.gethostbyname(socket.gethostname())
    except OSError:
        candidate = ""
    if candidate not in ("", "127.0.0.1", "0.0.0.0"):
        host = candidate
print(host or "")
PYEOF
  )
  if [ -z "$host" ]; then
    echo "rendezvous: rank 0 could not determine its own address" >&2
    exit 1
  fi
  mkdir -p "$exchange_dir"
  # Staged and renamed: the readers reach this file over NFS, where they could
  # otherwise observe it empty or half written.
  printf '%s:%s\n' "$host" "$port" > "${address_file}.partial"
  mv "${address_file}.partial" "$address_file"
  echo "rendezvous: rank 0 on ${NODE_NAME:-an unnamed node} published ${host}:${port}" >&2
  printf '%s:%s\n' "$host" "$port"
  exit 0
fi

waited=0
while [ "$waited" -lt "$budget" ]; do
  # A readdir rather than a test on the path: this directory is on NFS, where a
  # cached negative lookup would outlive rank 0's write.
  if ls "$exchange_dir" 2>/dev/null | grep -qx 'rendezvous'; then
    address=$(cat "$address_file" 2>/dev/null || true)
    if [ -n "$address" ]; then
      echo "rendezvous: node ${node_rank} joins ${address} after ${waited}s" >&2
      printf '%s\n' "$address"
      exit 0
    fi
  fi
  sleep 5
  waited=$(( waited + 5 ))
done

echo "rendezvous: node ${node_rank} waited ${budget}s and rank 0 never published ${address_file}" >&2
exit 1
