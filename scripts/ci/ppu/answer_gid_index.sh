#!/usr/bin/env bash
# Print the RoCE GID index a multi-node Answer group must be told to use, for
# NCCL_IB_GID_INDEX.
#
# Why this exists: pccl picks the GID itself, and on these hosts it picks the
# wrong one. Its automatic choice prefers a RoCE v2 GID derived from the
# device's IPv4 address; the RoCE bonds here carry no IPv4 at all, so it falls
# back to index 0 -- the link-local fe80::<EUI-64> address formed from the MAC.
# A link-local source does not cross the L3 fabric between two nodes, so the
# group builds its queue pairs over the out-of-band TCP channel, reports
# "Connected all rings", and then dies on its first payload with
# IBV_WC_RETRY_EXC_ERR: the packets leave and the peer never acknowledges them.
#
# What the bonds do carry is a global ULA under fd03::/8, exposed as a
# consecutive pair of GIDs -- RoCE v1 at the lower index, RoCE v2 at the higher
# one. Measured on a two-node group of these boards, only the v2 entry of that
# pair carries traffic:
#
#   gid 0,1  fe80:...:MAC              link-local     hang / ibv_modify_qp ETIMEDOUT
#   gid 2    fd03:45c2:1:XXXX::1       global, v1     IBV_WC_RETRY_EXC_ERR
#   gid 3    fd03:45c2:1:XXXX::1       global, v2     all_reduce correct
#   gid 4,5  fe80:45c2:...             link-local     IBV_WC_RETRY_EXC_ERR
#
# So the rule below is: among the GIDs that are not link-local, take the
# highest index, which is the RoCE v2 entry of the routable address. That
# resolved to 3 on every device of both nodes measured, but it is derived
# rather than written down because the table is built in the order the kernel
# added the addresses, and a node that ordered them differently would
# otherwise fail silently back to an unroutable GID.
#
# The GID type would say v1/v2 outright, but ports/1/gid_attr/types is absent
# on this driver, so index order within the address pair is what is left.
#
# Prints one integer on stdout. Exits non-zero, with the reason on stderr, if
# no routable GID exists or if the devices disagree -- both are conditions the
# group must not start under, because a silent fallback costs a whole run.
set -u

sysfs="${ANSWER_GID_SYSFS_ROOT:-/sys/class/infiniband}"

if [ ! -d "$sysfs" ]; then
  echo "gid: no RDMA devices under $sysfs" >&2
  exit 1
fi

chosen=""
chosen_device=""

for device_path in "$sysfs"/*; do
  [ -d "$device_path" ] || continue
  device=$(basename "$device_path")
  gid_dir="$device_path/ports/1/gids"
  [ -d "$gid_dir" ] || continue

  best=""
  for gid_file in "$gid_dir"/*; do
    index=$(basename "$gid_file")
    # The directory holds one file per table slot, so a slot that is not a
    # number is not a GID and a slot that is all zeroes is unpopulated.
    case "$index" in
      '' | *[!0-9]*) continue ;;
    esac
    value=$(cat "$gid_file" 2>/dev/null || true)
    case "$value" in
      '' | 0000:0000:0000:0000:0000:0000:0000:0000) continue ;;
      # Link-local, in either of the two shapes these devices present: the
      # standard fe80:0000:... and the fe80:<something> the driver also emits.
      fe80:*) continue ;;
    esac
    # Compared as numbers, not in the order the glob happened to hand them
    # over: a table with more than ten slots enumerates as 0 1 10 11 ... 2,
    # where the last name seen is not the highest index.
    if [ -z "$best" ] || [ "$index" -gt "$best" ]; then
      best="$index"
    fi
  done

  if [ -z "$best" ]; then
    echo "gid: $device exposes no routable GID; its bonds carry only link-local addresses" >&2
    exit 1
  fi

  if [ -z "$chosen" ]; then
    chosen="$best"
    chosen_device="$device"
  elif [ "$best" != "$chosen" ]; then
    # NCCL_IB_GID_INDEX is one number for every device, so devices that
    # disagree cannot all be served. Refusing here is the cheap failure.
    echo "gid: $device wants index $best but $chosen_device wants $chosen" >&2
    exit 1
  fi
done

if [ -z "$chosen" ]; then
  echo "gid: found no RDMA device with a GID table under $sysfs" >&2
  exit 1
fi

echo "gid: using index $chosen (agreed by every device under $sysfs)" >&2
printf '%s\n' "$chosen"
