#!/usr/bin/env bash
# Print the network interface a multi-node Answer group must bind gloo to, for
# GLOO_SOCKET_IFNAME.
#
# Why this exists: gloo carries the CPU side of every process group SGLang
# creates, and left to itself it picks its address by resolving the pod's own
# hostname. These pods run with hostNetwork and the host's resolver, and on
# several of these nodes that hostname has no address. Two failure modes follow
# from the one cause, and both were measured:
#
#   * torch warns "Unable to resolve hostname to a (local) address ... Manually
#     set the network interface to bind to with GLOO_SOCKET_IFNAME" and falls
#     back to loopback. Ranks that fall back then advertise 127.0.0.1 to peers
#     on the other node, which reach it as their own loopback and are refused.
#   * gloo raises out of the resolver instead, killing the server at
#     `new_group` before a weight is read: "[enforce fail at
#     gloo/transport/tcp/device.cc:99] rv == 0. -5 vs 0", where -5 is glibc's
#     EAI_NODATA -- the name resolved to no address at all.
#
# Measured on run 34109451085, a two-node group whose rank 0 landed on
# na131t-cloud-swu12: three ranks warned and fell back, one raised, and all ten
# cases were recorded as server_start_failed. The same cause killed three
# single-board entries of run 34085800820 on swu10/swu12/swu15.
#
# The single-board script answers this with `lo`, which is correct there and
# only there: its ranks are processes in one network namespace. A multi-node
# group needs the interface that actually carries traffic to its peers, which is
# derived here rather than named: these hosts differ in how their bonds are
# numbered, and a wrong name would put the group back on loopback.
#
# The derivation is the address the group already agreed to meet at. Rank 0
# publishes that address, so on rank 0 it is a local address and on every other
# node it is the address whose route selects the interface facing rank 0 --
# either way the kernel's own source-address choice for that destination names
# the interface gloo has to use. Written inline against the standard library,
# like the rendezvous script and for the same reason: this runs before the
# editable install, and it must not depend on the tree under test.
#
# Prints one interface name on stdout. Exits non-zero, with the reason and the
# interface inventory on stderr, if the address belongs to no interface -- a
# silent fallback to loopback costs a whole multi-hour run, so it is refused.
#
# Usage: answer_gloo_iface.sh <host:port>
set -u

address="${1:-${SGLANG_PPU_ANSWER_DIST_INIT_ADDR:-}}"
if [ -z "$address" ]; then
  echo "gloo: usage: answer_gloo_iface.sh <host:port>" >&2
  exit 2
fi

ANSWER_GLOO_TARGET="$address" python3 - <<'PYEOF'
import fcntl
import os
import socket
import struct
import sys

SIOCGIFADDR = 0x8915

target = os.environ["ANSWER_GLOO_TARGET"]
# Split at the last colon, so a bracketed IPv6 literal keeps its own.
host, _, port = target.rpartition(":")
host = host.strip("[]")
if not host:
    print(f"gloo: {target!r} is not a <host>:<port> address", file=sys.stderr)
    raise SystemExit(1)
try:
    port = int(port)
except ValueError:
    port = 29500


def local_address_towards(destination, destination_port):
    """The source address the kernel would use to reach destination."""

    # A UDP connect sends nothing; it only selects a source address. The same
    # probe the rendezvous script uses, aimed at the peer rather than a public
    # address, so a host with several interfaces yields the one facing the peer.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((destination, destination_port))
        return sock.getsockname()[0]


def interface_addresses():
    """Every interface with an IPv4 address, as {name: address}."""

    found = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, name in socket.if_nameindex():
            try:
                packed = fcntl.ioctl(
                    sock.fileno(),
                    SIOCGIFADDR,
                    struct.pack("256s", name.encode("utf-8")[:15]),
                )
            except OSError:
                # An interface with no IPv4 address; not a candidate.
                continue
            found[name] = socket.inet_ntoa(packed[20:24])
    return found


try:
    source = local_address_towards(host, port)
except OSError as error:
    print(f"gloo: no route from this node to {host}: {error}", file=sys.stderr)
    raise SystemExit(1)

addresses = interface_addresses()
for name, address in addresses.items():
    if address == source and name != "lo":
        print(f"gloo: binding to {name} ({address}), the interface towards {host}", file=sys.stderr)
        print(name)
        raise SystemExit(0)

inventory = ", ".join(f"{name}={address}" for name, address in sorted(addresses.items()))
print(
    f"gloo: the source address towards {host} is {source}, which belongs to no "
    f"non-loopback interface here; refusing to let gloo fall back to loopback. "
    f"Interfaces: {inventory or '(none with an IPv4 address)'}",
    file=sys.stderr,
)
raise SystemExit(1)
PYEOF
