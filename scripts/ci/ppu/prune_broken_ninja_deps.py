#!/usr/bin/env python3
"""Delete .ninja_deps files that ninja cannot load without crashing.

A .ninja_deps that ends mid-record makes ninja segfault as it starts, before it
runs a single build edge, and the pip ninja wrapper reports that as `ninja
exited with status 245`. Nothing downstream recovers from it: the build
directory can hold a perfectly good .so and every later run still dies there,
because loading the deps log happens first. Run 34207512571 hit it on the one
module every mxfp4 lane needs, and the same file had already been served to
four earlier failures.

Deleting the file is the whole repair. ninja treats a target built with
`deps = gcc` and no deps-log entry as dirty, so the module is compiled once more
and the log is written again from scratch -- seconds per module, against a lane
that otherwise cannot start.

Only structural damage is looked for, and the parse deliberately mirrors
ninja's own reader (DepsLog::Load) rather than guessing: the header and version,
then records whose 4-byte header carries the payload size in its low 31 bits and
the record type in the top bit. A record that claims more bytes than the file
has left, or claims none at all, is damage ninja would walk off the end of.
Anything that parses cleanly is left alone, because a valid log is exactly what
makes the cache worth seeding.

Usage: prune_broken_ninja_deps.py <cache_dir>
Prints one line per file removed and exits 0 whether or not it removed any; a
cache that cannot be repaired is still a cache worth compiling into.
"""

import os
import struct
import sys

DEPS_HEADER = b"# ninjadeps\n"


def first_fault(path):
    """Return a description of the first structural fault, or None if clean."""
    with open(path, "rb") as handle:
        data = handle.read()

    if not data.startswith(DEPS_HEADER):
        return "missing ninjadeps header"
    offset = len(DEPS_HEADER)
    if offset + 4 > len(data):
        return "truncated before version"
    offset += 4

    while offset + 4 <= len(data):
        (header,) = struct.unpack("<I", data[offset : offset + 4])
        offset += 4
        size = header & 0x7FFFFFFF
        if size == 0:
            return "zero-length record at byte %d" % (offset - 4)
        if offset + size > len(data):
            return "record at byte %d wants %d bytes, %d left" % (
                offset - 4,
                size,
                len(data) - offset,
            )
        # A path record is a name, 0xff padding to a 4-byte boundary, and a
        # 4-byte checksum; one shorter than the checksum cannot be a record.
        if not header >> 31 and size < 4:
            return "path record at byte %d is %d bytes" % (offset - 4, size)
        offset += size

    # ninja appends whole records, so a tail too short to be one is a torn write.
    if offset != len(data):
        return "%d trailing bytes after the last record" % (len(data) - offset)
    return None


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("usage: prune_broken_ninja_deps.py <cache_dir>\n")
        return 2
    cache_dir = sys.argv[1]
    if not os.path.isdir(cache_dir):
        return 0

    for entry in sorted(os.listdir(cache_dir)):
        path = os.path.join(cache_dir, entry, ".ninja_deps")
        if not os.path.isfile(path):
            continue
        try:
            fault = first_fault(path)
        except OSError as error:
            # Unreadable is not the same as damaged; leave it to ninja to report.
            print(
                "JIT cache:       could not read %s/.ninja_deps (%s)" % (entry, error)
            )
            continue
        if fault is None:
            continue
        try:
            os.remove(path)
        except OSError as error:
            print(
                "JIT cache:       %s/.ninja_deps is damaged (%s) and could not be "
                "removed (%s)" % (entry, fault, error)
            )
            continue
        print(
            "JIT cache:       dropped %s/.ninja_deps (%s); it will be rebuilt"
            % (entry, fault)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
