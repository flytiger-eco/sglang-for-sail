#!/bin/bash
# Point tvm-ffi's JIT build directory at the pod's own filesystem, seeded from
# the shared cache, and drop any deps log in that seed that ninja cannot load.
# Sourced, not executed, because the export has to reach the process that will
# compile.
#
# The failure this exists for is `ninja exited with status 245` -- the pip ninja
# wrapper reporting SIGSEGV as SystemExit(256-11) -- with stdout and stderr both
# empty, in a build directory that already held a valid .so and gained no new
# .ninja_log record. ninja segfaulted at startup, before any build edge ran, and
# had nothing to compile.
#
# The cause is a torn .ninja_deps in the shared cache, not where ninja runs. The
# file for silu_and_mul_post_quant_mxfp4 at block_n 256 ends 36 bytes short of a
# complete record, with 1655 bytes of garbage after the last good one; ninja
# segfaults loading it, reproducibly, even under `ninja -n` on a local disk, and
# it is the only damaged log of the 35 in that cache. That is also the whole
# pattern of failures: every mxfp4 lane needs that module and died (Qwen3.8,
# Kimi-K2.6, MiniMax-M2.7, four different nodes), no fp8 lane touches it and
# none died. Moving the build off the shared mount alone does not help, because
# cp -a copies the damage faithfully -- run 34207512571 proved that, failing on
# a local disk -- so the seed is pruned before anything compiles.
#
# Keeping the build local is still worth doing, and is why the damage cannot be
# re-created here: the shared cache is a hostPath onto an Alibaba Cloud Extreme
# NAS export mounted `vers=3,nolock,local_lock=all`, and nolock there is not a
# tuning choice. rpcinfo against the export's server (11.161.52.65) lists
# portmapper, status, nfs v3 and mountd v3 and no nlockmgr at all, so no mount
# option can give the FileLock tvm-ffi takes any reach beyond one host, and two
# nodes appending to one deps log is how a record gets torn in the first place.
#
# cp -a carries the mtimes the up-to-date checks compare. The seed is
# best-effort: compiling what did not arrive is slower but correct, and a cache
# that cannot be read is not a reason to fail a measurement. Nothing is copied
# back, so a module compiled here is compiled again next run -- that is the
# price of keeping every ninja invocation off the NAS, and it also means this
# pruning never writes to the shared cache, which other lines are reading.
#
# One qualification on that isolation: a seeded build.ninja names its source by
# absolute path under the shared cache, so unless tvm-ffi rewrites the file, a
# seeded module still reads its generated source from the mount. Only reads --
# every output, the deps log included, is written relative to the build
# directory, which is local, and a torn record is a write.
#
# Usage: source scripts/ci/ppu/use_local_jit_cache.sh
PPU_JIT_CACHE_SHARED="${PPU_JIT_CACHE_SHARED:-/root/.cache/tvm-ffi}"
PPU_JIT_CACHE_LOCAL="${PPU_JIT_CACHE_LOCAL:-/root/.cache/tvm-ffi-local}"

mkdir -p "$PPU_JIT_CACHE_LOCAL"
if [ -d "$PPU_JIT_CACHE_SHARED" ]; then
  cp -a "$PPU_JIT_CACHE_SHARED"/. "$PPU_JIT_CACHE_LOCAL"/ ||
    echo 'Shared JIT cache seed incomplete; this run compiles what is missing'
fi
export TVM_FFI_CACHE_DIR="$PPU_JIT_CACHE_LOCAL"
PPU_JIT_CACHE_ENTRIES=$(find "$PPU_JIT_CACHE_LOCAL" -mindepth 1 -maxdepth 1 | wc -l)
echo "JIT cache:       $TVM_FFI_CACHE_DIR seeded with $PPU_JIT_CACHE_ENTRIES entries"
# Located next to this file rather than relative to the caller's directory, so
# that sourcing it from a workflow step and from the board driver both work.
python3 "$(dirname "${BASH_SOURCE[0]}")/prune_broken_ninja_deps.py" "$PPU_JIT_CACHE_LOCAL" ||
  echo 'JIT cache:       deps-log check did not complete; ninja may still crash'
