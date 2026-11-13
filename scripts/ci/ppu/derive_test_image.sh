#!/usr/bin/env bash
# Name the image the nightly build pushed, and the date variants a re-run can
# produce.
#
# The tag rule has to match Compute image metadata in release-docker-ppu-nightly.yml
# character for character: lowercased owner, / turned into - in the branch, UTC
# date, first 7 of the sha. Whichever file owns the rule, the other has to follow,
# which is why it lives here once and prepare calls it.
#
# Several candidates come out because the date is the one part that can differ.
# The build stamps `date -u +%Y%m%d` when it executes, not when the run was
# created, and a re-run keeps created_at -- so a re-run that lands after 00:00 UTC
# on the following day names a tag neither of the first two dates can guess.
#
# Reads:
#   PPU_DERIVE_TAG_INPUT   an image the caller named; when set it is used verbatim
#                          and no build run is consulted
#   PPU_DERIVE_SHA         head_sha of the build run; empty means "no build behind
#                          this trigger", which keeps the source path
#   PPU_DERIVE_BRANCH      head_branch of the build run
#   PPU_DERIVE_STARTED     run_started_at of the build run (falls back to CREATED)
#   PPU_DERIVE_CREATED     created_at of the build run
#   PPU_DERIVE_OWNER       repository owner, lowercased here
#   PPU_DERIVE_IMAGE       image path, default ghcr.io/<owner>/sglang-ppu-nightly
#   PPU_DERIVE_SUFFIX      tag suffix, default cuda13.0.0-sdk2.2.0
#   PPU_DERIVE_OUTPUT      file appended to, default $GITHUB_OUTPUT
#
# Writes:
#   tag=<first candidate, or the caller's image verbatim>
#   candidates=<space separated, in the order they should be tried>
#
# Usage: bash scripts/ci/ppu/derive_test_image.sh
set -uo pipefail

sha="${PPU_DERIVE_SHA:-}"
branch="${PPU_DERIVE_BRANCH:-}"
started="${PPU_DERIVE_STARTED:-}"
created="${PPU_DERIVE_CREATED:-}"
owner="${PPU_DERIVE_OWNER:-}"
image="${PPU_DERIVE_IMAGE:-}"
tag_input="${PPU_DERIVE_TAG_INPUT:-}"
suffix="${PPU_DERIVE_SUFFIX:-cuda13.0.0-sdk2.2.0}"
out="${PPU_DERIVE_OUTPUT:-${GITHUB_OUTPUT:-/dev/stdout}}"

write() {
  printf '%s\n' "$1" >>"${out}"
}

if [ -n "${tag_input}" ]; then
  echo "image under test, as given: ${tag_input}"
  write "tag=${tag_input}"
  write "candidates=${tag_input}"
  exit 0
fi

if [ -z "${sha}" ]; then
  # A manual dispatch with no build behind it: keep the fixed base image and the
  # source install.
  echo 'no build run behind this trigger; keeping the base image and the source install'
  write 'tag='
  write 'candidates='
  exit 0
fi

if [ -z "${owner}" ] || [ -z "${branch}" ]; then
  echo '::error::PPU_DERIVE_OWNER and PPU_DERIVE_BRANCH are required when a build sha is given' >&2
  exit 1
fi

owner_lc=$(printf '%s' "${owner}" | tr '[:upper:]' '[:lower:]')
# Docker tags forbid "/" and some git-allowed chars, so sanitize the branch name
# (e.g. feat/foo -> feat-foo) exactly as the build does.
branch_slug=$(printf '%s' "${branch}" | sed 's|/|-|g; s/[^A-Za-z0-9._-]/-/g')
short_sha="${sha:0:7}"
if [ -z "${image}" ]; then
  image="ghcr.io/${owner_lc}/sglang-ppu-nightly"
fi

# Anything that is not a UTC timestamp is dropped rather than fed through: GitHub
# renders an unset run_started_at as the string "null", and cutting that down would
# produce a tag with "null" where the date belongs.
day_of() {
  case "$1" in
    [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*)
      printf '%s' "$1" | cut -c1-10 | tr -d '-'
      ;;
  esac
}

started_day="$(day_of "${started:-${created}}")"
created_day="$(day_of "${created}")"

next_day=''
# GNU date first, since the runners are Linux. BSD date is tried second so a
# contributor can run this on a mac, and it takes -v before -f. The output is
# checked rather than the exit status: BSD date reads -d as its own flag, exits 0
# and prints a full date in the local locale, which an `||` fallback never sees.
next_day_from() {
  case "$1" in
    [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]) printf '%s' "$1" ;;
  esac
}
if [ -n "${created_day}" ]; then
  next_day="$(next_day_from "$(date -u -d "${created_day} + 1 day" +%Y%m%d 2>/dev/null || true)")"
  if [ -z "${next_day}" ]; then
    next_day="$(next_day_from "$(date -u -v+1d -j -f '%Y%m%d' "${created_day}" +%Y%m%d 2>/dev/null || true)")"
  fi
fi

# Tried in this order: the default branch can stand still for several nights, so
# the same commit gets rebuilt under more than one date tag and only the first
# candidate belongs to this build.
candidates=''
for day in "${started_day}" "${created_day}" "${next_day}"; do
  [ -n "${day}" ] || continue
  candidate="${image}:${branch_slug}-${day}-${suffix}-${short_sha}"
  case " ${candidates} " in
    *" ${candidate} "*) continue ;;
  esac
  candidates="${candidates:+${candidates} }${candidate}"
done

if [ -z "${candidates}" ]; then
  echo '::error::no date could be read off the build run, so no tag can be named' >&2
  exit 1
fi

echo "candidate images under test: ${candidates}"
write "tag=${candidates%% *}"
write "candidates=${candidates}"
