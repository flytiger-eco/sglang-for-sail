#!/usr/bin/env bash
# Everything a rehearsal can check without submitting a pod.
#
# This is the "static" half of .github/workflows/nightly-ppu-rehearsal.yml. It is
# a file rather than inline YAML so the same assertions run on a laptop: the parts
# of the change that used to be validated by hand (date candidates, digest
# handling, the health check in source mode) become falsifiable.
#
# Exits non-zero if any assertion fails. Prints one line per assertion.
#
# Usage: bash scripts/ci/ppu/rehearsal_preflight.sh
set -uo pipefail

repo_root="${GITHUB_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${repo_root}" || exit 1

failures=0
fail() {
  echo "::error::[preflight] $*"
  failures=$((failures + 1))
}
pass() {
  echo "[preflight] ok   $*"
}

# ---------------------------------------------------------------- syntax ---- #
for wf in nightly-test-ppu nightly-ppu-issue nightly-ppu-rehearsal release-docker-ppu-nightly; do
  if python3 -c 'import sys, yaml; yaml.safe_load(open(sys.argv[1]))' \
    ".github/workflows/${wf}.yml" 2>/dev/null; then
    pass "parse .github/workflows/${wf}.yml"
  else
    fail "parse .github/workflows/${wf}.yml"
  fi
done

for sh in scripts/ci/ppu/*.sh; do
  if bash -n "${sh}"; then
    pass "bash -n ${sh}"
  else
    fail "bash -n ${sh}"
  fi
done

if python3 -m py_compile python/sglang/test/kits/answer_eval_kit.py 2>/dev/null; then
  pass "py_compile answer_eval_kit.py"
else
  fail "py_compile answer_eval_kit.py"
fi

# ------------------------------------------------------- derive candidates -- #
# Table driven, because the whole point of the derivation is the cases nobody
# types by hand: a re-run that crossed midnight, a branch with a slash, a commit
# the default branch has not moved off for several nights.
DERIVE="scripts/ci/ppu/derive_test_image.sh"
SHA=5dd9dbd2fed34c3a4c1f8bb32d0e6ff9a2b7c111
IMAGE=ghcr.io/flytiger-eco/sglang-ppu-nightly
SUFFIX=cuda13.0.0-sdk2.2.0

check_derive() {
  local name="$1" expected="$2"
  shift 2
  local out actual
  out="$(mktemp)"
  if ! env "$@" PPU_DERIVE_OUTPUT="${out}" bash "${DERIVE}" >/dev/null 2>&1; then
    fail "derive/${name}: exited non-zero"
    rm -f "${out}"
    return
  fi
  actual="$(sed -n 's/^candidates=//p' "${out}")"
  if [ "${actual}" = "${expected}" ]; then
    pass "derive/${name}"
  else
    fail "derive/${name}: got '${actual}' want '${expected}'"
  fi
  rm -f "${out}"
}

# Started and created on the same UTC day: the next day is still tried, because
# the build stamps execution time and the re-run may land on the other side.
check_derive same-day \
  "${IMAGE}:v0.5.18-20261110-${SUFFIX}-5dd9dbd ${IMAGE}:v0.5.18-20261111-${SUFFIX}-5dd9dbd" \
  PPU_DERIVE_SHA="${SHA}" PPU_DERIVE_BRANCH=v0.5.18 \
  PPU_DERIVE_STARTED=2026-11-10T05:00:03Z PPU_DERIVE_CREATED=2026-11-10T04:59:58Z \
  PPU_DERIVE_OWNER=Flytiger-Eco

# A re-run that started on the 11th for a run created on the 10th: the execution
# date comes first, the created date second, and the 12th collapses into the 11th.
check_derive rerun-next-day \
  "${IMAGE}:v0.5.18-20261111-${SUFFIX}-5dd9dbd ${IMAGE}:v0.5.18-20261110-${SUFFIX}-5dd9dbd" \
  PPU_DERIVE_SHA="${SHA}" PPU_DERIVE_BRANCH=v0.5.18 \
  PPU_DERIVE_STARTED=2026-11-11T00:30:00Z PPU_DERIVE_CREATED=2026-11-10T04:59:58Z \
  PPU_DERIVE_OWNER=Flytiger-Eco

# A calendar rollover: created on the 31st, re-run on the 1st.
check_derive month-rollover \
  "${IMAGE}:v0.5.18-20261101-${SUFFIX}-5dd9dbd ${IMAGE}:v0.5.18-20261031-${SUFFIX}-5dd9dbd" \
  PPU_DERIVE_SHA="${SHA}" PPU_DERIVE_BRANCH=v0.5.18 \
  PPU_DERIVE_STARTED=2026-11-01T00:30:00Z PPU_DERIVE_CREATED=2026-10-31T04:59:58Z \
  PPU_DERIVE_OWNER=Flytiger-Eco

# A branch with a slash has to be sanitized the way the build sanitizes it.
check_derive branch-slash \
  "${IMAGE}:ci-foo-20261110-${SUFFIX}-5dd9dbd ${IMAGE}:ci-foo-20261111-${SUFFIX}-5dd9dbd" \
  PPU_DERIVE_SHA="${SHA}" PPU_DERIVE_BRANCH=ci/foo \
  PPU_DERIVE_STARTED=2026-11-10T05:00:03Z PPU_DERIVE_CREATED=2026-11-10T04:59:58Z \
  PPU_DERIVE_OWNER=Flytiger-Eco

# GitHub renders an unset run_started_at as the string "null". Cutting that down
# would name a tag ...-null-..., so it has to be dropped and the created date used.
check_derive null-started \
  "${IMAGE}:v0.5.18-20261110-${SUFFIX}-5dd9dbd ${IMAGE}:v0.5.18-20261111-${SUFFIX}-5dd9dbd" \
  PPU_DERIVE_SHA="${SHA}" PPU_DERIVE_BRANCH=v0.5.18 \
  PPU_DERIVE_STARTED=null PPU_DERIVE_CREATED=2026-11-10T04:59:58Z \
  PPU_DERIVE_OWNER=Flytiger-Eco

# An image the caller named is used verbatim, digest and all.
check_derive caller-given \
  "ghcr.io/x/y@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" \
  PPU_DERIVE_TAG_INPUT=ghcr.io/x/y@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  PPU_DERIVE_SHA="${SHA}" PPU_DERIVE_BRANCH=v0.5.18 PPU_DERIVE_OWNER=Flytiger-Eco

# No build behind the trigger: the source path, with empty outputs.
check_derive no-build '' PPU_DERIVE_OWNER=Flytiger-Eco

# A sha with no readable date anywhere cannot be named, and silently emitting a
# tag with an empty part would send the pods after an image nobody built.
if PPU_DERIVE_SHA="${SHA}" PPU_DERIVE_BRANCH=v0.5.18 PPU_DERIVE_OWNER=o \
  PPU_DERIVE_STARTED=null PPU_DERIVE_CREATED=null PPU_DERIVE_OUTPUT=/dev/null \
  bash "${DERIVE}" >/dev/null 2>&1; then
  fail 'derive/no-date: exited 0, expected 1'
else
  pass 'derive/no-date'
fi

# A build run with no branch or owner is a bug in the caller, not a quiet source
# run, so it has to fail rather than emit a tag with an empty part.
if PPU_DERIVE_SHA="${SHA}" PPU_DERIVE_OUTPUT=/dev/null bash "${DERIVE}" >/dev/null 2>&1; then
  fail 'derive/missing-owner: exited 0, expected 1'
else
  pass 'derive/missing-owner'
fi

# --------------------------------------------------------- digest pinning -- #
RESOLVE="scripts/ci/ppu/resolve_image_digest.sh"
DIGEST=sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

check_resolve() {
  local name="$1" want_exit="$2"
  shift 2
  local envfile out
  envfile="$(mktemp)"
  out="$(mktemp)"
  env "$@" GITHUB_ENV="${envfile}" bash "${RESOLVE}" >"${out}" 2>&1
  local got=$?
  if [ "${got}" != "${want_exit}" ]; then
    fail "resolve/${name}: exit ${got}, want ${want_exit}: $(tail -2 "${out}" | tr '\n' ' ')"
  elif [ "${want_exit}" = 1 ] && [ -s "${envfile}" ]; then
    # A strict failure must not leave half a pin behind: an empty REF written by a
    # give-up would happily override the caller's image in later steps.
    fail "resolve/${name}: failed but still wrote $(tr '\n' ' ' <"${envfile}")"
  else
    pass "resolve/${name}"
  fi
  rm -f "${envfile}" "${out}"
}

check_resolve digest-passthrough 0 \
  PPU_RESOLVE_INPUT="ghcr.io/x/y@${DIGEST}" PPU_RESOLVE_PREFIX=PPU_TEST_IMAGE PPU_RESOLVE_STRICT=1
check_resolve malformed-digest-lenient 0 \
  PPU_RESOLVE_INPUT='ghcr.io/x/y@sha256:abc' PPU_RESOLVE_PREFIX=PPU_TEST_IMAGE PPU_RESOLVE_STRICT=0
check_resolve malformed-digest-strict 1 \
  PPU_RESOLVE_INPUT='ghcr.io/x/y@sha256:abc' PPU_RESOLVE_PREFIX=PPU_TEST_IMAGE PPU_RESOLVE_STRICT=1
# A tag that cannot exist: strict has to fail without writing anything.
check_resolve missing-tag-strict 1 \
  PPU_RESOLVE_INPUT='ghcr.io/x/y:nope-00000000-nope' PPU_RESOLVE_PREFIX=PPU_TEST_IMAGE PPU_RESOLVE_STRICT=1

# ----------------------------------------------------------- health check -- #
# Source mode is the path every run took before this change; it has to stay a
# no-op so the source nightly is not affected by any of the above.
if PPU_TEST_MODE=source bash scripts/ci/ppu/image_healthcheck.sh "$(mktemp -d)" >/dev/null 2>&1; then
  pass 'healthcheck/source-mode-is-a-noop'
else
  fail 'healthcheck/source-mode-is-a-noop'
fi

# ------------------------------------------------------- issue tracker ------ #
# The finish-job lookup. Kept as a table so the three shapes are pinned: absent
# (no such job), null (the job exists but never ran) and success.
finish_query='[.jobs[] | select(.name == "nightly-test-ppu-finish")]
  | if length == 0 then "absent" else (.[0].conclusion // "null") end'

check_finish() {
  local name="$1" payload="$2" want="$3" got
  got="$(printf '%s' "${payload}" | jq -r "${finish_query}" 2>/dev/null)"
  if [ "${got}" = "${want}" ]; then
    pass "issue-tracker/${name}"
  else
    fail "issue-tracker/${name}: got '${got}' want '${want}'"
  fi
}

check_finish absent '{"jobs":[{"name":"prepare","conclusion":"skipped"}]}' absent
check_finish skipped '{"jobs":[{"name":"nightly-test-ppu-finish","conclusion":null}]}' null
check_finish green '{"jobs":[{"name":"nightly-test-ppu-finish","conclusion":"success"}]}' success

# ---------------------------------------------------------------- verdict -- #
echo
if [ "${failures}" -eq 0 ]; then
  echo "[preflight] all assertions passed"
  exit 0
fi
echo "::error::[preflight] ${failures} assertion(s) failed"
exit 1
