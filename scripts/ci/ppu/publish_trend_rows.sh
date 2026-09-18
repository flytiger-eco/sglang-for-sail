#!/bin/bash
# Append this run's performance measurements to the nightly-test-data branch.
#
# Run once per workflow, after every measuring job of that workflow has finished,
# whatever each of them concluded: a night that could not measure is itself a
# point in the series, and dropping it would leave a gap indistinguishable from a
# night on which nothing was scheduled.
#
# The rows are not derived here. Each measuring job's artifact already carries the
# trend.jsonl that write_report_files wrote next to its report, so this script
# only files those rows on the data branch -- it never reinterprets a number.
#
# Rows are filed under the test_id they carry rather than under a name parsed out
# of the artifact, because test_id is part of the series key: partitioning by it
# is the same grouping a reader of the series has to do anyway.
#
# One file per (test_id, run, attempt), never appended to once written. Several of
# these workflows finish on the same night, so the branch has more than one writer
# and the push below retries; distinct paths keep that retry from ever having to
# merge two writers' content.
#
# Run from a checkout of the data branch, with this script reached through a
# second checkout of the source: the data branch carries no scripts of its own,
# and giving it a copy of these would make it something that has to be kept in
# step with the code, which is exactly what an orphan data branch is for avoiding.
#
# Reads:
#   INCOMING_DIR       directory the run's artifacts were downloaded into
#   TREND_DATA_BRANCH  branch to file the rows on
#   GITHUB_RUN_ID      \
#   GITHUB_RUN_ATTEMPT  > what makes a filename unique, and what attributes it
#   GITHUB_WORKFLOW    /
#
# Usage: bash <source checkout>/scripts/ci/ppu/publish_trend_rows.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCOMING_DIR="${INCOMING_DIR:-incoming}"
TREND_DATA_BRANCH="${TREND_DATA_BRANCH:-nightly-test-data}"

if [ ! -e .git ]; then
  echo "::error::run this from a checkout of ${TREND_DATA_BRANCH}, not from ${PWD}" >&2
  exit 1
fi

if [ ! -d "${INCOMING_DIR}" ]; then
  echo "::warning::no artifacts were downloaded, so there are no rows to file"
  exit 0
fi

# Staged by a Python pass rather than by find plus cp: the rows have to be
# grouped by the test_id inside them, and a malformed row must stop the run
# before it reaches the branch, because a series is only as good as the worst
# line anyone ever appended to it.
python3 "${HERE}/stage_trend_rows.py"

if [ -z "$(git status --porcelain data 2>/dev/null)" ]; then
  echo "nothing new to file on ${TREND_DATA_BRANCH}"
  exit 0
fi

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add data
git commit --quiet --message "data(perf): ${GITHUB_WORKFLOW:-perf} run ${GITHUB_RUN_ID:-?} attempt ${GITHUB_RUN_ATTEMPT:-?}"

# Rebase and retry rather than force: every writer only adds paths no other
# writer uses, so a rebase onto whoever won the race replays cleanly, and a force
# would discard their rows. Bounded, with a jittered wait, so that a genuinely
# broken push fails the step instead of spinning.
for attempt in 1 2 3 4 5 6; do
  if git push --quiet origin "HEAD:${TREND_DATA_BRANCH}"; then
    echo "filed on ${TREND_DATA_BRANCH} at $(git rev-parse --short HEAD)"
    exit 0
  fi
  echo "push ${attempt} lost the race for ${TREND_DATA_BRANCH}; rebasing"
  sleep $(( attempt * 5 + RANDOM % 10 ))
  git fetch --quiet origin "${TREND_DATA_BRANCH}"
  git rebase --quiet "origin/${TREND_DATA_BRANCH}"
done

echo "::error::could not file the rows on ${TREND_DATA_BRANCH} after six attempts"
exit 1
