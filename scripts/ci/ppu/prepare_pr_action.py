"""Constrain the pinned upstream worker action's default NAS mounts for PRs.

The upstream action has no read-only mount input. Apply one checked patch to its
local checkout, never to the upstream repository or any nightly invocation.
Custom host_volumes remain writable and must name only PR-scoped destinations.
"""

import subprocess
import sys
from pathlib import Path

REVISION = "67ef781e31924dc04b31cc01b7c537a797add97e"
MOUNT_LINE = "  DEFAULT_NAS_VOLUME_MOUNTS+=\"          mountPath: ${_mpath}\"$'\\n'"


def constrain(source):
    if source.count(MOUNT_LINE) != 1:
        raise ValueError("Pinned worker mount contract changed; refusing writable NAS")
    for mount in (
        "nas-aisw|/nas_aisw|/nas_aisw",
        "wl-nas|/mnt/wl_nas|/wl_nas",
        "wl-nas-direct|/wl_nas|/wl_nas",
    ):
        if mount not in source:
            raise ValueError(f"Missing expected default mount: {mount}")
    return source.replace(
        MOUNT_LINE,
        MOUNT_LINE
        + "\n  DEFAULT_NAS_VOLUME_MOUNTS+=\"          readOnly: true\"$'\\n'",
    )


SUBMIT_LINE = '      run: bash "$GITHUB_ACTION_PATH/scripts/submit.sh"'


def observe_before_submit(source):
    if source.count(SUBMIT_LINE) != 1:
        raise ValueError("Pinned worker submit contract changed")
    return source.replace(
        SUBMIT_LINE,
        '''      run: |
        export PPU_ENV_JOB_NAME="$JOB_NAME" PPU_ENV_PR_TIMINGS=true
        export PPU_ENV_OBSERVER_LIMIT="$((INPUT_TIMEOUT_MINUTES * 60))"
        bash "$GITHUB_WORKSPACE/scripts/ci/ppu/observe_pod_environment.sh" start \\
          "/wl_nas/pr-ci/evidence/$GITHUB_RUN_ID/$GITHUB_RUN_ATTEMPT/$GITHUB_JOB_NAME/environment-pods.tsv"
        bash "$GITHUB_ACTION_PATH/scripts/submit.sh"''',
    )


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    revision = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError("Unexpected worker action revision")
    target = directory / "scripts/submit.sh"
    action = directory / "action.yml"
    constrained = constrain(target.read_text())
    observed = observe_before_submit(action.read_text())
    target.write_text(constrained)
    action.write_text(observed)
