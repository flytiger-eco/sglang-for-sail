"""Custom setuptools_scm version/local scheme for sglang packaging.

Builds the package version from the current git branch name and commit id
(e.g. 0.5.15+abcd123), without relying on git tags.

The version_scheme extracts a numeric version from the branch name when it
starts with a version-like string (e.g. v0.5.15 -> 0.5.15); otherwise it
falls back to the tag version.  The local_scheme appends the commit id.

Both functions first try the ScmVersion attributes (branch, node) that
setuptools_scm may have already populated.  If those are None (which
happens when setuptools_scm couldn't run git itself, e.g. due to CWD
issues), we fall back to running git directly in the repository root
(obtained from version.config.root) to avoid working-directory problems.
"""

import re
import subprocess
from typing import Any


def _get_repo_root(version: Any) -> str | None:
    """Return the repository root from the ScmVersion config, if available."""
    config = getattr(version, "config", None)
    if config is not None:
        root = getattr(config, "root", None)
        if root is not None:
            return str(root)
    return None


def _run_git(version: Any, *args: str) -> str:
    """Run git in the repository root to avoid CWD issues during build."""
    cwd = _get_repo_root(version)
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def version_scheme(version: Any) -> str:
    """Derive the version number from the git branch name.

    If the branch name starts with a version-like string (e.g. v0.5.15),
    extract the numeric version (0.5.15).  Otherwise fall back to the
    tag version that setuptools_scm already resolved.
    """
    branch = getattr(version, "branch", None) or ""
    if not branch:
        branch = _run_git(version, "rev-parse", "--abbrev-ref", "HEAD")
    m = re.match(r"^v?(\d+\.\d+\.\d+)", branch)
    if m:
        return m.group(1)
    # Fall back to the tag version (e.g. "0.5.17" from v0.5.17, or "0.0.0"
    # from the synthesized describe string when no tags exist).
    return str(version.tag)


def local_scheme(version: Any) -> str:
    """Use the commit id as the local version identifier.

    Tries version.node first (setuptools_scm may have already collected it).
    If that is None, runs git in the repo root to get the short commit hash.
    Strips the leading 'g' prefix (from git describe format) and truncates
    to 7 characters to produce a clean short hash.
    """
    node = getattr(version, "node", None) or ""
    if not node:
        node = _run_git(version, "rev-parse", "--short=7", "HEAD")
    if node:
        # Strip "g" prefix from git describe format (e.g. "g7e44c9b" -> "7e44c9b")
        if node.startswith("g"):
            node = node[1:]
        # Truncate to 7-character short hash
        node = node[:7]
        return f"+{node}"
    return ""
