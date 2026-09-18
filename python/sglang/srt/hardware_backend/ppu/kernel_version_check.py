"""PPU sgl-kernel version guard.

On PPU we deliberately run the vendor's ``sglang-kernel`` fork -- its
distribution version carries a ``+...ppu...`` local segment (observed:
``0.4.3+v0.1.0.ppu2.1.1``) -- rather than the upstream CUDA wheel that
``python/pyproject.toml`` pins (``sglang-kernel==0.4.6.post1``).

The upstream startup assertion in ``entrypoints/engine.py`` is ``if _is_cuda:``
gated, so it never fires on PPU; and if it did it would *wrongly* reject the
PPU fork, because ``assert_pkg_version`` is a minimum-version check and
``0.4.3`` < ``0.4.6.post1``. The 0.4.3-vs-0.4.6 gap is the intended fork
divergence, not a fault.

This guard is the PPU-appropriate form of that check. Instead of pinning the
upstream version, it asserts the installed kernel *is* a PPU build (fork
marker present) and meets a minimum base version. The real failure it
surfaces is an image that silently ships the wrong kernel -- a stray upstream
CUDA wheel, or a PPU build older than what this source tree expects -- which
would otherwise run with mismatched ops and no signal.

Default behavior is a loud WARNING (non-fatal). Set
``SGLANG_PPU_KERNEL_VERSION_STRICT=1`` to raise instead.
"""

import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

from packaging import version as pkg_version

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# The installed sglang-kernel must be a PPU build (its local version segment
# carries this marker) and at least this base version. Bump _MIN_BASE_VERSION
# only when this source tree starts to require a newer PPU kernel op.
_PPU_FORK_MARKER = "ppu"
_MIN_BASE_VERSION = "0.4.3"
# importlib.metadata normalizes hyphen/underscore, but query both to be safe.
_DIST_NAMES = ("sglang-kernel", "sglang_kernel")


def _installed_kernel_version() -> Optional[str]:
    for name in _DIST_NAMES:
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return None


def check_ppu_kernel_version() -> None:
    """Verify the installed sglang-kernel is the expected PPU build.

    Logs a WARNING on mismatch by default; raises ``RuntimeError`` when
    ``SGLANG_PPU_KERNEL_VERSION_STRICT`` is set.
    """
    strict = envs.SGLANG_PPU_KERNEL_VERSION_STRICT.get()
    expected = (
        f"a PPU build (version contains '{_PPU_FORK_MARKER}', "
        f"base >= {_MIN_BASE_VERSION})"
    )

    def _report(msg: str) -> None:
        if strict:
            raise RuntimeError(msg)
        logger.warning(
            "%s (set SGLANG_PPU_KERNEL_VERSION_STRICT=1 to make this fatal)", msg
        )

    installed = _installed_kernel_version()
    if installed is None:
        _report(
            "PPU kernel version check: sglang-kernel is not installed; "
            f"expected {expected}"
        )
        return

    if _PPU_FORK_MARKER not in installed.lower():
        _report(
            f"PPU kernel version check: installed sglang-kernel {installed!r} is "
            f"not a PPU build; expected {expected}. A non-PPU (upstream CUDA) "
            "kernel does not match the PPU op set."
        )
        return

    base = installed.split("+", 1)[0]
    try:
        too_old = pkg_version.parse(base) < pkg_version.parse(_MIN_BASE_VERSION)
    except pkg_version.InvalidVersion:
        logger.warning(
            "PPU kernel version check: could not parse a base version from %r; "
            "skipping the min-version comparison",
            installed,
        )
        return

    if too_old:
        _report(
            f"PPU kernel version check: installed sglang-kernel {installed!r} "
            f"base {base} is older than the required minimum {_MIN_BASE_VERSION}; "
            f"expected {expected}"
        )
        return

    logger.info("PPU kernel version check: sglang-kernel %s OK", installed)
