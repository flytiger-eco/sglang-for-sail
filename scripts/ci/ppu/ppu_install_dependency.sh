#!/bin/bash
set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
PIP_INSTALL="python3 -m pip install --no-cache-dir"

# SAIL SDK v2.1.1 PyPI source (replaces the old art-pub.eng.t-head.cn index)
PPU_PIP_INDEX="https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple"

if [[ -n "${PPU_ARTIFACTORY_USER:-}" && -n "${PPU_ARTIFACTORY_PASSWORD:-}" ]]; then
    echo "machine pkg.flytiger-eco.com login ${PPU_ARTIFACTORY_USER} password ${PPU_ARTIFACTORY_PASSWORD}" > ~/.netrc
    chmod 600 ~/.netrc
fi

git config --global --add safe.directory "${REPO_ROOT}"

# Only pip/wheel. Deliberately NOT setuptools or dill: the v2.1.1 image pins them
# to satisfy constraints that an --upgrade breaks (observed in run 31663500015):
#   setuptools 84.0.0 -> torch 2.11.0 requires setuptools<82
#   dill 0.4.1        -> datasets 3.1.0 requires dill<0.3.9 (image ships 0.3.6)
${PIP_INSTALL} --upgrade pip wheel

# ==================== PPU Dependencies (SAIL SDK v2.1.1) ==================== #
# The v2.1.1 base image already ships the whole PPU stack, and at versions NEWER
# than the v0.5.13 user guide's PyPI section lists. Verified in-image 2026-08-13:
#
#   torch              2.11.0+v0.1.0.ppu2.1.1
#   sglang             0.5.13+v0.1.0.ppu2.1.1
#   sglang-kernel      0.4.3+v0.1.0.ppu2.1.1     <- has the `fwd` op
#   flashinfer-python  0.6.12+v0.1.0.ppu2.1.1
#   triton             3.6.0+v0.2.0.ppu2.1.1
#   apache-tvm-ffi     0.1.9
#   transformers       5.8.1
#
# So we deliberately do NOT reinstall them. Two reasons this matters:
#
#  1. The guide's PyPI list pins flashinfer_python==0.6.8.post1, which would
#     DOWNGRADE the image's 0.6.12 — and contradicts the guide's own stated
#     requirement of flashinfer>=0.6.11.post1 (section 2). The image is right.
#  2. Those wheels carry a +v0.1.0.ppu2.1.1 local version. A bare `==0.4.3`
#     --force-reinstall could pull a differently-built artifact for the same
#     public version. Leaving the image's stack alone avoids the whole question.
#
# Only install what the image genuinely lacks. Verified in-image 2026-08-13 that
# these are already present and need no action:
#   z3-solver     4.13.0.0                  (note: `==4.13.0` would force a reinstall)
#   sglang-router 0.3.2+v0.1.0.ppu2.1.1     (PPU build; the guide's bare ==0.3.2
#                                            could replace it with a generic one)
#
# uvicorn is the one real gap: image ships 0.29.0, sglang's multi-worker path needs
# timeout_worker_healthcheck (added in 0.37.0).
${PIP_INSTALL} /nas_aisw/datasets/packages/uvicorn-0.37.0-py3-none-any.whl --force-reinstall --no-deps

# dill: image ships 0.3.6, but Python 3.12's ABC implementation makes
# _abc._abc_data unpicklable with dill<0.3.8, crashing custom_logit_processor
# tests. Upgrade to 0.3.8 (still satisfies datasets' dill<0.3.9 constraint).
${PIP_INSTALL} "dill>=0.3.8,<0.3.9"

# ==================== Install SGLang from source ==================== #
rm -f "${REPO_ROOT}/python/pyproject.toml"
cp "${REPO_ROOT}/python/pyproject_other.toml" "${REPO_ROOT}/python/pyproject.toml"
# tracing: the v2.1.1 image dropped opentelemetry (v2.1.0 shipped it), and
# all_ppu doesn't pull it in. test_tracing needs it to exercise the OTLP path.
cd "${REPO_ROOT}" && ${PIP_INSTALL} -v -e "python[all_ppu,tracing]" --no-build-isolation

# ==================== sgl-kernel: PR wheel / source build / PyPI ==================== #
# Priority 1: install the PR-built wheel downloaded by the build-sgl-kernel
#   CI job (SGL_KERNEL_WHEEL_DIR points at the artifact directory), so tests
#   exercise kernels built from the PR's own code.
# Priority 2: if this PR touches sgl-kernel source but no wheel is available,
#   rebuild from source. Detect by checking git diff against the previous
#   commit. Force with SGL_KERNEL_BUILD_FROM_SOURCE=1.
# Priority 3: the PyPI wheel installed above.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_kernel_source_changed() {
    if git -C "${REPO_ROOT}" rev-parse --verify HEAD~1 >/dev/null 2>&1; then
        git -C "${REPO_ROOT}" diff --name-only HEAD~1 -- sgl-kernel/ | grep -qE '\.(py|cc|cu|cpp|h|cuh|toml)$'
    else
        return 1
    fi
}

# Note: the PR build names its wheel sglang_kernel-* (setup_ppu.py dist name).
# Install with --no-deps, mirroring ppu_build_kernel.sh for source builds.
if [[ -n "${SGL_KERNEL_WHEEL_DIR:-}" ]] && ls "${SGL_KERNEL_WHEEL_DIR}"/sglang_kernel*.whl >/dev/null 2>&1; then
    echo "Installing PR-built sgl-kernel wheel from ${SGL_KERNEL_WHEEL_DIR}..."
    ${PIP_INSTALL} --force-reinstall --no-deps "${SGL_KERNEL_WHEEL_DIR}"/sglang_kernel*.whl
elif [[ "${SGL_KERNEL_BUILD_FROM_SOURCE:-0}" == "1" ]] || _kernel_source_changed; then
    echo "sgl-kernel source changed (or SGL_KERNEL_BUILD_FROM_SOURCE=1) — building from source..."
    bash "${SCRIPT_DIR}/ppu_build_kernel.sh"
else
    echo "sgl-kernel source unchanged — using PyPI wheel."
fi

# ==================== EIC SDK + mooncake-barex (for disaggregation tests) ==================== #
EIC_PKG_DIR=/nas_aisw/datasets/packages/eic-sdk
if [[ -d "${EIC_PKG_DIR}" ]]; then
    echo "Installing EIC SDK components for PPU disaggregation..."
    dpkg -i --force-overwrite "${EIC_PKG_DIR}"/ali-rdma-core_*.deb || echo "WARNING: ali-rdma-core install failed"
    dpkg -i "${EIC_PKG_DIR}"/accl-barex-cuda13-*.deb || echo "WARNING: accl-barex install failed"
    dpkg -i "${EIC_PKG_DIR}"/u2mm.deb || echo "WARNING: u2mm install failed"
    dpkg -i "${EIC_PKG_DIR}"/eic-tools_*.deb 2>/dev/null || echo "WARNING: eic-tools install failed (non-critical)"
    if [[ -f "${EIC_PKG_DIR}/libunicm.so" ]]; then
        cp "${EIC_PKG_DIR}/libunicm.so" /usr/lib64/
    fi
    if [[ -d /usr/lib64 ]] && ! grep -q '/usr/lib64' /etc/ld.so.conf /etc/ld.so.conf.d/*.conf 2>/dev/null; then
        echo "/usr/lib64" > /etc/ld.so.conf.d/usr-lib64.conf
    fi
    ldconfig
    ${PIP_INSTALL} --no-deps --force-reinstall \
        "${EIC_PKG_DIR}"/mooncake_transfer_engine-0.3.6.post1-*.whl
    echo "EIC SDK installation complete."
else
    echo "WARNING: EIC SDK not found at ${EIC_PKG_DIR}, disaggregation tests may fail."
fi
