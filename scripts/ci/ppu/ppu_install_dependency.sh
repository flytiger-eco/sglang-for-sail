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

${PIP_INSTALL} --upgrade pip setuptools wheel dill

# ==================== PPU Dependencies (SAIL SDK v2.1.1) ==================== #
# Versions per SGLang-for-SAIL v0.5.13 user guide (2026-08-12).
# No more +v0.1.0.ppu2.1.0 suffixes — the new PyPI index has clean versions.
${PIP_INSTALL} sglang_kernel==0.4.3 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall
${PIP_INSTALL} flashinfer_python==0.6.8.post1 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall
${PIP_INSTALL} deep_ep==1.0.0 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall
${PIP_INSTALL} deep_gemm==1.0.0 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall
${PIP_INSTALL} tilelang==0.1.8 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall
${PIP_INSTALL} flash_mla==2.0.0 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall
${PIP_INSTALL} sglang_router==0.3.2 -i ${PPU_PIP_INDEX}
${PIP_INSTALL} z3-solver==4.13.0

# ==================== Install SGLang from source ==================== #
rm -f "${REPO_ROOT}/python/pyproject.toml"
cp "${REPO_ROOT}/python/pyproject_other.toml" "${REPO_ROOT}/python/pyproject.toml"
cd "${REPO_ROOT}" && ${PIP_INSTALL} -v -e "python[all_ppu]" --no-build-isolation

# ==================== sgl-kernel: source build when needed ==================== #
# Default: use the PyPI wheel installed above.
# If this PR touches sgl-kernel source, rebuild from source so CI actually tests
# the new code. Detect by checking git diff against the merge base.
# Force with SGL_KERNEL_BUILD_FROM_SOURCE=1.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_kernel_source_changed() {
    if git -C "${REPO_ROOT}" rev-parse --verify HEAD~1 >/dev/null 2>&1; then
        git -C "${REPO_ROOT}" diff --name-only HEAD~1 -- sgl-kernel/ | grep -qE '\.(py|cc|cu|cpp|h|cuh|toml)$'
    else
        return 1
    fi
}

if [[ "${SGL_KERNEL_BUILD_FROM_SOURCE:-0}" == "1" ]] || _kernel_source_changed; then
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
