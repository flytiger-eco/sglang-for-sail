#!/bin/bash
set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
PIP_INSTALL="python3 -m pip install --no-cache-dir"

PPU_PIP_INDEX="https://art-pub.eng.t-head.cn/artifactory/api/pypi/pypi_index/simple"

if [[ -n "${PPU_ARTIFACTORY_USER:-}" && -n "${PPU_ARTIFACTORY_PASSWORD:-}" ]]; then
    echo "machine art-pub.eng.t-head.cn login ${PPU_ARTIFACTORY_USER} password ${PPU_ARTIFACTORY_PASSWORD}" > ~/.netrc
    chmod 600 ~/.netrc
fi

git config --global --add safe.directory "${REPO_ROOT}"

${PIP_INSTALL} --upgrade pip setuptools wheel dill

# ==================== PPU Dependencies ==================== #
# Install/upgrade PPU-specific packages from T-HEAD Artifactory
# Use --no-build-isolation so sdist builds can see system torch
${PIP_INSTALL} sglang_kernel==0.4.2.post2+v0.1.0.ppu2.1.0 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation
${PIP_INSTALL} flashinfer_python==0.6.8.post1+v0.1.0.ppu2.1.0 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation
${PIP_INSTALL} deep_ep==1.0.0+v0.2.0.ppu2.1.0 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation
${PIP_INSTALL} deep_gemm==1.0.0+v0.2.0.ppu2.1.0 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation
${PIP_INSTALL} tilelang==0.1.8+v0.1.0.ppu2.1.0 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation
${PIP_INSTALL} flash_mla==2.0.0+v0.1.0.ppu2.1.0 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation
${PIP_INSTALL} z3-solver==4.13.0
${PIP_INSTALL} triton==3.6.0+v0.1.0.ppu2.1.0 -i ${PPU_PIP_INDEX} --force-reinstall --no-deps --no-build-isolation
${PIP_INSTALL} fla==1.0.0+v0.1.0.ppu2.1.0 -i ${PPU_PIP_INDEX} --force-reinstall --no-deps --no-build-isolation
${PIP_INSTALL} sglang==0.5.12+v0.1.0.ppu2.1.0 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation

# Upgrade uvicorn: Docker image ships 0.29.0, sglang needs timeout_worker_healthcheck (added in 0.37.0)
${PIP_INSTALL} /nas_aisw/datasets/packages/uvicorn-0.37.0-py3-none-any.whl --force-reinstall --no-deps

# ==================== Install SGLang from source ==================== #
rm -f "${REPO_ROOT}/python/pyproject.toml"
cp "${REPO_ROOT}/python/pyproject_other.toml" "${REPO_ROOT}/python/pyproject.toml"
cd "${REPO_ROOT}" && ${PIP_INSTALL} -v -e "python[all_ppu]"

# sgl-kernel is already installed from Artifactory (PPU Dependencies step above).
# Source build from setup_ppu.py requires internal GitLab access for cutlass;
# skip it in CI until the runner has connectivity to gitlab.alibaba-inc.com.

# ==================== EIC SDK + mooncake-barex (for disaggregation tests) ==================== #
# Install AcclBarex RDMA library and fic2 user-space driver from NAS cache.
# These are required for SGLANG_MOONCAKE_CUSTOM_MEM_POOL=BAREX to bypass EIC MR PA limit.
# NAS packages: /nas_aisw/datasets/packages/eic-sdk/ (one-time populated from ppu-v-disagg)
EIC_PKG_DIR=/nas_aisw/datasets/packages/eic-sdk
if [[ -d "${EIC_PKG_DIR}" ]]; then
    echo "Installing EIC SDK components for PPU disaggregation..."
    dpkg -i --force-overwrite "${EIC_PKG_DIR}"/ali-rdma-core_*.deb || echo "WARNING: ali-rdma-core install failed; fic2 RDMA may be unavailable"
    dpkg -i "${EIC_PKG_DIR}"/accl-barex-cuda13-*.deb || echo "WARNING: accl-barex install failed; BAREX mem pool will not work"
    dpkg -i "${EIC_PKG_DIR}"/u2mm.deb || echo "WARNING: u2mm install failed"
    dpkg -i "${EIC_PKG_DIR}"/eic-tools_*.deb 2>/dev/null || echo "WARNING: eic-tools install failed (missing unicm dep, non-critical)"
    # libunicm.so is required by mooncake but only available as RPM on host;
    # copy the shared object directly from NAS cache into the container's lib path
    if [[ -f "${EIC_PKG_DIR}/libunicm.so" ]]; then
        cp "${EIC_PKG_DIR}/libunicm.so" /usr/lib64/
    fi
    # u2mm.deb and libunicm install to /usr/lib64 which Ubuntu ldconfig doesn't search by default
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
