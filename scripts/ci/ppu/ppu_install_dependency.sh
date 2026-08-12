#!/bin/bash
set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
PIP_INSTALL="python3 -m pip install --no-cache-dir"

PPU_PIP_INDEX="https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple"
PPU_PIP_INDEX_LEGACY="https://art-pub.eng.t-head.cn/artifactory/api/pypi/pypi_index/simple"

if [[ -n "${PPU_ARTIFACTORY_USER:-}" && -n "${PPU_ARTIFACTORY_PASSWORD:-}" ]]; then
    {
        echo "machine pkg.flytiger-eco.com login ${PPU_ARTIFACTORY_USER} password ${PPU_ARTIFACTORY_PASSWORD}"
        echo "machine art-pub.eng.t-head.cn login ${PPU_ARTIFACTORY_USER} password ${PPU_ARTIFACTORY_PASSWORD}"
    } > ~/.netrc
    chmod 600 ~/.netrc
fi

git config --global --add safe.directory "${REPO_ROOT}"

${PIP_INSTALL} --upgrade pip setuptools wheel dill

# ==================== PPU Dependencies ==================== #
# Install/upgrade PPU-specific packages from SAIL PyPI source.
# Versions per SAIL install guide (v0.5.13). Use --no-build-isolation so
# sdist builds can see system torch.
${PIP_INSTALL} sglang_kernel==0.4.3 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation
${PIP_INSTALL} flashinfer_python==0.6.8.post1 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation
${PIP_INSTALL} deep_ep==1.0.0 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation
${PIP_INSTALL} deep_gemm==1.0.0 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation
${PIP_INSTALL} tilelang==0.1.8 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation
${PIP_INSTALL} flash_mla==2.0.0 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation
${PIP_INSTALL} z3-solver==4.13.0
${PIP_INSTALL} triton==3.6.0+v0.1.0.ppu2.1.0 -i ${PPU_PIP_INDEX_LEGACY} --force-reinstall --no-deps --no-build-isolation
${PIP_INSTALL} fla==1.0.0+v0.1.0.ppu2.1.0 -i ${PPU_PIP_INDEX_LEGACY} --force-reinstall --no-deps --no-build-isolation
${PIP_INSTALL} sglang==0.5.13 -i ${PPU_PIP_INDEX} --no-deps --force-reinstall --no-build-isolation

# ==================== Install SGLang from source ==================== #
rm -f "${REPO_ROOT}/python/pyproject.toml"
cp "${REPO_ROOT}/python/pyproject_other.toml" "${REPO_ROOT}/python/pyproject.toml"
cd "${REPO_ROOT}" && ${PIP_INSTALL} -v -e "python[all_ppu]"

# sgl-kernel is already installed from Artifactory (PPU Dependencies step above).
# Source build from setup_ppu.py requires internal GitLab access for cutlass;
# skip it in CI until the runner has connectivity to gitlab.alibaba-inc.com.