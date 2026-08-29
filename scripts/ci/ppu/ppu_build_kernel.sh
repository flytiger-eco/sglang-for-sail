#!/bin/bash
# Build sgl-kernel PPU wheel from source inside the CI container.
#
# Called by ppu_install_dependency.sh when it detects that the repo's kernel
# source has changed relative to the installed Artifactory wheel, or when
# SGL_KERNEL_BUILD_FROM_SOURCE=1 is explicitly set.
#
# Prerequisites (all satisfied inside the PPU CI Docker container):
#   - torch with PPU CUDA support (already installed in base image)
#   - PPU_SDK at /usr/local/PPU_SDK (baked into image)
#   - ninja (already installed)
#   - git (for cloning third-party: cutlass/flashinfer/triton)
#
# Environment variables (optional):
#   MAX_JOBS          — ninja parallelism (default: min(nproc*2/3, 32))
#   SGL_KERNEL_SKIP_THIRD_PARTY — set to 1 if third_party already populated
#   SGL_KERNEL_THIRD_PARTY_DIR  — override third_party location (e.g. NAS cache)

set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(cd "$(dirname "$0")/../../.." && pwd)}"
KERNEL_DIR="${REPO_ROOT}/python/sglang/kernels/aot"

echo "========================================"
echo "  PPU sgl-kernel: building from source"
echo "========================================"
echo "KERNEL_DIR: ${KERNEL_DIR}"
echo "MAX_JOBS: ${MAX_JOBS:-auto}"

cd "${KERNEL_DIR}"

# Parallelism: be conservative on shared runners (default 2/3 of cores, cap 32)
if [ -z "${MAX_JOBS:-}" ]; then
    MAX_JOBS=$(python3 -c "import os; print(min(os.cpu_count() * 2 // 3, 32))")
fi
export MAX_JOBS

# Build the wheel
echo "Building wheel (MAX_JOBS=${MAX_JOBS})..."
BUILD_START=$(date +%s)

python3 setup_ppu.py bdist_wheel 2>&1 | tee /tmp/sgl_kernel_build.log | \
    grep -E "^(Building|running|creating|Cloning|copying|nvcc|error:|warning:.*error)" || :
BUILD_RC=${PIPESTATUS[0]}

BUILD_END=$(date +%s)
BUILD_ELAPSED=$((BUILD_END - BUILD_START))

if [ "${BUILD_RC}" -ne 0 ]; then
    echo "ERROR: setup_ppu.py failed (exit ${BUILD_RC}) after ${BUILD_ELAPSED}s. Full log:"
    cat /tmp/sgl_kernel_build.log
    exit "${BUILD_RC}"
fi
echo "Build completed in ${BUILD_ELAPSED}s"

# Find and install the wheel
WHEEL=$(ls -t dist/sglang_kernel-*.whl 2>/dev/null | head -1)
if [ -z "${WHEEL}" ]; then
    echo "ERROR: No wheel produced. Full build log:"
    cat /tmp/sgl_kernel_build.log
    exit 1
fi

echo "Installing: ${WHEEL}"
pip install --force-reinstall --no-deps "${WHEEL}"

# Verify import works
python3 -c "import sgl_kernel; print(f'sgl_kernel {sgl_kernel.__version__} installed from source')"

echo "========================================"
echo "  PPU sgl-kernel build SUCCESS (${BUILD_ELAPSED}s)"
echo "========================================"
