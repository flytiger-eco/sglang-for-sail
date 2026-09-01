#!/bin/bash
set -ex

if [ $# -lt 2 ]; then
  echo "Usage: $0 <PYTHON_VERSION> <CUDA_VERSION> [ARCH]"
  exit 1
fi

PYTHON_VERSION="$1"          # e.g. 3.10
CUDA_VERSION="$2"            # e.g. 12.9
ARCH="${3:-$(uname -i)}"     # optional override

if [ "${ARCH}" = "aarch64" ]; then
  exit 1
else
  BASE_IMG="pkg.flytiger-eco.com/docker_build/pytorch:ubuntu24.04-py312.06"
fi

PY_TAG="cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}"

# Output directory for wheels
DIST_DIR="dist"
mkdir -p "${DIST_DIR}"

echo "----------------------------------------"
echo "Build configuration"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
echo "CUDA_VERSION:   ${CUDA_VERSION}"
echo "ARCH:           ${ARCH}"
echo "BASE_IMG:       ${BASE_IMG}"
echo "PYTHON_TAG:     ${PY_TAG}"
echo "Output:         ${DIST_DIR}/"
echo "----------------------------------------"

docker run --rm \
  --network=host \
  -v "$(pwd):/python/sglang/kernels/aot" \
  -w /python/sglang/kernels/aot \
  -e ARCH="${ARCH}" \
  "${BASE_IMG}" \
  bash -c '
set -ex

export PPU_SDK=/usr/local/PPU_SDK
export PPU_PATH=${PPU_SDK}
export PPU_HOME=${PPU_PATH}
export CUDA_SDK=${PPU_SDK}/CUDA_SDK
export CUDA_TOOLKIT_ROOT=${CUDA_SDK}
export CUDA_PATH=${CUDA_SDK}
export CUDA_HOME=${CUDA_SDK}
export CUDNN_HOME=${CUDA_SDK}
export CUDACXX=${CUDA_SDK}/bin/nvcc
export PATH=${CUDA_SDK}/bin:${PPU_SDK}/bin:${PPU_SDK}/asight/bin:${PPU_SDK}/ppu-smi/bin:${PATH}
export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=${CUDA_SDK}/lib64:${PPU_SDK}/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${CUDA_SDK}/lib64:${PPU_SDK}/lib:${LIBRARY_PATH}

wget --no-check-certificate -nv https://pkg.flytiger-eco.com/artifactory/generic-local/CUDA_SDK/v2.1.1/PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e.tar.gz -O /tmp/ppu.tar.gz
mkdir /tmp/ppu
tar --extract --file="/tmp/ppu.tar.gz" --directory=/tmp/ppu
mv /tmp/ppu/PPU_SDK /usr/local/
ln -s /usr/local/PPU_SDK/CUDA_SDK /usr/local/cuda-13.0
ln -s /usr/local/cuda-13.0 /usr/local/cuda
echo /usr/local/PPU_SDK/CUDA_SDK/lib >> /etc/ld.so.conf.d/ppu.conf
echo /usr/local/PPU_SDK/CUDA_SDK/lib64 >> /etc/ld.so.conf.d/ppu.conf
ldconfig
ldconfig -p | grep -q libcuda.so
ldconfig -p | grep -q /usr/local/PPU_SDK/CUDA_SDK/lib
source /usr/local/PPU_SDK/envsetup.sh
clang --version
nvcc --version
asys --version
ppu-smi --version
rm -rf /tmp/*

python3 -m pip install https://pkg.flytiger-eco.com/artifactory/pypi_generic/torch/2.11.0%2Bv0.1.0.ppu2.1.1/torch-2.11.0%2Bcu130ubuntu2404oe-cp312-cp312-linux_x86_64.whl

python3 setup_ppu.py sdist bdist_wheel

# python3 ./rename_wheels.sh

'

echo "Done. Wheels are in ${DIST_DIR}/"
ls -lh "${DIST_DIR}"/*.whl 2>/dev/null || true