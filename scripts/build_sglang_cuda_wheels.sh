#!/usr/bin/env bash
# Build CUDA-compatible sglang and sglang-kernel wheels from a pinned source ref.
#
# Defaults match the "SGLang CUDAFY temporary wheel upload" workflow:
#   SGLANG_SOURCE_REPO=<required source repository URL>
#   SGLANG_SOURCE_REF=yuhe/v1.0.0
#   SGLANG_PACKAGES=all              # all | sglang | sglang-kernel
#   BUILD_ROOT=/tmp/sglang-cuda-build
#   WHEEL_DIR=${BUILD_ROOT}/wheels
#   MAX_JOBS=$(nproc)
#
# SGLANG_SOURCE_DIR can point to an existing checkout to skip cloning.
# CUDAFY_SCRIPT can override the sibling scripts/cudafy_sail_sglang.py path.
set -eo pipefail

SOURCE_REPO="${SGLANG_SOURCE_REPO:-}"
SOURCE_REF="${SGLANG_SOURCE_REF:-yuhe/v1.0.0}"
PACKAGES="${SGLANG_PACKAGES:-all}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/sglang-cuda-build}"
WHEEL_DIR="${WHEEL_DIR:-${BUILD_ROOT}/wheels}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUDAFY_SCRIPT="${CUDAFY_SCRIPT:-${SCRIPT_DIR}/cudafy_sail_sglang.py}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

case "${PACKAGES}" in
    all | sglang | sglang-kernel) ;;
    *) echo "SGLANG_PACKAGES must be all, sglang or sglang-kernel" >&2; exit 1 ;;
esac

if [[ -z "${SGLANG_SOURCE_DIR:-}" && -z "${SOURCE_REPO}" ]]; then
    echo "Set SGLANG_SOURCE_REPO to the source repository URL, or SGLANG_SOURCE_DIR to an existing checkout" >&2
    exit 1
fi

test -f "${CUDAFY_SCRIPT}" || { echo "Missing cudafy script: ${CUDAFY_SCRIPT}" >&2; exit 1; }

# Match the workflow's network setup for the CUDA-compatible build image.
unset PIP_INDEX_URL || true
for key in $(git config --global --name-only --get-regexp '^url\.' || true); do
    git config --global --unset-all "$key"
done
git config --global --add safe.directory '*'
source "${PPU_SDK:-/usr/local/PPU_SDK}/envsetup.sh" || true

if [[ -z "${SGLANG_SOURCE_DIR:-}" ]]; then
    SGLANG_SOURCE_DIR="${BUILD_ROOT}/source"
    mkdir -p "${BUILD_ROOT}"
    # Keep full history and tags for setuptools-scm, matching the workflow build job.
    git clone --no-checkout "${SOURCE_REPO}" "${SGLANG_SOURCE_DIR}"
    git -C "${SGLANG_SOURCE_DIR}" fetch --no-tags origin "${SOURCE_REF}"
    git -C "${SGLANG_SOURCE_DIR}" checkout --detach FETCH_HEAD
fi
git -C "${SGLANG_SOURCE_DIR}" submodule update --init --recursive
SOURCE_COMMIT="$(git -C "${SGLANG_SOURCE_DIR}" rev-parse HEAD)"
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid source commit: ${SOURCE_COMMIT}" >&2; exit 1; }
echo "==> Source: ${SOURCE_REPO} ${SOURCE_REF} -> ${SOURCE_COMMIT}"

# The cudafy conversion and direct CUDA compilation require CUDA-compatible Torch.
python3 - <<'PY'
import torch
from torch.utils import cpp_extension

print(f"torch {torch.__version__} (cuda {torch.version.cuda})")
if torch.version.cuda is None or getattr(cpp_extension, "IS_PPU_EXTENSION", False):
    raise SystemExit("CUDA-compatible PPU Torch is required; hgcc / cuda-free is not supported")
PY

python3 -m pip install --no-input \
    build setuptools-scm setuptools-rust \
    'setuptools==75.8.0' 'wheel==0.45.1' 'packaging==24.2'

export_wheels() {
    local dist_dir="$1"
    # Record the source commit in the wheel build tag, as the upload workflow does.
    python3 -m wheel tags --build "1g${SOURCE_COMMIT:0:12}" --remove "${dist_dir}"/*.whl
    mkdir -p "${WHEEL_DIR}"
    cp "${dist_dir}"/*.whl "${WHEEL_DIR}/"
}

if [[ "${PACKAGES}" == all || "${PACKAGES}" == sglang ]]; then
    echo "==> Building sglang wheel"
    cd "${SGLANG_SOURCE_DIR}"
    # install_protoc.sh exits early when protoc exists; do not let it exit the builder.
    (source scripts/ci/utils/install_protoc.sh)
    source scripts/ci/utils/install_rustup.sh
    export PATH="${HOME}/.cargo/bin:${PATH}"
    cd python
    cp pyproject_other.toml pyproject.toml
    cp ../README.md ../LICENSE .
    python3 -m build --wheel
    export_wheels dist
fi

if [[ "${PACKAGES}" == all || "${PACKAGES}" == sglang-kernel ]]; then
    echo "==> Converting and building sglang-kernel wheel"
    cd "${SGLANG_SOURCE_DIR}/python/sglang/kernels/aot"
    test -f setup_sail.py || { echo "Missing setup_sail.py" >&2; exit 1; }
    python3 "${CUDAFY_SCRIPT}" --file "${PWD}/setup_sail.py"
    cp pyproject_sail.toml pyproject.toml
    python3 setup_sail.py bdist_wheel
    export_wheels dist
fi

echo "==> Built wheels:"
shopt -s nullglob
built_wheels=("${WHEEL_DIR}"/*.whl)
expected_wheels=1
[[ "${PACKAGES}" == all ]] && expected_wheels=2
if (( ${#built_wheels[@]} != expected_wheels )); then
    echo "Expected ${expected_wheels} wheel(s), found ${#built_wheels[@]} in ${WHEEL_DIR}" >&2
    exit 1
fi
ls -l "${WHEEL_DIR}"
