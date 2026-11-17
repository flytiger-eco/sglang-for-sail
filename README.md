# SGLang-for-sail User Guide

## Repository relationship

[`t-head/sglang-for-sail`](https://github.com/t-head/sglang-for-sail) provides the
SAIL-adapted SGLang source, including the Python package and AOT kernels.
[`flytiger-eco/sglang-for-sail`](https://github.com/flytiger-eco/sglang-for-sail)
provides the CUDA-compatible build conversion and packaging integration for that
source.

The compilation steps below use the source from `t-head` and the
[`cudafy_sail_sglang.py`](https://github.com/flytiger-eco/sglang-for-sail/blob/ppu_cicd_workflow/scripts/cudafy_sail_sglang.py)
script from Flytiger's `ppu_cicd_workflow` branch. The script converts the AOT
`setup_sail.py` from ACTLIZE/native SAIL build configuration to CUTLASS and
CUDA-compatible compilation, without rewriting the kernel source files. Both
`sglang` and `sglang-kernel` are built from the same `t-head` checkout.

For the latest prebuilt packages, visit the [Flytiger website](https://www.flytiger-eco.com/home).

## Compilation

Use a build environment with Git, a host C++ compiler, Ninja, the CUDA-compatible
PPU SDK, and matching Torch. For example:

```text
ghcr.io/flytiger-eco/sglang-ppu-nightly:release-v0.5.23-20261116-cuda13.0.0-sdk2.2.0-cf244d4
```

This corresponds to Ubuntu 24.04, Python 3.12, CUDA compatibility runtime 13.0,
SDK 2.2.0, and Torch 2.13.0. GitHub and the Flytiger package registry must be
reachable. The SGLang build also needs Rust and Protobuf; the source checkout
provides setup helpers. Kernel setup fetches its pinned third-party dependencies.

### 1. Prepare the source and environment

Run the following steps in the same shell. Replace `release/v0.5.23` with the
source branch you want to build.

```bash
git clone --branch release/v0.5.23 --recurse-submodules \
  https://github.com/t-head/sglang-for-sail.git sglang-source
cd sglang-source
source /usr/local/PPU_SDK/envsetup.sh
```

Do not replace the image's Torch with an upstream CUDA wheel or force native
`PYTORCH_SAIL_ARCH` settings for this conversion path.

### 2. Build SGLang

```bash
source scripts/ci/utils/install_protoc.sh
source scripts/ci/utils/install_rustup.sh
cd python
cp pyproject_other.toml pyproject.toml
cp ../README.md ../LICENSE .
python3 -m pip install build wheel setuptools setuptools-scm setuptools-rust
python3 -m build --wheel
```

Output: `sglang-source/python/dist/*.whl`.

### 3. Convert and build sglang-kernel

Download the converter directly and pin the build tools to avoid packaging API
incompatibilities:

```bash
cd sglang/kernels/aot
wget -O cudafy_sail_sglang.py \
  https://raw.githubusercontent.com/flytiger-eco/sglang-for-sail/ppu_cicd_workflow/scripts/cudafy_sail_sglang.py
python3 -m pip install 'setuptools==75.8.0' 'wheel==0.45.1' 'packaging==24.2'
python3 cudafy_sail_sglang.py --file setup_sail.py
cp pyproject_sail.toml pyproject.toml
python3 setup_sail.py bdist_wheel
```

Output: `sglang-source/python/sglang/kernels/aot/dist/*.whl`.
The wheel distribution is named `sglang-kernel`; its import package is
`sgl_kernel`. Use `setup_sail.py` and `pyproject_sail.toml`, even after conversion.
No `SGL_KERNEL_SKIP_SAILIFY` setting is needed for the converted setup file.
