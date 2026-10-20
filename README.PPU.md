<div align="center" id="ppusglangtop">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>
</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://lmsys.org/blog/"><b>Community Blog</b></a> |
<a href="https://docs.sglang.io/"><b>Community Docs</b></a> |
<a href="https://roadmap.sglang.io/"><b>Roadmap</b></a> |
<a href="https://slack.sglang.io/"><b>Slack</b></a> |
<a href="https://meet.sglang.io/"><b>Weekly Meeting</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>Slides</b></a>
</p>

---

## Introduction

SGLang-for-SAIL is a PPU-adapted inference engine based on SGLang v0.5.21.
It provides runtime dependency, backend kernel, and deployment workflow adaptations for PPU AI accelerator chips.
It keeps SGLang's high-performance serving capabilities and integrates PPU platform optimizations for deploying large language models and multimodal models on PPU devices.

This document only covers the basic installation, verification, and deployment workflow.

## Requirements

Before installing SGLang-for-SAIL v0.5.21, make sure the SAIL SDK and required runtime components are available in your environment.

- SAIL SDK v2.2.0 or later
- Python 3.12
- PyTorch-for-SAIL 2.13.0 or later
- FlashInfer-for-SAIL 0.6.17 or later

## Quick Start

### Option 1: Build the Docker Image with ppu.Dockerfile

Using a Docker image is recommended because it avoids manual setup of the base runtime environment.
Build the image with [docker/ppu.Dockerfile](docker/ppu.Dockerfile): the build starts from the PPU PyTorch base image, compiles the PPU kernels and the model gateway, and installs SGLang-for-SAIL 0.5.21 in one pass, so no further installation is needed inside the container.

The build requires BuildKit (enabled by default in Docker 23 and later). Clone the repository and run the build from the repository root:

```bash
git clone --branch v0.5.21 https://github.com/flytiger-eco/sglang-for-sail.git
cd sglang-for-sail

# Runtime image: production environment
docker build -f docker/ppu.Dockerfile --target runtime -t sglang:ppu .

# Development image: additionally installs gdb, vim/tmux, the Rust toolchain, pre-commit, pytest, etc.
docker build -f docker/ppu.Dockerfile --target framework -t sglang:ppu-dev .
```

By default (`BRANCH_TYPE=local`) the build packages the current checkout, so make sure the working tree is at v0.5.21. Useful build arguments:

- `PPU_BASE_IMAGE`: PPU PyTorch base image (default `pkg.flytiger-eco.com/docker_release/pytorch:sdk2.2.0-2.13.0-ubuntu24.04-cuda13.0-py312-20261001`).
- `BRANCH_TYPE=remote` with `SGL_VERSION=0.5.21`: clone the release tag inside the build instead of using the local checkout.
- `MAX_JOBS`: parallel compile jobs for the kernel builds (default `16`).
- `PIP_DEFAULT_INDEX` / `UBUNTU_MIRROR`: pip and apt mirrors for restricted networks.

Then start a container from the built image:

```bash
IMAGE_NAME="sglang:ppu"   # or "sglang:ppu-dev" for the development image

sudo docker run -it \
  --name SGLang-for-SAIL_0.5.21 \
  --privileged \
  --ipc=host \
  --device=/dev/alixpu_ctl \
  --device=/dev/alixpu \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --init \
  --shm-size=8g \
  -v /path/to/host/:/path/to/container/ \
  ${IMAGE_NAME} bash
```

### Option 2: Install from PyPI

* Install SGLang-for-SAIL from PyPI index:

```bash
pip install sglang==0.5.21 --force-reinstall --no-deps -i https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple
```

### Option 3: Build from Source
* Please make sure using SAIL SDK or SAIL Docker Image
```bash
# setup build tools
pip install scikit-build-core cmake==3.31.2 uv

# setup rust
pip install setuptools_rust maturin[patchelf]
apt update && apt install -y libssl-dev protobuf-compiler
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# build and install sglang
cd python
rm -rf build dist *.egg-info
python -m uv build --no-build-isolation --wheel -Cbuild-dir=build --color=always .
pip install  dist/*.whl --force-reinstall --no-deps

# build and install sglang-kernel
cd sglang/kernels/aot
rm -rf build dist
python3 setup_ppu.py bdist_wheel
pip install  dist/*.whl --force-reinstall

# build and install sglang-router
cd ../../../../sgl-model-gateway/bindings/python
maturin build --release --out dist --features vendored-openssl
pip install dist/*.whl --force-reinstall
```

## Verify Installation

After entering the Docker container or completing the PyPI installation, run the following command to verify that SGLang is installed correctly:

```bash
python3 -c "import sglang; print(sglang.__version__)"
```

The command should print `0.5.21`.

## Deploy a Model

Start an SGLang server with a local model path.
The following example uses tensor parallelism across 8 PPU devices and enables the FA3 attention backend:

```bash
python3 -m sglang.launch_server \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8999 \
  --model-path /path/to/model \
  --tp-size 8 \
  --attention-backend fa3 \
  --mem-fraction-static 0.9 \
  --disable-radix-cache \
  --quantization w8a8_int8
```

Adjust `--model-path`, `--tp-size`, memory settings, and quantization options based on your model size and machine configuration.

## Send a Request

SGLang-for-SAIL is compatible with the OpenAI API.
After the service starts, send a chat completion request with the following command:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

curl -X POST http://127.0.0.1:8999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Who are you?"}
    ],
    "max_completion_tokens": 200,
    "top_k": 1
  }'
```
