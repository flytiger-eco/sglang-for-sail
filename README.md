<div align="center" id="ppusglangtop">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>


  <p>
    <a href="./README_zh.md">简体中文</a> | <b>English</b>
  </p>
</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://www.t-head.cn/product?id=7"><b>About T-Head</b></a> |
<a href="https://lmsys.org/blog/"><b>Community Blog</b></a> |
<a href="https://docs.sglang.io/"><b>Community Docs</b></a> |
<a href="https://roadmap.sglang.io/"><b>Roadmap</b></a> |
<a href="https://slack.sglang.io/"><b>Slack</b></a> |
<a href="https://meet.sglang.io/"><b>Weekly Meeting</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>Slides</b></a>
</p>

---

## Introduction

SGLang-for-SAIL is a PPU-adapted inference engine based on SGLang v0.5.13.
It provides runtime dependency, backend kernel, and deployment workflow adaptations for T-Head AI accelerator chips.
It keeps SGLang's high-performance serving capabilities and integrates PPU platform optimizations for deploying large language models and multimodal models on PPU devices.

This document only covers the basic installation, verification, and deployment workflow.
For quantization, environment variables, Attention Backend, DeepGEMM tuning, known issues, and model-specific instructions, see the SGLang-for-SAIL User Guide.

## Requirements

Before installing SGLang-for-SAIL v0.5.13, make sure the SAIL SDK and required runtime components are available in your environment.

- SAIL SDK v2.1.1 or later
- Python 3.12
- PyTorch-for-SAIL 2.10.0 or later
- FlashInfer-for-SAIL 0.6.8.post1 or later

For supported operating systems, CUDA Wrapper versions, and the full dependency list, see the SGLang-for-SAIL User Guide.

## Quick Start

### Option 1: Use the Docker Image

Using the SGLang-for-SAIL Docker image is recommended because it avoids manual setup of the base runtime environment.

```bash
IMAGE_NAME="<SGLang-for-SAIL_IMAGE>"

sudo docker run -it \
  --name SGLang-for-SAIL_0.5.13 \
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
pip install sglang==0.5.13 --force-reinstall --no-deps
```

* For source builds, see the SGLang-for-SAIL User Guide.

## Verify Installation

After entering the Docker container or completing the PyPI installation, run the following command to verify that SGLang is installed correctly:

```bash
python3 -c "import sglang; print(sglang.__version__)"
```

The command should print `0.5.13`.

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
For more deployment parameters and model recommendations, see SGLang-for-SAIL User Guide.

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

## More Documentation

Please refer to the SGLang-for-SAIL User Guide.
