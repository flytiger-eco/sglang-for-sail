<div align="center" id="ppusglangtop">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>


  <p>
    <b>简体中文</b> | <a href="./README.md">English</a>
  </p>
</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://www.t-head.cn/product?id=7"><b>关于平头哥</b></a> |
<a href="https://lmsys.org/blog/"><b>社区博客</b></a> |
<a href="https://docs.sglang.io/"><b>社区文档</b></a> |
<a href="https://roadmap.sglang.io/"><b>Roadmap</b></a> |
<a href="https://slack.sglang.io/"><b>Slack</b></a> |
<a href="https://meet.sglang.io/"><b>社区周会</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>Slides</b></a>
</p>

---

## 简介

SGLang-for-SAIL 是基于 SGLang v0.5.13 适配的 PPU 推理引擎版本，面向 T-Head AI 加速芯片提供运行时依赖、后端算子和部署流程适配。
该版本保留了 SGLang 的高性能 serving 能力，并结合 PPU 平台特性进行优化，可用于在 PPU 设备上部署大语言模型和多模态模型。

本文档仅介绍基础安装、安装验证和模型部署流程。
关于量化能力、环境变量、Attention Backend、DeepGEMM tuning、已知问题以及模型专项说明，请参考 SGLang-for-SAIL 用户指南。

## 环境要求

安装 SGLang-for-SAIL v0.5.13 前，请确认当前环境已准备好 SAIL SDK 和必要的运行时组件。

- SAIL SDK v2.1.1 或更高版本
- Python 3.12
- PyTorch-for-SAIL 2.10.0 或更高版本
- FlashInfer-for-SAIL 0.6.8.post1 或更高版本

支持的操作系统、CUDA Wrapper 版本以及完整依赖列表，请参考 SGLang-for-SAIL 用户指南。

## 快速开始

### 方式一：使用 Docker 镜像

推荐使用 SGLang-for-SAIL Docker 镜像启动环境，以避免手动配置基础运行时依赖。

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

### 方式二：通过 PyPI 安装

* 从 PyPI 源安装 SGLang-for-SAIL：

```bash
pip install sglang==0.5.13 --force-reinstall --no-deps
```

* 如需从源码编译，请参考 SGLang-for-SAIL 用户指南。

## 安装验证

进入 Docker 容器或完成 PyPI 安装后，运行以下命令确认 SGLang 已正确安装：

```bash
python3 -c "import sglang; print(sglang.__version__)"
```

该命令应输出 `0.5.13`。

## 部署模型

使用本地模型路径启动 SGLang server。
以下示例使用 8 张 PPU 设备进行 tensor parallel，并启用 FA3 attention backend：

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

请根据模型规模和机器配置调整 `--model-path`、`--tp-size`、显存参数以及量化选项。
更多部署参数和模型建议，请参考 SGLang-for-SAIL 用户指南。

## 发送请求

SGLang-for-SAIL 兼容 OpenAI API。
服务启动后，可以使用以下命令发送 chat completion 请求：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

curl -X POST http://127.0.0.1:8999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你是谁？"}
    ],
    "max_completion_tokens": 200,
    "top_k": 1
  }'
```

## 更多文档

请参考 SGLang-for-SAIL 用户指南。
