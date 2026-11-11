# syntax=docker/dockerfile:1.7
#
# Dockerfile.ppu — Step-5-Preview on Flytiger (PPU) 810E / 890P
# 依据钉钉文档《Step 5 preview sglang 推理最佳实践》§2.2 / §3.1 生成
#   https://alidocs.dingtalk.com/i/nodes/GZLxjv9VGqOq1ZxzsqGGYkA986EDybno
#
# 构建：
#   docker build -f Dockerfile.ppu -t step5-preview-sglang:ppu-sdk2.2.0 .
#
# 运行（示例，模型目录挂载到 /models）：
#   docker run -it --rm --shm-size=64g --network=host \
#     -v /path/to/Step-5-Preview-MoE-Quant-W-INT8-PerChannel-A-INT8-PerToken:/models/step5-preview \
#     step5-preview-sglang:ppu-sdk2.2.0 bash
#
# 拉起服务参考 §4.1 (810E) / §4.2 (890P) 的 sglang serve 命令。

# ─────────────────────────────────────────────────────────────────────────────
# 1. 基础镜像（sdk 2.2.0 / CUDA 13.0 / Python 3.12 / Ubuntu 24.04）
# ─────────────────────────────────────────────────────────────────────────────
ARG BASE_IMAGE=pkg.flytiger-eco.com/docker_release/pytorch:sdk2.2.0-2.13.0-ubuntu24.04-cuda13.0-py312-20261001
FROM ${BASE_IMAGE}

# ─────────────────────────────────────────────────────────────────────────────
# 2. 全局参数
# ─────────────────────────────────────────────────────────────────────────────
ARG SGLANG_REPO=https://github.com/t-head/sglang-for-sail.git
ARG SGLANG_BRANCH=support-step5-preview
ARG FLYTIGER_PYPI=https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────────────────────────────────────
# 3. 系统依赖（git 用于源码拉取 sglang；其余按基础镜像实际情况可裁剪）
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ─────────────────────────────────────────────────────────────────────────────
# 4. Step 2 — 源码编译安装 sglang 0.5.22 与 sglang-kernel 0.4.9
#    对应文档 §3.1 Step 2
# ─────────────────────────────────────────────────────────────────────────────
RUN pip install -U packaging && \
    pip install setuptools==81.0.0 && \
    git clone --depth=1 --branch ${SGLANG_BRANCH} ${SGLANG_REPO} sglang-for-sail

# 4.1 sglang wheel
RUN cd /workspace/sglang-for-sail/python && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl --no-deps

# 4.2 sglang-kernel wheel（必须带 SGL_KERNEL_SKIP_THIRD_PARTY=1，见 §8 Q1）
RUN cd /workspace/sglang-for-sail/python/sglang/kernels/aot && \
    SGL_KERNEL_SKIP_THIRD_PARTY=1 python setup_sail.py bdist_wheel && \
    pip install --no-deps --force-reinstall dist/sglang_kernel-*.whl

# ─────────────────────────────────────────────────────────────────────────────
# 5. Step 3 — 通用 Python 依赖
#    对应文档 §3.1 Step 3
# ─────────────────────────────────────────────────────────────────────────────
RUN pip install transformers==5.20.1

RUN pip install --no-deps \
        xgrammar==0.2.7 \
        tokenizers==0.23.3 \
        compressed_tensors

RUN pip install \
        pynvml orjson psutil aiohttp build fastapi pybase64 msgspec zmq \
        IPython httpx dill sentencepiece ray loguru uv uvicorn uvloop

RUN pip install \
        huggingface-hub==1.33.0 \
        safetensors==0.8.0 \
        apache-tvm-ffi==0.1.15 \
        nvidia-cutlass-dsl==4.8.0 \
        pyarrow==25.0.1

RUN pip install \
        "regex>=2025.10.22" \
        einops gguf openai setproctitle partial_json_parser \
        cuda-python python-multipart quack

# ─────────────────────────────────────────────────────────────────────────────
# 6. Step 4 — Flytiger 私有源安装 PPU 专用 wheel
#    对应文档 §3.1 Step 4，版本务必与文档保持一致（否则易触发 §8 Q1 kernel 加载失败）
# ─────────────────────────────────────────────────────────────────────────────
RUN pip install deep_gemm==1.1.0 \
        -i ${FLYTIGER_PYPI}

RUN pip install flash_attn_3==2.8.2+v0.2.0.ppu2.2.0 \
        -i ${FLYTIGER_PYPI}

RUN pip install flashinfer_python==0.6.17+v0.1.0.ppu2.2.0 \
        -i ${FLYTIGER_PYPI} --no-deps

RUN pip install flash_mla==2.0.0+v0.2.0.ppu2.2.0 \
        -i ${FLYTIGER_PYPI}

RUN pip install acext==1.1.0+v0.2.0.ppu2.2.0 \
        -i ${FLYTIGER_PYPI}

# ─────────────────────────────────────────────────────────────────────────────
# 7. Step 5 — 构建期自检
#    对应文档 §3.1 Step 5
# ─────────────────────────────────────────────────────────────────────────────
RUN python -c "import sglang; assert sglang.__version__ == '0.5.22', sglang.__version__; print('sglang', sglang.__version__)" && \
    python -c "import sglang_kernel, importlib.metadata as m; print('sglang-kernel', m.version('sglang-kernel'))" && \
    python -c "import flash_attn_3, flashinfer, flash_mla, acext, deep_gemm; print('PPU wheels OK')"

# ─────────────────────────────────────────────────────────────────────────────
# 8. 运行时配置
# ─────────────────────────────────────────────────────────────────────────────
ENV MODEL_ROOT=/models \
    SGLANG_HOST=0.0.0.0 \
    SGLANG_PORT=9876

# 810E 默认启动脚本（对应 §4.1）；890P 请传 SGLANG_EXTRA_ARGS="--disable-radix-cache" 并改端口，见 §4.2
# 使用 <<'EOF' 单引号 heredoc，避免 Dockerfile 提前展开 ${...}，保留运行时可变性
RUN <<'EOS' cat > /usr/local/bin/serve-step5-810e.sh
#!/usr/bin/env bash
set -euo pipefail
model="${MODEL_ROOT:-/models}/step5-preview"
exec sglang serve \
  --trust-remote-code \
  --model-path "${model}" \
  --tp-size 8 \
  --ep-size 8 \
  --moe-runner-backend triton \
  --attention-backend fa3 \
  --page-size 64 \
  --kv-cache-dtype bfloat16 \
  --mem-fraction-static 0.93 \
  --swa-full-tokens-ratio 0.03 \
  --chunked-prefill-size 8192 \
  --max-running-requests 64 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-multi-layer-eagle \
  --reasoning-parser step5 \
  --tool-call-parser step5 \
  --quantization w8a8_int8 \
  --host "${SGLANG_HOST:-0.0.0.0}" \
  --port "${SGLANG_PORT:-9876}" \
  ${SGLANG_EXTRA_ARGS:-}
EOS

RUN chmod +x /usr/local/bin/serve-step5-810e.sh

WORKDIR /workspace

EXPOSE 9876

CMD ["/usr/local/bin/serve-step5-810e.sh"]
