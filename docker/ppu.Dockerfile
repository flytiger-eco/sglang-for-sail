# syntax=docker/dockerfile:1
# Build the current branch from the repository root:
# docker build -f docker/ppu.Dockerfile --target runtime -t sglang:ppu .
# docker build -f docker/ppu.Dockerfile --target framework -t sglang:ppu-dev .
#
# Based on docker/ppu.Dockerfile from v0.5.21, retaining parallel kernel/gateway
# builds and framework/runtime targets with the current SAIL scripts.
# Support both CUDA-compatible PPU and native cuda-free SDK images.
# NVIDIA-only components and kernels-community CUDA cubins are not installed.
ARG PPU_BASE_IMAGE=pkg.flytiger-eco.com/docker_release/pytorch:sdk2.2.0-2.13.0-ubuntu24.04-cuda13.0-py312-20261001
# pyproject_other.toml no longer provides http2/ray extras; inference dependencies
# come from runtime_common, with ray installed separately for the DeepGEMM tuner.
# Avoid CUDA diffusion dependencies from all and torch-for-sail/triton-for-sail
# wrapper distributions from all_sail.
ARG BUILD_TYPE=runtime_common,tracing
ARG BRANCH_TYPE=local
ARG SGL_VERSION
ARG SGLANG_REPO=https://github.com/flytiger-eco/sglang-for-sail.git
ARG PIP_DEFAULT_INDEX
ARG PPU_PIP_INDEX=https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple
ARG UBUNTU_MIRROR
ARG MAX_JOBS=16
# Download the pinned sailify wheel directly instead of building from Git.
ARG SAILIFY_WHEEL_URL=https://download.t-head.cn/artifactory/pypi_tmp/2026-11-03-0938/sailify-1.0.0-py3-none-any.whl
ARG PYTORCH_SAIL_ARCH="ppu_10;ppu_15"
ARG SGL_KERNEL_VERSION=0.4.9
# Pin the same standalone CUDA configuration converter used by the wheel workflow.
ARG CUDAFY_SCRIPT_URL=https://raw.githubusercontent.com/flytiger-eco/cudafy-for-sail/17fc9b6dd6267000a2fcb58e0177051d3439052b/cudafy_sail_sglang.py
ARG CUDAFY_SCRIPT_SHA256=0b4ab5ddf536731b855981048d8a568e411d22937d70772035165886b76199c5
# Keep v0.5.21 PPU local-version pins for dependencies without temporary wheels
# to prevent CUDA package selection. sailify is required for build-time source
# conversion in setup_sail.py and for JIT compilation.
ARG PPU_PACKAGES="\
    acext==1.1.0+v0.2.0.ppu2.2.0 \
    cpm-kernels==1.0.11+v0.1.0.ppu2.2.0 \
    cuda-bindings==13.0.0+v0.1.0.ppu2.2.0 \
    cuda-python==13.0.0+v0.1.0.ppu2.2.0 \
    deep_ep==2.2.0+v0.1.0.ppu2.2.0 \
    fast-hadamard-transform==1.0.4.post1+v0.1.0.ppu2.2.0 \
    flashinfer_python==0.6.17+v0.1.0.ppu2.2.0 \
    mooncake==0.3.12.post1+v0.1.0.ppu2.2.0 \
    nvidia-dali-cuda130==1.53.0+v0.1.0.ppu2.2.0 \
    tilelang==0.1.11+v0.1.0.ppu2.2.0 \
    torch_memory_saver==0.0.9+v0.1.0.ppu2.2.0 \
    torchao==0.17.0+v0.1.0.ppu2.2.0 \
    torchcodec==0.15.0+v0.1.0.ppu2.2.0 \
    transformer-engine==2.5+v0.1.0.ppu2.2.0 \
    triton==3.7.0+v0.1.0.ppu2.2.0"
# The base image provides torch/torchvision; only add torchaudio.
ARG PPU_TORCH_PACKAGES="torchaudio==2.11.0+v0.1.0.ppu2.2.0"
# The 2026-11-11-2345 directory contains flash_attn and flash_attn_3; pin both URLs.
# These projects are excluded from PPU_PACKAGES so old index versions cannot
# replace the requested wheels.
ARG PPU_WHEEL_URLS="\
    https://pkg.flytiger-eco.com/artifactory/pypi_tmp/2026-11-11-2240/deep_gemm-1.1.0.dev294+v1.1.0.ppu2.2.0.g4869f1e3-cp312-cp312-linux_x86_64.whl \
    https://pkg.flytiger-eco.com/artifactory/pypi_tmp/2026-11-11-2133/flash_mla-2.1.0.dev5+v1.0.0.ppu2.2.0.ga85e3e5-cp312-cp312-linux_x86_64.whl \
    https://pkg.flytiger-eco.com/artifactory/pypi_tmp/2026-11-11-2345/flash_attn-2.8.2-cp312-cp312-linux_x86_64.whl \
    https://pkg.flytiger-eco.com/artifactory/pypi_tmp/2026-11-11-2345/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl \
    https://pkg.flytiger-eco.com/artifactory/pypi_tmp/2026-11-11-2252/pla-1.1.0-cp312-cp312-linux_x86_64.whl \
    https://pkg.flytiger-eco.com/artifactory/pypi_tmp/2026-11-11-2111/msa_for_sail-0.1.1+v0.1.0.ppu2.2.0-py3-none-any.whl"

FROM ${PPU_BASE_IMAGE} AS base

ARG PIP_DEFAULT_INDEX
ARG UBUNTU_MIRROR

# Use the base image interpreter containing SAIL torch, without a separate venv.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PPU_SDK=/usr/local/PPU_SDK \
    PPU_HOME=/usr/local/PPU_SDK
# Preserve the base image's CUDA_HOME/CUDA_PATH for its actual compiler mode.
ENV PATH="${PPU_SDK}/bin:${PATH}" \
    LD_LIBRARY_PATH="${PPU_SDK}/lib:${LD_LIBRARY_PATH}"

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Support both sources.list and deb822 ubuntu.sources on Ubuntu 24.04.
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
        for f in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources; do \
            if [ -f "$f" ]; then \
                sed -i -e "s|http://.*archive.ubuntu.com|$UBUNTU_MIRROR|g" \
                       -e "s|http://.*security.ubuntu.com|$UBUNTU_MIRROR|g" "$f"; \
            fi; \
        done; \
    fi

RUN --mount=type=cache,target=/var/cache/apt,id=ppu-base-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    netcat-openbsd \
    unzip \
    curl \
    wget \
    git \
    git-lfs \
    lsof \
    locales \
    build-essential \
    cmake \
    pkg-config \
    patchelf \
    ccache \
    libssl-dev \
    protobuf-compiler \
    libprotobuf-dev \
    libnuma1 \
    libnuma-dev \
    numactl \
    ffmpeg \
    libgl1 \
    libglib2.0-0t64 \
    libibverbs1 \
    libibverbs-dev \
    librdmacm1 \
    libibumad3 \
    ibverbs-providers \
    rdma-core \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN if [ -n "${PIP_DEFAULT_INDEX}" ]; then \
        python3 -m pip config set global.index-url "${PIP_DEFAULT_INDEX}"; \
    fi

# Disable build isolation for native builds to avoid installing CUDA torch.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip \
    && python3 -m pip install \
        "setuptools>=75" \
        "setuptools-rust>=1.11" \
        "setuptools-scm>=8" \
        wheel packaging ninja pybind11 cython scikit-build-core \
    && python3 -c "import platform, sys, torch; from torch.utils import cpp_extension; assert sys.version_info[:2] == (3, 12); assert platform.machine() == 'x86_64'; assert torch.version.cuda is not None, 'A GPU-enabled PPU Torch build is required'; print('base torch:', torch.__version__, 'cuda:', torch.version.cuda, 'sail:', getattr(torch.version, 'sail', None), 'native PPU extensions:', getattr(cpp_extension, 'IS_PPU_EXTENSION', False))"

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

########################################################
# PARALLEL STAGE 1: PPU dependency wheels and current-branch AOT kernels
########################################################
FROM base AS ppu_kernels_builder

ARG PPU_PIP_INDEX
ARG MAX_JOBS
ARG SAILIFY_WHEEL_URL
ARG PYTORCH_SAIL_ARCH
ARG PPU_PACKAGES
ARG PPU_TORCH_PACKAGES
ARG PPU_WHEEL_URLS
ARG SGL_KERNEL_VERSION
ARG CUDAFY_SCRIPT_URL
ARG CUDAFY_SCRIPT_SHA256

WORKDIR /build

# Process packages individually to identify failures. Some PPU sdists use
# os.rename to move prebuilt wheels; disable pip cache to keep temporary files
# and output on the same filesystem and avoid EXDEV.
RUN source "${PPU_SDK}/envsetup.sh" ppu \
    && mkdir -p /wheels \
    && wget --tries=3 --timeout=60 \
        -O /wheels/sailify-1.0.0-py3-none-any.whl "${SAILIFY_WHEEL_URL}" \
    && for spec in ${PPU_PACKAGES} ${PPU_TORCH_PACKAGES} ${PPU_WHEEL_URLS}; do \
        echo "==> Building ${spec}" \
        && MAX_JOBS=${MAX_JOBS} python3 -m pip wheel \
            --no-cache-dir --no-deps --no-build-isolation \
            --index-url "${PPU_PIP_INDEX}" --wheel-dir /wheels "${spec}" \
        || exit 1; \
    done \
    && ls -l /wheels

# Build AOT kernels from the current branch to avoid stale same-version index
# packages. Native builds require sailify and the relative kernels/jit/include
# directory layout. CUDA-compatible builds need CUTLASS, NVCC flags, CUDA link
# libraries, and direct relative sources instead of native sailify staging.
COPY python/sglang/kernels/aot /build/sglang/kernels/aot
COPY python/sglang/kernels/jit/include /build/sglang/kernels/jit/include
RUN source "${PPU_SDK}/envsetup.sh" ppu \
    && unset TORCH_CUDA_ARCH_LIST SGL_KERNEL_SKIP_SAILIFY SGL_KERNEL_SKIP_THIRD_PARTY \
    && export PYTORCH_SAIL_ARCH="${PYTORCH_SAIL_ARCH}" \
              MAX_JOBS="${MAX_JOBS}" SGL_KERNEL_COMPILE_THREADS="${MAX_JOBS}" \
              SGL_KERNEL_USE_SAILIFY=1 \
    && python3 -m pip install --no-deps --no-index /wheels/sailify-*.whl \
    && cd /build/sglang/kernels/aot \
    && { grep -qx "version = \"${SGL_KERNEL_VERSION}\"" pyproject_sail.toml \
         || { echo "AOT version does not match SGL_KERNEL_VERSION=${SGL_KERNEL_VERSION}"; exit 1; }; } \
    && cp pyproject_sail.toml pyproject.toml \
    && if python3 -c "from torch.utils import cpp_extension; import sys; sys.exit(0 if getattr(cpp_extension, 'IS_PPU_EXTENSION', False) else 1)"; then \
        echo "Building native PPU AOT kernels with hgcc and sailify"; \
    else \
        echo "Converting AOT configuration for the CUDA-compatible PPU toolchain" \
        && curl --fail --location --retry 3 --connect-timeout 30 \
            "${CUDAFY_SCRIPT_URL}" -o /tmp/cudafy_sail_sglang.py \
        && echo "${CUDAFY_SCRIPT_SHA256}  /tmp/cudafy_sail_sglang.py" | sha256sum -c - \
        && python3 /tmp/cudafy_sail_sglang.py --file setup_sail.py --no-backup \
        && rm -f /tmp/cudafy_sail_sglang.py \
        || exit 1; \
    fi \
    && python3 setup_sail.py bdist_wheel --dist-dir /wheels \
    && rm -rf /build/sglang \
    && ls -l /wheels

# Wrapper sdists such as mooncake may produce wheel filenames that differ from
# their .dist-info project names. Normalize filenames using actual metadata;
# subsequent constraints are also generated from installed distributions.
RUN python3 -c 'import glob, os, zipfile; \
[os.rename(w, os.path.join("/wheels", d.split("-")[0] + "-" + os.path.basename(w).split("-", 1)[1])) \
 for w in glob.glob("/wheels/*.whl") \
 for d in {n.split("/")[0] for n in zipfile.ZipFile(w).namelist() if n.endswith(".dist-info/METADATA")} \
 if d.split("-")[0] != os.path.basename(w).split("-")[0]]' \
    && ls -l /wheels

########################################################
# PARALLEL STAGE 2: Gateway binary and Python wheel
########################################################
FROM base AS gateway_builder

WORKDIR /build
COPY sgl-model-gateway /build/sgl-model-gateway

RUN --mount=type=cache,target=/root/.cache/pip \
    curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && python3 -m pip install maturin \
    && cd /build/sgl-model-gateway/bindings/python \
    && ulimit -n 65536 && maturin build --release --features vendored-openssl --out /build/gateway_wheels \
    && cd /build/sgl-model-gateway \
    && cargo build --release --bin sgl-model-gateway --features vendored-openssl \
    && cp target/release/sgl-model-gateway /build/sgl-model-gateway-bin \
    && rm -rf /root/.cargo /root/.rustup /build/sgl-model-gateway/target /build/sgl-model-gateway/bindings/python/target

########################################################
# STAGE 3: Install PPU wheels and SGLang runtime dependencies
########################################################
FROM base AS ppu_deps

ARG BUILD_TYPE
ARG PPU_PIP_INDEX

WORKDIR /sgl-workspace

COPY --from=ppu_kernels_builder /wheels /tmp/wheels/ppu
COPY --from=gateway_builder /build/gateway_wheels /tmp/wheels/gateway
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-deps /tmp/wheels/ppu/*.whl /tmp/wheels/gateway/*.whl \
    && for w in /tmp/wheels/ppu/*.whl /tmp/wheels/gateway/*.whl; do \
        basename "$w" | cut -d- -f1; \
    done > /sgl-workspace/ppu_requirements.txt \
    && rm -rf /tmp/wheels

# Generate constraints from installed versions: URLs are not ordinary version
# constraints, and dev/beta wheels cannot use the stable pins from v0.5.21.
# Freeze the torch family and all PPU wheels to prevent replacement during
# resolution, while allowing generic dependencies to meet current requirements.
RUN python3 -c 'from importlib.metadata import version; from pathlib import Path; \
names = {"torch", "torchvision", "torchaudio"} | set(Path("/sgl-workspace/ppu_requirements.txt").read_text().split()); \
print("\n".join(f"{n}=={version(n)}" for n in sorted(names)))' > /sgl-workspace/ppu_constraints.txt \
    && cat /sgl-workspace/ppu_constraints.txt

# Use a stub without setup.py so Python source changes do not rebuild the entire
# dependency layer. uv requires an explicit constraint for transformer-engine's
# transitive URL dependency. Keep the template's pyarrow cap for compatibility
# with numpy<2 builds in the base image.
COPY python/pyproject_other.toml /tmp/sglang_deps/python/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv \
    && echo "nvdlfw-inspect @ git+https://github.com/NVIDIA/nvidia-dlfw-inspect.git@v0.1" \
        >> /sgl-workspace/ppu_constraints.txt \
    && echo "pyarrow<26" >> /sgl-workspace/ppu_constraints.txt \
    && cd /tmp/sglang_deps/python \
    && mkdir -p sglang \
    && touch sglang/__init__.py README.md LICENSE \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv pip install --system \
        $(grep '==' /sgl-workspace/ppu_constraints.txt | cut -d= -f1 | sed 's/^/--only-binary /') \
        -c /sgl-workspace/ppu_constraints.txt \
        -r /sgl-workspace/ppu_requirements.txt \
        --index-url "${PPU_PIP_INDEX}" \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        ".[${BUILD_TYPE}]" ray \
    && python3 -m pip uninstall -y sglang \
    && cd /sgl-workspace \
    && rm -rf /tmp/sglang_deps \
    && python3 -m pip list --format=freeze | grep -v "^sglang==" > /sgl-workspace/constraints.txt

########################################################
# STAGE 4: Install SGLang sources from the current branch
########################################################
FROM scratch AS local_src
COPY . /src

FROM ppu_deps AS sglang_install

ARG BRANCH_TYPE
ARG BUILD_TYPE
ARG SGL_VERSION
ARG SGLANG_REPO

WORKDIR /sgl-workspace

COPY --from=local_src /src /tmp/local_src
RUN if [ "$BRANCH_TYPE" = "local" ]; then \
        cp -r /tmp/local_src /sgl-workspace/sglang; \
    elif [ "$BRANCH_TYPE" != "remote" ]; then \
        echo "BRANCH_TYPE must be local or remote" && exit 1; \
    elif [ -z "$SGL_VERSION" ]; then \
        git clone --depth=1 "${SGLANG_REPO}" /sgl-workspace/sglang; \
    else \
        git clone --depth=1 --branch "v${SGL_VERSION}" "${SGLANG_REPO}" /sgl-workspace/sglang; \
    fi \
    && rm -rf /tmp/local_src

# Use pyproject_other.toml instead of the removed pyproject_ppu.toml.
# Build Rust extensions according to the current pyproject allowlist and remove
# the Rust build toolchain from the runtime image.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cargo/registry \
    curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && cd /sgl-workspace/sglang \
    && cp python/pyproject_other.toml python/pyproject.toml \
    && CARGO_TARGET_DIR=/tmp/cargo-target python3 -m pip install \
        --no-deps --no-build-isolation -e "python[${BUILD_TYPE}]" \
    && rm -rf /tmp/cargo-target /root/.rustup \
    && find /root/.cargo -mindepth 1 -maxdepth 1 ! -name registry -exec rm -rf {} + \
    && mkdir -p /root/.cache/huggingface /root/.cache/sglang \
    && python3 -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__)"

ENV SGLANG_RUST_BUILD_MODE=never

########################################################
# Framework Image (dev)
########################################################
FROM sglang_install AS framework

WORKDIR /sgl-workspace

RUN --mount=type=cache,target=/var/cache/apt,id=ppu-framework-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    gdb ninja-build vim tmux htop zsh tree silversearcher-ag less infiniband-diags perftest \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
    && rustc --version && cargo --version

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -c /sgl-workspace/constraints.txt \
    pre-commit pytest black isort icdiff uv py-spy pandas matplotlib tabulate

COPY docker/configs/opt/.vimrc /root/.vimrc
COPY docker/configs/opt/.tmux.conf /root/.tmux.conf
COPY --from=gateway_builder /build/sgl-model-gateway-bin /usr/local/bin/sgl-model-gateway

WORKDIR /sgl-workspace/sglang
CMD ["/bin/bash"]

########################################################
# Runtime Image: retain the SDK toolchain for DeepGEMM/MSA/Triton JIT
########################################################
FROM sglang_install AS runtime

COPY --from=gateway_builder /build/sgl-model-gateway-bin /usr/local/bin/sgl-model-gateway
WORKDIR /sgl-workspace/sglang

# Keep build provenance at the end so metadata changes do not invalidate layers.
ARG SGLANG_BUILD_COMMIT=unknown
ARG SGLANG_BUILD_URL=
ARG SGLANG_IMAGE_TAG=local/sglang:ppu
ENV SGLANG_BUILD_COMMIT=${SGLANG_BUILD_COMMIT:-unknown} \
    SGLANG_BUILD_URL=${SGLANG_BUILD_URL:-} \
    SGLANG_IMAGE_TAG=${SGLANG_IMAGE_TAG:-local/sglang:ppu}
LABEL org.opencontainers.image.source="https://github.com/flytiger-eco/sglang-for-sail" \
      org.opencontainers.image.revision="${SGLANG_BUILD_COMMIT}" \
      org.opencontainers.image.version="${SGLANG_IMAGE_TAG}" \
      org.opencontainers.image.url="${SGLANG_BUILD_URL}" \
      ai.sglang.build.commit="${SGLANG_BUILD_COMMIT}" \
      ai.sglang.build.url="${SGLANG_BUILD_URL}" \
      ai.sglang.image.tag="${SGLANG_IMAGE_TAG}"

CMD ["/bin/bash"]
