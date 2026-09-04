# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import platform
import subprocess
from pathlib import Path

import torch
from setuptools import find_namespace_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
third_party = Path(
    os.environ.get("SGL_KERNEL_THIRD_PARTY_DIR", root / "build" / "_deps")
)
arch = platform.machine().lower()

# ======================= CUDA Version Detection ======================= #
cuda_version = torch.version.cuda
if cuda_version is None:
    raise RuntimeError("CUDA is not available in the current PyTorch installation.")

cuda_version_tuple = tuple(int(x) for x in cuda_version.split(".")[:2])
print(f"Detected CUDA version: {cuda_version} ({cuda_version_tuple})")


# ======================= PPU SDK Detection ======================= #
# The sailify-converted sources include the PPU SDK's native headers
# (<hggc_runtime.h>, <hggc_bf16.h>, <hggcTypedefs.h>, <hggc.h>, ...), and
# actlize's cutlass/half.h pulls <hggc_fp16.h>. Those live only in the PPU SDK
# tree -- CUDA_SDK/include ships just three hggc_* headers, none of which
# declares the runtime API -- so the SDK's include dir has to be on the search
# path or every such translation unit fails with "No such file or directory".
#
# Resolved the way the SDK's own envsetup.sh exports it, falling back to the
# default install prefix.
_PPU_SDK_DIR = Path(
    os.environ.get("PPU_SDK") or os.environ.get("PPU_PATH") or "/usr/local/PPU_SDK"
)
_PPU_SDK_INCLUDE = _PPU_SDK_DIR / "include"
if not (_PPU_SDK_INCLUDE / "hggc_runtime_api.h").exists():
    raise RuntimeError(
        f"PPU SDK headers not found under {_PPU_SDK_INCLUDE}. "
        "Set PPU_SDK (or PPU_PATH) to the SDK install prefix, e.g. "
        "/usr/local/PPU_SDK."
    )
print(f"Detected PPU SDK: {_PPU_SDK_DIR}")


# ======================= Third-Party Repository Info ======================= #
class _RepoInfo:
    """Configuration for a third-party git repository."""

    def __init__(self, name, git_repository, git_tag, git_shallow=False):
        self.name = name
        self.git_repository = git_repository
        self.git_tag = git_tag
        self.git_shallow = git_shallow
        self.source_dir = third_party / name


_ACTLIZE_REPO = _RepoInfo(
    name="actlize",
    git_repository="https://github.com/t-head/actlize",
    git_tag="129651181ed29ec3d3e61df786f610926b739a9b",
    git_shallow=False,
)

_ACTLIZE_DIR = Path(os.environ.get("SGL_KERNEL_ACTLIZE_DIR", _ACTLIZE_REPO.source_dir))

_TRITON_REPO = _RepoInfo(
    name="triton",
    git_repository="https://github.com/triton-lang/triton",
    git_tag="v3.6.0",
    git_shallow=False,
)

ALL_REPOS = [
    _ACTLIZE_REPO,
    _TRITON_REPO,
]


# ======================= Version ======================= #
def _get_version():
    with open(root / "pyproject_ppu.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')
    return "0.0.0"


# ======================= Build Configuration ======================= #
operator_namespace = "sgl-kernel"

# Determine CXX ABI compatibility with PyTorch
torch_cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
abi_flag = f"-D_GLIBCXX_USE_CXX11_ABI={torch_cxx11_abi}"

compile_threads = int(os.environ.get("SGL_KERNEL_COMPILE_THREADS", "32"))
if compile_threads < 1:
    compile_threads = 1

# ======================= hgcc (device compiler) Flags ======================= #
# No `-gencode`: hgcc takes `--gpu-architecture=ppu_XX` instead, and SAIL torch
# emits that itself from the PYTORCH_SAIL_ARCH environment variable (see
# torch.utils.cpp_extension._get_cuda_arch_flags). Passing a CUDA `-gencode` here
# makes hgcc fail outright, so the arch selection is left entirely to
# PYTORCH_SAIL_ARCH (`ppu_10`, `ppu_15`, or a semicolon-separated list).
sgl_kernel_cuda_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    "-DCUTE_USE_PACKED_TUPLE=1",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    "-DCUTLASS_VERSIONS_GENERATED",
    "-DCUTLASS_TEST_LEVEL=0",
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    f"--threads={compile_threads}",
    # Suppress warnings
    "-Xcompiler=-Wno-clang-format-violations",
    "-Xcompiler=-Wno-conversion",
    "-Xcompiler=-Wno-deprecated-declarations",
    "-Xcompiler=-Wno-terminate",
    "-Xcompiler=-Wfatal-errors",
    "-Xcompiler=-ftemplate-backtrace-limit=1",
    # `-Xcudafe` is an nvcc-only hand-off to cudafe++, which hgcc has no
    # counterpart for and rejects. The diagnostics it used to suppress (177
    # unused-variable, 2361) are nvcc frontend numbers with no hgcc analogue.
    f"-Xcompiler={abi_flag}",
    abi_flag,
]

# ======================= Include Directories ======================= #
# Ordering matters. torch appends -I$CUDA_HOME/include (the CUDA-compat
# surface) *after* these, so the PPU SDK listed here wins for the 15 headers
# that exist under both prefixes -- most importantly hggc_runtime.h, where only
# the PPU SDK copy declares hggcError_t/hggcStream_t/hggcGetDevice, and the
# angle-bracket sub-includes it performs (<driver_types.h>, <vector_types.h>)
# which must come from the same tree to stay self-consistent.
#
# This does not shadow anything CUDA needs: CUDA's own headers reach those same
# 15 names through quoted includes, which resolve next to the including file in
# CUDA_SDK/include first, and no header in torch, actlize or csrc/ includes any
# of them with angle brackets directly.
include_dirs = [
    str(root / "include"),
    str(root / "csrc"),
    str(_PPU_SDK_INCLUDE),
    str(_ACTLIZE_DIR / "include"),
    str(_ACTLIZE_DIR / "tools" / "util" / "include"),
    str(_ACTLIZE_DIR / "examples" / "77_blackwell_fmha"),
    str(_ACTLIZE_DIR / "examples" / "common"),
]

# ======================= Source Files ======================= #
common_sources = [
    "csrc/allreduce/custom_all_reduce.cu",
    "csrc/attention/merge_attn_states.cu",
    "csrc/common_extension_ppu.cc",
    "csrc/elementwise/concat_mla.cu",
    "csrc/elementwise/copy.cu",
    "csrc/elementwise/pos_enc.cu",
    "csrc/elementwise/topk.cu",
    "csrc/gemm/awq_kernel.cu",
    "csrc/gemm/per_token_group_quant_8bit.cu",
    "csrc/gemm/per_token_group_quant_8bit_v2.cu",
    "csrc/gemm/per_token_quant_fp8.cu",
    "csrc/gemm/gptq/gptq_kernel.cu",
    "csrc/grammar/apply_token_bitmask_inplace_cuda.cu",
    "csrc/kvcacheio/transfer.cu",
    "csrc/mamba/causal_conv1d.cu",
    "csrc/memory/weak_ref_tensor.cpp",
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/fused_qknorm_rope_kernel.cu",
    "csrc/moe/moe_sum.cu",
    "csrc/moe/moe_sum_reduce.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    "csrc/moe/moe_topk_sigmoid_kernels.cu",
    "csrc/moe/prepare_moe_input.cu",
    "csrc/quantization/gguf/gguf_kernel.cu",
    "csrc/speculative/eagle_utils.cu",
    "csrc/speculative/ngram_utils.cu",
    "csrc/speculative/packbit.cu",
]

# Libraries to link.
#
# The cuda_free PPU SDK ships no libcuda/libcublas/libcublasLt; the sailified
# sources call the native spellings instead, so link their PPU counterparts:
#   cudart  -> hggcrt1   (libhggcrt1.so -> libhggcrt.13.0.so; there is no
#                         unversioned libhggcrt.so to satisfy a plain -lhggcrt)
#   cublas  -> acblas    (defines acblasHgemm, used by csrc/gemm/gptq)
#   cublasLt-> acblasLt
# The CUDA driver API (libcuda) is dropped: no csrc translation unit calls it.
# `library_paths()` from SAIL torch already contributes $PPU_SDK/lib, so no
# explicit library_dirs is needed.
libraries = ["c10", "hggcrt1", "acblas", "acblasLt"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib"]


# ======================= Extension Modules ======================= #
def _make_common_ops():
    """common_ops for SM80/SM89."""
    nvcc_flags = sgl_kernel_cuda_flags + ["-use_fast_math"]
    return CUDAExtension(
        name="sgl_kernel.common_ops",
        sources=common_sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": nvcc_flags,
            "cxx": ["-O3", "-std=c++17", abi_flag],
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    )


def _make_spatial_ops():
    """Spatial ops extension for green contexts."""
    spatial_sources = [
        "csrc/spatial/greenctx_stream.cu",
        "csrc/spatial_extension.cc",
    ]

    return CUDAExtension(
        name="sgl_kernel.spatial_ops",
        sources=spatial_sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": list(sgl_kernel_cuda_flags),
            "cxx": ["-O3", "-std=c++17", abi_flag],
        },
        libraries=["c10", "hggcrt1"],
        extra_link_args=extra_link_args,
        py_limited_api=False,
    )


# ======================= Clone Third-Party Repositories ======================= #
def _clone_and_checkout(repo_path, repo_url, git_tag, git_shallow):
    """Clone a git repository and checkout a specific tag/commit."""
    repo_path = Path(repo_path)
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    if not repo_path.exists():
        clone_cmd = ["git", "clone"]
        if git_shallow:
            clone_cmd += ["--depth", "1"]
        clone_cmd += [repo_url, str(repo_path)]
        print(f"Cloning {repo_url} -> {repo_path}")
        subprocess.check_call(clone_cmd)
        subprocess.check_call(["git", "checkout", git_tag], cwd=repo_path)


if os.environ.get("SGL_KERNEL_SKIP_THIRD_PARTY", "0") != "1":
    for _repo in ALL_REPOS:
        _clone_and_checkout(
            _repo.source_dir,
            _repo.git_repository,
            _repo.git_tag,
            _repo.git_shallow,
        )


# ======================= Build Extension Modules List ======================= #
ext_modules = [
    _make_common_ops(),
    _make_spatial_ops(),
]


# ======================= Custom Build Extension ======================= #
class _CustomBuildExt(BuildExtension):
    """Custom build extension with ninja support."""

    pass


# ======================= Package Discovery ======================= #
_triton_kernels_root = _TRITON_REPO.source_dir / "python" / "triton_kernels"

_packages = find_namespace_packages(where="python")
_package_dir = {"": "python"}

# Discover triton_kernels packages (may not exist before first clone)
if _triton_kernels_root.exists():
    _triton_pkgs = find_namespace_packages(
        where=str(_triton_kernels_root),
        exclude=["tests", "tests.*"],
    )
else:
    _triton_pkgs = ["triton_kernels"]

for pkg in _triton_pkgs:
    _package_dir[pkg] = os.path.relpath(
        _triton_kernels_root / pkg.replace(".", os.sep), root
    )
_packages += _triton_pkgs


# ======================= Setup ======================= #
setup(
    name="sglang-kernel",
    version=_get_version(),
    packages=_packages,
    package_dir=_package_dir,
    ext_modules=ext_modules,
    cmdclass={"build_ext": _CustomBuildExt.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
