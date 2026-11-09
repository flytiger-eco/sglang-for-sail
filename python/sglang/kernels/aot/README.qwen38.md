# Qwen38 W4A8 on Hopper

This module integrates the frozen Qwen38 INT4 tiling, fast packed-weight
conversion, and fused gate-up epilogue into the AOT kernel package.
The former `sgl-kernel` source tree lives at `python/sglang/kernels/aot`.
Humming MXFP4 is an optional dependency-backed interface in the same package;
SGLang already supplies its Humming MoE runner and checkpoint loader.

## Shapes and scope

For global E=512, top-k=10, TP4/EP4, each rank owns 128 routed experts:

| Projection | Per-expert A | Logical B | Output |
|---|---|---|---|
| Gate + up | `[m_e,2560]` FP8 E4M3 | `[1280,2560]` INT4 | `[m_e,640]` FP8 E4M3 after SwiGLU/static quantization |
| Down | `[m_e,640]` FP8 E4M3 | `[2560,640]` INT4 | `[m_e,2560]` BF16 |

The grouped kernels accept contiguous rows or `[E,capacity,K]` masked buffers.
Device problem sizes determine valid rows, including empty experts. Host-known
row counts choose a frozen tile; forward does not read route counts back to CPU.
4K/5K/6K/7K/8K balanced prefill corresponds to 80/100/120/140/160 rows per expert.
Decode batch 32 corresponds to approximately 80 total rows on one EP4 rank,
not 80 rows per expert. Uneven distributions use the same valid-row metadata.

The prefill dispatch table uses channel/token/K tile dimensions:

| Global chunk | Gate-up tile | Gate schedule / cluster | Down tile |
|---|---|---|---|
| 4096 | 128/80/256 | cooperative / 2x1x1 | 256/80/128 |
| 5120 | 128/112/256 | cooperative / 1x1x1 | 128/128/128 |
| 6144 | 128/128/256 | cooperative / 1x1x1 | 128/128/128 |
| 7168 | 128/144/256 | cooperative / 1x1x1 | 128/160/128 |
| 8192 | 128/160/256 | cooperative / 1x1x1 | 128/160/128 |
| Decode batch about 32 | 64/32/256 | ping-pong / 1x1x1 | 128/32/128 (ping-pong) |

Down prefill tiles use cooperative scheduling and a 1x1x1 cluster. The
smallest decode cases additionally retain a K512 gate variant and a
cooperative down variant; see `select_w4a8_sm_config` for the exact hints.

The fused path uses FP32 GEMM accumulators for SwiGLU, then static E4M3
quantization. It does not insert the BF16 intermediate rounding of an unfused
GEMM + activation sequence. Numerical comparisons therefore use dequantized
FP32 references and format-appropriate tolerances, rather than requiring
bitwise equality with the unfused implementation.

## Checkpoint processing

`prepare_w4a8_sm_weight` runs once in `process_weights_after_loading`:

1. Validate canonical signed INT4, low nibble first, `[E,N,K/2]`, with logical
   group-128 scales `[E,N,K/128]`.
2. Convert two's-complement values to sign-magnitude. This kernel represents
   **[-7,7]**. A checkpoint containing **-8 is rejected**, never clipped.
3. Interleave gate/up channels and apply the complete CUTLASS N/K lane mapping
   and register permutation. A simple adjacent-K permutation is insufficient.
4. Multiply weight scales by 512 to compensate the E4M3 subnormal conversion;
   pack gate scales for K256/K512 tiles and down scales for K128 tiles.
5. Replace both projections only after both validate. Repeated processing is
   idempotent. No full dequantization/requantization is performed.

Checkpoint files retain their canonical format. Prepared runtime tensors are
kernel-specific and must not be saved as a canonical W4AFP8 checkpoint.

`SGLANG_W4AFP8_MOE_BACKEND` selects `auto` (default), `ours`, or `upstream`.
`auto` selects Ours only on Hopper with H=2560/I=640, group-128 weights and
static activations. Missing extension or incompatible weight values preserve
the upstream layout and log a warning. `ours` makes those conditions errors;
use it when validating deployment so fallback cannot be mistaken for Ours.
Other model shapes retain the upstream implementation.

The standard runner, DeepEP normal, and DeepEP low-latency paths are wired to
this loader. Normal dispatch consumes BF16 and quantizes to the checkpoint's
static scale. Low-latency dispatch consumes FP8 plus per-token-group scales
and requantizes to that static scale. Standard route metadata is normalized
to int32 before the C++ metadata kernel.

## Building

Build the full kernel wheel in the target SGLang environment (Hopper, CUDA
12.8 or newer, matching Torch/CUDA development dependencies):

```bash
python -m pip install --no-build-isolation --no-deps ./python/sglang/kernels/aot
```

`SGL_KERNEL_BUILD_W4A8_SM=ON` is the default CMake option. The resulting
`sgl_kernel/w4a8_sm_ops.so` is loaded lazily. It contains only the Qwen38
SM90a variants and does not replace existing `common_ops` operators.
Private CUTLASS mainloop headers preserve the measured implementation without
altering other kernels' CUTLASS templates.

For a focused module build against an existing compatible kernel installation:

```bash
cmake -S python/sglang/kernels/aot -B build-qwen38 \
  -DSGL_KERNEL_W4A8_SM_ONLY=ON \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-qwen38 --target w4a8_sm_ops -j2
```

Install the built module and the two Python adapters into an **isolated**
`sgl_kernel` package based on the same runtime environment. The focused build
is not a replacement for a complete SGLang wheel build. CMake fetches the
pinned CUTLASS revision; offline builds may supply
`-DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS=/path/to/cutlass-57e3cfb`.

## Humming MXFP4

`sgl_kernel.humming_mxfp4` exports `prepare_humming_mxfp4_moe` and
`humming_mxfp4_moe_mm`. Supply native, untransformed Humming E2M1 weights,
group-32 E8M0 scales and E4M3 activations. Load-time `transform()` prepares
Humming's layout once. Forward retains default heuristic configurations;
there is no tile search. Masked buffers use a capacity divisible by four,
with exact valid counts carried separately. Flattening is a view, not a copy.

For serving, use SGLang's existing `--quantization humming` and
`--moe-runner-backend humming` with a compatible Humming-format checkpoint.
To explicitly choose E4M3 tensorwise activation quantization:

```bash
export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3","group_size":0}'
```

MXFP4 weights are not interchangeable with an INT4 checkpoint. The adapter
neither silently requantizes INT4 nor overrides GPT-OSS's separate MXFP4
checkpoint-loading path. `humming-kernels` supplies its kernel implementation;
the target SGLang dependency already pins its version.

## Validation commands

Use an otherwise idle Hopper and select the intended physical GPU. All tests
operate on synthetic canonical checkpoint tensors, not production model
quality or network communication:

```bash
CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID python -m pytest -q \
  python/sglang/kernels/aot/tests/test_qwen38_w4a8_sm.py \
  python/sglang/kernels/aot/tests/test_humming_mxfp4_moe.py \
  python/sglang/kernels/aot/tests/test_qwen38_w4a8_runtime.py

CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID python \
  python/sglang/kernels/aot/benchmark/bench_qwen38_w4a8_sm.py \
  --output /path/to/new-results \
  --m 30 32 36 4096 5120 6144 7168 8192 --rounds 5 --iters 20
```

The benchmark validates all nonempty experts against FP32 dequantized
references. It captures a CUDA Graph after warmup and uses Kineto correlation
IDs to attribute GPU events to each replay. Each measurement sums GPU event
durations, excluding host time and inter-kernel gaps. Five guard replays are
excluded; each round takes the median of 20 calls and the result is the median
of five rounds. Raw traces and CUDA Event cross-checks are saved. This measures
single-rank operator work, not end-to-end TPOT or communication.

An optional `--baseline-configs FILE.json` compares the integrated module with
frozen benchmark libraries on identical tensors, checks their SHA256 hashes,
and requires byte-for-byte output agreement before timing both.

## Validation snapshot (2026-09-12)

Physical GPU4, H800 132 SM: 43 tests and 6 subtests passed after a clean
CMake build. This includes load-time packing, all 12 tile variants in 2D/3D,
changed routing under CUDA Graph replay, native Humming MXFP4, and the actual
SGLang loader/standard/DeepEP-normal/DeepEP-low-latency compute paths. DeepEP
payloads were supplied locally; this does not exercise four-rank communication.

Three independent benchmark processes (different data seeds) verified 48
byte-identical new-versus-frozen outputs. Independent parsing verified 480
Kineto traces, 9,600 formal replays and 19,200 GPU events. All per-expert
checks passed: worst relative L1 0.001847, minimum cosine 0.999668.

| Global M | Integrated gate+down (us) | Frozen gate+down (us) | Change |
|---|---|---|---|
| 30 | 106.831 | 106.752 | +0.07% |
| 32 | 113.552 | 113.920 | -0.32% |
| 36 | 124.127 | 124.352 | -0.18% |
| 4096 | 207.918 | 207.599 | +0.15% |
| 5120 | 254.365 | 253.440 | +0.36% |
| 6144 | 258.510 | 256.863 | +0.64% |
| 7168 | 280.096 | 278.783 | +0.47% |
| 8192 | 291.005 | 291.391 | -0.13% |

Values are sums of the gate/down medians across three independent processes.
The frozen baseline uses the same selected tiles and tensors. This comparison
checks integration regression; it is not a new Humming speedup measurement.

Runtime: Torch 2.9.1+cu129, CUDA/NVCC 12.9.86, Triton 3.5.1, Transformers
5.12.1, TVM-FFI 0.1.11, xgrammar 0.2.1. Tests used an isolated overlay of the
new module/adapters on the existing `sglang-kernel 0.4.1+mergefix0716` base
package. Humming source was `3c1561bceee85ec3fbc2e3caf87bf90391113af2`.
The branch's complete Torch 2.13/CUDA 13 wheel and full-model TPOT/quality
were not tested by this single-GPU validation. Rebuild the kernel wheel in
the deployment environment; do not copy a Torch 2.9 binary into Torch 2.13.

Clean-built `w4a8_sm_ops.so` SHA256:

```text
31c988b19dcb8f4b11e2ff9ccb57733cbed1405b84ef2395d64f7e1d79ea1710
```
