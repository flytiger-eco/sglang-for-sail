# Opt-in vision policy on v0.5.21

Adapted from `18d93fac3567116812d5849e5a5a2e275683ecad`.

The default visual tower remains unquantized. The new configuration fields live
in the `mm` namespace and are validated by the normal argument-resolution
pipeline, including the dummy-model path.

- `--vision-quantization fp8` quantizes unquantized visual checkpoint tensors
  online. Serialized quantized tensors and scale metadata are rejected.
- `--vision-model-path PATH` replaces only visual tensors; language-model
  tensors still come from `--model-path`. The separate checkpoint must supply
  all required visual weights and biases. Unknown, missing and duplicate visual
  tensors are rejected.
- `SGLANG_QUANTIZE_VIT=1` explicitly inherits a supported FP8 backbone policy.
  An unsupported inherited quantization policy is rejected.

These options require a supported native Qwen3VL-derived visual tower, serial
DefaultModelLoader loading and no weight cache. Separate visual weights require
pipeline parallel size 1. ModelOpt/INC and MXFP8 loading are not supported.
Literal visual exclusion paths are remapped without mutating the language
model's quantization policy.

On PPU, channelwise FP8 linears use DeepGEMM with normalized scale layouts.
Biased visual linears and arbitrary leading input dimensions preserve the
output shape and output dtype.

## Validation

Policy and real CPU-loader integration tests:
```bash
python3 -m pytest -q \
  test/registered/unit/multimodal/test_aim_vision_policy.py \
  test/registered/unit/multimodal/test_aim_vision_policy_integration.py
```

Real PPU numerical tests compare quantized matrix multiplication against an FP32
reference, covering two-dimensional and three-dimensional inputs, optional bias,
and vector/column channel scales:
```bash
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 python3 -m pytest -q \
  test/manual/test_aim_vision_fp8_ppu.py
```

End-to-end feature activation, performance and multimodal comparisons are recorded
separately for each tested source revision and container image. Unit-test success
alone does not establish end-to-end correctness.
