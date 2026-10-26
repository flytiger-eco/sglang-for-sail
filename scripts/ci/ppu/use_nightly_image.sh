#!/bin/bash
# Source only for an exact candidate-image run; never install or rebuild here.
if [[ "${PPU_TEST_BAKED_IMAGE:-0}" != "1" ]]; then
  echo "ERROR: use_nightly_image.sh requires PPU_TEST_BAKED_IMAGE=1" >&2
  return 1
fi
if [[ -n "${PPU_E2E_PLAN:-}" || "${PPU_INSTALL_PROFILE:-}" == "pr" ]]; then
  echo "ERROR: PPU_TEST_BAKED_IMAGE cannot be combined with PPU_E2E_PLAN or PPU_INSTALL_PROFILE=pr" >&2
  return 1
fi
if [[ ! "${PPU_BASE_IMAGE_DIGEST:-}" =~ ^sha256:[0-9a-f]{64}$ ||
      ! "${PPU_BASE_IMAGE:-}" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ||
      "${PPU_BASE_IMAGE##*@}" != "${PPU_BASE_IMAGE_DIGEST}" ]]; then
  echo "ERROR: PPU_BASE_IMAGE must be pinned to PPU_BASE_IMAGE_DIGEST (sha256:64hex)" >&2
  return 1
fi
if [[ ! "${PPU_CI_SOURCE_SHA:-}" =~ ^[0-9a-f]{40}$ ||
      "${SGLANG_BUILD_COMMIT:-}" != "${PPU_CI_SOURCE_SHA}" ]]; then
  echo "ERROR: SGLANG_BUILD_COMMIT must match the full PPU_CI_SOURCE_SHA" >&2
  return 1
fi
ppu_embedded_head=$(git -C /sgl-workspace/sglang rev-parse HEAD) || return 1
if [[ "$ppu_embedded_head" != "$PPU_CI_SOURCE_SHA" ]]; then
  echo "ERROR: embedded git HEAD must match SGLANG_BUILD_COMMIT and PPU_CI_SOURCE_SHA" >&2
  return 1
fi

cd /sgl-workspace/sglang || return 1
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export GITHUB_WORKSPACE=/sgl-workspace/sglang
export SGLANG_PPU_SOURCE_REVISION="$ppu_embedded_head"
python3 - <<'PY' || return 1
from pathlib import Path
import sglang

if not Path(sglang.__file__).resolve().is_relative_to(Path.cwd() / "python"):
    raise SystemExit("ERROR: not importing the baked image's SGLang Python package")
print(f"Baked SGLang: {sglang.__file__}")
PY

for ppu_config in SGLANG_PPU_{ANSWER,PERF,ACCURACY,PD_PERF}_TEST_CONFIG; do
  ppu_config_path="${!ppu_config:-}"
  if [[ "$ppu_config_path" == /workspace/source/* ]]; then
    printf -v "$ppu_config" '%s' "/sgl-workspace/sglang/${ppu_config_path#/workspace/source/}"
    export "${ppu_config?}"
  fi
done
unset ppu_embedded_head ppu_config ppu_config_path
