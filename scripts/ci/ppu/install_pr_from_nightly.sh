#!/bin/bash
# Dependency changes incompatible with the passing image require a new candidate image.
set -euo pipefail

readonly EMBEDDED_REPO=/sgl-workspace/sglang
REPO_ROOT="$(cd "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must name the PR checkout}" && pwd -P)"
if [[ "${REPO_ROOT}" -ef "${EMBEDDED_REPO}" ]]; then
    echo "ERROR: PR checkout must not be the embedded image source (${EMBEDDED_REPO})" >&2
    exit 1
fi
embedded_head=$(git -C "${EMBEDDED_REPO}" rev-parse HEAD)
if [[ ! "${SGLANG_BUILD_COMMIT:-}" =~ ^[0-9a-f]{40}$ || "${SGLANG_BUILD_COMMIT}" != "${embedded_head}" ]]; then
    echo "ERROR: SGLANG_BUILD_COMMIT must match embedded git HEAD (${embedded_head})" >&2
    exit 1
fi
pr_head=$(git -C "${REPO_ROOT}" rev-parse HEAD)
echo "PPU PR source: ${pr_head}; tested image source: ${embedded_head}"
export GITHUB_WORKSPACE="${REPO_ROOT}"

# Compare Git objects because the kernel builder stamps versions into working files.
kernel_changed=0
for path in python/sglang/kernels/aot python/sglang/kernels/jit/include scripts/ci/ppu/ppu_build_kernel.sh; do
    embedded_tree=$(git -C "${EMBEDDED_REPO}" rev-parse "HEAD:${path}")
    pr_tree=$(git -C "${REPO_ROOT}" rev-parse "HEAD:${path}")
    if [[ "${embedded_tree}" != "${pr_tree}" ]]; then
        kernel_changed=1
    fi
done
embedded_gateway=$(git -C "${EMBEDDED_REPO}" rev-parse HEAD:sgl-model-gateway)
pr_gateway=$(git -C "${REPO_ROOT}" rev-parse HEAD:sgl-model-gateway)
gateway_changed=0
if [[ "${embedded_gateway}" != "${pr_gateway}" ]]; then
    gateway_changed=1
fi

# A global pip check would reject the image's intentional mooncake-barex override.
check_requirements() {
    python3 - "${EMBEDDED_REPO}" "${REPO_ROOT}" "$1" "${kernel_changed}" "${gateway_changed}" <<'PY'
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

mode = sys.argv[3]
rebuilt = {name for name, changed in zip(("sglang-kernel", "sglang-router"), sys.argv[4:]) if changed == "1"}


def candidate(reason):
    raise SystemExit(f"ERROR: {reason}; validate a candidate image with these dependencies first (no dependency upgrades in PR jobs)")


checked = set()


def require_installed(spec, extra="", defer_rebuilt=False):
    req = Requirement(spec)
    if req.marker and not req.marker.evaluate({"extra": extra}):
        return
    if req.url:
        candidate(f"changed URL requirement {req}")
    if defer_rebuilt and canonicalize_name(req.name) in rebuilt:
        return
    key = (str(req), extra)
    if key in checked:
        return
    checked.add(key)
    try:
        installed = distribution(req.name)
    except PackageNotFoundError:
        candidate(f"missing {req}")
    if not req.specifier.contains(installed.version, prereleases=True):
        candidate(f"{req} is not satisfied by installed {installed.version}")
    available_extras = installed.metadata.get_all("Provides-Extra") or []
    for requested_extra in req.extras:
        if requested_extra not in available_extras:
            candidate(f"installed {req.name} does not provide extra {requested_extra}")
        for child in installed.requires or []:
            require_installed(child, requested_extra)


for relative in ("python/pyproject_ppu.toml", "sgl-model-gateway/bindings/python/pyproject.toml"):
    old, new = [tomllib.loads((Path(root) / relative).read_text()) for root in sys.argv[1:3]]
    if old.get("tool", {}).get("uv") != new.get("tool", {}).get("uv"):
        candidate(f"dependency source configuration changed in {relative}")
    old_build, new_build = old.get("build-system", {}), new.get("build-system", {})
    for key in ("build-backend", "backend-path"):
        if old_build.get(key) != new_build.get(key):
            candidate(f"{key} changed in {relative}")
    old_project, new_project = old["project"], new["project"]
    python_spec = new_project.get("requires-python", "")
    if not SpecifierSet(python_spec).contains(".".join(map(str, sys.version_info[:3])), prereleases=True):
        candidate(f"Python does not satisfy {python_spec}")
    # CI installs no project extras; requested dependency extras are checked recursively.
    groups = [(old_build.get("requires", []), new_build.get("requires", []), False),
              (old_project.get("dependencies", []), new_project.get("dependencies", []), True)]
    for before, after, runtime in groups:
        if mode == "preflight":
            for spec in sorted(set(after) - set(before)):
                require_installed(spec, defer_rebuilt=runtime)
        elif runtime:
            # Rebuilding can invalidate even a pin unchanged from the tested image.
            for spec in after:
                if canonicalize_name(Requirement(spec).name) in rebuilt:
                    require_installed(spec)
PY
}
check_requirements preflight

# Ignore inherited overrides that could silently reuse the image's Rust extensions.
export SGLANG_BUILD_RUST_EXTS=all
export CARGO_TARGET_DIR="${REPO_ROOT}/target/ppu-ci"
cp "${REPO_ROOT}/python/pyproject_ppu.toml" "${REPO_ROOT}/python/pyproject.toml"
python3 -m pip install --no-deps --no-build-isolation -e "${REPO_ROOT}/python"

_evidence="${PPU_INSTALL_EVIDENCE_DIR:-${PPU_EVIDENCE_DIR:-${REPO_ROOT}/ppu-install-evidence}}"
mkdir -p "${_evidence}"
_kernel_seconds=0
_kernel_rc=0
if [[ "${kernel_changed}" == 1 ]]; then
    _kernel_start=$SECONDS
    # The legacy cache key omits JIT headers and build scripts, so changed inputs bypass it.
    SGL_KERNEL_WHEEL_CACHE=0 bash "${REPO_ROOT}/scripts/ci/ppu/ppu_build_kernel.sh" || _kernel_rc=$?
    _kernel_seconds=$((SECONDS-_kernel_start))
else
    echo "Kernel inputs unchanged; retaining the tested image kernel"
fi
printf '{"aot_seconds":%s}\n' "${_kernel_seconds}" > "${_evidence}/aot.json"
if [[ "${_kernel_rc}" != 0 ]]; then
    exit "${_kernel_rc}"
fi

if [[ "${gateway_changed}" == 1 ]]; then
    _wheels=$(mktemp -d "${REPO_ROOT}/.ppu-gateway-wheels.XXXXXX")
    trap 'rm -rf "${_wheels}"' EXIT
    cd "${REPO_ROOT}/sgl-model-gateway/bindings/python"
    maturin build --release --features vendored-openssl --out "${_wheels}"
    cd "${REPO_ROOT}/sgl-model-gateway"
    cargo build --release --bin sgl-model-gateway --features vendored-openssl
    python3 -m pip install --no-deps --no-build-isolation --force-reinstall "${_wheels}"/*.whl
    install -m 755 "${CARGO_TARGET_DIR}/release/sgl-model-gateway" /usr/local/bin/sgl-model-gateway
else
    echo "Gateway inputs unchanged; retaining the tested image bindings and binary"
fi
check_requirements postbuild
