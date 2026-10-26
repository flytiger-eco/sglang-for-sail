#!/usr/bin/env bash
# Fail closed around the legacy best-effort resolver; PRs must never fall back to a tag.
set -euo pipefail

fail() {
  printf '::error::%s\n' "$1" >&2
  exit 1
}

image="${PPU_BASE_IMAGE:-}"
# Fully qualified registry (optional port) and Docker repository components.
host='[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?'
component='[a-z0-9]+(([._]|__|-+)[a-z0-9]+)*'
repository_pattern="${host}(\.${host})*(:[0-9]+)?/${component}(/${component})*"
[[ "$image" =~ ^${repository_pattern}:v0\.5\.18-latest$ ]] ||
  fail 'PPU_BASE_IMAGE must be a fully qualified repository:v0.5.18-latest; configure PPU_NIGHTLY_IMAGE_REPOSITORY.'
[[ -n "${GITHUB_ENV:-}" && -n "${GITHUB_OUTPUT:-}" ]] ||
  fail 'GITHUB_ENV and GITHUB_OUTPUT are required.'

script_dir="$(dirname -- "${BASH_SOURCE[0]}")"
repository="${image%:*}"
temporary="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/ppu-nightly-image.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT
# Isolate legacy output until validated. Never source registry-controlled text.
GITHUB_ENV="$temporary/env" GITHUB_OUTPUT="$temporary/output" \
  bash "$script_dir/resolve_image_digest.sh" || fail 'Nightly image lookup failed.'
[[ -f "$temporary/env" ]] || fail 'Nightly image lookup produced no identity.'

ref=''
digest=''
seen_ref=false
seen_digest=false
while IFS='=' read -r key value; do
  case "$key" in
    PPU_BASE_IMAGE_REF)
      [[ "$seen_ref" == false ]] || fail 'Duplicate image reference.'
      seen_ref=true
      ref="$value"
      ;;
    PPU_BASE_IMAGE_DIGEST)
      [[ "$seen_digest" == false ]] || fail 'Duplicate image digest.'
      seen_digest=true
      digest="$value"
      ;;
    *) fail 'Unexpected nightly image lookup output.' ;;
  esac
done < "$temporary/env"

[[ "$digest" =~ ^sha256:[a-f0-9]{64}$ ]] ||
  fail 'No valid successful nightly digest is available; refusing tag fallback.'
[[ "$ref" == "$repository@$digest" ]] ||
  fail 'Nightly reference does not match the requested repository and digest.'

printf 'PPU_BASE_IMAGE_REF=%s\nPPU_BASE_IMAGE_DIGEST=%s\n' "$ref" "$digest" >> "$GITHUB_ENV"
printf 'image=%s\ndigest=%s\n' "$ref" "$digest" >> "$GITHUB_OUTPUT"
