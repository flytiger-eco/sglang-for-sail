#!/bin/bash
# Install runner-owned EIC artifacts at build time so nightly tests the published image.
set -euo pipefail

EIC_PKG_DIR="${1:-/tmp/ppu-eic}"
rdma=("${EIC_PKG_DIR}"/ali-rdma-core_*.deb)
barex=("${EIC_PKG_DIR}"/accl-barex-cuda13-*.deb)
mooncake=("${EIC_PKG_DIR}"/mooncake_transfer_engine-0.3.6.post1-*.whl)
for artifact in "${rdma[@]}" "${barex[@]}" "${mooncake[@]}" "${EIC_PKG_DIR}/u2mm.deb"; do
    if [[ ! -f "${artifact}" ]]; then
        echo "ERROR: missing required artifact in ppu-eic context: ${artifact}" >&2
        exit 1
    fi
done

# Required component failures must fail the candidate build without dependency resolution.
dpkg -i --force-overwrite "${rdma[@]}"
dpkg -i "${barex[@]}"
dpkg -i "${EIC_PKG_DIR}/u2mm.deb"
tools=("${EIC_PKG_DIR}"/eic-tools_*.deb)
if [[ -f "${tools[0]}" ]]; then
    dpkg -i "${tools[@]}" || echo "WARNING: optional eic-tools install failed" >&2
fi
if [[ -f "${EIC_PKG_DIR}/libunicm.so" ]]; then
    mkdir -p /usr/lib64
    cp "${EIC_PKG_DIR}/libunicm.so" /usr/lib64/
fi
if [[ -d /usr/lib64 ]]; then
    mkdir -p /etc/ld.so.conf.d
    printf '/usr/lib64\n' > /etc/ld.so.conf.d/ppu-eic.conf
fi
ldconfig
# The intentional barex override must be part of the tested image, not a runtime substitution.
python3 -m pip install --no-deps --no-build-isolation --force-reinstall "${mooncake[@]}"
echo "EIC SDK installation complete."
