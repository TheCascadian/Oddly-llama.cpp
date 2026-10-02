#!/usr/bin/env bash
# Run a command with the project-local Level Zero loader and matched oneAPI.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runtime="$root/.runtime/level-zero-1.32.0"
if [[ ${1:-} == --setup ]]; then
    command -v curl >/dev/null
    command -v dpkg-deb >/dev/null
    staging=$(mktemp -d)
    trap 'rm -rf -- "$staging"' EXIT
    curl --fail --location --retry 3 'https://github.com/oneapi-src/level-zero/releases/download/v1.32.0/libze1_1.32.0%2Bu22.04_amd64.deb' -o "$staging/libze1.deb"
    echo 'ce124a8f9cd049bc3177d940b4aacacbc4c5087ca7c0a844639b9c9c9c23a9ec  '"$staging/libze1.deb" | sha256sum --check
    dpkg-deb --extract "$staging/libze1.deb" "$staging/extracted"
    if [[ -e "$runtime" ]]; then
        cmp "$runtime/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0" \
            "$staging/extracted/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0"
    else
        mkdir -p "$(dirname -- "$runtime")"
        mv "$staging/extracted" "$runtime"
    fi
    printf 'Installed isolated loader in %s\n' "$runtime"
    exit 0
fi
if [[ $# == 0 ]]; then
    echo "Usage: $0 --setup | COMMAND [ARG ...]" >&2
    exit 2
fi
loader="$runtime/usr/lib/x86_64-linux-gnu"
[[ -f "$loader/libze_loader.so.1.32.0" ]] || { echo "Run $0 --setup first" >&2; exit 1; }
echo 'e31b3a9e9efdd8ac4912cf617293babcfe30dd2caba30caa66664db82246ed95  '"$loader/libze_loader.so.1" | sha256sum --check --status || {
    echo 'Isolated loader checksum mismatch; refusing to launch' >&2
    exit 1
}
oneapi=${PRISM_ONEAPI_ROOT:-/opt/intel/oneapi}
[[ -d "$oneapi/compiler/2026.1/lib" && -d "$oneapi/mkl/2026.1/lib" ]] || { echo 'Matched oneAPI compiler and MKL 2026.1 required' >&2; exit 1; }
set +u
source "$oneapi/setvars.sh" --force >/dev/null 2>&1
set -u
export LD_LIBRARY_PATH="$loader:$oneapi/compiler/2026.1/lib:$oneapi/mkl/2026.1/lib:${LD_LIBRARY_PATH:-}"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
exec "$@"
