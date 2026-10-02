#!/usr/bin/env bash
# Start the local model router using this fork's working Intel build.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
server="$root/build-intel-all/bin/llama-server"
presets=${PRISM_MODELS_PRESET:-/mnt/Data/Projects/Models/presets.ini}
load_mode=${PRISM_LOAD_MODE:-mmap}
[[ -x "$server" ]] || { echo "Missing server: $server" >&2; exit 1; }
[[ -r "$presets" ]] || { echo "Cannot read presets: $presets" >&2; exit 1; }
export GGML_SYCL_PTQ1_T2=${GGML_SYCL_PTQ1_T2:-ffn}
export GGML_SYCL_T2_W8A8_MIN=${GGML_SYCL_T2_W8A8_MIN:-0}
export GGML_SYCL_FA_ONEDNN_MAX_KV=${GGML_SYCL_FA_ONEDNN_MAX_KV:-98304}
export LLAMA_ARG_SPEC_DRAFT_UBATCH=${LLAMA_ARG_SPEC_DRAFT_UBATCH:-64}
exec "$root/scripts/run-sycl-level-zero.sh" "$server" \
    --host "${PRISM_HOST:-127.0.0.1}" --port "${PRISM_PORT:-8091}" \
    --load-mode "$load_mode" \
    --models-preset "$presets" --models-max 1 --models-autoload "$@"
