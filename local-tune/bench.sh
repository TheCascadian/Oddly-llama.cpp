#!/usr/bin/env bash
# usage: bench.sh <build-dir> <label> [extra llama-bench args...]
# Runs llama-bench across the representative models on this machine; appends CSV to results/<label>.csv
set -euo pipefail
B=${1:?build dir}; L=${2:?label}; shift 2
export PATH=/opt/cuda/bin:$PATH
M=~/odysseus-local/models
Q=~/.lmstudio/models/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf
mkdir -p "$(dirname "$0")/results"; OUT="$(dirname "$0")/results/$L.csv"; : > "$OUT"
run() { # model ngl
  "$B/bin/llama-bench" -m "$1" -ngl "$2" -p 512 -n 128 -r 3 -o csv "${@:3}" 2>/dev/null | tail -n +2 >> "$OUT"
}
run "$M/deepseek-coder-1.3b-base-Q8_0.gguf" 99 "$@"
run "$M/qwen2.5-3b-instruct-q5_k_m.gguf" 99 "$@"
run "$Q" 22 "$@"     # hybrid: 9B Q4_K_M, 22 layers on GPU
run "$Q" 0 "$@"      # CPU only
echo "wrote $OUT"
