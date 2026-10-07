#!/usr/bin/env bash
# usage: bench.sh <build-dir> <label> [extra llama-bench args...]
# Runs llama-bench across the representative models on this machine; appends CSV to results/<label>.csv
# results/<label>.log gets START/TEST/DONE lines with how long each model and each test took.
set -euo pipefail
B=${1:?build dir}; L=${2:?label}; shift 2
export PATH=/opt/cuda/bin:$PATH
M=${BENCH_MODELS:-$HOME/odysseus-local/models}   # folder with the small models; set BENCH_MODELS to use your own
Q=${BENCH_9B:-$HOME/.lmstudio/models/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf}
OUT="$(python3 "$(dirname "$0")/paths.py" "$L.csv")"; : > "$OUT"
LOG="${OUT%.csv}.log"; : > "$LOG"; T0=$(date +%s)
TIMER='BEGIN{t=systime()} NR>1{print; fflush(); n=systime(); a=$(NF-7); d=$(NF-5); gsub(/"/,"",a); gsub(/"/,"",d); printf "%s   TEST %s@%s took=%ds\n", strftime("%T"), (a>0?"pp":"tg"), d, n-t >> lf; fflush(lf); t=n}'
run() { # model ngl
  [[ -f "$1" ]] || { echo "$(date +%T) SKIP $(basename "$1" .gguf | cut -c1-18) (file not found)" >> "$LOG"; return 0; }
  local t1; t1=$(date +%s); echo "$(date +%T) START $(basename "$1" .gguf | cut -c1-18) ngl=$2" >> "$LOG"
  "$B/bin/llama-bench" -m "$1" -ngl "$2" -p 512 -n 128 -r 3 -o csv "${@:3}" 2>/dev/null | awk -F, -v lf="$LOG" "$TIMER" >> "$OUT"
  echo "$(date +%T) DONE took=$(( $(date +%s) - t1 ))s" >> "$LOG"
}
run "$M/deepseek-coder-1.3b-base-Q8_0.gguf" 99 "$@"
run "${BENCH_MID:-$M/qwen2.5-3b-instruct-q5_k_m.gguf}" 99 "$@"   # mid-size model, all on GPU; set BENCH_MID to use another file
run "$Q" 22 "$@"     # hybrid: 9B Q4_K_M, 22 layers on GPU
run "$Q" 0 "$@"      # CPU only
echo "ALLDONE total=$(( $(date +%s) - T0 ))s" >> "$LOG"
echo "wrote $OUT"
