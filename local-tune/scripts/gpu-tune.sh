#!/usr/bin/env bash
# Guided GPU tuning for the 1660 Ti. Asks before each step, measures, and offers to undo.
# usage: local-tune/gpu-tune.sh        (run from a terminal; it asks for sudo when needed)
# Each step: apply -> bench.sh -> compare with the baseline -> keep or undo. Nothing changes without a "y".
set -uo pipefail
cd "$(dirname "$0")/.."
BUILD=build-live
STAMP=$(date +%m%d-%H%M)
BASE="gpu-base-$STAMP"
ask() { local a; read -r -p "$1 [y/N] " a; [[ $a == y || $a == Y ]]; }
tg() { # print pp/tg t/s for a bench label
  python3 - "$1" <<'P'
import csv, sys
for r in csv.reader(open(f"local-tune/results/{sys.argv[1]}.csv")):
    print(f"   {r[11].split('/')[-1][:28]:28} ngl={r[17]:>3} {'pp512' if int(r[-8]) else 'tg128'} {float(r[-2]):8.1f} t/s")
P
}
measure() { # label
  echo ">> benchmark $1 (about 2 minutes). Watch it in the viewer: python3 local-tune/watch.py"
  local-tune/bench.sh "$BUILD" "$1" -t 6 -fa 1 >/dev/null 2>&1 && tg "$1"
}
checkq() { # output check for any clock change: backend test + perplexity
  echo ">> correctness check"
  if build-exp/bin/test-backend-ops -o FLASH_ATTN_EXT 2>&1 | tail -3 | grep -q -i 'ok'; then echo "   test-backend-ops: ok"; else echo "   test-backend-ops: NOT CLEAN"; return 1; fi
  python3 local-tune/ppl.py run gpu-check >/dev/null 2>&1; grep -h -i -E 'final|ppl' "local-tune/results/ppl-gpu-check.csv" 2>/dev/null | tail -2
  echo "   Compare with 8.2 (+/- 0.14). A much higher number means the setting corrupts output: undo it."
  ask "   Is the perplexity within the normal range?"
}
smi() { nvidia-smi --query-gpu=persistence_mode,power.limit,clocks.max.sm,clocks.max.mem --format=csv,noheader; }

# --- automatic setup ---
# 1. live view: open it unless a watch.py is already running
if ! ps -C python3 -o args= | grep -q 'local-tune/watch.py'; then
  command -v konsole >/dev/null && setsid -f konsole --workdir "$PWD" -e bash -c 'while true; do python3 local-tune/watch.py; sleep 2; done' >/dev/null 2>&1
  sleep 3; ps -C python3 -o args= | grep -q 'local-tune/watch.py' && echo "viewer: opened" || echo "viewer: could not open, run python3 local-tune/watch.py yourself"
else echo "viewer: already running"; fi
# 2. sudo password once, kept alive while the script runs
sudo -v || { echo "sudo is needed"; exit 1; }
( while true; do sudo -n true; sleep 50; kill -0 "$$" 2>/dev/null || exit; done ) &
# 3. stop the gateway so no model sits in VRAM (and the server it starts); restarted at exit
GW=odysseus-gateway.service
if systemctl --user is-active --quiet "$GW"; then
  ask "Stop $GW for the test (it is restarted when the script ends)?" && systemctl --user stop "$GW" && { pkill -x llama-server; true; } && trap 'systemctl --user start "$GW"; echo "gateway restarted"' EXIT
fi
# 4. build-exp must have the test tool, build-live the bench tools
for f in "$BUILD/bin/llama-bench" "$BUILD/bin/llama-perplexity" build-exp/bin/test-backend-ops; do [[ -x $f ]] || { echo "missing $f"; exit 1; }; done
# 5. a GPU memory offset needs X11; say so now instead of at step 4
[[ ${XDG_SESSION_TYPE:-} == wayland ]] && echo "note: Wayland session, so step 4 (memory offset) will be skipped"
echo "Current GPU state (persistence, power limit, max core, max memory): $(smi)"
echo "Display VRAM in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
ask "Start with a baseline benchmark?" || exit 0
measure "$BASE"

echo; echo "== 1. Persistence mode (faster server start; does not change t/s)"
if ask "Enable persistence mode?"; then
  sudo nvidia-smi -pm 1 && echo "   on. It resets at reboot." && ask "   Keep it after reboot (systemctl enable nvidia-persistenced)?" && sudo systemctl enable --now nvidia-persistenced
fi

echo; echo "== 2. Power limit (efficiency; a lower limit may cost a few percent of t/s)"
echo "   Allowed range: 70-120 W. Current: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
if ask "Test a lower power limit?"; then
  read -r -p "   Watts to test (e.g. 100): " W
  sudo nvidia-smi -pl "$W" && measure "gpu-pl$W-$STAMP"
  ask "   Keep $W W?" || sudo nvidia-smi -pl 120
fi

echo; echo "== 3. Core clock lock (less jitter when the card hits its power cap)"
if ask "Test a core clock lock?"; then
  read -r -p "   Lock range min,max in MHz (e.g. 1500,1860): " R
  sudo nvidia-smi -lgc "$R" && measure "gpu-lgc-$STAMP"
  ask "   Keep the lock?" || sudo nvidia-smi -rgc
fi

echo; echo "== 4. Memory clock offset (the only setting that can raise decode speed; an overclock)"
echo "   Needs X11 with Coolbits=8 for nvidia-settings. Every step is followed by a correctness check."
if ask "Test a memory clock offset?"; then
  if ! command -v nvidia-settings >/dev/null || [[ -z ${DISPLAY:-} ]]; then
    echo "   nvidia-settings or an X display is missing, so this step cannot run here. Skipped."
  else
    for OFF in 250 500 750 1000; do
      ask "   Try +$OFF MHz transfer-rate offset?" || break
      nvidia-settings -a "[gpu:0]/GPUMemoryTransferRateOffsetAllPerformanceLevels=$OFF" >/dev/null 2>&1 || { echo "   the driver refused it (Coolbits?). Stopping."; break; }
      measure "gpu-mo$OFF-$STAMP"
      if ! checkq; then echo "   Failed the check. Going back to 0."; nvidia-settings -a "[gpu:0]/GPUMemoryTransferRateOffsetAllPerformanceLevels=0" >/dev/null 2>&1; break; fi
      LAST=$OFF
    done
    echo "   Highest offset that passed: ${LAST:-none}."
    ask "   Reset the offset to 0 now?" && nvidia-settings -a "[gpu:0]/GPUMemoryTransferRateOffsetAllPerformanceLevels=0" >/dev/null 2>&1
  fi
fi

echo; echo "== 5. Shared-memory huge pages (CPU layers of the 9B)"
echo "   Now: $(cat /sys/kernel/mm/transparent_hugepage/shmem_enabled)"
if ask "Test shmem_enabled=always?"; then
  echo always | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled >/dev/null && measure "gpu-thp-$STAMP"
  ask "   Keep it?" || echo never | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled >/dev/null
fi

echo; echo "Done. Results are in local-tune/results/ with labels $BASE and gpu-*-$STAMP."
echo "Tell Claude the stamp ($STAMP) and which steps you kept, and it will add the rows to TRIALS.md."
echo "Current state: $(smi)"
