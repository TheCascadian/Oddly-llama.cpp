# Porting the auto-overclock kit to another machine (instructions for an AI agent)

Goal: edit `gpu-push.py` (and optionally `gpu-tune.sh`, `watch.py`) so it finds the highest *stable* GPU memory and core clock offsets on this machine, using a llama.cpp workload as the stability test. It was written and run on: Ryzen 5 7600X, GTX 1660 Ti 6 GB, Linux, NVIDIA proprietary driver, Wayland, KDE. Result there: mem +1400 MHz-offset, core +105, about +5-8% decode.

## How it works (do not change the design unless the hardware forces it)
1. Sets V/F offsets through NVML (`nvmlDeviceSetMemClkVfOffset`, `nvmlDeviceSetGpcClkVfOffset`) via ctypes, in a `sudo python gpu-push.py _set <kind> <mhz>` subprocess. Offsets live in the driver only: reboot or crash resets them. Power limit uses `nvidia-smi -pl`.
2. Each step runs a deterministic workload: `llama-bench` decode, then `llama-perplexity` on a fixed corpus (4 chunks). A step passes only if: clean exit, no new `NVRM: Xid` kernel line, perplexity text equal to baseline digit for digit, decode speed >= 0.93 x best so far, temperature < 83 C.
3. Sweep memory up until the first failure, then core, back off `--margin` steps, then a soak (default 10 min) must pass. State is written to `results/gpu-oc.json` only after the soak passes. `--resume` re-verifies the saved pair (stepping down if it fails) and then only sweeps upward. `apply` re-sets saved values (use from a boot-time unit).

## Required edits for a different system (all in gpu-push.py unless noted)
| What | Where | Change to |
|---|---|---|
| Model file | `MODEL = ...` (line ~23) | A GGUF that fits in this GPU's VRAM entirely (`-ngl 99`) with the KV cache used below. Hardcoded path search under `~/.lmstudio/models`; replace with a path or env var. |
| llama.cpp build dir | `BUILD` (line ~22) | The build with `llama-bench` and `llama-perplexity`, built for this GPU's arch (e.g. `-DCMAKE_CUDA_ARCHITECTURES=86` for Ampere, 89 Ada, 120 Blackwell). |
| Perplexity corpus | `CORPUS` (line ~24) | Any fixed text file (~400 KB used). Keep it fixed; it is `results/ppl-corpus.txt`, copy any large plain-text file there. |
| KV cache flags | `-ctk q8_0 -ctv q4_0 -fa` in `workload()` | Drop the KV flags (or use `f16`) if the build lacks that quantized-KV pair. Keep the flags identical in bench and perplexity. |
| Threads | `-t 6` | Physical cores of the new CPU (only matters for non-offloaded work). |
| Context/depth | `-d 4096`, `-c 4096`, `--chunks 4` | Shrink if VRAM is tight; keep decode memory-bound. Workload should run 20-60 s per step. |
| Sweep limits | `opt` dict defaults | Mem/core max and steps. NVML offsets semantics differ by generation: the mem offset on Turing was half the displayed clock change. Check the real clock with `nvidia-smi -q -d CLOCK` after setting, and set steps so each is ~25-50 MHz real. Newer cards (Ada/Blackwell) accept larger core offsets (often +200..+300) but may reject mem offsets above a limit; the script already stops when the driver refuses. |
| Power limit | `--power` flag | Must be within `nvidia-smi -q -d POWER` Min/Max. Never exceed Max. |
| Temp ceiling | `t >= 83` in `workload()` | The card's real throttle point (see `nvidia-smi -q -d TEMPERATURE`, "GPU Slowdown Temp"), minus a few C. |
| GPU index | `nvmlDeviceGetHandleByIndex_v2(0, ...)` and `smi()` (no `-i`) | Pass the right index on multi-GPU (also add `-i N` to nvidia-smi calls and `CUDA_VISIBLE_DEVICES` for the workload). |
| Gateway service | `odysseus-gateway.service`, `pkill -x llama-server` in `main()` | Delete, or replace with whatever else uses the GPU. Nothing else may touch the GPU during a run. |
| Live view | `konsole ... watch.py` in `main()` | Replace `konsole` with the user's terminal, or remove. `watch.py` is optional; it renders `results/gpupush-*.log`. |
| sudo | `sudo -n` | Assumes passwordless within the cached sudo ticket (script runs `sudo -v` first and refreshes it). Fine on any sudo setup. |
| `gpu-tune.sh` | `BUILD=build-live`, `build-exp/bin/test-backend-ops`, gateway service lines | Optional guided front end. Fix paths or skip it. |

## Preflight on the new machine (verify before running the sweep)
1. `nvidia-smi` works; driver is the proprietary one (NVML offset calls do not exist on nouveau, and AMD/Intel need an entirely different mechanism: AMD uses `/sys/class/drm/card*/device/pp_od_clk_voltage` or `rocm-smi`/LACT; do not reuse the NVML code there, rewrite `set_offset` and `smi`).
2. `sudo python3 gpu-push.py _set mem 0; echo $?` returns 0. If nonzero, the driver rejects NVML offset calls (some laptops, some datacenter SKUs): stop and tell the user.
3. Run the workload once by hand and confirm: three consecutive perplexity runs give identical "Final estimate: PPL = x". If not, the stability check is not deterministic on this build/GPU; fix (e.g. pin `-t`, avoid non-deterministic kernels) before sweeping.
4. Tell the user it can hang the machine (power button) and that nothing should be saved or running on the GPU.

## Running
```
python3 gpu-push.py --mem-max <N> --core-max <N> --power <W>      # first run (3-run baseline, sweeps, soak)
python3 gpu-push.py --resume --power <W> --mem-max <N>            # later: re-verify, only go up
python3 gpu-push.py apply                                         # after each reboot (consider a systemd oneshot)
python3 gpu-push.py reset
```
Logs: `results/gpupush-<stamp>.log` (lines `TEST ... PASS/FAIL`, `PHASE ...`, `ALLDONE`). Result: `results/gpu-oc.json`.

## Known weak spots (verify/fix while porting)
- `sampler()` thread (power/temp during workload) was added late and is lightly tested.
- The 0.93 decode-regression threshold absorbs ~3% run-to-run noise; widen it if the new box is noisier.
- Perplexity equality is the key silent-corruption detector; keep it exact, do not round.
- Memory offset failures may appear as lower speed before wrong output: that is why decode is checked.
- Do not claim the result is "stable" to the user from the sweep alone; the soak passing is the bar, and longer real workloads should be watched.
