# Local tuning: Ryzen 5 7600X + GTX 1660 Ti (6 GB), CachyOS

Rebuild: `local-tune/build.sh` (CUDA 13.4 from `/opt/cuda`, gcc 16). Benchmark: `local-tune/bench.sh <build-dir> <label> -t 6`, summarize with `local-tune/summarize.py local-tune/results/<label>.csv`.

## Build flags
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release` (+ccache if present).
Kept at defaults after measuring: `GGML_CUDA_FORCE_MMQ` (no change, within 1%), `GGML_CUDA_HOPPER_Q1` (sm_90a only, left off).

## Code change
`src/models/qwen35.cpp`: on x86, layers that run on the CPU skip the fused raw-gate GDN path (it was 20-35% slower than the
separate sigmoid/softplus nodes). Env `GGML_GDN_RAW_GATES_DISABLE=1` did the same before this change.

## Results (llama-bench, -t 6, pp512 / tg128, t/s)
| model | placement | Oddly before | Oddly after | plain llama.cpp (same base) |
|---|---|---|---|---|
| deepseek-coder-1.3b Q8_0 | GPU | 856 / 158 | ~833 / ~152 (noise) | 829 / 150 |
| qwen2.5-3b Q5_K_M | GPU | 390 / 94 | ~378 / ~91 (noise) | 379 / 90 |
| Qwythos-9B Q4_K_M | ngl 22 hybrid | 153 / 15.6 | 150 / 20.6 | 151 / 20.3 |
| Qwythos-9B Q4_K_M | CPU | 142 / 7.7 | 139 / 9.4 | 139 / 9.4 |

## Runtime defaults (measured)
- Threads 6 (physical cores); 4/8 equal, 12 slightly worse.
- `-ub 512`, flash attention on (`-fa on`); smaller ubatch is slower.
- 9B Q4_K_M: `-ngl 26` is the ceiling at small context (27+ aborts with a CUDA OOM); use `-ngl 22` with `-c 16384 -ctk q8_0 -ctv q8_0`
  (4.5 GB VRAM used with the desktop running).
- Models up to ~3B fully offload (`-ngl 99`).
