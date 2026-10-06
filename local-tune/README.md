# Local tuning: Ryzen 5 7600X + GTX 1660 Ti (6 GB), CachyOS

Two code changes live on this branch. Both are measured on this machine only.

| # | Change | Effect | Commit |
|---|---|---|---|
| 1 | CUDA: stop using tensor-core kernels on GTX 16xx | prompt processing 2.9x to 4.2x faster | `a260d811e` |
| 2 | qwen35: skip fused raw-gate GDN path on CPU layers (x86) | 9B hybrid/CPU decode 20-35% faster | `52b730eb4` |

## Change 1: GTX 16xx kernel selection

### In one sentence
GTX 16xx cards report the same compute capability (7.5) as RTX 20xx but have no tensor cores, so the tensor-core (MMA) kernels ran through slow emulation; the card is now detected by name and gets the plain kernels.

### What you get (GTX 1660 Ti, old build vs new build, same session, 3 repetitions)

```
Prompt processing, tokens/sec (higher is better)

deepseek-1.3b Q8_0   GPU      old  860 |######
                              new 2901 |####################   3.4x
qwen2.5-3b Q5_K_M    GPU      old  388 |###
                              new 1331 |#########              3.4x
R1-distill-7B Q4_K_S GPU      old  168 |#
                              new  690 |#####                  4.1x
Qwythos-9B Q4_K_M    ngl 22   old  156 |#
                              new  500 |###                    3.2x
Qwythos-9B Q4_K_M    CPU      old  140 |#
                              new  401 |###                    2.9x
```

| Question | Answer |
|---|---|
| Reading a prompt | 2.9x to 4.2x faster in every case measured |
| Writing, short chat | unchanged (-1% to +3%) |
| Writing, long chat, q8_0/q4_0 KV on GPU | unchanged within noise (-6% to +9%, median -2%) |
| Writing, long chat, f16 KV on GPU at 8K | 7% slower (42.0 vs 45.0 t/s). This is the one cost. |
| Writing, KV in RAM (`-nkvo 1`) | unchanged (-4% to +2%) |
| Out-of-memory stops, f16 KV at 8K | 0 in 9 runs |
| Output quality | perplexity equal within error; `test-backend-ops -o FLASH_ATTN_EXT` passes |

### KV matrix (R1-distill-7B Q4_K_S, `-ngl 99 -fa 1`, old -> new, t/s)

| KV type | KV on | pp512 @0K | pp512 @8K | pp512 @16K | tg32 @0K | tg32 @8K | tg32 @16K |
|---|---|---|---|---|---|---|---|
| f16  | GPU | 168 -> 690 | 110 -> 439 | does not fit | 54.1 -> 55.0 | 45.0 -> 42.0 | does not fit |
| f16  | RAM | 170 -> 678 | 109 -> 423 | 77 -> 300 | 48.1 -> 48.9 | 15.9 -> 15.5 | 9.7 -> 9.3 |
| q8_0 | GPU | 164 -> 676 | 106 -> 429 | failed -> 314 | 51.2 -> 53.2 | 41.2 -> 40.2 | failed -> 32.6 |
| q8_0 | RAM | 172 -> 665 | 105 -> 396 | 77 -> 302 | 48.9 -> 49.9 | 21.2 -> 20.3 | 13.7 -> 13.9 |
| q4_0 | GPU | 160 -> 671 | 110 -> 427 | 81 -> 288 | 48.7 -> 52.9 | 41.3 -> 40.4 | 33.2 -> 31.1 |
| q4_0 | RAM | 171 -> 633 | 105 -> 403 | 80 -> 307 | 51.0 -> 50.5 | 26.1 -> 25.6 | 18.6 -> 18.3 |

The f16/GPU row is from the back-to-back run `compare-f16-confirm` (two passes per build); the other rows are from `kvmatrix-final-base` and `kvmatrix-final-fix`.
"failed" = the old build could not create the 16K context in that run. f16 KV at 16K never fits in 6 GB with either build.

### How it works
| Part | File | What it does |
|---|---|---|
| Detection | `ggml/src/ggml-cuda/ggml-cuda.cu`, `common.cuh` | cc 7.5 device named "GTX 16xx" is stored as `GGML_CUDA_CC_TURING_NO_MMA` |
| Quantized matmul | `mmq.cu`, `mmq.cuh`, `template-instances/mmq-no-mma-instance-*.cu` | such devices use the dp4a MMQ kernels instead of the MMA ones |
| Attention, batch size 1 | `fattn.cu` | always the vector kernel (the tile kernel was slower at long context) |
| Attention, larger batches | `fattn-common.cuh` | tile kernel drops its extra parallel blocks once the output fills the GPU, so no temporary buffer is allocated (this was the f16 out-of-memory cause) |

The `mmq-no-mma-instance-*.cu` files are generated: edit `template-instances/generate_cu_files.py` and rerun it, do not edit them by hand.

Why the remaining 7%: f16 KV decode at 8K on this card, t/s at 0K / 8K / 14K context.

| Attention kernel | 0K | 8K | 14K | Note |
|---|---|---|---|---|
| tile | 54.3 | 38.6 | 30.3 | first version of the fix; also ran out of memory in 3 of 6 runs |
| vector | 55.1 | 42.2 | 35.8 | chosen: fastest kernel that needs no tensor cores |
| tensor-core (old) | 54.3 | 45.4 | 40.5 | fastest here, but only together with the slow MMA matmul path |

### Switches
| Setting | Effect |
|---|---|
| (default) | new selection on GTX 16xx; other GPUs are not affected |
| `GGML_CUDA_FORCE_TURING_MMA=1` | old selection, for A/B checks |

### Keeping it healthy
- After rebasing on upstream: rerun `generate_cu_files.py`, rebuild, then `test-backend-ops -o FLASH_ATTN_EXT`.
- If upstream changes `fattn.cu` kernel selection, check that a batch-size-1 f16 run on this card still picks the vector kernel.
- Quick check that the fix is active: pp512 on the 1.3B model should be near 2900 t/s, not near 860.
- Full recheck: the three commands under "Reproduce" below; expected numbers are the tables above.
- Not done on purpose: quantized KV with shared GQA, KV streaming to RAM/NVMe.

### Reproduce
```
local-tune/build.sh
local-tune/bench.sh build <label> -t 6                # 4 model/placement rows, pp512 + tg128
KV_REPS=3 local-tune/kvmatrix.sh build                # KV type x GPU/RAM x 0K/8K/16K
python3 local-tune/compare.py run f16-confirm         # old vs new back to back, f16 KV (plan names build-base and build-exp)
```
Live view in a second terminal: `local-tune/watch.py` for bench and KV matrix, `local-tune/compare.py f16-confirm` for a comparison. Every runner writes a log with `START`, `TEST ... took=Ns`, `DONE ... took=Ns` and `ALLDONE total=Ns` lines, and the view shows a time table per row and context depth.

Result files in `local-tune/results/`:

| File | Content |
|---|---|
| `final-base.csv`, `final-fix.csv` | bench, old and new build |
| `kvmatrix-final-base.csv`, `kvmatrix-final-fix.csv` | KV matrix, old and new build |
| `compare-f16-confirm.*` | old vs new, f16 KV, two passes each |
| `compare-f16-decode.*`, `compare-f16-prompt.*` | attention kernel comparison |
| `exp*`, `kvmatrix-exp*`, `base1`, `kvmatrix-base1` | earlier experiments kept for reference |

## Change 2: qwen35 GDN path on CPU layers
`src/models/qwen35.cpp`: on x86, layers that run on the CPU skip the fused raw-gate GDN path (it was 20-35% slower than the
separate sigmoid/softplus nodes). Env `GGML_GDN_RAW_GATES_DISABLE=1` did the same before this change.

Measured before change 1 (llama-bench, -t 6, pp512 / tg128, t/s). The pp512 values are superseded by the chart above; tg128 still holds.

| model | placement | Oddly before | Oddly after | plain llama.cpp (same base) |
|---|---|---|---|---|
| deepseek-coder-1.3b Q8_0 | GPU | 856 / 158 | ~833 / ~152 (noise) | 829 / 150 |
| qwen2.5-3b Q5_K_M | GPU | 390 / 94 | ~378 / ~91 (noise) | 379 / 90 |
| Qwythos-9B Q4_K_M | ngl 22 hybrid | 153 / 15.6 | 150 / 20.6 | 151 / 20.3 |
| Qwythos-9B Q4_K_M | CPU | 142 / 7.7 | 139 / 9.4 | 139 / 9.4 |

## Setup
Rebuild: `local-tune/build.sh` (CUDA 13.4 from `/opt/cuda`, gcc 16). Benchmark: `local-tune/bench.sh <build-dir> <label> -t 6`, summarize with `local-tune/summarize.py local-tune/results/<label>.csv`.

## Build flags
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release` (+ccache if present).
Kept at defaults after measuring: `GGML_CUDA_FORCE_MMQ` (no change, within 1%), `GGML_CUDA_HOPPER_Q1` (sm_90a only, left off).

## Runtime defaults (measured)
- Threads 6 (physical cores); 4/8 equal, 12 slightly worse.
- `-ub 512`, flash attention on (`-fa on`); smaller ubatch is slower.
- 9B Q4_K_M: `-ngl 26` is the ceiling at small context (27+ aborts with a CUDA OOM); use `-ngl 22` with `-c 16384 -ctk q8_0 -ctv q8_0`
  (4.5 GB VRAM used with the desktop running).
- Models up to ~3B fully offload (`-ngl 99`).
