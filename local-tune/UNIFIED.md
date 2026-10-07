# Wrekt-llama.cpp on a 6 GB GTX 1660 Ti: unified record

Authoritative document for `local-tune/`. Written 2026-10-07 from README, TRIALS, ASSESSMENT, EXPERIMENT-DISCOVERY, BACKEND-PLAN and the context-capacity directive (now in [archive/](archive/)). Commit timeline: [HISTORY.md](HISTORY.md).
Speeds are tokens per second on this one machine; differences under about 3% are noise. Missing values read `(not recorded)`. Sources are cited as `*(source: results/<family>/<file>)*`; a bare file name means the family folder named by `scripts/paths.py`.

## 1. Abstract

A llama.cpp fork tuned for a Ryzen 5 7600X + GTX 1660 Ti (6 GB, Turing, no tensor cores), CachyOS, CUDA. Three code changes shipped, plus a GPU overclock and gateway settings:

| # | Change | Effect | Commit |
|---|---|---|---|
| 1 | CUDA: no tensor-core kernels on GTX 16xx | prompt reading 2.9x to 4.2x faster | `a260d811e` |
| 2 | qwen35: skip fused raw-gate GDN path on CPU layers (x86) | 9B hybrid/CPU decode 20-35% faster | `52b730eb4` |
| 3 | CUDA: q8_0 K / q4_0 V attention pair in the default build | 7B fits all layers at 16K: writing 11-16% faster | `afbf9a20c` |

Measured bottlenecks (§5): 7B and 4B weights are bandwidth-bound (83-89% of 331 GB/s), 7B attention growth is compute-bound, the 9B is limited by which bytes sit on the CPU. Context capacity on the 9B reaches 81,920 tokens at q4/q4 ub128 (§6). Everything below this line is the evidence.

## 2. Hardware and the 5754 MiB ceiling

| Part | Value |
|---|---|
| GPU | GTX 1660 Ti, 6144 MiB, driver 615.71.09 |
| CPU | AMD Ryzen 5 7600X, 6 cores, 12 threads |
| RAM | 62 GB, dual-channel DDR5 (about 50 GB/s measured on CPU layers); EXPO/FCLK state (not recorded) |
| Power / clocks | 120 W (stock 120 W per assessment; stock run 100 W), memory +2300 (6900 MHz), core +105, saved 1006-1325 |
| Build | CUDA 13.4 from `/opt/cuda`, gcc 16, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release`; kernel 6.18.52-1-cachyos-lts |
| Display | one display on the iGPU since 2026-10-06; the other stays on the 1660 Ti (desktop VRAM 660-890 MiB, was about 1 GB) |

**Ceiling.** The driver keeps about 390 MiB (`memory.reserved`). A context fails to create when used memory would pass **5754 MiB** (directive; TRIALS wrote "about 5750"). Every context failure in §6 is `allocation failure at load` at that line.

> ⚠ Contradiction: GPU bandwidth. BACKEND-PLAN/TRIALS use "288 GB/s" (the plan's 4B ceiling 105 t/s, "effective rate 222 GB/s = 77% of 288"); EXPERIMENT-DISCOVERY uses "331 GB/s theoretical (192 pins at 13.8 Gbps)" for the overclocked memory. Adjudicated: 288 is the stock figure, 331 the overclocked one. The audit's "226 vs 288 GB/s" and "15-20% headroom" have no measurement behind them.

Kept at defaults after measuring: `GGML_CUDA_FORCE_MMQ` (within 1%), `GGML_CUDA_HOPPER_Q1` (sm_90a only). Runtime defaults: threads 6 (4/8 equal, 12 slightly worse); `-ub 512`, `-fa on`; models up to ~3B fully offload (`-ngl 99`). CPU governor `performance`; swap is zram (zstd); shmem THP `never`; no `mitigations=` or `amd_iommu=` on the kernel command line.

### GPU overclock (setting, not code)

`scripts/gpu-push.py`, 7B as test load, decode at 4K context:

| Run | Result |
|---|---|
| Stock, 100 W | 43.8 t/s |
| 100 W sweep | memory +1400, core +105: 47.5 t/s (+8%), 13 soak passes *(source: hardware/gpupush-1006-1250.log)* |
| 120 W, resumed upward | memory +2300 (6950 MHz at +2400 passed too), core +105: 50.7 t/s (+16% over stock), 14 of 14 soak passes, 50.4-50.7 t/s, 121-123 W, 73-75 C, perplexity 12.8239 each time, run time 1252 s *(source: hardware/gpupush-1006-1325.log, gpu-oc.json)* |

A step passes only on a clean exit, no new Xid line, perplexity equal digit for digit, decode not under 93% of the best, under 83 C. Core +120 passed (50.9), +135 failed twice on perplexity (12.8248; 12.8234). 100 W memory reached the +1500 cap without failing (6001-6750 MHz real clock; the offset is half the displayed change). Re-verification at 120 W: 49.1 and 50.2 t/s (47.5 at 100 W with same offsets, so about +5% from power). Power/memory split is not separable; splits under 3% are noise. System switches (100 W limit, locked clocks, memory +250, huge pages) gave no gain; 100 W cost 1-3% *(source: hardware/gpu-base-1006-1229.csv, gpu-pl100-*, gpu-lgc-*, gpu-mo250-*, gpu-thp-*)*. Full bench after soak: decode +5% to +14% *(source: hardware/full-1006-1356.csv)*. Power column of the 100 W run (42-48 W) was a sampler bug, fixed for the second run. The 12:29-12:39 system runs are labelled by file name only; read as 100 W, locked clocks, +250, huge pages.

Offsets reset at reboot. `gpu-push.py apply` restores them. The state file `results/hardware/gpu-oc.json` is per machine and git-ignored.

> ⚠ Contradiction: TRIALS 2d says "A boot-time systemd unit for it is not set up"; EXPERIMENT-DISCOVERY §9 says all rows were taken "with `gpu-oc.service` applied". Adjudicated: the unit exists. Its `ExecStart` still names `local-tune/gpu-push.py`, which now works through a shim (`local-tune/gpu-push.py` execs `scripts/gpu-push.py`).

## 3. Models

Suite `shortlist` ran kept and new models together. Write/read are llama-bench; one chat and all slots go through the gateway.

| Model | Status | File | Writes | At 8K | Reads | One chat | All slots | Items/s | VRAM | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-coder-1.3b | Kept | 1.33 GB | 167 | 108 | 3,027 | 183 | 465 (4) | - | 3,155 MiB | ngram-simple |
| gemma4-e2b | Kept | 2.44 GB | 137 | 117 | 2,032 | 132 | 349 (8) | 18.9 | 1,587 MiB | forced-label classify |
| qwythos-9b | Kept | 5.24 GB | 26.4 | 22.1 | 541 | 26.1 | - | - | 4,322 MiB | ngram-simple |
| embeddinggemma-300m | Kept | 0.31 GB | - | - | - | - | - | 454 | 428 MiB | 768 dimensions |
| qwen3-reranker-0.6b | Kept | 0.6 GB | - | - | - | - | - | 38.9 | 1,896 MiB | right document ranked first |
| virbiusguard | Added | 0.45 GB | 348 | 159 | 6,504 | 313 | 640 (4) | - | 1,388 MiB | |
| minicpm5-2b | Added | 1.45 GB | 136 | 83.0 | 1,970 | 133 | 305 (8) | 23.9 | 2,125 MiB | forced-label classify |
| qwen3.5-4b | Added | 2.64 GB | 78.4 | 62.8 | 986 | 77.6 | - | - | 3,151 MiB | ngram-simple |
| gemma4-e2b-mtp | Tested, not added | 2.44 GB | 125 | - | - | 125 | 324 (8) | 15.3 | 1,909 MiB | MTP drafter 0.83x, text changed |
| sharp-spark-4b | Tested, not added | - | 65.7 | 58.2 | 987 | 65.3 | - | - | 4,038 MiB | tool calls 3 of 6; 900 MiB larger than qwen3.5-4b |
| qwen3.5-4b-mtp | Tested, not added | 2.64 GB | 63.3 | - | - | 63.3 | - | - | 3,688 MiB | MTP drafter 0.65x |
| sharp-minicpm 2B Q6_K_XL | Dropped | - | 107 | 71.8 | 1,875 | - | - | - | - | lost first round, file deleted |
| qwen2.5-3b Q5_K_M | Dropped | - | 103 | 73.4 | 1,410 | - | - | - | - | replaced by gemma4-e2b as swarm worker |
| agents-a1 4B Q4_K_M | Dropped | - | 78.7 | 70.4 | 1,020 | - | - | - | - | lost first round, file deleted |
| ternary-bonsai 8B PQ2_0 | Dropped | - | 69.6 | 44.4 | 638 | - | - | - | - | lost first round, file deleted |
| deepseek-r1-distill 7B Q4_K_S | Excluded | - | 54.0 | 40.3 | 682 | - | - | - | - | still served, left out of suite runs |

*(source: suites/suite-shortlist.csv, suite-added.csv, assess.json)*

> ⚠ Contradiction: the qwythos-9b row (26.4 t/s, 4,322 MiB) is the `shortlist` run and predates the later placement C and the Q4_K output file. Adjudicated: kept as history; the current 9B line is placement C (§4 row 11, §6).

**Job checks** (`scripts/jobs.py`; they show whether a model can do the job, not which of two passing models is better):

| Model | Job | Score | Detail |
|---|---|---|---|
| deepseek-coder-1.3b | Writes code | 3/4 | asserts |
| gemma4-e2b | Routes requests | 16/16 | 7.9 items/s |
| gemma4-e2b | Calls tools | 6/6 | every call valid |
| qwythos-9b | Calls tools | 6/6 | every call valid |
| qwythos-9b | Writes code | 3/4 | asserts |
| virbiusguard | Blocks attacks | 11/12 | 1 attack missed; 0 safe blocked; 151 ms per verdict |
| minicpm5-2b | Routes requests | 16/16 | 10.5 items/s |
| minicpm5-2b | Calls tools | 6/6 | every call valid |
| qwen3.5-4b | Calls tools | 6/6 | every call valid |
| qwen3.5-4b | Writes code | 4/4 | asserts |
| gemma4-e2b-mtp | Drafter speed-up | 0.83x | TEXT DIFFERS; one stream 133 to 110, 8 slots 340 to 288 |
| sharp-spark-4b | Writes code | 4/4 | asserts |
| sharp-spark-4b | Calls tools | 3/6 | missed run_shell, create_document, read_file |
| qwen3.5-4b-mtp | Drafter speed-up | 0.65x | same text; one stream 78 to 51 |

Gateway settings now: 9B placement C (§4, §6); 7B `-ngl 99 --parallel 1 -c 16384 -ctk q8_0 -ctv q4_0 -fa on`; `--no-mmproj-offload` for the 9B; q8_0 KV for 9B, 3B, 1.3B; `--spec-type ngram-simple` for 9B, 3B, 1.3B. `vram_mb` is 4500 for both big models (measured about 4670 MiB each, so the gateway check `free - 250 MiB >= vram_mb` is about 150 MiB lenient). Virbiusguard runs on the CPU as served (112 t/s, 145 on 2 slots, prompt about 300 t/s) so it never evicts the agent model.

## 4. Change ledger

| # | Area | Change | Measured on | Before | After | Gain | Where | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | Kernel | No tensor-core kernels on GTX 16xx | 1.3B reads a prompt | 860 | 2,901 | 3.4x | `a260d811e` | throughput/final-base, final-fix |
| 2 | Kernel | same | 3B reads | 388 | 1,331 | 3.4x | same | same |
| 3 | Kernel | same | 7B reads (Q4_K_S, GPU) | 168 | 690 | 4.1x | same | same |
| 4 | Kernel | same | 9B reads, 22 layers on GPU | 156 | 500 | 3.2x | same | same |
| 5 | Kernel | same | 9B reads, CPU only | 140 | 401 | 2.9x | same | same |
| 6 | Backend | qwen35: no fused GDN path on CPU layers | 9B writes, ngl 22 | 15.6 | 20.6 | +32% | `52b730eb4` | throughput/base, patched |
| 7 | Backend | same | 9B writes, CPU only | 7.7 | 9.4 | +22% | same | same |
| 8 | Kernel | q8_0 K / q4_0 V pair default | 7B writes, empty | 48.7 | 54.0 | +11% | `afbf9a20c` | kv/compare-kvmix-7b-confirm |
| 9 | Kernel | same | 7B writes, 15K | 28.4 | 32.9 | +16% | same | same |
| 10 | Settings | More GPU layers | 9B -ngl 22 to 25, 16K, 0K | 21.3 | 24.6 | +16% | `models.conf` | throughput/compare-ngl-9b-16k |
| 11 | Settings | 9B placement C (Q4_K output, `-ngl 99 -ot blk\.[45]\.=CPU,blk\.1[3467]\.=CPU`) | 9B through gateway, 0K / 8K / 15K vs shipped `-ngl 25` | 26.9 / 21.6 / 18.9 | 30.7 / 30.0 / 29.2 | see S1, S3 | `models.conf` | §6 |
| 12 | Settings | More GPU layers | 7B -ngl 24 to 99 | 35.1 | 53.1 (53.6 bench) | 1.5x | `models.conf` | throughput/compare-ngl-7b-16k |
| 13 | Settings | ngram-simple | 9B edits code | 24.1 | 170 (169.6) | 7.0x | `models.conf` | speculative/spec-edit-qwythos9b |
| 14 | Settings | ngram-simple | 3B edits code | 86.7 | 274 (273.7) | 3.2x | `models.conf` | speculative/spec-tune-qwen3b |
| 15 | Hardware | OC memory +2300, core +105, 120 W | 7B writes 4K | 43.8 | 50.7 | +16% | `gpu-push.py` | §2 |
| 16 | Models | swarm worker qwen2.5-3b -> gemma4-e2b | writes, empty | 103 | 136 | +33% | `models.conf` | throughput/newmodels |
| 17 | Models | same | reads | 1,410 | 2,139 | 1.5x | same | same |
| 18 | Tooling | one-token guard verdict | tool-call check | 214 and 239 ms | 101 and 111 ms (about 45 ms prompt cached) | about 2x | `suites` | §7 backend plan |

Edits without a speed number: gateway source moved into the repo (`~/odysseus-local/gateway.py` is a link to it; new `GET /conf` and a jobs column in `models.conf`); `models.conf` is the one model list (gateway, `suite.py`, `assess.py` read it); MCP swarm server uses gemma4-e2b with 8 slots; Odysseus `agent_loop.py` browser tools gated, documents created strictly for gemma (local patch, not in the image); catalog of 927 small GGUF models, 16 shortlisted, 5 downloaded and tested, 3 added; `suite.py` phase 3 job checks; virbiusguard checks every Odysseus tool call.

### Change 1 detail: GTX 16xx kernel selection

GTX 16xx reports compute capability 7.5 like RTX 20xx but has no tensor cores, so MMA kernels ran through slow emulation. The card is now detected by name and stored as `GGML_CUDA_CC_TURING_NO_MMA`. Old build vs new build, same session, 3 repetitions, prompt processing t/s:

| Model | Old | New | Gain |
|---|---|---|---|
| deepseek-1.3b Q8_0 GPU | 860 | 2901 | 3.4x |
| qwen2.5-3b Q5_K_M GPU | 388 | 1331 | 3.4x |
| R1-distill-7B Q4_K_S GPU | 168 | 690 | 4.1x |
| Qwythos-9B Q4_K_M ngl 22 | 156 | 500 | 3.2x |
| Qwythos-9B Q4_K_M CPU | 140 | 401 | 2.9x |

Other effects: writing in short chat unchanged (-1% to +3%); long chat q8_0/q4_0 KV on GPU unchanged within noise (-6% to +9%, median -2%); f16 KV on GPU at 8K 7% slower (42.0 vs 45.0 t/s, the one cost); KV in RAM (`-nkvo 1`) unchanged (-4% to +2%); out-of-memory stops with f16 KV at 8K: 0 in 9 runs; perplexity equal within error; `test-backend-ops -o FLASH_ATTN_EXT` passes.

KV matrix, R1-distill-7B Q4_K_S, `-ngl 99 -fa 1`, old -> new, t/s *(source: kv/kvmatrix-final-base.csv, kvmatrix-final-fix.csv, throughput/compare-f16-confirm)*:

| KV | On | pp512 @0K | @8K | @16K | tg32 @0K | @8K | @16K |
|---|---|---|---|---|---|---|---|
| f16 | GPU | 168 -> 690 | 110 -> 439 | does not fit | 54.1 -> 55.0 | 45.0 -> 42.0 | does not fit |
| f16 | RAM | 170 -> 678 | 109 -> 423 | 77 -> 300 | 48.1 -> 48.9 | 15.9 -> 15.5 | 9.7 -> 9.3 |
| q8_0 | GPU | 164 -> 676 | 106 -> 429 | failed -> 314 | 51.2 -> 53.2 | 41.2 -> 40.2 | failed -> 32.6 |
| q8_0 | RAM | 172 -> 665 | 105 -> 396 | 77 -> 302 | 48.9 -> 49.9 | 21.2 -> 20.3 | 13.7 -> 13.9 |
| q4_0 | GPU | 160 -> 671 | 110 -> 427 | 81 -> 288 | 48.7 -> 52.9 | 41.3 -> 40.4 | 33.2 -> 31.1 |
| q4_0 | RAM | 171 -> 633 | 105 -> 403 | 80 -> 307 | 51.0 -> 50.5 | 26.1 -> 25.6 | 18.6 -> 18.3 |

"failed" = old build could not create the 16K context. f16 KV at 16K never fits with either build. q4_0 rows are valid as speeds only (K q4_0 breaks the 7B, §7).

Implementation: detection in `ggml-cuda.cu`, `common.cuh`; `mmq.cu`, `mmq.cuh`, `template-instances/mmq-no-mma-instance-*.cu` (generated by `generate_cu_files.py`, never edit by hand) use dp4a MMQ kernels; `fattn.cu` always picks the vector kernel at batch size 1; `fattn-common.cuh` tile kernel drops extra parallel blocks once output fills the GPU (this removed the f16 out-of-memory cause). f16 KV decode at 8K, 0K / 8K / 14K: tile 54.3 / 38.6 / 30.3 (first fix; out of memory in 3 of 6 runs); vector 55.1 / 42.2 / 35.8 (chosen); tensor-core (old) 54.3 / 45.4 / 40.5 (fastest, only with the slow MMA matmul). Switch: `GGML_CUDA_FORCE_TURING_MMA=1` restores the old selection. After rebasing upstream: rerun `generate_cu_files.py`, rebuild, `test-backend-ops -o FLASH_ATTN_EXT`; quick health check is pp512 near 2900 t/s on the 1.3B, not 860.

### Change 2 detail

`src/models/qwen35.cpp`: on x86, CPU layers skip the fused raw-gate GDN path (20-35% slower than separate sigmoid/softplus nodes). `GGML_GDN_RAW_GATES_DISABLE=1` did the same before. pp512 / tg128 *(measured before change 1; pp512 superseded)*:

| Model | Placement | Oddly before | Oddly after | plain llama.cpp |
|---|---|---|---|---|
| deepseek-coder-1.3b Q8_0 | GPU | 856 / 158 | ~833 / ~152 (noise) | 829 / 150 |
| qwen2.5-3b Q5_K_M | GPU | 390 / 94 | ~378 / ~91 (noise) | 379 / 90 |
| Qwythos-9B Q4_K_M | ngl 22 hybrid | 153 / 15.6 | 150 / 20.6 | 151 / 20.3 |
| Qwythos-9B Q4_K_M | CPU | 142 / 7.7 | 139 / 9.4 | 139 / 9.4 |

### Change 3 detail

`fattn.cu`, `CMakeLists.txt`: vector kernel for K q8_0 / V q4_0 is in the default build; other mixed pairs still need `-DGGML_CUDA_FA_ALL_QUANTS=ON`. On the 7B K needs q8_0, V q4_0 stays inside perplexity error and frees about 130 MiB at 16K, which lets the last layer go to the GPU. `compare-kvmix-7b-confirm`, `-c 16384`, two passes:

| Setup | 0K | 8K | 15K | Perplexity |
|---|---|---|---|---|
| `-ngl 28`, K q8_0 / V q8_0 (before) | 48.2-48.7 | 35.1-35.3 | 28.3-28.4 | 8.181 |
| `-ngl 99`, K q8_0 / V q4_0 (now) | 53.4-54.0 | 40.2-40.3 | 32.8-32.9 | 8.204 |

Error +/- 0.14; f16 KV gives 8.166; prompt reading equal. The pair alone is not a speed setting: at the same layer count it only saves memory. After `fattn*.cu` changes `lab.py` runs `test-backend-ops -o FLASH_ATTN_EXT`, perplexity and speed (suite `attention`).

## 5. Bottleneck model

Tool: CUPTI activity tracer via `CUDA_INJECTION64_PATH` (`prof/cupti_trace.cpp`, `prof/build.sh`, `prof/analyze.py`). No hardware counters: `RmProfilingAdminOnly` is 1, no root, no Nsight/`ncu`/`perf`. DRAM traffic = bytes read / kernel time; cache hit rates are inferred. Tracer cost 2-5% (59.8 t/s untraced, 58.6 traced). Register use from `cuobjdump -res-usage` *(source: profiling/prof-kernel-resources.tsv)*. Raw traces (310 MB) are not in the repo; plans regenerate them; summaries are `profiling/prof-<model>-d<depth>.json`.

ms per decoded token *(source: profiling/compare-prof-*.csv, prof-hwstate.txt)*:

| Model | Depth | Wall | GPU busy | Weights | Attention | quantize_q8_1 | Other | GPU idle |
|---|---|---|---|---|---|---|---|---|
| 7B | 0 | 16.79 | 16.41 | 14.88 | 0.59 | 0.30 | 0.63 | 0.38 |
| 7B | 4096 | 20.03 | 19.59 | 15.16 | 3.49 | 0.30 | 0.63 | 0.44 |
| 7B | 8192 | 22.67 | 22.28 | 14.92 | 6.42 | 0.31 | 0.64 | 0.39 |
| 7B | 12288 | 26.51 | 26.01 | 15.30 | 9.74 | 0.31 | 0.67 | 0.50 |
| 7B | 15360 | 28.73 | 28.28 | 15.20 | 12.12 | 0.31 | 0.65 | 0.45 |
| 9B, 25 layers GPU | 0 | 39.26 | 15.85 | 14.23 | 0.11 | 0.28 | 1.23 | 23.41 |
| 9B | 4096 | 43.86 | 16.40 | 14.16 | 0.73 | 0.29 | 1.22 | 27.46 |
| 9B | 8192 | 46.59 | 16.78 | 14.03 | 1.26 | 0.27 | 1.21 | 29.81 |
| 9B | 12288 | 54.16 | 18.50 | 14.94 | 1.99 | 0.29 | 1.28 | 35.66 |
| 9B | 15360 | 54.28 | 18.32 | 14.34 | 2.45 | 0.28 | 1.24 | 35.96 |
| 4B | 0 | 13.17 | 12.57 | 10.32 | 0.16 | 0.30 | 1.80 | 0.60 |
| 4B | 8192 | 15.05 | 14.49 | 10.54 | 1.82 | 0.29 | 1.84 | 0.56 |

The 7B trace at 15360 failed once on context creation and passed on retry (`compare-prof-7b15`); a 14336 trace is also stored.

| Cost | Share | Class | Evidence |
|---|---|---|---|
| 7B and 4B weights (`mul_mat_vec_q`) | 89% of the 7B token at 0K, 53% at 15K | bandwidth | 275-294 GB/s against 331 theoretical; flat with depth |
| 7B attention at depth | 3% at 0K, 43% at 15K | compute | time does not follow bytes, block count or K thread layout |
| 9B CPU layers at 0K | 56% of token | system-RAM bandwidth | 1077 MB on CPU in about 22 ms, about 50 GB/s |
| 9B CPU attention at depth | +12.6 ms 0K to 15K, 23% at 15K | compute on wrong device | layers 3 and 7 on CPU: 6.3 ms/layer vs 0.41 on GPU |
| Launch | 0.5-0.7 ms CPU per token, overlapped | negligible | one `cudaGraphLaunch`/token; gap 0.2-0.4 us; GPU busy 96-98% |
| Synchronization | 18 stream syncs/token on 7B | negligible | 17 return at once, 1 real wait |
| Allocation | none | - | zero memory events per decoded token |
| CPU/GPU transfer | 0.06 ms (7B), 0.14 ms (9B) | negligible | 7B: 8 uploads (95-118 KiB) + logits (592 KiB); 9B: 28 uploads, 7 downloads, 7 graph launches, 92 syncs (split graph) |
| `quantize_q8_1` | 0.30 ms, 169 calls | small | 1.8% at 0K; Q, K, V quantize the same input 3 times |

Hottest kernels, 7B at 0K (678 kernels/token) and 15K:

| Kernel | Op | Calls | ms 0K | ms 15K | us/call 0K | Grid / block | Regs |
|---|---|---|---|---|---|---|---|
| `mul_mat_vec_q<Q4_K, fused>` | fused FFN gate+up | 28 | 7.17 | 7.20 | 256 | 18944 / 32x2 | 56 |
| `mul_mat_vec_q<Q4_K>` | q, k, v, o, ffn_down | 132 | 5.55 | 5.76 | 42 | 512-3584 / 32x2 | 43 |
| `mul_mat_vec_q<Q6_K>` | output 151665 x 3584 | 1 | 1.61 | 1.69 | 1614 | 151665 / 32x2 | 44 |
| `flash_attn_ext_vec<128, q8_0, q4_0>` | FLASH_ATTN_EXT | 28 | 0.54 | 12.05 | 19 | 1x5x28 / 32x4 | 234 |
| `mul_mat_vec_q<Q5_K>` | MUL_MAT | 7 | 0.51 | 0.51 | 73 | 512 / 32x2 | 48 |
| `quantize_q8_1` | mat-vec input | 169 | 0.30 | 0.31 | 1.8 | 74 / 256 | 18 |
| `rms_norm_f32` | RMS_NORM | 57 | 0.24 | 0.24 | 4.1 | 1 / 1024 | - |
| `rope_neox` | ROPE | 56 | 0.10 | 0.10 | 1.7 | - | - |
| `fwht_cuda<64>`, `<128>` | Hadamard rotation | 112 | 0.17 | 0.17 | 1.5 | - | - |
| `k_set_rows_quant` | KV write | 56 | 0.13 | 0.13 | 2.3 | - | - |
| `flash_attn_combine_results` | attention 2nd pass | 28 | 0.05 | 0.07 | 1.8 | - | - |

Only `flash_attn_ext_vec` grows with context: 0.79 ms per 1000 tokens on the 7B (28 layers, 28 query / 4 KV heads); 0.22 ms per 1000 on the 4B (8 attention layers); 0.16 on the 9B GPU part (6 attention layers). Everything else is flat within 4%. On the 9B the GPU-idle time grows 12.6 ms 0K to 15K while GPU attention grows 2.3 ms: that is CPU attention of layers 3 and 7.

| Model | Layers | Query / KV heads | Head size | Attention layers | Largest matrix |
|---|---|---|---|---|---|
| 7B (qwen2) | 28 | 28 / 4 (ratio 7) | 128 | all 28 | FFN 3584 x 18944 Q4_K |
| 9B (qwen35) | 32 | 16 / 4 (ratio 4) | 256 | 8 (3, 7, .., 31) | output 248320 x 4096 Q6_K, 834 MB |
| 4B (qwen35) | 33 | 16 / 4 | 256 | 8 | output 248320 x 2560 Q6_K, 521 MB |

> ⚠ Contradiction: README says the 7B "fits all 29 layers"; the 7B has 28 transformer layers. Adjudicated: 29 offloadable layers = 28 blocks + output layer (`-ngl 28` leaves the output on the CPU).

**Weights.** 7B at 0K reads 4.25 GB per token in 14.9 ms: fused gate/up 294 GB/s, unfused Q4_K 275, output layer 274 (83-89% of 331). Nothing in MMQ/MMVQ (DP4A use, layout, registers 41-61, shared memory 128-256 bytes) can give 5% at batch 1; only fewer bytes per token help.

**7B attention.** The vector kernel reads each KV head once per query head (7 times). All reads from DRAM would be 72 MB/layer at 12K = 207 GB/s, which looks like a limit but is not. KV type test at 8K *(source: profiling/compare-prof-kvtype)*: f16/f16 (512 B/row) 208 us/call, q8_0/q8_0 (272 B) 220, q8_0/q4_0 (208 B) 232, q4_0/q4_0 (144 B) 218; a 3.5x byte change moves time under 12% and the biggest format is fastest, so repeated reads are cache hits. Timing-only ablations at 12K: no V read/accumulate saves 4.05 ms, no K read/dot saves 4.67 ms (of 9.7). The kernel uses 234 registers (255 for head size 256): only 2 blocks of 128 threads per SM. Verdict: compute-bound on instruction count per head row, K and V in equal parts; mechanism below that needs counters not exposed here.

**9B.** GPU busy is 34-40% of the token; it waits for 8 CPU layers. The CPU attention loop for quantized KV (`ggml-cpu/ops.cpp:8622`, `:8668`) does one generic `vec_dot` per K row and one `to_float` + `mad` per V row: 6.3 ms/layer at 15K, 15x the GPU kernel.

Finding outside the plan: `-ctk q4_0 -ctv q8_0` has no CUDA attention kernel in this build (`fattn.cu:444` accepts mixed types only as K q8_0 / V q4_0); the run falls back to CPU attention and looks hung at 8K. `compare-prof-kvtype.plan` still lists that variant; it was stopped by hand.

**New hierarchy after round 2** (§9 below): 9B CPU blocks (704 MiB read per token, about half the token: 32 ms vs about 16 ms GPU, estimate from the old trace, not re-traced) dominant and open; 9B attention growth 31.1 to 29.5 t/s 0K to 15K (about 5%) closed; 7B and 4B weights closed; 7B attention compute, mechanism open. Each 117 MiB block moved to the GPU has been worth 4-6%.

Model-side effective memory rate while writing: 222 GB/s on the 4B (77% of 288), 188 GB/s MiniCPM, 136 GB/s guard; small models are held back by fixed cost per token, which more slots recover (MiniCPM 134 t/s on one stream, 659 on 32).

## 6. Context capacity

Objective (directive): the largest stable context on the 7B and 9B without catastrophic quality loss. Metric: largest context where the model recalls all planted codes (`3/3`). Throughput policy: any configuration at >= 5 t/s that increases usable context is accepted.

> ⚠ Contradiction: the directive defines the pass condition as "3/3 hidden code tests in `jobs.py`". `scripts/ctxprobe.py` actually fills the context with text and asks for three planted keys (`ALPHA`, `BETA`, `GAMMA`); `keys` = how many were recalled. `jobs.py` (code asserts, 3/4 and 4/4 in §3) is not run by the probe. Adjudicated: "3/3 codes" below means three planted keys recalled. Needle/tool/code checks at long context are `(not recorded)`.

> ⚠ Contradiction: directive names repository `TheCascadian/Oddly-llama.cpp` branch `local-1660ti`. The work lives on `master` of `/home/wrekt/src/Oddly-llama.cpp` (user confirmed "typo, work on master").

Memory model (directive, verified): static GPU model 3858 MiB (placement C, 6 GDN blocks on CPU = 704 MiB); the driver ceiling is 5754 MiB; the compute buffer scales with context and shrinks with `-ub`. At 80K, `ub128`: model 3858 + KV 770 + compute 356 = peak 5700, free 54 MiB. 96K needs +154 MiB KV and about +68 MiB compute: a deficit of about 220 MiB. Compute buffer under `ub128`: 84 MiB at 16K to 356 MiB at 80K (about 4.25 MiB per 1K tokens); under `ub512` 384 MiB at 64K.

Architecture: 9B (Qwen3.5 hybrid) has 8 of 32 layers with attention, KV about 9.6 KiB/token, so the 9B ceiling is set by weight residency and compute buffer. 7B (pure transformer) has 28 attention layers, KV about 21 KiB/token (2.2x), so it is KV-bound.

### Certified ceilings (9B placement C, 3/3 codes) *(source: context/ctx-c1-9b.csv)*

| Variant | Max stable ctx | Peak VRAM | Headroom vs 5754 | t/s at ctx | pp t/s | Result |
|---|---|---|---|---|---|---|
| q8/q8 ub512 | 49,152 | 5749 MiB | 5 MiB | 22.3 | 370.1 | 3/3 |
| q8/q4 ub512 | 49,152 | 5567 MiB | 187 MiB | 21.7 | 363.8 | 3/3 |
| q4/q4 ub512 | 65,536 | 5502 MiB | 252 MiB | 19.8 | 337.9 | 3/3 |
| q4/q4 ub128 | 81,920 | 5700 MiB | 54 MiB | 18.9 | 252.1 | 3/3 |
| q4/q4 ub64 | 81,920 | 5693 MiB | 61 MiB | 20.0 | 224.0 | 3/3 *(source: context/ctx-t1-ub.csv)* |

Failures, all `allocation failure at load`: 64K on q8/q8 and q8/q4, 80K on ub512, 96K on ub128, 96K on ub64 (98,304), 80K on ub32 (ub32 peak 5706 reported at the failure).

### Full probe rows (ctx, result, peak, free, mdl, kv, cmpute, t/s ctx, pp t/s, codes)

| Variant | ctx | result | peak | free | mdl | kv | cmpute | t/s ctx | pp t/s | codes |
|---|---|---|---|---|---|---|---|---|---|---|
| q8/q8 ub512 | 16,384 | ok | 5055 | 699 | 3858 | 322 | 144 | 28.9 | 487.2 | 3/3 |
| | 20,480 | ok | 5137 | 617 | 3858 | 390 | 164 | 27.6 | 472.9 | 3/3 |
| | 24,576 | ok | 5226 | 528 | 3858 | 458 | 184 | 27.4 | 456.1 | 3/3 |
| | 32,768 | ok | 5400 | 354 | 3858 | 594 | 224 | 25.9 | 426.0 | 3/3 |
| | 40,960 | ok | 5666 | 88 | 3858 | 730 | 264 | 23.0 | 400.7 | 3/3 |
| | 49,152 | ok | 5749 | 5 | 3858 | 866 | 304 | 22.3 | 370.1 | 3/3 |
| | 65,536 | fail | (not recorded) | - | - | - | - | - | - | - |
| q8/q4 ub512 | 16,384 | ok | 4866 | 888 | 3858 | 258 | 144 | 26.3 | 471.2 | 3/3 |
| | 20,480 | ok | 4938 | 816 | 3858 | 310 | 164 | 25.4 | 458.0 | 3/3 |
| | 24,576 | ok | 5010 | 744 | 3858 | 362 | 184 | 25.1 | 443.9 | 3/3 |
| | 32,768 | ok | 5255 | 499 | 3858 | 466 | 224 | 23.8 | 414.3 | 3/3 |
| | 40,960 | ok | 5332 | 422 | 3858 | 570 | 264 | 22.3 | 395.4 | 3/3 |
| | 49,152 | ok | 5567 | 187 | 3858 | 674 | 304 | 21.7 | 363.8 | 3/3 |
| | 65,536 | fail | (not recorded) | - | - | - | - | - | - | - |
| q4/q4 ub512 | 16,384 | ok | 4848 | 906 | 3858 | 194 | 144 | 26.6 | 472.5 | 3/3 |
| | 20,480 | ok | 4888 | 866 | 3858 | 230 | 164 | 26.2 | 460.1 | 3/3 |
| | 24,576 | ok | 5018 | 736 | 3858 | 266 | 184 | 25.1 | 444.2 | 3/3 |
| | 32,768 | ok | 5071 | 683 | 3858 | 338 | 224 | 24.1 | 415.6 | 3/3 |
| | 40,960 | ok | 5166 | 588 | 3858 | 410 | 264 | 23.0 | 394.8 | 3/3 |
| | 49,152 | ok | 5278 | 476 | 3858 | 482 | 304 | 22.1 | 374.4 | 3/3 |
| | 65,536 | ok | 5502 | 252 | 3858 | 626 | 384 | 19.8 | 337.9 | 3/3 |
| | 81,920 | fail | (not recorded) | - | - | - | - | - | - | - |
| q4/q4 ub128 | 16,384 | ok | 4776 | 978 | 3858 | 194 | 84 | 27.6 | 383.5 | 3/3 |
| | 20,480 | ok | 4828 | 926 | 3858 | 230 | 101 | 27.1 | 376.3 | 3/3 |
| | 24,576 | ok | 4879 | 875 | 3858 | 266 | 118 | 25.4 | 362.3 | 3/3 |
| | 32,768 | ok | 4983 | 771 | 3858 | 338 | 152 | 25.9 | 344.7 | 3/3 |
| | 40,960 | ok | 5266 | 488 | 3858 | 410 | 186 | 23.1 | 323.3 | 3/3 |
| | 49,152 | ok | 5291 | 463 | 3858 | 482 | 220 | 22.1 | 305.2 | 3/3 |
| | 65,536 | ok | 5475 | 279 | 3858 | 626 | 288 | 20.3 | 275.4 | 3/3 |
| | 81,920 | ok | 5700 | 54 | 3858 | 770 | 356 | 18.9 | 252.1 | 3/3 |
| q4/q4 ub64 | 49,152 | ok | 5486 | 268 | 3858 | 482 | 206 | 23.7 | 263.5 | 3/3 |
| | 65,536 | ok | 5655 | 99 | 3858 | 626 | 272 | 21.5 | 240.2 | 3/3 |
| | 81,920 | ok | 5693 | 61 | 3858 | 770 | 338 | 20.0 | 224.0 | 3/3 |
| | 98,304 | fail | (not recorded) | - | - | - | - | - | - | - |
| q4/q4 ub32 | 49,152 | ok | 5263 | 491 | 3858 | 482 | 199 | 23.6 | 194.2 | 3/3 |
| | 65,536 | ok | 5543 | 211 | 3858 | 626 | 264 | 21.6 | 181.7 | 3/3 |
| | 81,920 | fail | 5706 (at failure) | - | 3858 | 770 | 329 | - | - | - |

The `free` column is 5754 minus peak. Smaller `ub` runs show a higher base VRAM (desktop, `vram_base` column: 553-649 MiB for ub64/ub32 against 526-541 for ub128), which hides part of the compute saving in the peak. Earlier 8K points: q8/q8 8,192 t/s 30.5 pp 526.5 at 20 s; 12,288 t/s 29.7 pp 506.8.

### Compute buffer by `-ub` (ctx 16K / 64K, other columns the same) *(source: context/ctx-m1-ub.csv)*

| ub | cmpute 16K | cmpute 64K | peak 16K | peak 64K |
|---|---|---|---|---|
| 512 | 144 | 384 | 4873 | 5541 |
| 256 | 104 | 320 | 4827 | 5474 |
| 128 | 84 | 288 | 4806 | 5443 |
| 64 | 74 | 272 | 4795 | 5425 |

KV and model sizes by type *(source: context/ctx-m2-slope.csv)*: 9B C q8/q8 at 8,192: kv 186, cmpute 41, peak 4760; at 32,768: kv 594, cmpute 140, peak 5263. 9B C f16/f16 (fa) at 8,192: kv 306, cmpute 36, peak 4871; at 24,576: kv 818, cmpute 38, peak 5385. 7B q8/q4 (model 3952, cpu 292): 8,192 kv 182 cmpute 21 peak 4823; 16,384 kv 364 cmpute 38 peak 5021; 32,768 kv 728 cmpute 72 peak 5419; 49,152 fail. 4B q8/q8 (model 2616, cpu 497): 8,192 kv 186 cmpute 40 peak 3555; 32,768 kv 594 cmpute 139 peak 4014; 65,536 kv 1138 cmpute 271 peak 4689. The 7B placement shows its ceiling between 32,768 and 49,152 (not certified; probe `codes`/t/s (not recorded) for these rows).

> ⚠ Contradiction: the CSV reports `cpu_model` = 2788 MiB for every 9B row, but placement C is 704 MiB on the CPU (the directive and §6 memory model). Not reconciled: 2788 is probably the host-mapped copy of the whole file (mmap), not the layers placed on the CPU. Treat 704 MiB as the weights read per token.

### Track status against the directive

| Track | Goal | Status |
|---|---|---|
| 1 ub64 / workspace clamp | 96K stable, cmpute at 80K <= 220 MiB, pp >= 150 | **Aborted by its own rule**: cmpute at 80K 338 MiB (saves 18 MiB against ub128, abort threshold < 30 MiB); 96K fails; pp 224.0 t/s was fine |
| 2 imatrix + IQ3 FFN | mdl < 3500 MiB, 112K-128K, 3/3, ΔPPL <= +0.08 | not started |
| 3 DuoAttention on 7B | KV -45% at 32K, 48K ctx, needle 100% | not started |
| 4 GDN block pruning | -234 MiB, +16K ctx, code >= 3/4 | not started |
| 5 CPU/GPU KV streaming | 128K, >= 6.0 t/s | not started (`-nkvo 1` cost today in §11) |
| Forced vec kernel (build-allq, q4/q4 ub128) | smaller cmpute | 49,152 ok: cmpute 55 MiB, peak 5211 MiB (vs 220 MiB and 5291 on the normal kernel); 65,536 and above run was cut off, csv empty *(source: context/ctx-t1v-vec.log, ctx-t1v-vec/)* |

Context matrix protocol: 32,768 -> 40,960 -> 49,152 -> 65,536 -> 81,920 -> 98,304 -> 114,688 -> 131,072. Log per step: ctx, result, VRAM peak (<= 5754), free, gpu mdl, gpu kv, cmpute, t/s ctx, pp t/s, codes.

### NEXT THREE CONTEXT EXPERIMENTS

Ranked strictly by context capacity gained per effort:

1. **Finish the forced vec-kernel ceiling run** (`ctx-t1v-vec`, build-allq, `-ctk q4_0 -ctv q4_0`, ub128): at 48K the compute buffer is 55 MiB instead of 220, which at 96K is about 160 MiB of the 220 MiB deficit and may close it outright. Run 65,536 -> 131,072, certify codes at each step, and compare pp t/s. This is the only measured lever that moves cmpute by more than the ub64 savings.
2. **Track 2: imatrix + IQ3 FFN requantization of the 9B** (keep attention Q4_K+, `output.weight` Q4_K). Needs `mdl` < 3500 MiB (>= 358 MiB saved), enough for 112K-128K when stacked with experiment 1; gate on WikiText-2 ΔPPL <= +0.08 and 3/3 codes. Abort at ΔPPL +0.12 or decode < 8 t/s.
3. **Track 5 bridge: KV on host for the oldest layers/tokens** starting with `-nkvo 1` on the 9B (KV 9.6 KiB/token = 1.2 GB at 128K, small against 62 GB RAM). Measure the decode curve at 32K, 64K, 96K, 128K; success is 128K with no allocation failure at >= 6.0 t/s. Fallback if experiments 1-2 do not reach 96K.

(Track 3, DuoAttention on the 7B, is the only route to raise the 7B past about 32K; it ranks below these because the objective's 9B ceiling moves further per hour of work.)

## 7. Experiment log

### E1-E3 and S1-S3 (EXPERIMENT-DISCOVERY)

**E1. 9B: both CPU attention layers to the GPU: WIN.** Bottleneck: GPU idle grows 12.6 ms 0K to 15K; CPU attention costs 6.3 ms/layer at 15K against 0.41 on GPU. Experiment: `-ngl 30 -ot blk\.[45689]\.=CPU` against `-ngl 25`; CPU weights 1094 MB against 1077 MB; 3 repetitions, two interleaved passes *(source: profiling/compare-e1-9b-attn-gpu)*. Ceiling was about 11.8 ms of 54.3 ms at 15K (+28%).

| Depth | `-ngl 25` | attention on GPU | Change |
|---|---|---|---|
| 0 | 25.91, 26.44 | 26.06, 26.31 | 0% |
| 8192 | 21.87, 21.46 | 25.06, 25.12 | +16% |
| 15360 | 18.49, 18.91 | 24.35, 24.37 | +30% |

Perplexity 5.3574 +/- 0.144 against 5.3588 +/- 0.144 (`ppl-e1-9b-attn-gpu`, 4 chunks). Prompt speed 516.5 against 514.9 t/s at 2048 tokens. Peak VRAM 5446 against 5407 MiB. Verdict: keep (superseded by S1/S3).

**E2. 7B: attention blocks along the KV axis: rejected.** Env knob `GGML_CUDA_FATTN_PB` (`prof/fattn-knobs.patch`), 12K: default (5) 39.0 and 38.3; 1: 34.4; 2: 35.8; 10: 37.8; 20: 37.0; 40: CUDA error (knob only). Default is best; occupancy is not the limit *(source: profiling/compare-e2-7b-pb)*.

**E3. 7B: K dot over 8 threads per row: rejected.** `GGML_CUDA_FATTN_VEC_NT_KQ=8`: registers 234 to 150; 38.74 and 38.42 live, 38.97 and 38.12 changed *(source: profiling/compare-e3-7b-kq8)*.

Diagnostics: K/V ablations via `GGML_CUDA_FATTN_ABLATE=1|2`, llama-bench direct, 2 reps: 45.74 +/- 0.37 and 47.09 +/- 0.15 against 38.6 (no csv, never shown in `watch.py`). The kernel source is back to the committed state; `prof/fattn-knobs.patch` holds the three knobs.

**S1. 9B placement: WIN, e1-drop9 is the byte-minimal stable set.** Variants: shipped `-ngl 25` (1024 MiB CPU), E1 (1041), e1-drop9 `-ngl 30 -ot blk\.[4568]\.=CPU` (903), e1-drop69 `blk\.[458]\.` (765), 7 small GDN blocks `-ngl 99` (821). Block sizes (MiB): GDN 137.9 (0, 1, 2, 6, 9, 12, 18, 21, 24, 28, 29, 30) or 117.3 (4, 5, 8, 10, 13, 14, 16, 17, 20, 22, 25, 26); attention 125.9 (3, 15, 27, 31) or 112.5 (7, 11, 19, 23); `output.weight` 795.7 Q6_K.

llama-bench *(source: profiling/compare-s1-9b-place)*:

| Variant | 0K | 8K | 12K | 15K |
|---|---|---|---|---|
| shipped | 26.1, 24.4 | 21.9, 21.5 | 20.0, 20.4 | 18.3, 18.5 |
| E1 | 26.0, 25.6 | 24.8, 25.1 | 24.5, 24.6 | 23.5, 24.2 |
| e1-drop9 | 27.7, 27.7 | 26.3, 26.6 | 26.0, 26.0 | 24.2, 25.5 |
| e1-drop69 | 27.0, 29.6 | fail, 28.2 | fail | fail |

Gateway *(source: profiling/compare-s1-9b-served)*:

| Variant | 0K | 8K | 12K | 15K | 16K soak |
|---|---|---|---|---|---|
| shipped | 26.9, 26.8 | 21.6, 21.5 | 20.2, 21.6 | 18.9, 19.0 | 18.8, 19.1 |
| E1 | 26.9, 26.7 | 24.2, 25.3 | 24.0, 24.4 | 24.9, 25.2 | 23.4, 24.0 |
| e1-drop9 | 28.6, 28.6 | 25.6, 25.8 | 25.4, 25.9 | 25.3, 25.4 | 26.9, 27.6 |

e1-drop9 against shipped: +6% at 0K, +19% at 8K, +23% at 12K, +34% at 15K; every soak passed. Fewer CPU bytes fail (765 and 821 run out of memory at 8K-15K). Perplexity 5.3574 shipped, 5.3588 E1, 5.3588 e1-drop9, each +/- 0.144 (`ppl-s1-9b-place`); job checks of the end-to-end suite were not run. The `min8` and `min7` rows are invalid (llama-bench splits `-ot` on commas, so those had 3 blocks on the CPU; the separator inside one run is `;`). Proven: the 9B is limited by which bytes sit on the CPU; each 138 MiB GDN block on the CPU costs about 3-6% at every depth; the stable floor for the Q6_K file at 16K is about 900 MiB on the CPU.

**S2. 7B shared-GQA vector kernel: rejected.** `flash_attn_ext_vec<128, 7, Q8_0, Q4_0>`, 7 query heads of one KV head as columns, batch 1, Turing, no softcap. Patch: `profiling/s2-7b-sharedkv-rejected.patch`. Registers `REG:255 STACK:176` against `REG:234 STACK:0` (spills). End-to-end, two passes *(source: profiling/compare-s2-7b-sharedkv)*: 0K 57.6, 57.1 against 59.7, 57.4; 12K 39.0, 38.4 against 38.6, 38.1 (+1%); 15K 35.9, 35.9 against 35.2, 34.8 (+2.5%). Not run: `test-backend-ops -o FLASH_ATTN_EXT` for the new cases, traced attention time, perplexity. Rejected on spilling and gain under 5%.

**S3. 9B Q4_K output matrix: WIN through residency.** `llama-quantize --allow-requantize --output-tensor-type q4_k <9B gguf> build-exp/qwythos-9b-outq4k.gguf Q4_K_M`: only `output.weight` changed, 795.7 to 545.6 MiB. llama-bench *(source: profiling/compare-s3-9b-outq)*:

| Variant | CPU MiB | 0K | 8K | 15K |
|---|---|---|---|---|
| A: Q6_K, e1-drop9 | 903 | 27.7, 27.9 | fail, 26.6 | fail, 24.7 |
| B: Q4_K, same placement | 903 | 28.3, 27.6 | 27.4, 27.3 | 25.3, 25.5 |
| C: Q4_K `-ngl 99 -ot blk\.[45]\.=CPU;blk\.1[3467]\.=CPU` | 704 | 31.1, 31.0 | 30.8, 30.3 | 29.6, 29.4 |
| Q4_K `blk\.[45]\.;blk\.1[367]\.` | 587 | 34.4, 34.5 | fail, 32.7 | fail, 31.1 |

Gateway *(source: profiling/compare-s3-9b-served)*: A 27.6, 28.3 at 0K; 25.3, 24.3 at 8K; 25.4, 25.0 at 15K; soak 26.9, 26.8. C 30.7, 31.0; 30.0, 29.4; 29.2, 30.2; soak 27.7, 28.1. C against A +11%, +20%, +18%; all soaks passed; peak 5231 MiB (C) against 5391 (A). Perplexity 4.0059 +/- 0.050 (Q6_K) against 4.0206 +/- 0.050 (Q4_K), 16 chunks (`ppl-s3-9b-outq`), +0.4%. Quantization alone is worth 3% (B); the gain is two more GPU blocks. The 5-block set is faster but hit a CUDA error once. The first pass of A failed on context creation within 5 s; cause not found, the same line passed every gateway soak.

> ⚠ Contradiction: E1 verdict says "Not applied to the gateway; that is a configuration change for the owner" while §10 next action 1 (EXPERIMENT-DISCOVERY) says "Done: the 9B is served from the Q4_K-output file" with placement C. Adjudicated: E1 itself was never applied; S1+S3 (placement C) was. §10 of the original also points to the file in `build-exp/`; it now sits next to the other model files (stale path).

### Plan outcome and the 2026-10-05/06 trials (TRIALS §1-§2e)

Plan from the CUDA backend audit:

| # | Plan item | Outcome | Evidence |
|---|---|---|---|
| 1 | Route quantized KV to the f16-converting attention kernel | Tested, no-go | kv/kvmatrix-exp1-forced*.csv |
| 2 | CUDA Graphs A/B | Tested, no gain, defaults kept | throughput/compare-graphs-* |
| 3 | MMA detection fix for GTX 16xx | Shipped `a260d811e` | |
| 4 | Non-MMA build A/B | Done | final-base vs final-fix, compare-f16-* |
| 5 | Shared-GQA kernel for quantized KV | Dropped (item 1 was its gate and failed); later tried as S2 and rejected | |
| 6 | EXPO / DDR5 speed and FCLK | Not tested (BIOS; RAM speed not read, needs root `dmidecode`) | |
| 7 | Speculative decoding | `ngram-simple` applied; `draft-simple` no-go; DFlash, MTP, EAGLE3, DSpark need own draft models | speculative/spec-* |
| 8 | MoE offload | Does not apply (none of the five gateway models is MoE) | |
| 9 | ik_llama upper-bound probe | Done in 2c: no gain | |
| 10 | Turbo4 KV type and rotation | Not tested: no such KV type or flag in this tree | |
| 11 | Pinned host memory | Not tested; only matters with KV in RAM or MoE | |

**Item 1 in numbers** (tg32 t/s, normal kernel -> forced f16-converting kernel, 1 repetition): MMA build q8_0 @8K 41.7 -> 34.0, @16K 33.6 -> no result; q4_0 @8K 42.0 -> 37.8, @16K 31.5 -> 31.5. Non-MMA q8_0 @8K 41.5 -> 34.4, @16K 32.5 -> no result; q4_0 @8K 41.5 -> 33.3, @16K 33.0 -> 23.5. Forced kernel slower or equal everywhere. ("no result" = row missing in csv.)

**Item 2** (t/s, first / second pass; "on" current build, "opt" `GGML_CUDA_GRAPH_OPT=1`, "off" `-DGGML_CUDA_GRAPHS=OFF`, 3 repetitions):

| Model, test | on | opt | off |
|---|---|---|---|
| 1.3B Q8_0 GPU, tg128 | 145.9 / 150.8 | 153.8 / 153.3 | 146.8 / 149.7 |
| 1.3B, pp512 | 2956 / 2817 | 3048 / 3040 | 3026 / 3042 |
| 7B q8_0 KV, tg32 @0K | 53.4 / 52.5 | 52.9 / 52.8 | 53.4 / 53.4 |
| 7B tg32 @8K | 39.3 / 40.6 | 40.4 / 40.2 | 40.6 / 40.5 |
| 7B pp512 @0K | 689 / 676 | 674 / 674 | 681 / 675 |
| 7B pp512 @8K | 425 / 432 | 433 / 431 | 433 / 428 |
| 9B ngl 22, tg64 | 20.8 / 21.0 | 20.9 / 20.8 | 21.1 / 21.2 |
| 9B ngl 22, pp512 | 482 / 478 | 491 / 494 | 495 / 472 |

No difference larger than the gap between two passes. `GRAPH_OPT=1`: 1.3B writes 2-5% faster with less spread (+-0.2 against +-3 to +-9), nothing on 7B/9B. The 16K depth failed to create its context in all six runs (desktop held about 1 GB; a session memory limit, not a graph result).

**Item 7: speculative decoding** (`spectest.py`, 400 tokens, temp 0, 3 reps, `-c 8192 -ctk q8_0 -ctv q8_0`, t/s):

| Model | Prompt | none | ngram-simple | ngram-mod |
|---|---|---|---|---|
| qwen2.5-3b Q5_K_M GPU | rewrite a file, one rename | 85.3 | 391.0 (4.6x) | 488.8 (5.7x) |
| qwen2.5-3b | open question | 88.6 | 87.9 | 98.9 |
| Qwythos-9B ngl 22 | rewrite | 19.9 | 90.2 (4.5x) | 117.3 (5.9x) |
| Qwythos-9B | open question | 19.9 | 20.2 | 24.1 |

`ngram-simple`: identical text in all 12 answers, 94-97% of drafted tokens accepted. `ngram-mod` remembers earlier requests, so its second and third answers differ and the open-question gain is not like-for-like. The rewrite prompt is the best case.

**Backlog trials (2026-10-06)**, all on `build-live`, desktop VRAM 660-760 MiB:

| Trial | Outcome | Evidence |
|---|---|---|
| GPU layers after display move | Applied: 9B 22 -> 25, 7B 24 -> 28 | compare-ngl-* |
| ngram-simple on code edits | Applied to 9B, 3B, 1.3B | spec-tune-qwen3b, spec-edit-qwythos9b, spec-single-1p3b, spec-load-1p3b |
| ngram-simple n/m, `ngram-map-k`, `ngram-map-k4v` | Defaults kept | spec-tune-qwen3b |
| ngram-simple on 7B, 2000-token answers | 1.4x to 2.0x but answer changes; not applied | spec-r1-7b-long |
| `draft-simple` | No-go | spec-draft-qwen3b, spec-r1-7b |
| `GRAPH_OPT=1` under server load | No gain | spec-load-1p3b, spec-single-1p3b |
| `GGML_CUDA_DISABLE_FUSION=1` | Slower, default kept | compare-fusion-* |
| `-ub` 1024 and 2048 | 1.5-3% for 70-900 MiB, default kept | compare-ubatch-* |
| `REGISTER_HOST`, `--poll`, `--prio`, pinning | No gain | compare-host-9b-hybrid |

GPU layer count at 16K, q8_0 KV (`-d 0,15360 -p 512 -n 32`; peak VRAM includes desktop):

| Model | ngl | tg32 @0K | @15K | pp512 @15K | Peak VRAM |
|---|---|---|---|---|---|
| Qwythos-9B | 22 (old) | 21.3 | 15.2 | 385 | 4990 MiB |
| | 24 | 23.5 | 16.9 | 392 | 5224 |
| | 25 (gateway then) | 24.6 | 17.7 | 396 | 5335 |
| | 26 | 25.6 | 18.8 to 20.8 | 376 to 400 | 5507 |
| | 27 | 27.2 | 21.7 | 388 to 403 | 5616 |
| | 28, 29 | 28.7, 30.6 | does not fit | does not fit | - |
| R1-distill-7B | 24 (old) | 35.1 | 18.2 | 315 | 4784 |
| | 26 | 39.9 | 22.2 | 320 | 5155 |
| | 27 | 40.3 to 44.1 | 24.7 | 329 | 5293 |
| | 28 (gateway then) | 48.2 | 28.5 | 330 | 5422 |
| | 99 (all 29) | 53.6 | 33.5 | 329 | 5574 |

Chosen values left about 330 MiB below the limit with the desktop at 750 MiB: 9B +15% at 0K and +16% at 15K, 7B +37% and +57%. 7B with all layers is 11-18% faster again but left 180 MiB; the gateway check did not load it (it needed the second display off the card; it works after the display move and the V q4 saving).

**ngram-simple on code-edit prompts** (400 tokens, none -> ngram-simple, t/s):

| Model | rewrite | edit | refactor | free |
|---|---|---|---|---|
| qwen2.5-3b GPU | 86.6 -> 386.9 (4.5x) | 86.7 -> 273.7 (3.2x) | 86.7 -> 254.7 (2.9x) | 88.4 -> 88.4 |
| Qwythos-9B ngl 25 | - | 24.1 -> 169.6 (7.0x) | 24.0 -> 52.7 (2.2x) | - |
| R1-distill-7B GPU | - | 51.5 -> 51.4 | 51.5 -> 53.3 | 52.7 -> 52.6 |
| deepseek-coder 1.3B raw completion | 139.4 -> 239.9 (1.7x) one request; 345.4 -> 391.5 (1.13x) with 4 at once | | | |

Text identical except the 7B refactor answer and the first 1.3B answer under load. The 7B thinks first, so there is nothing to copy. 7B with 2000 tokens (`spec-r1-7b-long`, `-ngl 28`, 16K): edit 44.3 -> 88.9 (2.0x), refactor 45.2 -> 62.5 (1.4x), text differs; the edit run with the draft was still thinking at 2000 tokens and gave no code, the run without gave code at 1760: worse on one of two prompts, not applied. Size n: n=8 and n=6 are 9-25% faster than n=12 on code but draft on the open question too (0-1% accepted), costing 2-7% there and changing text; n=16 and m=24 slower, m=96 equal. `ngram-map-k`/`-k4v` match default on rewrite, nothing on edit/refactor.

**Draft-model speculative decoding** (qwen2.5-0.5b-instruct Q8_0 for qwen2.5-3b, t/s):

| Variant | edit | refactor | free |
|---|---|---|---|
| none | 86.6 | 86.5 | 88.4 |
| ngram-simple | 273.3 | 254.9 | 88.2 |
| draft-simple n-max 3 | 109.2 | 102.4 | 68.1 |
| n-max 8 | 121.0 | 98.3 | 38.7 |
| n-max 16 | 132.3 | 105.1 | 28.3 |
| n-max 16, p-min 0.75 | 143.9 | 126.2 | 72.4 |
| ngram-simple + line above | 277.5 | 214.5 | 72.3 |

Best 1.7x on code, open question 18-68% slower in every draft variant. The 7B cannot use this draft (BOS token differs; no GPU room). The 9B has no small same-vocabulary model here.

**Switches with no gain:** `GRAPH_OPT=1` 1.3B server 139.4 / 139.3 against 139.2 / 139.2 (one request), 345.4 / 343.9 against 343.7 / 343.2 (4 at once); `DISABLE_FUSION=1` decode 5% slower on 1.3B (143 against 151), 3-9% 7B, 1-4% 9B; `-ub 1024 / 2048` 2048-token prompt: 1.3B 2710 -> 2796 -> 2814, 3B 1227 -> 1257 -> 1263, 7B 645 -> 655 -> does not fit, 9B 486 and 477 -> 484 and 520 -> does not fit (not repeatable); 9B hybrid `REGISTER_HOST=1`, `--poll 0/100`, `--prio 2`, `-C 0x3F --cpu-strict 1`: all 20.5-21.3 against baselines 21.3 and 19.1.

**KV precision trials (2b)** (perplexity runner `ppl.py`):

| Trial | Outcome | Evidence |
|---|---|---|
| K q8 / V q4 on the 7B, all 29 layers | Applied: +11% at 0K, +16% at 15K | compare-kvmix-7b-confirm, ppl-kv-7b |
| q4_0 for K on the 7B | Rejected: perplexity 8.17 -> 1534 | ppl-kv-7b, ppl-kq4-check |
| Mixed precision as a speed setting | No gain, memory only | compare-kvmix-7b |
| K q8 / V q4 on the 9B | No gain; `-ngl 26` candidate with q8/q8 | ppl-kv-9b, compare-kvmix-9b-ngl |
| `--fit` automatic layer count | Exists; needs `-fitt 512` to match fixed count | spec-fit-7b |

Details: `-ngl 99` q8/q8 failed at 15K in that session, q8/q4 ran all four passes. K q4_0 breaks the 7B on every path: 1534 on GPU, 4227 on the old build, 4064 with CPU attention, against 12.8 for q8/q8 (4 chunks). 9B perplexity is flat: 3.716 f16, 3.709 q8/q8, 3.708 q8/q4, 3.725 q4/q4; V q4 frees about 40 MiB. 9B `-ngl 26` q8/q8: 25.6 / 22.9 / 21.0 against 24.6 / 20.9 / 18.3 at `-ngl 25`; `-ngl 27` q8/q4 once 27.3 / 24.1 / 22.2 with 420 MiB left; not applied (headroom). `--fit on` picks 25 of 29 layers with margin 1024 MiB (34.5 t/s); `-fitt 512` or `-fitt 256` picks 29 (48.5 against 47.2 t/s). Applied 2026-10-06: `build-live` rebuilt and the 7B line is `-ngl 99 --parallel 1 -c 16384 -ctk q8_0 -ctv q4_0 -fa on` (old file `models.conf.bak-kvmix`); the gateway was restarted. Not built: recent-token high-precision window, in-kernel dequantize (already there), shared GQA loads / tiled kernel (rejected), Turbo4/Turbo3/2-bit KV (no code). Not done: audit of KV and workspace allocation.

> ⚠ Contradiction: perplexity values. 7B: 12.8239 (4 chunks of 4096, `gpu-push.py` corpus) vs 8.181 / 8.204 / 8.166 (`ppl.py`, 16 chunks of the repo docs) vs 12.8 (4 chunks, K-q4 check). 9B: 3.716 / 3.709 / 3.708 / 3.725 (KV types) vs 5.3574 / 5.3588 (E1, S1, 4 chunks) vs 4.0059 / 4.0206 (S3, 16 chunks). Adjudicated (user asked me to settle): these differ by corpus and chunk count and are not comparable across experiments; only same-run pairs are valid. TRIALS' "16 chunks of 4096 tokens" for `ppl.py` is wrong for the 9B KV run, which used 8 chunks.

**Leftover leads (2c)**:
- ik_llama.cpp (fdb8e67) as an upper bound: no gain, prompt reading 3.6x slower, writing equal or slower. `bench.sh -t 6 -fa 1`, pp512 / tg128, ik against this tree: 1.3B 844 / 153.6 vs 3036 / 152.0; 3B 377 / 85.3 vs 1356 / 93.3; 9B ngl 22 153 / 15.2 vs 496 / 21.1; 9B CPU 145 / 8.0 vs 403 / 8.5 (the ik csv has the test name as last column, so `summarize.py` reads it wrong; use column -3). *(source: throughput/ik-probe, live-probe)*
- DFlash draft head for the 9B: no-go. `-ngl 20` ran out of memory; at `-ngl 14`, none -> DFlash with the draft on CPU: edit 13.7 -> 24.6 (1.8x), refactor 13.7 -> 24.2 (1.8x), free 13.5 -> 14.9 (+10%). At `-ngl 25` the plain model writes 24 and ngram-simple reaches 139 (edit, 8K) and 44 (refactor).
- 9B at 16K, two passes, 0K / 8K / 15K: `-ngl 25` 24.5 / 20.6 / 18.0 and 22.9 / 19.7 / 17.7; `-ngl 26` 25.1 / 22.9 / 20.8 and 25.6 / 22.8 / 21.1; `-ngl 27` 27.2 / 22.9 / no result and 26.2 / 23.7 / 21.9. Peak VRAM 5649 / 5671 / 5706 against about 5750. `-ngl 27` failed to create 15K once: not applied.
- 7B (`-ngl 99`, K q8_0 / V q4_0, 2000 tokens): none 50.1 / 51.0; n=12 103.9 text differs / 79.9 differs; n=24 110.1 same text (2.2x) / 65.9 differs; n=48 103.0 same / 62.8 differs. Answers finish (759 and 961 tokens). Refactor differs in every variant, worse or only different (not recorded). Not applied.
- Root-needing items were skipped that session (sudo asked for a password); power/memory/THP ran later (§2). EXPO/DDR5 left for the BIOS pass.

**Backend plan for the added models (2e)** (qwen3.5-4b, minicpm5-2b, virbiusguard; one change kept, and it is not a backend change; nothing in the source changed). Baselines: qwen3.5-4b 78.4 t/s writes, 986 reads, tools 6/6, code 4/4; minicpm5-2b 136 / 305 on 8 / reads (not recorded), routing 16/16 at 23.9 items/s; virbiusguard GPU 348 / 640 on 4 / reads (not recorded), guard 11/12 at 151 ms; virbiusguard CPU 112 / 145 on 2 / about 300, toolguard 19/22 at 101-111 ms. Ceiling for the 4B writing: about 105 t/s (2.7 GB per token at 288 GB/s). Plan rules (still the standing method, §9): baseline rerun in the same run; same conditions (overclock on, no other model loaded, 3 repetitions, reverse order when needed); under 3% is noise; correct first (`test-backend-ops`, same text at temp 0, job scores); ceiling check before code; one change at a time then stack; only winners stay; every result recorded; visible in `watch.py`; plan names `be-<id>-<what>`.

| Step | Result | Numbers | Files |
|---|---|---|---|
| One-token guard verdict | **kept** | tool-call check 214 and 239 ms down to 101 and 111 ms in the job (one cold read of the system prompt included), about 45 ms with the prompt cached; same verdicts on all 34 cases; no probability limit beat 0.5 (two missed tool calls score under 0.001: a model limit) | suites/suite-guard1.csv, suite-guard1-repeat.csv |
| B1 batch size (`-ub` 256 to 2048) | no gain | 4B reads 4096 tokens at 955 to 971 t/s for every value; two runs of 512 differ by 1.1% | throughput/compare-be-b1-batch.csv |
| B2 slot count | not applied | MiniCPM long output all slots 322 (8), 468 (12), 568 (16), 642 (24), 659 (32), same VRAM; routing 24.4, 25.1, 23.9, 21.1, 18.0 items/s; guard on GPU 642 (4), 847 (8) | suites/suite-be-b2.csv |
| B3 build flags (LTO) | no gain | guard CPU 125.5 and 128.7 (live) vs 127.5 and 126.9 (LTO); 12 threads slower than 6 (94-98); 4B GPU unchanged; AVX-512, VNNI, BF16 already compiled in | throughput/compare-be-b3-cpu.csv, -gpu.csv |
| B4 sampling on GPU (`-bs`) | no gain | one stream 133.3 to 134.1 (MiniCPM), 310.6 to 307.9 (guard GPU), 77.5 to 78.2 (4B); all slots 641-658 guard, under 3%; not verified that the server sampled on the GPU with these sampler settings | suites/suite-be-b4.csv |
| B5 mat-vec kernel shape | no gain | card uses its own table (`MMVQ_PARAMETERS_TURING`, 2 warps K-quants). 1 warp: 4B 79.7 -> 70.8, MiniCPM 144 -> 129, guard 350 -> 316; 4 warps 72.7, 136, 311; 8 warps 57.3, 99, 235 | throughput/compare-be-b5-w1.csv, -w4, -w8 |
| B6 output layer limited to allowed tokens | dropped at ceiling check | output layer is 18.5% (4B), 15.6% (MiniCPM), 32.7% (guard) of bytes per written token, but the guard writes one token after reading about 14 and the router two after about 17: gain about 3% and 2% | model file headers |

Routing is limited by batch steps per item, not by GPU or gateway (28.8 items/s through the gateway, 29.0 direct). `max_tokens` 1 with word labels gave 48-66 items/s against 36-50 (three repeats, same accuracy); one-letter labels were faster (72) but only 49 of 96 right. Two plan assumptions were wrong: the card already has its own kernel table (B5), and the routing test already used 96 items (B2). The final suite run and second pass of B1/B2 were skipped because the backend is unchanged.

**Baseline correction (TRIALS §3).** The audit's "q4_0 is 19% faster than q8_0 at depth" does not hold. Decode at 16K, KV on GPU (tg32): `kvmatrix-base0` old, 1 rep: q8_0 28.2, q4_0 33.6; `kvmatrix-base1` old, 1: 33.6, 31.5; `kvmatrix-exp4-nomma`, 1: 32.5, 33.0; `kvmatrix-final-base` old, 3: failed, 33.2; `kvmatrix-final-fix` new, 3: 32.6, 31.1. The 28.2 is a single run no later run approached; the derived per-head re-read cost of 17.2 ms is overstated. Figures with saved results: 158 and 94 t/s decode at depth 0 (1.3B, 3B; change-2 table), q8_0 KV 55 -> 28 t/s from 0K to 16K and q4_0 33.6 at 16K (`kvmatrix-base0.csv`: 55.1, 28.2).

## 8. Hypotheses H1-H13

Score is expected gain x confidence x workload share / cost on a rough 1-5 scale. Gains are end-to-end decode.

| # | Hypothesis | Expected gain | Confidence | Workload | Cost | Falsified if | State |
|---|---|---|---|---|---|---|---|
| H1 | 9B: attention layers on GPU, GDN layers on CPU | +16% at 8K, +30% at 15K | measured | 9B with context | flags only | - | Done, win (E1) |
| H2 | Shared-GQA vector kernel: load and dequantize each K/V row once for all query heads of a KV head (instruction count, not bandwidth) | 7B +10 to +25% at 15K, 0 at 0K; qwen35 under 5% | low to medium | 7B with context | high (234-255 registers) | a 7-column build not faster at 12K, or stack/local memory use | Rejected (S2): spills, +1% at 12K |
| H3 | 9B: Q4_K output matrix instead of Q6_K frees 263 MB for two more GPU layers | about +12% at every depth | medium | 9B | medium | perplexity outside its error, or gain under 5% | Done, win (S3) |
| H4 | 9B: smallest GDN layers on CPU (123 MB instead of 145 MB), about 110 MB less on CPU | +3 to +5% | medium | 9B | flags only, needs VRAM | no change or out of memory at 16K | Done, win (S1, e1-drop9) |
| H5 | CPU attention loop for quantized KV: row-blocked K dot and V accumulate without per-row `to_float` | large for any CPU attention layer, zero once H1 applied | medium | low after H1 | medium | perf shows the loop DRAM-bound | Open, low priority |
| H6 | 4B: Q4_K output matrix (521 MB Q6_K is about 15% of the token) | about +5% | medium | 4B | medium, perplexity gate | gain under 3% with repeats | Open, borderline |
| H7 | Attention V half: one scaled conversion instead of `__vsubss4` and per-byte int-to-float | at most 2-4% at 12K | low | 7B with context | low | V ablation share does not move | Open, below the 5% bar |
| H8 | One `quantize_q8_1` for Q, K, V and one fused QKV mat-vec | at most 1.5% | high | all | medium | - | Rejected by Amdahl |
| H9 | Fewer launches or syncs, tiny-kernel fusion | at most 1.2% (0.2 ms of gaps; "Other" 0.65 ms) | high | all | medium | - | Rejected by Amdahl |
| H10 | MMVQ compute changes (DP4A, layout, registers) | 0 at batch 1 | high | all | - | - | Rejected: 83-89% of bandwidth already |
| H11 | Attention occupancy (register diet, block count) | 0 | measured | - | - | - | Rejected (E2, E3) |
| H12 | Attention is bandwidth-bound; smaller KV type or load-once reuse saves traffic | 0 | measured | - | - | - | Rejected (KV type test); agrees with q8_0 vs q4_0 result |
| H13 | Workspace or allocation churn per token | 0 | measured | - | - | - | Rejected: zero memory events per token |

> ⚠ Contradiction: H2's note says it "was dropped earlier by proxy (plan item 5, gated on the f16-converting kernel)" and EXPERIMENT-DISCOVERY §5 says the shared-GQA idea was "Not tested here, see H2", while the State column says Rejected by S2. Adjudicated: the proxy drop was real, the idea was then tested directly as S2 and rejected. The reason for H2 changed from bandwidth (disproved) to instruction count (also not confirmed by S2).

External ideas checked against the code *(source: EXPERIMENT-DISCOVERY §5)*: mixed quantized KV falls back to CPU attention without `GGML_CUDA_FA_ALL_QUANTS` (llama.cpp issue 24485): confirmed, the fork adds one exception (K q8_0 / V q4_0); GQA-aware attention (FlashAttention-V arXiv 2608.18656, PyTorch int4 decoding post): the CUDA tile kernel has it (`ncols2`, `ntiles_z_gqa`), the vector kernel does not (`head / gqa_ratio`, `fattn-vec.cuh`); L2 prefetch hints in `mmvq.cu:674-690` (commit `d8f26eec7`) cover packed low-bit formats only, nothing to hide on Q4_K at 83-89% of bandwidth; per-hardware MMVQ warp count and MMQ crossover (`25ae3a9b3`, `2b5621094`) not swept; upstream changes since the fork point not checked (no fetched refs). The two papers were read from search summaries only: leads, not evidence.

## 9. Job checks and quality gates

- **Pass bar for any change:** correct first, fast second. In order: `test-backend-ops` for every op touched (`ops:` step in `config/suites.conf`); same text at temperature 0 (`jobs.py draft`) unless rounding changes, then `ppl.py` within 0.5%; job scores of the model must not drop (`suite.py run <name> --only=<model>`).
- **Noise:** 3% (`lab.py`). A difference under 3% in both runs is no change. Use at least 3 repetitions (`KV_REPS=3`) at 16K.
- **Gateway fit:** `free - 250 MiB >= vram_mb`.
- **Quality gates per track:** Track 2 WikiText-2 ΔPPL <= +0.08 (abort +0.12); Track 4 ΔPPL <= +0.10 on 16 chunks, code assertions >= 3/4, tool schema validation; Track 3 needle-in-a-haystack 100% (abort under 90%).
- **Context pass rule: 3/3 hidden codes.** A context length is certified when the probe recalls all three planted keys (`keys` 3/3) at that length with VRAM peak <= 5754 MiB and >= 5 t/s (see the §6 contradiction callout about `jobs.py`).
- **Jobs in `jobs.py`:** routing, tool calls, code that must pass asserts (3/4 or 4/4 shown in §3), guard verdicts (`guard`, `toolguard` read the model's full JSON; `guard1`, `toolguard1` start the answer and read the probability of the one verdict token, about half the time per verdict, same verdicts), drafter speed with unchanged text. `python3 local-tune/scripts/jobs.py <model> <job>` runs one.
- **Perplexity check for KV format changes:** `python3 local-tune/scripts/ppl.py run <name>` (plan `ppl-<name>.plan`). 7B K q8_0 error +/- 0.14.
- **Not recorded for long context:** needle recall, code asserts and tool checks at 49K-81K.

## 10. Rejected approaches

| Approach | Quantified cause | Source |
|---|---|---|
| Force quantized KV to the f16-converting kernel | slower or equal in every cell (q8_0 @8K 41.7 -> 34.0; lost the q8_0 16K case) | §7 item 1 |
| Shared-GQA vector kernel (S2) | 255 registers + 176 stack (spills); +1% at 12K, +2.5% at 15K | §7 S2 |
| Attention block count (E2) | default 5 best: 39.0 vs 34.4, 35.8, 37.8, 37.0 | §7 E2 |
| K thread layout 8 threads (E3) | 38.74 / 38.42 vs 38.97 / 38.12 | §7 E3 |
| Bandwidth explanation of attention growth | 3.5x byte change moves time under 12% | §5 |
| MMQ/MMVQ kernel work at batch 1 | 83-89% of bandwidth already | §5 |
| QKV fusion / shared input quantization | ceiling 1.5% | §8 H8 |
| Hadamard rotation kernels | 0.17 ms per token, 1% | §5 |
| Launch, sync, allocation, transfer | together under 3% of a token | §5 |
| K at q4_0 on 7B | perplexity 8.17 -> 1534 (4227 old build, 4064 CPU attention) | §7 2b |
| q4_0 K / q8_0 V mixed pair | no CUDA kernel; CPU fallback "hangs" at 8K | §5 |
| `draft-simple` model draft | open question 18-68% slower in every variant | §7 |
| DFlash draft head for 9B | needs fewer layers (ngl 14); gain 1.8x vs 7.0x ngram-simple | §7 2c |
| MTP drafters | 0.83x (text changed) and 0.65x | §3 |
| ngram-simple on 7B | thinking text repeated; one of two prompts worse | §7 |
| ik_llama.cpp | reads 3.6x slower, writes equal or slower | §7 2c |
| CUDA Graphs on/off, GRAPH_OPT, fusion off, `-ub 1024/2048`, host-memory switches | no gain, or slower | §7 |
| Backend steps B1-B6 | all within noise; B6 gain 3% and 2% | §7 |
| Fewer CPU bytes than 903 MiB with Q6_K output (765, 821) | out of memory at 8K-15K | §7 S1 |
| 5-block CPU set (587 MiB) | one CUDA error | §7 S3 |
| ub64 and ub32 for 96K / 80K | cmpute saving 18 MiB at 80K (ub64), 96K fails; ub32 fails at 80K | §6 |
| `-ngl 27` with 9B at 16K | failed to create 15K once | §7 2c |
| Recent-token high-precision KV window | V q4 already inside error, K below q8 fails at any age | §7 2b |
| Turbo4/Turbo3/2-bit KV | no code in this tree | §7 2b |
| MoE offload | no MoE model among the gateway models | §7 |

Leads still unmeasured (not rejected): EXPO/DDR5 speed and FCLK, C-state and IRQ affinity, SMT off, `mitigations=off` (trades security), mmap vs direct I/O and GPUDirect Storage (load time only), pinned host memory for KV in RAM, RoPE/YaRN extension (`--rope-scaling` changes quality, not speed), KV eviction/compression research code (Rolling KV, KVMem, StreamingLLM, DuoAttention), lower-bit KV (CommVQ, 1-2 bit).

## 11. Active directive: Context Capacity Maximization

Verbatim intent of the user-supplied `SYSTEM DIRECTIVE: CONTEXT CAPACITY MAXIMIZATION HARNESS (GTX 1660 Ti 6GB / TURING NON-MMA)` (the original text is not stored in the repo):

- **Primary objective:** maximize the largest stable, functional context window for the 7B and 9B without catastrophic quality loss. Metric: maximum verified stable context (tokens) that passes 3/3 hidden codes. Throughput policy: >= 5 t/s accepted if it increases usable context.
- **Verified ground truth:** the 9B placement C table and critical findings in §6 (ceiling exactly 5754 MiB; compute buffer linear in context and `ub`; 96K deficit about 220 MiB; 9B vs 7B divergence).
- **Tracks** (details and status in §6): 1 `ub64` + workspace clamp, goal cmpute <= 220 MiB at 80K and 96K stable with pp >= 150, abort pp < 100 or savings < 30 MiB. 2 imatrix + `IQ3_K`/`IQ3_XXS` FFN, goal mdl < 3500 MiB, 112K-128K, 3/3, ΔPPL <= +0.08, abort ΔPPL > +0.12 or decode < 8 t/s. 3 DuoAttention head-selective pruning on the 7B (128-token rolling window on streaming heads), goal KV -45% at 32K and 49,152 ctx with 100% needle recall, abort retrieval < 90%. 4 ShortGPT-style GDN block pruning on the 9B (2 blocks, 117.3 MiB each), goal >= 234 MiB saved and +16K ctx with code >= 3/4, abort ΔPPL > +0.10 (16 chunks). 5 mixed CPU/GPU KV streaming (62 GB DDR5 at about 50 GB/s), goal 131,072 ctx at >= 6.0 t/s, abort < 3.0 t/s.
- **Test matrix and logging columns:** §6.
- **Required deliverable:** measured VRAM/context breakdown comparing ub128 against new configurations (§6); exact headroom delta against 5754 MiB (§6 table); maximum certified ceiling per configuration (§6); quality verification (needle recall, code assertions, tool checks: (not recorded)); rejected approaches with quantified causes (§10); end with `### NEXT THREE CONTEXT EXPERIMENTS` (§6).

## 12. Tooling and reproduce commands

Layout: `scripts/` (code), `config/` (`assess.conf`, `suites.conf`, `models.example.conf`; `models.conf` git-ignored), `results/<family>/` (throughput, kv, speculative, quality, profiling, context, hardware, suites, lab; resolved by `scripts/paths.py`), `prof/` (CUPTI tracer, plans; stays in place), `img/`, `archive/` (superseded documents), `oc-kit/` (standalone copies of `gpu-push.py`, `gpu-tune.sh`, a short `watch.py` and `AGENT.md`, on purpose: the folder is copied to another machine alone).

```
# build, bench
local-tune/scripts/build.sh                                # CUDA 13.4, gcc 16
local-tune/scripts/bench.sh build <label> -t 6             # 4 model/placement rows, pp512 + tg128; summarize with scripts/summarize.py
KV_REPS=3 local-tune/scripts/kvmatrix.sh build             # KV type x GPU/RAM x 0K/8K/16K
python3 local-tune/scripts/compare.py run f16-confirm      # old vs new back to back (plans build-base, build-exp)
python3 local-tune/scripts/spectest.py build <label> <model.gguf> <server args>
python3 local-tune/scripts/ppl.py run <name>

# models, suite, assessment
cp local-tune/config/models.example.conf local-tune/config/models.conf   # your GGUF files
python3 local-tune/scripts/gateway.py &                    # serves them on :8700 (GATEWAY_CONF, GATEWAY_PORT, GATEWAY_BINDS, LLAMA_SERVER)
python3 local-tune/scripts/watch.py                        # live view, second terminal (watch.py <label> for a saved run)
python3 local-tune/scripts/suite.py run base [--only=a,b] [--skip=c]   # 1 llama-bench, 2 served speed, 3 job checks
python3 local-tune/scripts/suite.py run next
python3 local-tune/scripts/assess.py next base             # archive/ASSESSMENT.md, img/hero.svg, img/hero.png, img/models.svg
python3 local-tune/scripts/lab.py auto                     # watches sources, reruns suites, writes the results page
python3 local-tune/scripts/ctxprobe.py run <plan>          # context ceiling (results/context/ctx-<plan>.csv/.log/.plan)
python3 local-tune/scripts/gpu-push.py [--resume] [--power 120] [--mem-max 1500] [--mem-step 100] [--core-max 300] [--core-step 15] [--soak 10] [--margin 1]
python3 local-tune/scripts/gpu-push.py apply               # restore saved offsets
python3 local-tune/scripts/gpu-push.py reset
local-tune/prof/build.sh && python3 local-tune/prof/mkplans.py <trace-dir>
python3 local-tune/prof/analyze.py <trace-dir>/7b-d12288.tsv --json local-tune/results/profiling/prof-7b-d12288.json
```

Notes: `suite.py` stores each run's model list in `suite-<name>.conf.json`, so an old run renders after `models.conf` changes; `GATEWAY_URL` points it at another gateway. `served.py` runs a private gateway on port 8701 with the llama-server line under test (3 chats per depth, then 3 chats filling the 16K context as the soak). `lab.py run [suite]`, `lab.py accept`; add a suite with one line in `config/suites.conf`. `assess.py` sorts models by run (in both runs: kept; only in new: new); before/after pairs are in `config/assess.conf` as references to result files. `ledger.py` writes `ledger.html` (git-ignored). Every runner writes `START`, `TEST ... took=Ns`, `DONE ... took=Ns`, `ALLDONE total=Ns` lines for `watch.py`. An embedding model loaded in the real gateway takes 422 MiB and breaks 9B benches: `curl -s -X POST localhost:8700/unload/embeddinggemma-300m`. `gpu-tune.sh` is a guided front end for persistence mode, power limit and huge pages. How to add a trial: run it with `compare.py` (plan `compare-<name>.plan`) or `spectest.py` (`spec-<label>.plan`), keep csv and log under a new label, use at least 3 repetitions, add a row to §7 and move the lead out of §10's unmeasured list.

## 13. Commit timeline

Moved to [HISTORY.md](HISTORY.md) (22 commits from `52b730eb4` on 2026-10-05 19:34 to the consolidation commits of 2026-10-07).

## 14. Open questions and next actions

1. **Context ceiling (§6):** finish the forced-vec run; Track 2; KV streaming. `### NEXT THREE CONTEXT EXPERIMENTS` lists them in order.
2. Run the end-to-end suite with job checks on the placement-C 9B line (S1/S3 correctness checks were perplexity only).
3. Trace the 9B on placement C and write its VRAM budget at 16K; the "32 ms vs 16 ms" split is an estimate from the pre-S1 trace.
4. If root is available: set `RmProfilingAdminOnly` to 0 and read L2 hit rate and DRAM bytes for `flash_attn_ext_vec` (settles the 7B attention mechanism).
5. Open hypotheses H5, H6, H7 (§8).
6. Outside the backend: `max_tokens` 1 in the swarm caller; 16 slots for MiniCPM if more than 8 callers ever send long output; the Odysseus prompt that changes between agent rounds makes the 4B read 7,000 to 15,400 tokens again each round.
7. EXPO / DDR5 speed and FCLK in the BIOS, then rerun the 9B hybrid/CPU rows and `-nkvo 1` rows.
8. DFlash candidate `EntityDeletr/Qwen3.5-9B-DFlash-GGUF` (914 MB) was checked in 2c: no-go.
9. A longer real-workload overclock soak and an Xid log check over days.
10. Operational: update `gpu-oc.service` `ExecStart` to `local-tune/scripts/gpu-push.py` (works now via shim); restart the gateway (pid 51664 still runs the pre-move code); `ggml/src/ggml-cuda/fattn.cu` has an uncommitted 6-line change that this consolidation did not touch.
11. The 9B `cpu_model` 2788 MiB vs 704 MiB question (§6 callout).

## 15. Changelog of this document

- 2026-10-07: created from README, TRIALS, ASSESSMENT, EXPERIMENT-DISCOVERY, BACKEND-PLAN and the directive. Superseded originals moved to `archive/` with headers; scripts moved to `scripts/`, config to `config/`, results grouped by family; paths updated in all scripts (new `scripts/paths.py`).
- Assumptions: `free` in §6 is 5754 minus recorded peak; the ctx probe's `3/3` means three planted keys recalled; TRIALS' "16 chunks" for the 9B KV perplexity corrected to 8; the directive's `jobs.py` wording left as quoted and flagged; the 7B ceiling 32,768-49,152 is read from the `m2-slope` probe, not certified; ub64/ub32 peaks are read as including a higher desktop base.
- 2026-10-07: added §16 (coverage matrix).
- Contradictions adjudicated: GPU bandwidth 288 vs 331 GB/s; ceiling "about 5750" vs 5754; perplexity corpora; directive `jobs.py` vs ctxprobe keys; repo/branch name; `gpu-oc.service` "not set up" vs applied; E1 not applied vs next action 1 done; H2 dropped by proxy vs rejected by S2; "29 layers" on a 28-layer 7B; stale `build-exp` path; `cpu_model` 2788 vs 704 MiB (unreconciled); stale 9B numbers in the shortlist row.

## 16. Experiment coverage matrix

✔ run and recorded, ◐ partly run (noted), ○ not run yet, – does not apply. Largest gaps: the 4B (several levers never tried on it) and directive tracks 2–5.

### Experiment × model

| Experiment | 1.3B | 3B | 4B (qwen3.5) | 7B | 9B | Small 2B-class (gemma4-e2b, MiniCPM) |
|---|---|---|---|---|---|---|
| GTX 16xx kernel selection (change 1) | ✔ | ✔ | ○ | ✔ | ✔ (GPU and CPU) | ○ |
| GDN CPU path (change 2) | ✔ (no effect) | ✔ (no effect) | ○ (qwen35 model) | – | ✔ | – |
| q8_0 K / q4_0 V pair | ○ | ○ | ◐ (q8/q8 context run only) | ✔ | ◐ (perplexity, no speed gain) | ○ |
| GPU layer count / placement | – | – | ○ | ✔ | ✔ (E1, S1, S3) | – |
| ngram-simple speculation | ✔ | ✔ | ○ | ✔ (not applied) | ✔ | ○ |
| draft-model speculation | – | ✔ | ○ | ◐ (BOS mismatch, no room) | ○ (no matching small model) | – |
| MTP / DFlash drafters | – | – | ✔ (0.65×) | – | ✔ (DFlash no-go) | ✔ (0.83×) |
| CUDA Graphs on/off | ✔ | ○ | ○ | ✔ | ✔ | ○ |
| Fusion off, -ub 1024/2048 | ✔ | ✔ | ◐ (B1 -ub sweep only) | ✔ | ✔ | ○ |
| Attention kernel changes (E2, E3, S2) | – | – | ○ | ✔ (all rejected) | ○ (head size 256) | – |
| Q4_K output matrix | – | – | ○ (H6) | – | ✔ (S3) | ○ |
| Profiling trace (CUPTI) | ○ | ○ | ✔ | ✔ | ✔ (pre-S1 placement) | ○ |
| Context ceiling | ○ | ○ | ◐ (to 65K, uncertified) | ◐ (OK at 32K, fails at 49K, uncertified) | ✔ (certified) | ○ |
| GPU overclock | – | – | ○ | ✔ | ○ | ○ |
| Backend plan B1–B6 | – | – | ✔ | – | – | ✔ |

### Variants inside each experiment

| Experiment | Run | Not run yet |
|---|---|---|
| KV type (7B) | f16, q8/q8, q8/q4, q4/q4, K q4 / V q8 | recent-token high-precision window, Turbo4/Turbo3, 2-bit KV |
| 9B placement | shipped -ngl 25, E1, e1-drop9, e1-drop69, 7 small GDN blocks, A/B/C, a 5-block set | -ngl 27 repeated; a 9B Q8 or Q5 output file |
| -ub for context | 512, 256, 128, 64, 32 | -ub 16; -ub 64 with a clean desktop-memory baseline |
| Kernel for context | normal; forced vec at 49,152 only | forced vec at 65,536 to 131,072 |
| Context length | 8K to 81,920 (9B); 98,304 failed on ub128 and ub64 | 114,688; 131,072; 9B f16 past 24K; 7B certified at 32K |
| ngram settings | simple default, n/m sweep, map-k, map-k4v, mod | none open |
| Overclock | memory to +2400, core to +135, 100 W and 120 W, soak 14 of 14 | a multi-day soak with an Xid check; the 9B as the test load |
| Hardware | 100 W, locked clocks, memory +250, huge pages | EXPO/FCLK, C-states, SMT off, mitigations=off |
| Directive tracks 2–5 | none | IQ3 FFN, DuoAttention on the 7B, GDN pruning, KV streaming (-nkvo 1) |
| Quality at long context | 3 planted keys recalled | needle recall, code asserts, tool calls at 49K to 81K |
| End-to-end suite on placement C | perplexity only | job checks (routing, tools, code) |
