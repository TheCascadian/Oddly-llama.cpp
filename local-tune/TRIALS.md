# Trials: what was planned, what was measured, what is still open

Companion to [README.md](README.md). The README describes the two changes that shipped; this file records the plan behind them, the trials that did not ship, and the ideas nobody has measured yet.
All numbers are from this machine only (Ryzen 5 7600X, GTX 1660 Ti 6 GB), R1-distill-7B Q4_K_S, `-ngl 99 -fa 1`, unless stated.

## 1. The plan and its outcome

The plan came from an audit of the CUDA backend on this card. Items are in the order of its revised priority list.

| # | Plan item | Outcome | Evidence |
|---|---|---|---|
| 1 | Route quantized KV to the f16-converting attention kernel | Tested, no-go | `kvmatrix-exp1-forced.csv`, `kvmatrix-exp1-forced-nomma.csv` |
| 2 | CUDA Graphs A/B | Tested, no gain, defaults kept | `compare-graphs-*` |
| 3 | MMA detection fix for GTX 16xx | Shipped | commit `a260d811e` |
| 4 | Non-MMA build A/B | Done | `final-base.csv` vs `final-fix.csv`, `compare-f16-*` |
| 5 | Shared-GQA kernel for quantized KV | Dropped | item 1 was its gate and failed |
| 6 | EXPO / DDR5 speed and FCLK check | Not tested | BIOS setting; RAM speed not read (needs root `dmidecode`) |
| 7 | Speculative decoding | Tested: `ngram-simple` applied to the gateway. `draft-simple` no-go | `spec-*`, see below and section 2. DFlash, MTP, EAGLE3 and DSpark need their own draft models, not tested |
| 8 | MoE offload (`--n-cpu-moe`, expert pinning) | Does not apply | none of the five gateway models is a MoE model |
| 9 | ik_llama upper-bound probe | Not tested | separate fork, needs its own build |
| 10 | Turbo4 KV type and rotation, KLD check | Not tested | no such KV type or flag in this tree |
| 11 | Pinned host memory for expert / KV data | Not tested | only matters with KV in RAM or a MoE model |

### Item 1 in numbers

Decode speed (tg32, t/s) with q8_0 or q4_0 KV on the GPU, normal kernel -> forced f16-converting kernel. One repetition per cell.

| Build | KV type | @8K | @16K |
|---|---|---|---|
| MMA (`kvmatrix-base1` -> `kvmatrix-exp1-forced`) | q8_0 | 41.7 -> 34.0 | 33.6 -> no result |
| MMA | q4_0 | 42.0 -> 37.8 | 31.5 -> 31.5 |
| non-MMA (`kvmatrix-exp4-nomma` -> `kvmatrix-exp1-forced-nomma`) | q8_0 | 41.5 -> 34.4 | 32.5 -> no result |
| non-MMA | q4_0 | 41.5 -> 33.3 | 33.0 -> 23.5 |

"no result" = the row is missing from the csv, the run did not complete at that depth.
The forced kernel was slower or equal in every cell and lost the q8_0 16K case, so it was not kept.

### Item 2 in numbers

Three variants, each run twice in the order on, opt, off, on, opt, off, 3 repetitions per run. "on" is the current build (`GGML_CUDA_GRAPHS=ON`), "opt" adds `GGML_CUDA_GRAPH_OPT=1`, "off" is `build-nograph` (`-DGGML_CUDA_GRAPHS=OFF`). Values are t/s for the first / second pass.

| Model, test | on | opt | off |
|---|---|---|---|
| 1.3B Q8_0 on GPU, tg128 | 145.9 / 150.8 | 153.8 / 153.3 | 146.8 / 149.7 |
| 1.3B Q8_0 on GPU, pp512 | 2956 / 2817 | 3048 / 3040 | 3026 / 3042 |
| 7B q8_0 KV, tg32 @0K | 53.4 / 52.5 | 52.9 / 52.8 | 53.4 / 53.4 |
| 7B q8_0 KV, tg32 @8K | 39.3 / 40.6 | 40.4 / 40.2 | 40.6 / 40.5 |
| 7B q8_0 KV, pp512 @0K | 689 / 676 | 674 / 674 | 681 / 675 |
| 7B q8_0 KV, pp512 @8K | 425 / 432 | 433 / 431 | 433 / 428 |
| 9B ngl 22, tg64 | 20.8 / 21.0 | 20.9 / 20.8 | 21.1 / 21.2 |
| 9B ngl 22, pp512 | 482 / 478 | 491 / 494 | 495 / 472 |

Graphs on against off: no difference larger than the gap between two passes of the same variant.
`GGML_CUDA_GRAPH_OPT=1`: the 1.3B model writes 2-5% faster with less spread (+-0.2 against +-3 to +-9 t/s); nothing on the 7B and 9B. Too small to change a default.
The 16K depth of the 7B plan failed to create its context in all six runs. The desktop held about 1 GB of VRAM at the time, so this is a memory limit of the session and not a result about graphs.

### Item 7 in numbers: speculative decoding (`--spec-type`)

`local-tune/spectest.py` on llama-server, temperature 0, thinking off, 400 tokens per answer, 3 repetitions, `-c 8192 -ctk q8_0 -ctv q8_0`. Writing speed in t/s.

| Model | Prompt | none | ngram-simple | ngram-mod |
|---|---|---|---|---|
| qwen2.5-3b Q5_K_M, GPU | rewrite a file with one rename | 85.3 | 391.0 (4.6x) | 488.8 (5.7x) |
| qwen2.5-3b Q5_K_M, GPU | open question | 88.6 | 87.9 | 98.9 |
| Qwythos-9B Q4_K_M, ngl 22 | rewrite a file with one rename | 19.9 | 90.2 (4.5x) | 117.3 (5.9x) |
| Qwythos-9B Q4_K_M, ngl 22 | open question | 19.9 | 20.2 | 24.1 |

- `ngram-simple`: 4.5x on the rewrite prompt, no cost on the open question, and the output text is identical to the run without it in all 12 answers. 94-97% of its drafted tokens were accepted.
- `ngram-mod`: faster again, but it remembers earlier requests. On the repeated open question the second and third answers differ from the baseline text, and its gain there comes from seeing the same question again. Not a like-for-like result.
- The rewrite prompt is the best case (the answer is almost a copy of the prompt). Real code edits will land between the two rows.
- Applied to the gateway `models.conf` after the code-edit test in section 2.

## 2. Backlog trials, 2026-10-06

All on `build-live`, after one display moved to the iGPU (desktop VRAM about 660-760 MiB, was about 1 GB). Each variant ran twice in turn unless noted, 3 repetitions per run.

| Trial | Outcome | Evidence |
|---|---|---|
| GPU layer count after the display move | Applied: 9B `-ngl 22` -> 25, 7B `-ngl 24` -> 28 | `compare-ngl-*` |
| `ngram-simple` on code-edit prompts | Applied to the 9B, 3B and 1.3B models | `spec-tune-qwen3b`, `spec-edit-qwythos9b`, `spec-single-1p3b`, `spec-load-1p3b` |
| `ngram-simple` size n / m, `ngram-map-k`, `ngram-map-k4v` | Defaults kept | `spec-tune-qwen3b` |
| `ngram-simple` on the 7B with 2000-token answers | 1.4x to 2.0x, but the answer changes. Not applied | `spec-r1-7b-long` |
| Draft-model speculative decoding (`draft-simple`) | No-go | `spec-draft-qwen3b`, `spec-r1-7b` |
| `GGML_CUDA_GRAPH_OPT=1` under server load | No gain | `spec-load-1p3b`, `spec-single-1p3b` |
| `GGML_CUDA_DISABLE_FUSION=1` | Slower, default kept | `compare-fusion-*` |
| `-ub` 1024 and 2048 | 1.5-3% for 70-900 MiB of VRAM, default kept | `compare-ubatch-*` |
| `GGML_CUDA_REGISTER_HOST`, `--poll`, `--prio`, core pinning | No gain | `compare-host-9b-hybrid` |

### GPU layer count (`-ngl`) at 16K context, q8_0 KV

t/s, `-d 0,15360 -p 512 -n 32`. Peak VRAM includes the desktop.

| Model | ngl | tg32 @0K | tg32 @15K | pp512 @15K | Peak VRAM |
|---|---|---|---|---|---|
| Qwythos-9B Q4_K_M | 22 (old gateway value) | 21.3 | 15.2 | 385 | 4990 MiB |
| | 24 | 23.5 | 16.9 | 392 | 5224 MiB |
| | 25 (new gateway value) | 24.6 | 17.7 | 396 | 5335 MiB |
| | 26 | 25.6 | 18.8 to 20.8 | 376 to 400 | 5507 MiB |
| | 27 | 27.2 | 21.7 | 388 to 403 | 5616 MiB |
| | 28, 29 | 28.7, 30.6 | does not fit | does not fit | - |
| R1-distill-7B Q4_K_S | 24 (old gateway value) | 35.1 | 18.2 | 315 | 4784 MiB |
| | 26 | 39.9 | 22.2 | 320 | 5155 MiB |
| | 27 | 40.3 to 44.1 | 24.7 | 329 | 5293 MiB |
| | 28 (new gateway value) | 48.2 | 28.5 | 330 | 5422 MiB |
| | 99 (all 29) | 53.6 | 33.5 | 329 | 5574 MiB |

- The driver keeps about 390 MiB (`memory.reserved`), so a context fails to create when used memory would pass about 5750 MiB, not 6144.
- Chosen values leave about 330 MiB below that limit with the desktop at 750 MiB: 9B +15% at 0K and +16% at 15K, 7B +37% and +57%.
- 7B with all layers on the GPU is 11-18% faster again but leaves 180 MiB, and the gateway (`free - 250 MiB >= vram_mb`) does not load it. It needs the second display off the card.
- The gateway `vram_mb` values are now 4500 for both models. Measured use is about 4670 MiB each, so the gateway check is about 150 MiB more lenient than real use.

### `ngram-simple` on code-edit prompts

`spectest.py`, 400 tokens per answer, 3 repetitions, `-c 8192 -ctk q8_0 -ctv q8_0`. Writing speed in t/s, none -> `ngram-simple` (defaults).
"edit" asks for one changed function, "refactor" for type hints and a docstring on every function; both print the full file again.

| Model | rewrite | edit | refactor | free |
|---|---|---|---|---|
| qwen2.5-3b Q5_K_M, GPU | 86.6 -> 386.9 (4.5x) | 86.7 -> 273.7 (3.2x) | 86.7 -> 254.7 (2.9x) | 88.4 -> 88.4 |
| Qwythos-9B Q4_K_M, ngl 25 | - | 24.1 -> 169.6 (7.0x) | 24.0 -> 52.7 (2.2x) | - |
| R1-distill-7B Q4_K_S, GPU | - | 51.5 -> 51.4 | 51.5 -> 53.3 | 52.7 -> 52.6 |
| deepseek-coder 1.3B Q8_0, raw completion | 139.4 -> 239.9 (1.7x) on one request, 345.4 -> 391.5 (1.13x) with 4 requests at once | | | |

- Output text is identical to the run without it, except the 7B refactor answer and the first 1.3B answer under load.
- The 7B model thinks first, so the 400 tokens are reasoning text and there is nothing to copy. No gain measured, not applied there.
- 7B again with 2000 tokens per answer (`spec-r1-7b-long`, `-ngl 28`, 16K context): edit 44.3 -> 88.9 t/s (2.0x), refactor 45.2 -> 62.5 t/s (1.4x). The text is not the same as without it. Refactor ends with the same code. Edit does not: the run without it gives the code after 1760 tokens, the run with it is still in the thinking part at 2000 tokens and gives no code. The model repeats its own reasoning and `ngram-simple` copies that. Faster, but a worse answer on one of two prompts, so not applied.
- Size n (lookup length): n=8 and n=6 are 9-25% faster than n=12 on the three code prompts, but they draft on the open question too, with 0-1% accepted. That costs 2-7% there and changes the text. n=16 and m=24 are slower, m=96 is equal. Defaults kept.
- `ngram-map-k` and `ngram-map-k4v` match the default on rewrite and do nothing on edit and refactor.

### Draft-model speculative decoding

qwen2.5-0.5b-instruct Q8_0 as draft for qwen2.5-3b, both on the GPU, t/s.

| Variant | edit | refactor | free |
|---|---|---|---|
| none | 86.6 | 86.5 | 88.4 |
| `ngram-simple` | 273.3 | 254.9 | 88.2 |
| `draft-simple`, n-max 3 (default) | 109.2 | 102.4 | 68.1 |
| `draft-simple`, n-max 8 | 121.0 | 98.3 | 38.7 |
| `draft-simple`, n-max 16 | 132.3 | 105.1 | 28.3 |
| `draft-simple`, n-max 16, p-min 0.75 | 143.9 | 126.2 | 72.4 |
| `ngram-simple` + the line above | 277.5 | 214.5 | 72.3 |

- Best case 1.7x on code, but the open question is 18-68% slower in every draft variant. `ngram-simple` alone is faster on code and free on the rest.
- The 7B model cannot use this draft: the server refuses it (BOS token differs), and the draft does not fit on the GPU next to the 7B model anyway.
- The 9B model has no small model with the same vocabulary here.

### Switches with no gain

| Switch | Result |
|---|---|
| `GGML_CUDA_GRAPH_OPT=1`, 1.3B on llama-server | 139.4 / 139.3 against 139.2 / 139.2 t/s on one request, 345.4 / 343.9 against 343.7 / 343.2 with 4 at once. The 2-5% from llama-bench does not show on the server |
| `GGML_CUDA_DISABLE_FUSION=1` | decode 5% slower on the 1.3B (143 against 151), 3-9% on the 7B, 1-4% on the 9B hybrid; prompt speed equal |
| `-ub 1024` / `2048`, 2048-token prompt | 1.3B 2710 -> 2796 -> 2814, 3B 1227 -> 1257 -> 1263, 7B 645 -> 655 -> does not fit. 9B hybrid 486 and 477 -> 484 and 520 -> does not fit, not repeatable |
| 9B hybrid: `GGML_CUDA_REGISTER_HOST=1`, `--poll 0` / `100`, `--prio 2`, `-C 0x3F --cpu-strict 1` | all 20.5-21.3 t/s decode against 21.3 and 19.1 for the two baseline passes |

## 2b. KV precision trials, 2026-10-06

Scope: a list of 11 KV ideas, less the ones already rejected. Runner for perplexity: `ppl.py` (16 chunks of 4096 tokens from the repo docs).

| Trial | Outcome | Evidence |
|---|---|---|
| K q8 / V q4 on the 7B, all 29 layers | Applied. +11% writing at 0K, +16% at 15K | `compare-kvmix-7b-confirm`, `ppl-kv-7b` |
| q4_0 for K on the 7B | Rejected: perplexity 8.17 -> 1534 | `ppl-kv-7b`, `ppl-kq4-check` |
| Mixed precision as a speed setting | No gain, memory only | `compare-kvmix-7b` |
| K q8 / V q4 on the 9B | No gain; `-ngl 26` is a candidate with q8/q8 | `ppl-kv-9b`, `compare-kvmix-9b-ngl` |
| `--fit` automatic layer count | Exists. Needs `-fitt 512` to match the fixed count | `spec-fit-7b` |

R1-distill 7B, writing t/s, two passes each (`compare-kvmix-7b-confirm`):

| Setup | 0K | 8K | 15K | Perplexity |
|---|---|---|---|---|
| `-ngl 28`, K q8 / V q8 (gateway before) | 48.2-48.7 | 35.1-35.3 | 28.3-28.4 | 8.181 |
| `-ngl 99`, K q8 / V q4 (gateway now) | 53.4-54.0 | 40.2-40.3 | 32.8-32.9 | 8.204 |

- Perplexity error is +/- 0.14, f16 KV gives 8.166. Prompt reading is equal. `test-backend-ops -o FLASH_ATTN_EXT` passes.
- `-ngl 99` with q8/q8 failed at 15K in this session; with q8/q4 it ran in all four passes. V q4 frees about 130 MiB at 16K.
- Code: the q8_0/q4_0 pair is added to the default vector-kernel list (`fattn.cu`, `CMakeLists.txt`). Other mixed pairs still need `-DGGML_CUDA_FA_ALL_QUANTS=ON`.
- Applied 2026-10-06: `build-live` is rebuilt and the 7B line in the gateway `models.conf` is `-ngl 99 --parallel 1 -c 16384 -ctk q8_0 -ctv q4_0 -fa on` (old file: `models.conf.bak-kvmix`). The gateway reads the file at start, so it was restarted.
- K at q4_0 breaks this model on every path: 1534 on the GPU, 4227 on the old build and 4064 with attention on the CPU, against 12.8 for q8/q8 (4 chunks). So it is the format, not a kernel. The q4_0 rows of the README KV matrix are valid as speeds only.
- The 9B keeps its perplexity with every type (3.716 f16, 3.709 q8/q8, 3.708 q8/q4, 3.725 q4/q4). Its KV is small, so V q4 frees about 40 MiB. `-ngl 26` runs with q8/q8: 25.6 / 22.9 / 21.0 t/s against 24.6 / 20.9 / 18.3 at `-ngl 25`. `-ngl 27` with q8/q4 ran once at 27.3 / 24.1 / 22.2 with 420 MiB left. Not applied: `-ngl 25` was chosen for headroom.
- `--fit on` picks 25 of 29 layers with its default 1024 MiB margin (34.5 t/s). With `-fitt 512` or `-fitt 256` it picks 29, equal to the fixed count (48.5 against 47.2 t/s).
- Not built, with the reason:
  - Recent-token high-precision window: V q4 is already inside the error and K below q8 fails at any token age, so a window has nothing to win.
  - Dequantize inside the kernel, q8 K with DP4A: already in the vector kernel.
  - Shared GQA loads, tiled long-context kernel: rejected before (plan item 5; tile against vector in the README).
  - Turbo4 / Turbo3 and 2-bit KV: no code in this tree. The K result rules out fewer bits for K on the 7B; only V could go lower.
- Not done: the audit of KV and workspace allocation. One note from the kernel: with GQA the vector kernel reads each K/V head once per query head (7 times on the 7B).

## 2c. Leftover leads, 2026-10-06 (second session)

All on `build-live`, display on the iGPU (desktop 800-890 MiB).

| Trial | Outcome | Evidence |
|---|---|---|
| ik_llama.cpp (fdb8e67, built here) as an upper bound | No gain over this tree. Reads a prompt 3.6x slower, writes equal or slower | `ik-probe`, `live-probe` |
| DFlash draft head for the 9B (`draft-dflash`) | No-go: slower than `ngram-simple` and needs fewer GPU layers | `spec-dflash-9b`, `spec-dflash-9b2` |
| 9B `-ngl 26` / `27` with q8_0 K / q4_0 V, two passes | `-ngl 26` is stable. `-ngl 27` failed to create the 15K context in one pass. Not applied | `compare-ngl9b-confirm` |
| `ngram-simple` lookup length on the 7B, 2000 tokens | n=24 gives the same text as no draft on "edit" at 2.2x. Not applied | `spec-r1-7b-long2` |

- ik_llama.cpp, `bench.sh -t 6 -fa 1`, pp512 / tg128 t/s, ik against this tree: 1.3B 844 / 153.6 against 3036 / 152.0, 3B 377 / 85.3 against 1356 / 93.3, 9B ngl 22 153 / 15.2 against 496 / 21.1, 9B CPU 145 / 8.0 against 403 / 8.5. One pass of 3 repetitions each. The ik csv has the test name as last column, so `summarize.py` reads it wrong: use column -3 for t/s.
- DFlash, 9B, 8K context, q8_0 KV. At `-ngl 20` the draft run ran out of memory. At `-ngl 14`, t/s none -> DFlash with the draft on the CPU: edit 13.7 -> 24.6 (1.8x), refactor 13.7 -> 24.2 (1.8x), free 13.5 -> 14.9 (+10%). With the draft on the GPU it does not fit. At `-ngl 25` the plain model writes 24 t/s and `ngram-simple` reaches 139 (edit, 8K) and 44 (refactor), so DFlash gives nothing a layer count does not.
- 9B at 16K, writing t/s at 0K / 8K / 15K, two passes. `-ngl 25`: 24.5 / 20.6 / 18.0 and 22.9 / 19.7 / 17.7. `-ngl 26`: 25.1 / 22.9 / 20.8 and 25.6 / 22.8 / 21.1. `-ngl 27`: 27.2 / 22.9 / no result and 26.2 / 23.7 / 21.9. Peak VRAM 5649 / 5671 / 5706 MiB against a limit of about 5750. Prompt reading equal. Perplexity was not repeated (9B is flat across KV types, section 2b).
- 7B with the gateway setting (`-ngl 99`, K q8_0 / V q4_0), 2000 tokens allowed, t/s and whether the text equals the run without a draft:

| Variant | edit | refactor |
|---|---|---|
| none | 50.1 | 51.0 |
| n=12 (default) | 103.9, text differs | 79.9, differs |
| n=24 | 110.1, same text (2.2x) | 65.9, differs |
| n=48 | 103.0, same text | 62.8, differs |

  The answers now finish (759 and 961 tokens), unlike the `-ngl 28` run in section 2. Refactor differs from the baseline in every variant, and nothing here says if it is worse or only different. Not applied.
- Not run in this session: anything that needs root (governor, THP, persistence mode, power limit, memory clock, kernel parameters) because `sudo` asks for a password. The power limit, memory clock and THP were run later, see section 2d. EXPO / DDR5 is left for the BIOS pass.

## 2d. GPU clocks and power, 2026-10-06 (third session)

Run with `sudo` in the user's session. R1-distill 7B Q4_K_S, `-ngl 99`, K q8_0 / V q4_0, decode at 4K context, `llama-perplexity` on 4 chunks of 4096 tokens (stock value 12.8239). Tool: `gpu-push.py`.

| Trial | Outcome | Evidence |
|---|---|---|
| System switches with bench.sh: 100 W power limit, locked GPU clocks, memory offset +250, huge pages | No gain. Nothing beat the stock run; 100 W cost 1-3% | `gpu-base-*`, `gpu-pl100-*`, `gpu-lgc-*`, `gpu-mo250-*`, `gpu-thp-*` |
| Overclock sweep at a 100 W limit | Saved memory +1400, core +105. Decode 43.8 to 47.5 t/s (+8%) | `gpupush-1006-1250.log` |
| Overclock resumed at 120 W, upward only | Saved memory +2300, core +105. Decode 50.7 t/s (+16% over stock at 100 W) | `gpupush-1006-1325.log`, `gpu-oc.json` |
| Full `bench.sh` after the soak (offsets probably still active) | Decode +5% to +14% over the same build before | `full-1006-1356`, `live-probe` |

- 100 W run: memory passed every step to the +1500 cap (real clock 6001 to 6750 MHz; the offset is half the displayed change), so its limit was not found. Core +120 passed and +135 failed on perplexity (12.8248). Saved memory +1400 (one step of margin) and core +105, then 13 soak passes. The power column of this run read 42-48 W because the sampler read after the workload; it was fixed for the second run.
- 120 W run: the saved pair re-verified at 49.1 and 50.2 t/s (47.5 at 100 W with the same offsets, so about +5% from the power limit). Memory climbed to +2400 (6950 MHz) with no failure, decode 50.2-50.7 t/s. Core +120 passed (50.9), +135 failed again (12.8234). Saved memory +2300 and core +105, then 14 soak passes: 50.4-50.7 t/s, 121-123 W, 73-75 C, perplexity 12.8239 each time. Run time 1252 s.
- The split between power limit and memory clock is not separable from these runs: at the same offsets the power limit gave about +5%, and memory +1400 to +2300 added about 1 t/s. Single passes; splits under 3% are noise.
- The 12:29-12:39 system runs are labelled by file name only; I read them as power limit 100 W, locked clocks, memory offset +250 and huge pages.
- Offsets live in the driver and reset at reboot or on a crash. `gpu-push.py apply` sets the saved pair again. A boot-time systemd unit for it is not set up.
- Not done: a longer real-workload soak, and a check of the Xid log over days of use.

## 3. Baseline, with one correction

### Figures that have saved results
| Figure from the audit | Where it is |
|---|---|
| 158 and 94 t/s decode at depth 0 (1.3B and 3B models) | README, change 2 table, "Oddly before" column |
| q8_0 KV decode 55 -> 28 t/s from 0K to 16K | `kvmatrix-base0.csv` (55.1 and 28.2) |
| q4_0 KV decode 33.6 t/s at 16K | `kvmatrix-base0.csv` |

### The 16K figure did not repeat
Decode at 16K with KV on the GPU (tg32, t/s):

| Run | Build | Reps | q8_0 | q4_0 |
|---|---|---|---|---|
| `kvmatrix-base0` | old | 1 | 28.2 | 33.6 |
| `kvmatrix-base1` | old | 1 | 33.6 | 31.5 |
| `kvmatrix-exp4-nomma` | non-MMA trial | 1 | 32.5 | 33.0 |
| `kvmatrix-final-base` | old | 3 | failed | 33.2 |
| `kvmatrix-final-fix` | new | 3 | 32.6 | 31.1 |

The 28.2 is a single run and no later run came near it. q8_0 and q4_0 are within about 2 t/s of each other at 16K, in either direction.
So the audit's "q4_0 is 19% faster than q8_0 at depth" does not hold, and the per-head re-read cost of 17.2 ms that was derived from that gap is overstated.
Items 1 and 5 were aimed at that gap. Part of the gap was measurement noise, which is a likely reason item 1 found nothing to win.

### Figures with no measurement behind them
- 226 vs 288 GB/s memory bandwidth.
- 15-20% headroom.

Both are calculations from the audit. Nothing in `results/` supports or contradicts them.

## 4. Backlog: unverified leads

Nothing below has been measured here, except where a line says what was checked. The ideas past the audit came from a chat, and the PR numbers, forks, tools and speedups it cited have not been checked. Treat each line as a lead to verify, not as a result.

### Build and runtime switches
- ik_llama.cpp as an upper-bound probe on the same models (plan item 9). Cloned to `~/src/ik_llama.cpp` (commit fdb8e67), not built: the build of external code was not permitted in the agent session. Build it by hand, then run `bench.sh` on its build directory.
- Draft types that need their own draft model (`draft-mtp`, `draft-dflash`, `draft-eagle3`, `draft-dspark`). Checked 2026-10-06: each needs a head trained for the exact target model. No published head for Qwen2.5-3B, R1-distill-Qwen-7B or deepseek-coder-1.3b. `draft-mtp` needs MTP tensors in the model file and the Qwythos 9B file has none. One candidate is left: `EntityDeletr/Qwen3.5-9B-DFlash-GGUF` (914 MB, trained for base Qwen3.5-9B; Qwythos is a fine-tune of it). It is downloaded to `~/.lmstudio/models/EntityDeletr/Qwen3.5-9B-DFlash-GGUF/`, but the run was not permitted in the agent session. To test: `spectest.py` on the 9B with `--spec-type draft-dflash -md <file> -ngld 0`, and with `-ngl 20 -ngld 99` against `none` at `-ngl 20`.
- Pinned host memory for KV in RAM (plan item 11). Only the `-nkvo 1` rows could gain.

### Already in place (checked on this machine)
- `--no-mmproj-offload` for the 9B model, and q8_0 KV for the 9B, 3B and 1.3B models, in the gateway `models.conf`.
- `--spec-type ngram-simple` for the 9B, 3B and 1.3B models, `-ngl 25` for the 9B (section 2). The 7B runs `-ngl 99` with K q8_0 / V q4_0 (section 2b); before that `-ngl 28` with q8_0 for both.
- CPU governor is `performance`.
- Swap is zram (zstd), no disk swap.
- Threads pinned to the 6 physical cores by count (`-t 6`); 4, 8 and 12 were measured in the README.

### System tuning
State today: shmem THP is `never`, the kernel command line has no `mitigations=` or `amd_iommu=` entry, and the 1660 Ti still drives one display (660-760 MiB of VRAM in use by the desktop).
- One display is on the iGPU since 2026-10-06; the other one stays on the 1660 Ti. Moving it would free the room for the 7B model with all layers on the GPU (section 2).
- EXPO / DDR5 speed and FCLK (plan item 6). Affects CPU layers and KV in RAM only. Not measured: needs the BIOS. After a change, rerun `bench.sh` (9B hybrid and CPU rows) and the `-nkvo 1` rows of `kvmatrix.sh`.
- C-state limits, IRQ affinity, explicit CPU affinity, SMT off, real-time scheduling.
- shmem THP, proactive compaction, swappiness, zram against zswap. Huge pages for model and KV memory.
- GPU persistence mode (set by `gpu-push.py`, not measured alone).
- Power limit and GPU memory and core clock offsets: measured, section 2d. They gave +16% together. Any change needs an output check (perplexity), not only t/s.
- `mitigations=off` and `amd_iommu=off` kernel parameters. Both trade security or isolation for speed, they are not free.
- Model loading: mmap against direct I/O, GPUDirect Storage, a separate NVMe for models. Load time only, not t/s.

### Long context
- KV streaming to RAM or NVMe, CPU-offloaded KV. Listed in the README as not done on purpose; the `-nkvo 1` rows of the KV matrix show the cost today (13.9 against 32.6 t/s at 16K, q8_0).
- Lower-bit KV: Turbo4 / TurboQuant (plan item 10), CommVQ, 1-2 bit KV. Checked 2026-10-06: no such KV type in this tree or in the ik_llama.cpp clone, so there is nothing to measure. Each would need new code and a quality check (KLD or perplexity) before a speed number means anything.
- KV eviction or compression schemes: Rolling KV, KVMem, StreamingLLM, DuoAttention. Research code, not in this tree.
- RoPE / YaRN context extension (`--rope-scaling`). Exists in this tree; changes output quality, not speed.

### Does not apply to the current models
- MoE offloading (`--cpu-moe`, `--n-cpu-moe`), expert pinning, shmem huge-page tuning for expert weights. The flags exist, but none of the five gateway models is a MoE model.

## How to add a trial
Run it with the scripts in the README "Reproduce" section (a `compare-<name>.plan` for build or env A/Bs, a `spec-<label>.plan` with `spectest.py` for server-side options), keep the csv and log in `results/` under a new label, use at least 3 repetitions (`KV_REPS=3`) for anything at 16K, and add a row to section 1 or 2 and move the lead out of section 4.
