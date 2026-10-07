# Local tuning: Ryzen 5 7600X + GTX 1660 Ti (6 GB), CachyOS

Three code changes live in this repository, plus a GPU overclock that is a setting and not code (see "GPU overclock" below). All are measured on this machine only.
The interactive version of these results is [report.html](report.html); the overview with charts is in the repository [README](../README.md).
Trials that did not ship and ideas not yet measured are in [TRIALS.md](TRIALS.md).
The current state in one page, with every model old and new and one picture, is [ASSESSMENT.md](ASSESSMENT.md) (see "Models, suite and assessment").

| # | Change | Effect | Commit |
|---|---|---|---|
| 1 | CUDA: stop using tensor-core kernels on GTX 16xx | prompt processing 2.9x to 4.2x faster | `a260d811e` |
| 2 | qwen35: skip fused raw-gate GDN path on CPU layers (x86) | 9B hybrid/CPU decode 20-35% faster | `52b730eb4` |
| 3 | CUDA: build the q8_0 K / q4_0 V attention pair by default | 7B fits all 29 layers at 16K: writing 11-16% faster | `afbf9a20c` |

Day-to-day workflow: keep `python3 local-tune/lab.py auto` open in a terminal. It reruns the checks that cover a source file when that file changes, and writes the results page again (see "Reproduce").

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
python3 local-tune/spectest.py build <label> <model.gguf> <server args>   # llama-server A/B: speculative decoding, env switches, parallel load (variants in results/spec-<label>.plan)
```
Live view in a second terminal: `local-tune/watch.py`. It follows the newest run (KV matrix, comparison or spectest); `local-tune/watch.py <label>` shows a saved one. Every runner writes a log with `START`, `TEST ... took=Ns`, `DONE ... took=Ns` and `ALLDONE total=Ns` lines, and the view shows a time table per row and context depth.

One view for everything: `python3 local-tune/lab.py auto`. It shows every suite, watches the source files named in `suites.conf`, and on a change rebuilds `build-exp`, runs the suites that cover the changed files against `build-live`, and writes `ledger.html` again. `lab.py run [suite]` runs now, `lab.py accept` takes the current source as checked, `lab.py` alone is view only. Add a suite with one line in `suites.conf`.

Perplexity check for KV format changes: `python3 local-tune/ppl.py run <name>` (plan in `results/ppl-<name>.plan`).

Results page: `python3 local-tune/ledger.py` writes `local-tune/ledger.html` from the files in `results/`. It lists every decision with its verdict and deciding number, then the charts. After a new trial, add one row to `DEC` in `ledger.py` and run it again.

### Models, suite and assessment

One file names the models: `models.conf` (format in [models.example.conf](models.example.conf)). `gateway.py` serves them on port 8700 and starts each `llama-server` on first use. Nothing else holds a model path: `suite.py` and `assess.py` ask the gateway (`GET /conf`).
```
cp local-tune/models.example.conf local-tune/models.conf    # your GGUF files, one line each
python3 local-tune/gateway.py                                # GATEWAY_CONF, GATEWAY_PORT, GATEWAY_BINDS, LLAMA_SERVER change the defaults
python3 local-tune/suite.py run <name> [--only=a,b] [--skip=c]   # 1 llama-bench per model, 2 served speed, 3 job checks
python3 local-tune/assess.py <new run> <base run>            # ASSESSMENT.md, img/hero.svg, img/hero.png, img/models.svg
```
- `suite.py` stores the model list of each run in `results/suite-<name>.conf.json`, so an old run still renders after `models.conf` changes. Point it at another gateway with `GATEWAY_URL`.
- Job checks are in `jobs.py`: routing, tool calls, code that must pass asserts, guard verdicts, and drafter speed with unchanged text. A model gets the checks named in the last field of its `models.conf` line. `python3 local-tune/jobs.py <model> <job>` runs one.
- Guard jobs: `guard` and `toolguard` read the model's full JSON answer; `guard1` and `toolguard1` start the answer for the model and read the probability of the one verdict token (about half the time per verdict, same verdicts). `toolguard` uses the `<tool>: <input>` form that Odysseus sends before it runs a tool.
- [BACKEND-PLAN.md](BACKEND-PLAN.md) is the plan for backend changes for the added models, with its outcome; the numbers are in [TRIALS.md](TRIALS.md) section 2e.
- `assess.py` sorts models by run: in both runs is kept, only in the new run is new. Dropped and excluded models, and the before and after pairs, are listed in `assess.conf` as references to result files.
- On this machine `~/odysseus-local/gateway.py` is a link to `local-tune/gateway.py`, and `~/odysseus-local/models.conf` is the model list. Odysseus only knows the gateway URL.
- `oc-kit/` holds its own copies of `gpu-push.py`, `gpu-tune.sh` and a short `watch.py` on purpose: the folder is meant to be copied to another machine by itself.

Result files in `local-tune/results/`:

| File | Content |
|---|---|
| `final-base.csv`, `final-fix.csv` | bench, old and new build |
| `kvmatrix-final-base.csv`, `kvmatrix-final-fix.csv` | KV matrix, old and new build |
| `compare-f16-confirm.*` | old vs new, f16 KV, two passes each |
| `compare-f16-decode.*`, `compare-f16-prompt.*` | attention kernel comparison |
| `compare-graphs-*` | CUDA Graphs on / GRAPH_OPT / off, see TRIALS.md |
| `compare-ngl-*` | GPU layer count at 16K context, see TRIALS.md |
| `compare-fusion-*`, `compare-ubatch-*`, `compare-host-*` | switches with no gain, see TRIALS.md |
| `spec-*` | llama-server tests: speculative decoding and load, see TRIALS.md |
| `compare-kvmix-*`, `ppl-*` | K / V precision pairs: speed and perplexity, see TRIALS.md section 2b |
| `compare-lab-*`, `ppl-lab-*` | plans that `lab.py` runs on a source change |
| `suite-<name>.csv`, `.log`, `.conf.json`, `compare-<name>.*` | one `suite.py` run: served speed and job checks, its model list, its llama-bench rows |
| `assess.json` | everything `assess.py` put in ASSESSMENT.md, as data |
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

## Change 3: q8_0 K / q4_0 V attention pair
`ggml/src/ggml-cuda/fattn.cu`, `CMakeLists.txt`: the vector kernel for K q8_0 with V q4_0 is in the default build. Before, a mixed pair ran attention on the CPU unless the build had `-DGGML_CUDA_FA_ALL_QUANTS=ON`. Other mixed pairs still need that flag.

Why this pair: on the 7B, K needs q8_0 (q4_0 K breaks the model), but V at q4_0 stays inside the perplexity error. V q4_0 frees about 130 MiB at 16K, and that is what lets the last layer go to the GPU.

R1-distill 7B Q4_K_S, `-c 16384`, writing t/s, two passes each (`compare-kvmix-7b-confirm`, `ppl-kv-7b`):

| Setup | 0K | 8K | 15K | Perplexity |
|---|---|---|---|---|
| `-ngl 28`, K q8_0 / V q8_0 (before) | 48.2-48.7 | 35.1-35.3 | 28.3-28.4 | 8.181 |
| `-ngl 99`, K q8_0 / V q4_0 (now) | 53.4-54.0 | 40.2-40.3 | 32.8-32.9 | 8.204 |

- Perplexity error is +/- 0.14; f16 KV gives 8.166. Prompt reading speed is equal.
- The pair alone is not a speed setting: at the same layer count it only saves memory.
- No gain on the 9B (its KV is small). Details in TRIALS.md section 2b.
- After a change to `fattn*.cu`: `lab.py` runs `test-backend-ops -o FLASH_ATTN_EXT`, the perplexity check and the speed comparison (suite `attention`).

## GPU overclock (a setting, not a code change)

`gpu-push.py` finds the highest memory and core offsets that stay correct, using the 7B model as the test load. Run it as your normal user; it asks for `sudo` once for the NVML offset calls and `nvidia-smi -pl`.

| Result (7B decode, 4K context) | |
|---|---|
| Stock, 100 W | 43.8 t/s |
| Saved: memory +2300 (6900 MHz), core +105, 120 W | 50.7 t/s (+16%), 14 of 14 soak passes, perplexity 12.8239 |

```sh
python3 local-tune/gpu-push.py [--resume] [--power 120] [--mem-max 1500] [--mem-step 100] [--core-max 300] [--core-step 15] [--soak 10] [--margin 1]
python3 local-tune/gpu-push.py apply   # set the saved offsets again, they are lost at reboot
python3 local-tune/gpu-push.py reset
```

- A step passes only on a clean exit, no new Xid line, perplexity equal digit for digit, decode not under 93% of the best, and under 83 C.
- `--resume` applies the saved pair and `--power`, re-verifies it (stepping core, then memory, down if it no longer holds), then only sweeps upward. The state file `results/gpu-oc.json` is written only after the final soak passes; the old one is copied to `gpu-oc.json.<stamp>`. It is per machine and ignored by git.
- `gpu-tune.sh` is a guided front end for the other system switches (persistence mode, power limit, huge pages). `watch.py` shows the run live.
- `oc-kit/` holds copies of the scripts and an `AGENT.md` that tells an AI agent what to edit for a different machine. Details and the step tables are in TRIALS.md section 2d.

## Setup
Rebuild: `local-tune/build.sh` (CUDA 13.4 from `/opt/cuda`, gcc 16). Benchmark: `local-tune/bench.sh <build-dir> <label> -t 6`, summarize with `local-tune/summarize.py local-tune/results/<label>.csv`.

## Build flags
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release` (+ccache if present).
Kept at defaults after measuring: `GGML_CUDA_FORCE_MMQ` (no change, within 1%), `GGML_CUDA_HOPPER_Q1` (sm_90a only, left off).

## Runtime defaults (measured)
- Threads 6 (physical cores); 4/8 equal, 12 slightly worse.
- `-ub 512`, flash attention on (`-fa on`); smaller ubatch is slower.
- 9B Q4_K_M with `-c 16384 -ctk q8_0 -ctv q8_0`: `-ngl 25` (24.6 t/s, was 21.3 at `-ngl 22`). `-ngl 27` is the last one that fits, 28 does not.
- 7B Q4_K_S with `-c 16384 -ctk q8_0 -ctv q4_0`: `-ngl 99`, all 29 layers (53.7 t/s; was 48.2 at `-ngl 28` with q8_0 / q8_0). This is the gateway setting since 2026-10-06 (change 3).
- Do not use `-ctk q4_0` on the 7B: perplexity goes from 8.2 to above 1500. `-ctv q4_0` is safe. The q4_0 rows of the KV matrix above are valid as speeds only.
- These layer counts assume one display on the 1660 Ti (about 750 MiB of desktop VRAM). Table in TRIALS.md section 2.
- `--spec-type ngram-simple` on llama-server: 2.2x to 7x writing speed when the answer repeats text from the prompt (code edits), no cost otherwise.
- Models up to ~3B fully offload (`-ngl 99`).
