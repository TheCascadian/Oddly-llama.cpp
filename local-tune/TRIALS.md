# Trials: what was planned, what was measured, what is still open

Companion to [README.md](README.md). The README describes the two changes that shipped; this file records the plan behind them, the trials that did not ship, and the ideas nobody has measured yet.
All numbers are from this machine only (Ryzen 5 7600X, GTX 1660 Ti 6 GB), R1-distill-7B Q4_K_S, `-ngl 99 -fa 1`, unless stated.

## 1. The plan and its outcome

The plan came from an audit of the CUDA backend on this card. Items are in the order of its revised priority list.

| # | Plan item | Outcome | Evidence |
|---|---|---|---|
| 1 | Route quantized KV to the f16-converting attention kernel | Tested, no-go | `kvmatrix-exp1-forced.csv`, `kvmatrix-exp1-forced-nomma.csv` |
| 2 | CUDA Graphs A/B | Not tested | see backlog |
| 3 | MMA detection fix for GTX 16xx | Shipped | commit `a260d811e` |
| 4 | Non-MMA build A/B | Done | `final-base.csv` vs `final-fix.csv`, `compare-f16-*` |
| 5 | Shared-GQA kernel for quantized KV | Dropped | item 1 was its gate and failed |
| 6-11 | Remaining plan items | Not tested | nothing in the repo touches them |

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

## 2. Baseline, with one correction

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

## 3. Backlog: unverified leads

Nothing below has been measured here. The ideas past the audit came from a chat, and the PR numbers, forks, tools and speedups it cited have not been checked. Treat each line as a lead to verify, not as a result.

### Build and runtime switches (cheapest to measure)
- CUDA Graphs. All three build dirs already have `GGML_CUDA_GRAPHS=ON`, so the open A/B is `GGML_CUDA_GRAPHS=OFF` against the current build, and `GGML_CUDA_GRAPH_OPT=1` against unset.
- `--spec-type ngram-simple` on llama-server (with `--spec-ngram-simple-size-n` / `-m`). Needs a server-side test, `llama-bench` does not cover it.
- Plan items 6-11.

### Already in place
- `--no-mmproj-offload` for the 9B model, and q8_0 KV for all four LLMs, in the gateway `models.conf`.

### System tuning
- `mitigations=off` and `amd_iommu=off` kernel parameters. Both trade security or isolation for speed, they are not free.
- GPU memory overclock. Risk of silent corruption; any test needs an output check (perplexity or `test-backend-ops`), not only t/s.
- Huge pages for model and KV memory.

### Long context
- KV streaming to RAM or NVMe. Listed in the README as not done on purpose.
- Other huge-context ideas from the chat.

### Does not apply to the current models
- MoE offloading, expert pinning, shmem huge-page tuning for expert weights. None of the five gateway models is a MoE model.

## How to add a trial
Run it with the scripts in the README "Reproduce" section, keep the csv and log in `results/` under a new label, use at least 3 repetitions (`KV_REPS=3`) for anything at 16K, and add a row to section 1 or move the lead out of section 3.
