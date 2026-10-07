> **Superseded** by [UNIFIED.md](../UNIFIED.md) (§§6-8) on 2026-10-07. Kept for history; relative links below may be stale.

# Experiment discovery: where decode time goes on the 7600X + GTX 1660 Ti

Date: 2026-10-06. Tree `d954ac6f4` (master), `build-live` (llama-bench build `bafd6a875`), CUDA Graphs on.
Hardware state is in `results/prof-hwstate.txt`: overclock memory +2300 / core +105, 120 W, driver 615.71.09.

This report replaces flag searching with a measured cost model. One new win came out of it (9B, +30% decode at 15K context, section 6). The attention kernel on the 7B was narrowed down but not improved.

## 1. Method and limits

- Tool: a CUPTI activity tracer loaded with `CUDA_INJECTION64_PATH` (`prof/cupti_trace.cpp`, `prof/build.sh`). It records every kernel, copy, synchronization, allocation and runtime API call with timestamps. `prof/analyze.py` cuts the trace into decode tokens and prints the per-token table.
- Not available: hardware counters. `RmProfilingAdminOnly` is 1 and there is no root, so CUPTI cannot read DRAM or L2 traffic, and Nsight, `ncu` and `perf` are not installed. DRAM traffic below is bytes read divided by kernel time. Cache hit rates are inferred from experiments, not read.
- Tracer cost: 2-5% (59.8 t/s untraced, 58.6 traced, `compare-prof-smoke.csv`). Every traced depth also has an untraced row in the same csv.
- Register and shared-memory use per kernel: `cuobjdump -res-usage` on `libggml-cuda.so`, saved in `results/prof-kernel-resources.tsv`.

Reproduce:

```
local-tune/prof/build.sh
python3 local-tune/prof/mkplans.py <trace-dir>
python3 local-tune/compare.py run prof-7b     # also prof-9b, prof-4b, prof-7b15
python3 local-tune/prof/analyze.py <trace-dir>/7b-d12288.tsv --json local-tune/results/prof-7b-d12288.json
python3 local-tune/prof/analyze.py <trace-dir>/7b-d12288.tsv --seq    # event order of one token
```

The raw traces are 310 MB and are not in the repo. The plans regenerate them. The summaries are `results/prof-<model>-d<depth>.json`.

## 2. Measured bottleneck hierarchy

Milliseconds per decoded token, from the traces. "Weights" is every `mul_mat_vec_q` kernel. "Other" is all remaining GPU kernels.

| Model | Depth | Wall | GPU busy | Weights | Attention | quantize_q8_1 | Other | GPU idle |
|---|---|---|---|---|---|---|---|---|
| 7B | 0 | 16.79 | 16.41 | 14.88 | 0.59 | 0.30 | 0.63 | 0.38 |
| 7B | 4096 | 20.03 | 19.59 | 15.16 | 3.49 | 0.30 | 0.63 | 0.44 |
| 7B | 8192 | 22.67 | 22.28 | 14.92 | 6.42 | 0.31 | 0.64 | 0.39 |
| 7B | 12288 | 26.51 | 26.01 | 15.30 | 9.74 | 0.31 | 0.67 | 0.50 |
| 7B | 15360 | 28.73 | 28.28 | 15.20 | 12.12 | 0.31 | 0.65 | 0.45 |
| 9B, 25 layers on GPU | 0 | 39.26 | 15.85 | 14.23 | 0.11 | 0.28 | 1.23 | 23.41 |
| 9B | 4096 | 43.86 | 16.40 | 14.16 | 0.73 | 0.29 | 1.22 | 27.46 |
| 9B | 8192 | 46.59 | 16.78 | 14.03 | 1.26 | 0.27 | 1.21 | 29.81 |
| 9B | 12288 | 54.16 | 18.50 | 14.94 | 1.99 | 0.29 | 1.28 | 35.66 |
| 9B | 15360 | 54.28 | 18.32 | 14.34 | 2.45 | 0.28 | 1.24 | 35.96 |
| 4B | 0 | 13.17 | 12.57 | 10.32 | 0.16 | 0.30 | 1.80 | 0.60 |
| 4B | 8192 | 15.05 | 14.49 | 10.54 | 1.82 | 0.29 | 1.84 | 0.56 |

The 7B traced run at 15360 failed once on context creation (VRAM) and passed on the retry (`compare-prof-7b15`). A 14336 trace is also stored.

Classification of each major cost:

| Cost | Share | Class | Evidence |
|---|---|---|---|
| 7B and 4B weights (`mul_mat_vec_q`) | 89% of the 7B token at 0K, 53% at 15K | bandwidth | 275-294 GB/s effective against 331 GB/s theoretical (192 pins at 13.8 Gbps). Does not move with context depth |
| 7B attention at depth | 3% at 0K, 43% at 15K | compute, not bandwidth | Section 4: time does not follow bytes read, block count or K thread layout |
| 9B CPU layers at 0K | 56% of the token | bandwidth (system RAM) | 1077 MB of weights on the CPU in about 22 ms, about 50 GB/s on dual-channel DDR5 |
| 9B CPU attention at depth | +12.6 ms from 0K to 15K, 23% of the token at 15K | compute on the wrong device | Two attention layers (3 and 7) sit on the CPU. 6.3 ms per layer on the CPU against 0.41 ms per layer on the GPU |
| Launch overhead | 0.5-0.7 ms CPU per token, overlapped | launch overhead, negligible | One `cudaGraphLaunch` per token on the 7B and 4B. Median gap between kernels 0.2-0.4 us. GPU busy 96-98% |
| Synchronization | 18 stream syncs per token on the 7B | synchronization, negligible | 17 return at once (before small input copies), 1 is the real wait |
| Allocation | none | allocation | Zero memory events per decoded token in every trace |
| CPU/GPU transfer | 0.06 ms on the 7B, 0.14 ms on the 9B | CPU/GPU transfer, negligible | 7B: 8 uploads (95-118 KiB) and the logits download (592 KiB). 9B: 28 uploads, 7 downloads, 7 graph launches, 92 syncs per token because the graph is split |
| `quantize_q8_1` before each mat-vec | 0.30 ms, 169 calls | redundant memory traffic and launch overhead, small | 1.8% at 0K. Q, K and V projections quantize the same input three times |

## 3. Hottest kernels and context scaling

7B at 0K (678 kernels per token) and at 15K:

| Kernel | GGML op and source | Calls | ms at 0K | ms at 15K | us/call at 0K | Grid / block | Registers |
|---|---|---|---|---|---|---|---|
| `mul_mat_vec_q<Q4_K, fused>` | MUL_MAT, fused FFN gate and up, `mmvq.cu` | 28 | 7.17 | 7.20 | 256 | 18944 / 32x2 | 56 |
| `mul_mat_vec_q<Q4_K>` | MUL_MAT: q, k, v, o, ffn_down, `mmvq.cu` | 132 | 5.55 | 5.76 | 42 | 512-3584 / 32x2 | 43 |
| `mul_mat_vec_q<Q6_K>` | MUL_MAT, output layer 151665 x 3584 | 1 | 1.61 | 1.69 | 1614 | 151665 / 32x2 | 44 |
| `flash_attn_ext_vec<128, q8_0, q4_0>` | FLASH_ATTN_EXT, `fattn-vec.cuh` | 28 | 0.54 | 12.05 | 19 | 1 x 5 x 28 / 32x4 | 234 |
| `mul_mat_vec_q<Q5_K>` | MUL_MAT | 7 | 0.51 | 0.51 | 73 | 512 / 32x2 | 48 |
| `quantize_q8_1` | input of every mat-vec, `quantize.cu` | 169 | 0.30 | 0.31 | 1.8 | 74 / 256 | 18 |
| `rms_norm_f32` | RMS_NORM | 57 | 0.24 | 0.24 | 4.1 | 1 / 1024 | - |
| `rope_neox` | ROPE | 56 | 0.10 | 0.10 | 1.7 | - | - |
| `fwht_cuda<64>`, `fwht_cuda<128>` | Hadamard rotation of Q, K, V (`llama_mul_mat_hadamard`, `fwht.cu`) | 112 | 0.17 | 0.17 | 1.5 | - | - |
| `k_set_rows_quant` q8_0 and q4_0 | SET_ROWS, KV write | 56 | 0.13 | 0.13 | 2.3 | - | - |
| `flash_attn_combine_results` | second pass of the attention op | 28 | 0.05 | 0.07 | 1.8 | - | - |

Context scaling, 0K to 15K:

- Only one kernel grows: `flash_attn_ext_vec`. On the 7B it grows by 0.79 ms per 1000 tokens of context (28 layers, 28 query heads, 4 KV heads). Every other kernel is flat within 4%.
- 4B and 9B (qwen35: 16 query heads, 4 KV heads, head size 256, attention in one layer of four): 0.22 ms per 1000 tokens on the 4B (8 attention layers), 0.16 on the 9B GPU part (6 attention layers).
- On the 9B the fastest-growing cost is not on the GPU. The GPU-idle time grows by 12.6 ms from 0K to 15K while GPU attention grows by 2.3 ms. That is the CPU attention of layers 3 and 7.

Shapes that matter:

| Model | Layers | Query / KV heads | Head size | Attention layers | Largest matrix |
|---|---|---|---|---|---|
| 7B (qwen2) | 28 | 28 / 4 (ratio 7) | 128 | all 28 | FFN 3584 x 18944 Q4_K |
| 9B (qwen35) | 32 | 16 / 4 (ratio 4) | 256 | 8 (layers 3, 7, .., 31) | output 248320 x 4096 Q6_K, 834 MB |
| 4B (qwen35) | 33 | 16 / 4 (ratio 4) | 256 | 8 | output 248320 x 2560 Q6_K, 521 MB |

## 4. Memory and synchronization findings

Weights:

- 7B at 0K reads 4.25 GB of weights per token in 14.9 ms. The fused gate/up kernel reaches 294 GB/s, the unfused Q4_K calls 275 GB/s, the output layer 274 GB/s. That is 83-89% of the 331 GB/s the overclocked memory can deliver in theory.
- Consequence: nothing in MMQ/MMVQ (DP4A use, block layout, register count 41-61, shared memory 128-256 bytes) can give 5% at batch 1 on this card. The kernels already wait on memory. Only fewer bytes per token help.

Attention on the 7B (the only cost that grows):

- The kernel reads each KV head once per query head, 7 times on the 7B. If every read went to DRAM that would be 72 MB per layer at 12K, 207 GB/s, which looks like a bandwidth limit. It is not:
- KV type test at 8K (`compare-prof-kvtype`, traces `prof-7b-kv-*-d8192.json`): f16/f16 (512 bytes per head row) 208 us per call, q8_0/q8_0 (272 bytes) 220 us, q8_0/q4_0 (208 bytes) 232 us, q4_0/q4_0 (144 bytes) 218 us. A 3.5 times change in bytes changes the time by under 12%, and the largest format is the fastest. The f16 case would need more than the physical bandwidth if the repeated reads were not served from cache. So the repeated reads are mostly cache hits and the kernel is not bandwidth-bound.
- Timing-only ablations at 12K (kernel output is wrong on purpose): without the V read and accumulation the token drops by 4.05 ms; without the K read and dot product by 4.67 ms. K and V cost about the same, and together they are nearly all of the 9.7 ms.
- Register pressure is real but is not the limit. The kernel uses 234 registers (255 for head size 256), so only 2 blocks of 128 threads fit on one SM. Cutting the K path to 150 registers (E3) and forcing 1 to 20 blocks along the KV axis (E2) did not speed it up.
- Verdict: compute-bound on instruction count per head row, K and V in equal parts. Class: compute (with dequantization inside the V half). The mechanism below that level needs counters this machine does not expose.

9B:

- GPU busy is only 34-40% of the token. The GPU waits for 8 CPU layers.
- The CPU attention loop for a quantized KV cache (`ggml-cpu/ops.cpp:8622` and `:8668`) does one type-generic `vec_dot` per K row and one `to_float` plus `mad` per V row: 6.3 ms per layer at 15K, 15 times the GPU kernel.

Synchronization and launch: see the table in section 2. There is no per-token allocation, no workspace growth and no measurable launch gap. These surfaces are closed.

Finding outside the plan: `-ctk q4_0 -ctv q8_0` has no CUDA attention kernel in this build (`fattn.cu:444` accepts mixed types only as K q8_0 with V q4_0, and `GGML_CUDA_FA_ALL_QUANTS` is off). The run falls back to CPU attention and looks hung at 8K. `compare-prof-kvtype.plan` still lists that variant; it was stopped by hand.

## 5. External ideas checked against the code

| Idea | Source | Status in this tree |
|---|---|---|
| Mixed quantized KV types fall back to CPU attention without `GGML_CUDA_FA_ALL_QUANTS` | llama.cpp issue 24485 | Confirmed, see the finding above. The fork adds one exception (K q8_0 / V q4_0) |
| GQA-aware attention that reuses the shared KV tensors | FlashAttention-V, arXiv 2608.18656 (CPU vector architectures); PyTorch "int4 decoding" post | The CUDA tile kernel already has it (`ncols2`, `ntiles_z_gqa` in `fattn-common.cuh`). The vector kernel used for batch 1 does not (`head / gqa_ratio`, `fattn-vec.cuh`). Not tested here, see H2 |
| L2 prefetch hints in the mat-vec kernel | this tree, commit `d8f26eec7` | Present in `mmvq.cu:674-690` for the packed low-bit formats only. The Q4_K path already runs at 83-89% of bandwidth, so there is nothing for a prefetch to hide |
| Per-hardware MMVQ warp count and MMQ crossover | this tree, `25ae3a9b3`, `2b5621094` | Not swept, by the rules of this task. Section 4 shows the weight path is bandwidth-bound at batch 1 |
| Upstream changes since the fork point | - | Not checked. The `llamacpp` remote has no fetched refs in this clone |

The two papers were read from search summaries only. They are leads, not evidence.

## 6. Experiments performed

Each line: bottleneck, hypothesis, ceiling, experiment, result, correctness, verdict.

### E1. 9B: put both CPU attention layers on the GPU - WIN

- Bottleneck: GPU idle grows 12.6 ms from 0K to 15K; CPU attention costs 6.3 ms per layer at 15K against 0.41 ms on the GPU.
- Hypothesis: with the same number of weight bytes on the CPU, moving layers 3 and 7 to the GPU and GDN layers to the CPU removes the growth.
- Ceiling: about 11.8 ms of 54.3 ms at 15K (+28%), nothing at 0K.
- Experiment: `-ngl 30 -ot blk\.[45689]\.=CPU` against `-ngl 25`. CPU weights 1094 MB against 1077 MB. No code change. `compare-e1-9b-attn-gpu`, 3 repetitions, each variant run twice, interleaved.

| Depth | `-ngl 25` | attention on GPU | Change |
|---|---|---|---|
| 0 | 25.91, 26.44 | 26.06, 26.31 | 0% |
| 8192 | 21.87, 21.46 | 25.06, 25.12 | +16% |
| 15360 | 18.49, 18.91 | 24.35, 24.37 | +30% |

- Correctness: perplexity 5.3574 +/- 0.144 against 5.3588 +/- 0.144 (`ppl-e1-9b-attn-gpu`, 4 chunks). Prompt speed unchanged: 516.5 against 514.9 t/s at 2048 tokens (`compare-e1-9b-attn-gpu-pp`).
- Cost: peak VRAM 5446 MiB against 5407 MiB (the KV cache of two more layers).
- Verdict: keep. Not applied to the gateway; that is a configuration change for the owner to make (see next actions).

### E2. 7B: more or fewer attention blocks along the KV axis - rejected

- Hypothesis: 234 registers allow 2 blocks per SM, so the kernel is starved of warps and a different block count helps.
- Experiment: env knob `GGML_CUDA_FATTN_PB` in `launch_fattn` (`prof/fattn-knobs.patch`), 12K, `compare-e2-7b-pb`.
- Result: default (5) 39.0 and 38.3 t/s; 1: 34.4; 2: 35.8; 10: 37.8; 20: 37.0. A value of 40 ended in a CUDA error (knob only, not a product path).
- Verdict: the default is the best point. Occupancy is not the limit.

### E3. 7B: K dot product over 8 threads per row instead of 32 - rejected

- Hypothesis: the 5-step warp reduction per K row dominates; the HIP build already uses 4 threads.
- Experiment: compile-time `GGML_CUDA_FATTN_VEC_NT_KQ=8` (same patch), 12K, `compare-e3-7b-kq8`. Registers fell from 234 to 150.
- Result: 38.74 and 38.42 t/s live, 38.97 and 38.12 t/s with the change. No difference.
- Verdict: rejected. Thread layout of the K half is not the limit either.

### Diagnostics (not candidates)

- KV type against kernel time, section 4. `compare-prof-kvtype`.
- K and V ablations, section 4. Compile-time `GGML_CUDA_FATTN_ABLATE=1` or `2` in the same patch. These two runs were started with llama-bench directly, 2 repetitions each (45.74 +/- 0.37 and 47.09 +/- 0.15 t/s against 38.6), so they have no csv and did not show in `watch.py`.

The kernel source is back to the committed state. `prof/fattn-knobs.patch` holds the three knobs.

## 7. Ranked hypotheses

Score is expected gain x confidence x workload share / cost, on a rough 1-5 scale per factor. Gains are end-to-end decode.

| # | Hypothesis | Expected gain | Confidence | Workload | Cost | Falsified if | State |
|---|---|---|---|---|---|---|---|
| H1 | 9B: attention layers on the GPU, GDN layers on the CPU | +16% at 8K, +30% at 15K | measured | 9B with context | flags only | - | Done, win (E1) |
| H2 | Shared-GQA vector kernel: load and dequantize each K/V row once for all query heads of a KV head. The reason is instruction count, not bandwidth | 7B: +10 to +25% at 15K, 0 at 0K. qwen35: under 5% | low to medium | 7B with context | high: the kernel already uses 234-255 registers | a 7-column build is not faster at 12K, or `cuobjdump` shows stack or local memory use | Rejected (S2): spills, +1% at 12K |
| H3 | 9B: Q4_K output matrix instead of Q6_K frees 263 MB of VRAM for two more GPU layers | about +12% at every depth | medium | 9B | medium: requantize, perplexity gate | perplexity moves outside its error, or gain under 5% | Done, win (S3) |
| H4 | 9B: choose the smallest GDN layers for the CPU (123 MB each instead of 145 MB), about 110 MB less on the CPU | +3 to +5% | medium | 9B | flags only, needs VRAM | no change, or out of memory at 16K | Done, win (S1, e1-drop9) |
| H5 | CPU attention loop for quantized KV: row-blocked K dot and V accumulate without the per-row `to_float` | large for any CPU attention layer, zero once H1 is applied | medium | low after H1 | medium | perf shows the loop is DRAM-bound | Open, low priority |
| H6 | 4B: Q4_K output matrix (521 MB Q6_K is about 15% of the token) | about +5% | medium | 4B | medium, perplexity gate | gain under 3% with repeats | Open, borderline |
| H7 | Attention V half: replace `__vsubss4` and per-byte int-to-float with one scaled conversion | at most 2-4% at 12K | low | 7B with context | low | V ablation share does not move | Open, below the 5% bar |
| H8 | One `quantize_q8_1` for Q, K and V and one fused QKV mat-vec | at most 1.5% | high | all | medium | - | Rejected by Amdahl |
| H9 | Fewer launches or syncs, tiny-kernel fusion (norm, rope, fwht, set_rows) | at most 1.2% (0.2 ms of gaps), "Other" kernels total 0.65 ms | high | all | medium | - | Rejected by Amdahl |
| H10 | MMVQ compute changes: DP4A path, block layout, registers | 0 at batch 1 | high | all | - | - | Rejected: 83-89% of bandwidth already |
| H11 | Attention occupancy (register diet, block count) | 0 | measured | - | - | - | Rejected (E2, E3) |
| H12 | Attention is bandwidth-bound, so a smaller KV type or load-once reuse saves DRAM traffic | 0 | measured | - | - | - | Rejected (KV type test). Agrees with the earlier q8_0 against q4_0 result in TRIALS.md |
| H13 | Workspace or allocation churn per token | 0 | measured | - | - | - | Rejected: zero memory events per token |

H2 was dropped earlier by proxy (plan item 5, gated on the f16-converting kernel). The new evidence changes the reason for it, not its odds: the old case was bandwidth and that case is now disproved; the remaining case is fewer instructions per head row.

## 8. Rejected and why

- Bandwidth explanation of attention growth: time is independent of bytes per row across a 3.5 times range.
- Attention occupancy and thread layout: E2 and E3, no gain.
- Launch, sync, allocation, transfer: together under 3% of a token, measured.
- MMQ/MMVQ kernel work at batch 1: the weight path sits at 83-89% of theoretical bandwidth.
- QKV fusion and shared input quantization: ceiling 1.5%.
- Hadamard rotation kernels: 0.17 ms per token, 1%.

## 9. Second round (2026-10-06, overclock active: memory +2300, core +105, 120 W)

All rows below were taken with `gpu-oc.service` applied, so they do not compare with the tables above. Gateway rows come from `served.py`, which runs a private gateway on port 8701 with the llama-server line under test: 3 chats per depth, then 3 chats that fill the 16K context (the soak).

Block sizes of the 9B in MiB: GDN blocks are 137.9 (0, 1, 2, 6, 9, 12, 18, 21, 24, 28, 29, 30) or 117.3 (4, 5, 8, 10, 13, 14, 16, 17, 20, 22, 25, 26); attention blocks are 125.9 (3, 15, 27, 31) or 112.5 (7, 11, 19, 23); `output.weight` is 795.7 as Q6_K. With `-ngl 30` blocks 0 to 2 stay on the CPU and `output.weight` is on the GPU.

### S1. 9B placement - WIN, e1-drop9 is the byte-minimal stable set

- Variants: shipped `-ngl 25` (1024 MiB on the CPU), E1 `-ngl 30 -ot blk\.[45689]\.=CPU` (1041), e1-drop9 `-ngl 30 -ot blk\.[4568]\.=CPU` (903), e1-drop69 `blk\.[458]\.` (765), and 7 small GDN blocks with `-ngl 99` (821).
- llama-bench (`compare-s1-9b-place`, t/s, two passes):

| Variant | 0K | 8K | 12K | 15K |
|---|---|---|---|---|
| shipped | 26.1, 24.4 | 21.9, 21.5 | 20.0, 20.4 | 18.3, 18.5 |
| E1 | 26.0, 25.6 | 24.8, 25.1 | 24.5, 24.6 | 23.5, 24.2 |
| e1-drop9 | 27.7, 27.7 | 26.3, 26.6 | 26.0, 26.0 | 24.2, 25.5 |
| e1-drop69 | 27.0, 29.6 | fail, 28.2 | fail | fail |

- Gateway (`compare-s1-9b-served`, t/s, two passes):

| Variant | 0K | 8K | 12K | 15K | 16K soak |
|---|---|---|---|---|---|
| shipped | 26.9, 26.8 | 21.6, 21.5 | 20.2, 21.6 | 18.9, 19.0 | 18.8, 19.1 |
| E1 | 26.9, 26.7 | 24.2, 25.3 | 24.0, 24.4 | 24.9, 25.2 | 23.4, 24.0 |
| e1-drop9 | 28.6, 28.6 | 25.6, 25.8 | 25.4, 25.9 | 25.3, 25.4 | 26.9, 27.6 |

- e1-drop9 against shipped through the gateway: +6% at 0K, +19% at 8K, +23% at 12K, +34% at 15K. Every soak passed.
- Sets with fewer CPU bytes fail: e1-drop69 (765) and the 7-block set (821, in `compare-s3-9b-outq`) run out of memory at 8K to 15K.
- Correctness: perplexity 5.3574 shipped, 5.3588 E1, 5.3588 e1-drop9, each +/- 0.144 (`ppl-s1-9b-place`). The job checks of the end-to-end suite were not run.
- The `min8` and `min7` rows in `compare-s1-9b-place` are invalid: llama-bench splits `-ot` on commas into separate runs (the separator inside one run is `;`), so those rows had 3 blocks on the CPU. Ignore them.
- What it proves: the 9B is limited by which bytes sit on the CPU. Attention blocks on the CPU cost time that grows with depth; each 138 MiB GDN block on the CPU costs about 3 to 6% at every depth; the stable floor for the Q6_K file at 16K is about 900 MiB on the CPU.

### S2. 7B shared-GQA vector kernel - rejected

- Change: `flash_attn_ext_vec<128, 7, Q8_0, Q4_0>` with the 7 query heads of one KV head as columns, so each K and V row is loaded and dequantized once. Batch 1, Turing, no softcap only. Patch: `results/s2-7b-sharedkv-rejected.patch`. The source is back to the committed state.
- Registers (`cuobjdump -res-usage`): `REG:255 STACK:176` against `REG:234 STACK:0`. The kernel spills.
- End-to-end, same build, toggled by env (`compare-s2-7b-sharedkv`, t/s, two passes): 0K 57.6, 57.1 against 59.7, 57.4; 12K 39.0, 38.4 against 38.6, 38.1 (+1%); 15K 35.9, 35.9 against 35.2, 34.8 (+2.5%).
- Not run: `test-backend-ops -o FLASH_ATTN_EXT` for the new cases, traced attention time, perplexity.
- Verdict: rejected on two criteria, spilling and a gain under 5%. Sharing the K/V row load does not reduce the 7B attention cost in a useful way; the dequantize work per row is not what the kernel spends its time on, which agrees with the K and V ablations in section 4.

### S3. 9B Q4_K output matrix - WIN through residency

- File: `llama-quantize --allow-requantize --output-tensor-type q4_k <9B gguf> build-exp/qwythos-9b-outq4k.gguf Q4_K_M`. Only `output.weight` changed, 795.7 MiB to 545.6 MiB.
- llama-bench (`compare-s3-9b-outq`, t/s, two passes):

| Variant | CPU MiB | 0K | 8K | 15K |
|---|---|---|---|---|
| A: Q6_K, e1-drop9 | 903 | 27.7, 27.9 | fail, 26.6 | fail, 24.7 |
| B: Q4_K, same placement | 903 | 28.3, 27.6 | 27.4, 27.3 | 25.3, 25.5 |
| C: Q4_K, `-ngl 99 -ot blk\.[45]\.=CPU;blk\.1[3467]\.=CPU` | 704 | 31.1, 31.0 | 30.8, 30.3 | 29.6, 29.4 |
| Q4_K, `blk\.[45]\.;blk\.1[367]\.` | 587 | 34.4, 34.5 | fail, 32.7 | fail, 31.1 |

- The first pass of A failed on context creation within 5 s; the cause was not found and the same line passed every gateway soak.
- Gateway (`compare-s3-9b-served`, t/s, two passes): A 27.6, 28.3 at 0K; 25.3, 24.3 at 8K; 25.4, 25.0 at 15K; soak 26.9, 26.8. C 30.7, 31.0 at 0K; 30.0, 29.4 at 8K; 29.2, 30.2 at 15K; soak 27.7, 28.1. C against A: +11%, +20%, +18%. Every soak passed. Peak VRAM 5231 MiB for C against 5391 MiB for A.
- Correctness: perplexity 4.0059 +/- 0.050 for Q6_K against 4.0206 +/- 0.050 for Q4_K (`ppl-s3-9b-outq`, 16 chunks), +0.4%, inside the error.
- Verdict: keep C. The quantization alone is worth 3% (B); the gain comes from two more blocks on the GPU. The 5-block set is faster but hit a CUDA error once, so it is not stable.

### New bottleneck hierarchy

| Model | Cost | Share | State |
|---|---|---|---|
| 9B | CPU blocks, 704 MiB of weights read from system RAM per token | about half of the token: 32 ms per token at 0K against about 16 ms of GPU work (estimate from the section 2 trace, not re-traced) | dominant, open |
| 9B | attention growth with depth | 31.1 to 29.5 t/s from 0K to 15K, about 5% | closed by S1 |
| 7B, 4B | weights (`mul_mat_vec_q`) | 53 to 89% | at 83 to 89% of the bandwidth limit, closed |
| 7B | attention at depth | 43% at 15K | compute; E2, E3 and S2 rejected, mechanism still open |

The next single dominant measured bottleneck is the 9B's CPU residency: 6 GDN blocks on the CPU still cost about as much time as the whole GPU part. Each block moved to the GPU has been worth 4 to 6%. The lever is VRAM, about 117 MiB per block, and the smallest proof is a VRAM budget of the loaded 9B at 16K (weights, KV cache, compute buffers, desktop) to find bytes that can be given up.

## 10. Exact next actions

1. Done: the 9B is served from the Q4_K-output file. `qwythos-9b-outq4k.gguf` now sits next to the other model files (no longer in `build-exp/`), and in `~/odysseus-local/models.conf` the `qwythos-9b` line uses that path with `-ngl 25` replaced by `-ngl 99 -ot blk\.[45]\.=CPU,blk\.1[3467]\.=CPU` (llama-server separates overrides with a comma). Without the new file, `-ngl 30 -ot blk\.[4568]\.=CPU` on the current file is the proven fallback.
2. Run the end-to-end suite with its job checks on the new line.
3. The overclock offsets are lost at reboot unless `gpu-oc.service` runs; all numbers in section 9 depend on it. An embedding model loaded in the real gateway takes 422 MiB and breaks 9B benches: `curl -s -X POST localhost:8700/unload/embeddinggemma-300m`.
4. Trace the 9B on the new line and write the VRAM budget at 16K.
5. If root becomes available: set `RmProfilingAdminOnly` to 0 and read L2 hit rate and DRAM bytes for `flash_attn_ext_vec`. That settles the open mechanism in section 4.
