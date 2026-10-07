# Backend plan for the added models

Six backend changes for qwen3.5-4b, minicpm5-2b and virbiusguard on the GTX 1660 Ti. Each one has a goal in numbers, a
ceiling check that runs before any code is written, benchmark steps, and a rule for keeping or dropping it.
Nothing here is measured yet except the baseline column.

Model files are not changed by this plan.

## Outcome (2026-10-06)

All steps were run. No backend change was kept: B1, B3, B4 and B5 showed no gain, B2 gave more long-output throughput that nothing uses today,
and B6 was dropped at its ceiling check. The one-token guard verdict (outside the backend) was kept. Numbers and files are in [TRIALS.md](TRIALS.md), section 2e.
Two assumptions of this plan were wrong: the card already has its own kernel table (B5), and the served routing test already uses 96 items (B2, step 1).
The final suite run and the second pass of B1 and B2 were skipped because the backend is unchanged.

## Baseline today

From `results/suite-shortlist.csv` and `results/suite-added.csv`, build `build-live`, overclock on (memory +2300, core +105, 120 W).

| Model | Writes, one stream | All slots | Reads a prompt | Job |
|---|---|---|---|---|
| qwen3.5-4b | 78.4 t/s | 1 slot | 986 t/s | tools 6/6, code 4/4 |
| minicpm5-2b | 136 t/s | 305 t/s on 8 | not recorded | routing 16/16 at 23.9 items/s |
| virbiusguard, GPU | 348 t/s | 640 t/s on 4 | not recorded | guard 11/12 at 151 ms |
| virbiusguard, CPU (as served now) | 112 t/s | 145 t/s on 2 | about 300 t/s | one-token verdict: guard 11/12 at 146 ms, toolguard 19/22 at 101 to 111 ms; about 45 ms once the system prompt is cached |

The rough ceiling for the 4B is 105 t/s when writing: one token reads about 2.7 GB of weights and the card moves about 288 GB/s.

## Rules for every change

1. **The baseline is run again every time.** A candidate is only compared with a baseline from the same run: both variants
   in one `compare.py` plan, or two `suite.py` runs back to back. A stored number from an earlier day is never the baseline.
2. **Same conditions.** Overclock applied, no other model loaded (`nvidia-smi` shows only the benchmark), 3 repetitions.
   If `compare.py` runs variants in file order, each plan gets a second run with the order reversed, so warm-up and heat do not favour one side.
3. **3 percent is noise.** `lab.py` already uses this limit. A difference under 3 percent in both runs counts as no change.
4. **Correct first, fast second.** A candidate must pass, in this order:
   - `test-backend-ops` for every op it touches (`ops:` step in `suites.conf`);
   - the same text as the baseline at temperature 0 (the check that `jobs.py draft` does), unless the change is expected to alter rounding, and then `ppl.py` must stay within 0.5 percent;
   - the job scores of the model (`suite.py run <name> --only=<model>`): no score may drop.
5. **Ceiling before code.** Each code change starts with a measurement that says how much it can win at most. If the ceiling is under its goal, the change is dropped without being written.
6. **One change at a time, then stack.** Every change is first measured alone against the current baseline. After a change is kept:
   - it is committed and `build-live` is rebuilt from that commit;
   - `suite.py run base-<n>` runs for the three models and becomes the new baseline (`assess.py base-<n> base-<n-1>`);
   - all later changes are measured against this new baseline.
7. **Only winners stay.** A change is kept when it reaches its goal on at least one model and makes no model slower by more than 3 percent and no job score lower.
   Two changes that both win alone are also measured together; if the pair is not better than the better one alone, only that one stays.
8. **Every result is recorded.** Kept or dropped, the change gets a row in `assess.conf` with its two references and a line in `TRIALS.md` with the reason.
9. **Visible while it runs.** Every run goes through `compare.py`, `suite.py` or `lab.py`, so it shows in `watch.py`. Plans are named `be-<id>-<what>`.

Source changes are developed in `build-exp` and switched by an environment variable where possible, so one build can run both sides.

## Order

Settings first, because they cost nothing and move the baseline that the code changes are judged against. Then the code changes, cheapest ceiling check first.
After the last kept code change, the two setting sweeps (B1, B2) run once more, because a faster kernel can move the best batch size and slot count.

| Step | Change | Kind | For |
|---|---|---|---|
| B1 | Batch size sweep | setting | 4B reads |
| B2 | Slot count sweep | setting | minicpm5-2b, guard on GPU |
| B3 | Build flags | build | guard on CPU |
| B4 | Sampling on the GPU (`-bs`) | setting, code exists | minicpm5-2b, guard on GPU |
| B5 | Matrix-vector kernels on Turing | code | all three, writing |
| B6 | Output layer limited to the allowed tokens | code | constrained output only |

## B1. Batch size sweep

- **Goal:** qwen3.5-4b reads a prompt at 1085 t/s or more (10 percent over 986) with unchanged writing speed and VRAM that still fits beside the guard.
- **Why:** in Odysseus the agent re-reads 7,000 to 15,400 tokens per round, so reading speed is most of the wait.
- **Steps:**
  1. Plan `be-b1-batch`: `llama-bench -p 512,4096 -n 64 -r 3`, variants `-b 2048 -ub` 256, 512 (current default), 1024, 2048.
  2. Record VRAM per variant; a variant over the model's `vram` field in `models.conf` is out.
  3. Take the best variant and run `suite.py run be-b1 --only=qwen3.5-4b` against a fresh baseline run.
- **Keep if:** 10 percent or more on reading at 4096 tokens, writing within 3 percent.
- **Drop if:** best variant gains under 3 percent. Result goes into `models.conf` as `-ub`, no source change.

## B2. Slot count sweep

- **Goal:** minicpm5-2b routes 30 items/s or more (25 percent over 23.9). For the guard on GPU: 800 t/s or more on all slots (25 percent over 640).
- **Steps:**
  1. Extend the `classify` job with a longer item list (64 items) so a 16 or 32 slot run is not over before it is measured. Rerun the 8 slot baseline with the new list first.
  2. Gateway variants of minicpm5-2b with `--parallel` 8, 12, 16, 24, 32 and `-c` scaled so each slot keeps 2048 tokens. Run `suite.py run be-b2-<n> --only=minicpm5-2b` for each.
  3. Same for virbiusguard on GPU with 4, 8, 16.
- **Keep if:** items/s reaches the goal and one-stream latency grows by no more than 10 percent.
- **Drop if:** throughput is flat after 8 slots (then the GPU is already compute-bound).

## B3. Build flags

- **Goal:** the guard on CPU answers a tool-call check in 38 ms or less with the system prompt cached (15 percent under 45 ms).
- **Ceiling check:** read the `system_info` line of `build-live/bin/llama-server`. If AVX512, AVX512_VNNI and AVX512_BF16 are already 1, only LTO is left and the expected gain is small.
- **Steps:**
  1. `build-exp` with `-DGGML_LTO=ON`; if AVX-512 was off, also `-DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON`.
  2. Plan `be-b3-cpu`: `llama-bench -ngl 0 -t 6 -p 128 -n 16 -r 5` on the guard file, `build-live` against `build-exp`. Thread counts 6 and 12 as extra variants.
  3. `jobs.py virbiusguard toolguard` on both builds through the gateway (`LLAMA_SERVER=` selects the build), 3 runs each, median.
  4. One GPU plan for the 4B to prove LTO does not slow the CUDA path.
- **Keep if:** verdict time reaches the goal, same verdicts on all 22 cases.

## B4. Sampling on the GPU

The switch exists in this tree (`-bs`, `--backend-sampling`, marked experimental), so this is a trial of existing code.

- **Goal:** 8 percent or more writing speed on minicpm5-2b and on the guard when it runs on GPU. Nothing expected on the 4B.
- **Ceiling check:** time per token at 348 t/s is 2.9 ms. Measure the CPU share with `perf top` on a running guard; if sampling and the logits copy together are under 5 percent, stop here.
- **Steps:**
  1. Served comparison, since `llama-bench` does not sample: two gateway variants per model, with and without `-bs`, `suite.py run be-b4-off` and `be-b4-on` (phase 2: one stream and all slots).
  2. Same text at temperature 0 for both variants.
  3. Grammar check: the `classify` job uses a grammar. If `-bs` does not work together with a grammar, the change is only usable for free text and is judged on that.
- **Keep if:** goal reached on one model, job scores unchanged.

## B5. Matrix-vector kernels on Turing

Writing one token is a chain of quantized matrix-vector products (`ggml/src/ggml-cuda/mmvq.cu`). The kernel shape is chosen by
`calc_nwarps` and `calc_rows_per_block` from a per-device table; the GTX 16xx currently takes the generic entries.

- **Goal:** qwen3.5-4b writes at 86 t/s or more (10 percent over 78.4; the ceiling is about 105). No model slower.
- **Ceiling check:**
  1. `test-backend-ops perf -o MUL_MAT -b CUDA0` for the tensor types the three models use (read them from the GGUF header first: Q4_K, Q6_K and whatever the guard uses), at the row and column sizes of each model. This gives bytes per second per kernel.
  2. Compare with 288 GB/s. A kernel already above 90 percent of the card's bandwidth has nothing left; only kernels under 80 percent are worked on.
  3. If every kernel is above 90 percent, the 78 to 105 gap is not in these kernels and the change is dropped. The remaining time is then looked for with `nsys` on one `llama-bench -n 128` run (attention, normalisation, graph launch).
- **Implementation:**
  1. A device table entry for Turing without tensor cores (the same test that selects the kernels of change 1).
  2. Environment variables in `build-exp` to override warps and rows per block per type, for the sweep only. They are removed before the commit.
  3. Sweep warps 1, 2, 4, 8 and rows per block 1, 2 for each slow type with the `perf` mode above; put the winners into the table.
- **Benchmark:**
  1. `ops:MUL_MAT` must pass.
  2. Plan `be-b5-mmvq`: `llama-bench -fa 1 -d 0,4096,8192 -p 512 -n 64 -r 3` for the three model files, `build-live` against `build-exp`, both orders.
  3. `ppl.py` on the 4B: must be identical, the arithmetic does not change.
  4. `suite.py run be-b5` for the three models, then the 9B and 7B through `lab.py run` to prove they did not get slower.
- **Keep if:** goal reached on the 4B, or 5 percent or more on two models, and the 9B and 7B are within 3 percent.

## B6. Output layer limited to the allowed tokens

The last step of every written token multiplies the hidden state with the whole output matrix (vocabulary size times model width).
When only a known set of tokens can come next, only those rows are needed.

- **Correction to my earlier ranking:** this only saves time on tokens that are *written*. The guard and the router write very few tokens per request
  (with a one-token verdict, exactly one), and the server computes the output layer only for the last prompt token. So for their real work the gain is
  close to nothing. It pays off only for long output from a fixed small vocabulary, which none of the three models do today. It is listed last for that reason.
- **Goal:** 15 percent or more writing speed on a small model under a fixed token set. If the ceiling check gives less, the change is not written.
- **Ceiling check (no code):**
  1. From the GGUF header: size of `output.weight` in bytes divided by the bytes of all tensors read per token. This is the largest possible gain.
  2. Count written tokens per request in the `guard`, `toolguard` and `classify` jobs. Gain per request = share from step 1 times written tokens divided by all token passes.
  3. Stop if step 2 gives under 5 percent for every job, which I expect.
- **Implementation, only if the check passes:**
  1. Server option `--logit-subset <file>` with one token text per line. At load, the listed rows of `output.weight` are copied into a small F16 tensor.
  2. In the graph builders used by the three models (`src/models/qwen35.cpp`, `gemma4.cpp`, the MiniCPM one), the final `build_lora_mm(model.output, ...)` uses the small tensor. The graph stays static, so CUDA graph reuse is not affected.
  3. The logits buffer keeps its full size; rows outside the subset are set to minus infinity, so samplers and grammars need no change.
- **Benchmark:**
  1. `ops:MUL_MAT` and a new unit check: the logits of the subset rows equal the full run within F16 rounding.
  2. Served comparison with and without the option: `suite.py run be-b6-off` and `be-b6-on`, jobs `guard`, `toolguard`, `classify`.
  3. Same verdicts and same routes on every case.
- **Keep if:** goal reached and all job scores equal.

## After the last step

1. Rerun B1 and B2 on the final build.
2. `suite.py run final-backend` for all served models, then `assess.py final-backend shortlist`, so the report shows the whole effect against today's numbers.
3. `TRIALS.md` gets the list of dropped changes with their ceiling numbers, so they are not tried again without a new reason.

## Not in this plan

- Guard verdict from one token: done (`results/suite-guard1.csv`, `suite-guard1-repeat.csv`). Same verdicts on all 34 cases; a tool-call check went from 214 to 239 ms down to 101 to 111 ms
  in the job, where each job includes one cold read of the system prompt (about 1.1 s). With the prompt cached a verdict takes about 45 ms.
  No probability limit beat 0.5: the two missed tool calls score under 0.001, so they are a limit of the model, not of the threshold.
- Requantized or pruned model files.
- The Odysseus prompt that changes between rounds and defeats the prompt cache.
