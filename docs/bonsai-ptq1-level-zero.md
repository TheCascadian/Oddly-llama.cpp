# Bonsai PTQ1 on B580 with an isolated Level Zero loader

## Implemented

The project-local launcher uses Level Zero loader 1.32.0 and compiler/oneMKL 2026.1 without changing installed system packages. It verifies both the downloaded package and the loader binary. Download it with:

```sh
./scripts/run-sycl-level-zero.sh --setup
```

The PTQ1 T2 kernels are adapted from Torchit's arc-b580 snapshot 8160e3b. TernSYCL's license and attribution are retained in `ggml/src/ggml-sycl/ternsycl/`. They build as `libprism-sycl-t2`, independently of oneMKL's external device-image import setting. This fixes the missing-device-image error for IGC-provided matrix and memory instructions. Both the native memory instructions and the optional standard-SYCL replacements passed numerical checks.

The normal PTQ1 path remains the default. Enable T2 with `GGML_SYCL_PTQ1_T2=all`, or use `ffn` to convert only feed-forward weights. Selection checks the actual buffer device, tensor type and shape. Split buffers and unsupported layouts retain the normal path. T2 weights reserve 34 bytes per 128 weights instead of 28, approximately 21.4% additional storage for converted tensors. Temporary conversion space is also needed.

Temporary activation storage uses the backend memory pool. Zero activation scales are bounded to avoid fp16 overflow. Standard tensor reads, writes and generic operations restore PTQ1 first. Conversion checks canonical input encoding so that restoration is byte-exact; noncanonical encodings are declined. SYCL graph capture is disabled when T2 is selected because conversion performs blocking transfers.

## Build and launch

Use the existing matched oneAPI build. For a new build:

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl-t2 -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx \
  -DMKL_DIR=/opt/intel/oneapi/mkl/2026.1/lib/cmake/mkl \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-sycl-t2 --target llama-cli llama-bench llama-server -j 4
GGML_SYCL_PTQ1_T2=all ./scripts/run-sycl-level-zero.sh \
  ./build-sycl-t2/bin/llama-cli -m /path/to/Bonsai-PTQ1.gguf \
  --device SYCL0 -ngl 99 -c 4096 -ctk q4_0 -ctv q4_0 -fa on \
  -b 512 -ub 512 --single-turn -p 'Write a short Python function.'
```

`-DGGML_SYCL_T2_PORTABLE_LOADS=ON` builds the standard-SYCL memory-load alternative. Native memory helpers are the default and were faster in the measured prompt test. The linker warns that the Intel builtin functions are undefined in LLVM bitcode; IGC resolves them when loading the device code. Actual runtime checks, not those warnings alone, establish support.

## Measurements

B580, driver 26.35.39758.10, IGC 2.41.5, isolated loader 1.32.0, compiler/oneMKL 2026.1. Model: `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf` in the local Models directory. Same GPU and model, full offload, q4_0 K/V, flash attention, batch/ubatch 512. Three repetitions of pp512 and tg128, no speculation. Values are mean +/- sample standard deviation, tokens/second.

| Path | Prompt pp512 | Decode tg128 |
| --- | ---: | ---: |
| Existing path, T2 disabled | 568.21 +/- 4.12 | 8.04 +/- 0.03 |
| T2 with portable loads | 363.49 +/- 0.53 | 32.65 +/- 0.02 |
| T2 with native memory helpers | 715.17 +/- 5.75 | 32.89 +/- 0.27 |
| Native T2, oneDNN prompt threshold 512 | 674.12 +/- 3.50 | 33.22 +/- 0.15 |

The native T2 route was about 26% faster for this prompt test and 4.09 times faster for decode. The oneDNN threshold of 512 did not improve prompt speed, so it remains disabled by default. These short-context microbenchmarks do not establish the published speculative-edit or populated-128K speeds.

## Correctness and limits

- The existing PTQ1 MUL_MAT backend suite passed 38/38 supported cases; 40 other selected cases were skipped because the CPU reference did not support them.
- The standalone kernel check passed 40 shapes for each of two activation-scale settings, including batch tails, zero inputs, byte-exact restoration and noncanonical-input fallback. Native and portable loads both passed. See `artifacts/benchmarks/ptq1-t2-level-zero/check-kernel.cpp` and its logs.
- One fixed-seed, temperature-zero, 128-token Bonsai completion matched exactly with T2 enabled and disabled. This is a bounded correctness check, not a general quality evaluation.
- Plain generation completed at context capacity 131072, with a short prompt, at 27.5 t/s in one CLI run. This does not mean 128K tokens were resident.
- The first 128K MTP configuration (`all`, ubatch 512, draft ubatch 256) exhausted GPU physical memory. Do not use that configuration as a validated MTP preset.

A lower-memory MTP configuration completed at capacity 131072: `GGML_SYCL_PTQ1_T2=ffn`, `GGML_SYCL_T2_W8A8_MIN=0`, `GGML_SYCL_FA_ONEDNN_MAX_KV=98304`, `LLAMA_ARG_SPEC_DRAFT_UBATCH=64`, main `-ub 128 -b 512`, q4_0 caches for both contexts, `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 4`. The short 128-token run measured 30.7 t/s generation; the assistant output matched the baseline. This is a working memory configuration, not proof of the published MTP/edit throughput or filled-context performance.

Raw measurements and outputs are in `artifacts/benchmarks/ptq1-t2-level-zero/`.

## Research references

- [Intel compute-runtime 26.35.39758.10](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10) lists Level Zero 1.32.0 among its build components.
- [Intel compiler import classification](https://github.com/intel/llvm/blob/sycl/llvm/lib/SYCLPostLink/ModuleSplitter.cpp) explains how external-image imports can produce the missing-symbol error.
- [Intel SYCL-TLA](https://github.com/intel/sycl-tla) remains a matrix-kernel reference. A replacement kernel was unnecessary after isolating the existing Torchit device library.
