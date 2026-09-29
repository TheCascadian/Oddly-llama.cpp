# Intel Arc backend selection

This Prism build exposes the Vulkan, SYCL, and OpenVINO backends as normal
llama.cpp devices. The target device used during validation was an Intel Arc
B580.

## Terminal

Source the oneAPI runtime before running the Intel compiler-built binaries:

```bash
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH="/usr/lib:/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/dnnl/2026.1/lib:/opt/intel/oneapi/mkl/2026.1/lib:${LD_LIBRARY_PATH:-}"
```

List devices and select a backend with the standard llama.cpp flags:

```bash
llama-cli --list-devices
llama-cli -m /path/to/model.gguf --device Vulkan0 -ngl 99
llama-cli -m /path/to/model.gguf --device SYCL0 -ngl 99
GGML_OPENVINO_DEVICE=GPU llama-cli -m /path/to/model.gguf --device OPENVINO0 -ngl 99
```

`GGML_OPENVINO_ENABLE_LARGE_ALLOCATIONS=1` is an experimental opt-in for
weights larger than the GPU plugin's normal per-allocation limit. It is not
enabled by the WebUI and is not validated for the B580 27B models.

The Prism SYCL backend defaults to `ONEAPI_DEVICE_SELECTOR=opencl:gpu` when no
selector is supplied. This is the stable path for the Arc B580 with the
oneDNN-backed kernels. Set `ONEAPI_DEVICE_SELECTOR` explicitly when testing a
different SYCL runtime path.

## Level Zero API support

`GGML_SYCL_SUPPORT_LEVEL_ZERO_API` defaults to `ON`. The SYCL target enables it
when CMake finds both `level_zero/ze_api.h` and the Level Zero loader. CMake
reports the resolved header and loader paths; the discovered header directory
is added to the target include path. To select the Level Zero SYCL runtime for
a test, set `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` after sourcing oneAPI.

The B580 is discovered by `sycl-ls` as `[level_zero:gpu:0]`. The Level Zero
enabled fork builds and passes focused Q4_K SYCL backend operation tests on
that device. A Qwen3.5-9B Q4_K_M end-to-end benchmark currently fails during
`MUL_MAT` with `could not create a memory object`; use the default OpenCL path
for full model runs until that runtime issue is resolved.

The SYCL backend has an experimental fused Q5_K gate-up GLU path alongside the
fork's existing reordered Q4_K row-pair path. Enable Q5_K fusion with
`GGML_SYCL_ENABLE_Q5K_GLU_FUSION=1`. On the available Qwen3.5 Q5_K model, B580
OpenCL `llama-bench` at p128/n64/r5 measured 52.67±0.09 decode tokens/s with
fusion off and 52.16±0.12 with it on; prompt processing was effectively
unchanged (1017.1±2.4 vs 1018.3±4.1 tokens/s). It remains opt-in because decode
was about 1% slower. A fixed-seed 24-token generation matched with fusion on
and off. The newer mixed Q5_K/IQ4_XS plain-layout path is not included because
the current model inventory has no matching model for model-level validation.

The SYCL Flash Attention tile fallback now has an 8:1 GQA dispatch for 512-wide
attention, matching Gemma 4 E2B global attention. This is used when the oneDNN
Flash Attention route is disabled or unavailable (`GGML_SYCL_FA_ONEDNN=0`). On
the B580, the isolated long-context operation (8 query heads, 1 KV head, width
512, KV length 49,152) measured 563.35 us/run with the new dispatch versus
1249.49 us/run with the prior dispatch. The short KV=512 operation was within
noise (42.22 vs 42.59 us/run). With oneDNN disabled, Q4 model decode measured
86.04 vs 80.81 tokens/s while Q8 measured 57.93 vs 60.96 tokens/s, so the
full-model result is mixed and does not show a consistent speedup. Both local
Gemma 4 E2B Q4 and Q8 GGUFs passed the earlier short smoke through the default
oneDNN route. With oneDNN disabled, a 32-token CLI generation segfaulted for
both models on both the new and old tile dispatch, so fallback CLI inference
remains unverified; the matching `llama-bench` A/B runs completed. The
49,152-token backend correctness fixture exceeds the existing 5e-4 tolerance
on both the old and new dispatch (errors 0.001095 and 0.001089 respectively);
the long case remains a performance fixture only, with the passing KV=512 case
covering correctness.

SYCL FWHT uses a wide work-group kernel for Hadamard widths 1024, 2048, 4096,
and 8192. `GGML_SYCL_DISABLE_FWHT_WIDE=1` disables these kernels for comparison.
The B580 OpenCL PTQ1_0 Bonsai 2 27B benchmark at p128/n64/r3 measured
215.93±0.85 prompt and 7.163±0.008 decode tokens/s with the wide kernels
disabled, versus 234.18±0.56 prompt and 7.304±0.011 decode tokens/s with them
enabled. This is about 8.4% faster prompt processing and 2.0% faster decode in
that run. Fixed-seed CLI output matched with the kernels enabled and disabled,
and the SYCL `MUL_MAT_HADAMARD` backend tests passed 27/27, including the
supported wide widths.

## Router and WebUI

Start the router with the models directory:

```bash
llama-server --models-dir /home/oddsoul/models --models-max 1 --port 8080
```

The model picker exposes Auto, Vulkan, OpenVINO, and SYCL. The selected value
is persisted locally and is applied to the next model load. Router requests
accept only `--device` (`Vulkan0`, `SYCL0`, or `OPENVINO0`) and
`--n-gpu-layers`; other extra arguments are rejected. OpenVINO GPU selection
is configured in the child process, so the router itself does not need a
global `GGML_OPENVINO_DEVICE` setting.

## Validation matrix

The table includes recorded checks against the current and earlier model
inventories. Availability was rechecked on 2026-09-29 in both
`/mnt/Data/Projects/Models` and `/mnt/Data/Models`; rows marked historical are
not currently available for repeat testing.

| Model | Availability | Vulkan | SYCL | OpenVINO GPU |
| --- | --- | --- | --- | --- |
| Llama 3.2 1B Q4_K_M | historical | pass | pass | pass |
| Qwen2.5 Coder 7B Q4_K_M | present | not measured here | not measured here | pass (stateful generation) |
| Qwen3.5 9B Q4_K_M | present | pass | pass | prompt warmup fails (`res=-3`) |
| Gemma 4 12B QAT UD-Q4_K_XL | historical | generated | generated | generated |
| Qwen 3.8 27B GSQ IQ3_XXS MTP | historical | generated | generated | not validated: initialization timeout |
| Qwen 3.8 27B UD-Q2_K_XL | historical | generated | generated | unsupported: GPU memory allocation |
| Ternary Bonsai 2 27B PQ2_0 | present | pass (native PQ2 MMQ; f16 fallback available) | generated (native MMVQ) | unsupported: GPU memory allocation |
| Ternary Bonsai 2 27B PTQ1_0 | present | pass (Vulkan, as recorded in the LocalDesign evaluation) | inference completed (native MMVQ; OpenCL B580 device trace confirmed) | not validated |
| Bonsai 27B PQ2_0 | historical | pass (native PQ2 MMQ; f16 fallback available) | generated (native MMVQ) | unsupported: GPU memory allocation |

The OpenVINO GPU plugin is available on the B580. On Qwen2.5 Coder 7B Q4_K_M
with stateful execution, the existing KV-state sequence-axis relayout measured
2777.52±269.83 prompt and 39.86±0.34 decode tokens/s when disabled, versus
2829.99±161.50 prompt and 43.63±0.28 decode tokens/s when enabled
(p128/n64/r3). The decode gain was about 9.5%; the prompt difference was within
the sample spread. Fixed-seed 32-token generation matched exactly. This relayout
is already present in the fork and is enabled for OpenVINO GPU; disable it with
`GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT=1` for comparison. The Qwen3.5 OpenVINO
GPU run still fails its prompt warmup, so this Qwen2.5 result does not resolve
the separate Qwen3.5 GatedDeltaNet limitation.

Gemma uses a reasoning-style response format, so a short generation may begin
with a thinking marker rather than the requested literal answer. Vulkan PQ2 uses
an integer-dot MMQ path when the device exposes the required extension, with the
GPU chunked-dequant + regular f16 matmul path retained as a safe fallback under
other shapes. Large weight tensors are split into block-aligned dispatches under
the B580 workgroup limit. SYCL uses a native PQ2_0 MMVQ path, while OpenVINO
still rejects these 27B files because the GPU allocation exceeds available
memory. Runtime logs from these checks are kept in `artifacts/runtime/` in the
development worktree; `ptq1_0-sycl-device-openat.log` records the PTQ1_0 run's
SYCL/OpenCL driver and B580 render-node opens.
