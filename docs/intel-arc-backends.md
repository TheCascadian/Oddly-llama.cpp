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

The local `/mnt/Data/Models` inventory was used for runtime checks:

| Model | Vulkan | SYCL | OpenVINO GPU |
| --- | --- | --- | --- |
| Llama 3.2 1B Q4_K_M | pass | pass | pass |
| Gemma 4 12B QAT UD-Q4_K_XL | generated | generated | generated |
| Qwen 3.8 27B GSQ IQ3_XXS MTP | generated | generated | not validated: initialization timeout |
| Qwen 3.8 27B UD-Q2_K_XL | generated | generated | unsupported: GPU memory allocation |
| Ternary Bonsai 2 27B PQ2_0 | pass (native PQ2 MMQ; f16 fallback available) | generated (native MMVQ) | unsupported: GPU memory allocation |
| Ternary Bonsai 2 27B PTQ1_0 | pass (Vulkan, as recorded in the LocalDesign evaluation) | inference completed (native MMVQ; OpenCL B580 device trace confirmed) | not validated |
| Bonsai 27B PQ2_0 | pass (native PQ2 MMQ; f16 fallback available) | generated (native MMVQ) | unsupported: GPU memory allocation |

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
