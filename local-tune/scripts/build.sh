#!/usr/bin/env bash
# Rebuild Wrekt-llama.cpp tuned for: Ryzen 5 7600X + GTX 1660 Ti (sm_75, no tensor cores), CUDA from /opt/cuda.
# usage: local-tune/scripts/build.sh [build-dir]   (default: build)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=/opt/cuda/bin:$PATH
B=${1:-build}
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  $(command -v ccache >/dev/null && echo -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
cmake --build "$B" -j"$(nproc)" --target llama-cli llama-server llama-bench llama-perplexity
echo "built: $B/bin"
