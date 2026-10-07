#!/usr/bin/env bash
# Build the CUPTI tracer. usage: local-tune/prof/build.sh
set -euo pipefail
cd "$(dirname "$0")"
C=/opt/cuda/targets/x86_64-linux
g++ -O2 -fPIC -shared -o libcupti_trace.so cupti_trace.cpp -I"$C/include" -L"$C/lib" -lcupti -Wl,-rpath,"$C/lib"
echo "built: $PWD/libcupti_trace.so"
