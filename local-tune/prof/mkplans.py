#!/usr/bin/env python3
"""Write the compare.py plans for the decode profile: one traced and one untraced run per model and context depth.
usage: mkplans.py <trace-dir>   then: python3 local-tune/compare.py run prof-7b (prof-9b, prof-4b)"""
import glob, os, sys

T = os.path.abspath(sys.argv[1])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INJ = os.path.join(ROOT, "prof", "libcupti_trace.so")
H = os.path.expanduser("~")
M7 = glob.glob(H + "/.lmstudio/models/mradermacher/DeepSeek-R1-Distill-Qwen-7B-Uncensored-i1-GGUF/*Q4_K_S.gguf")[0]
M9 = H + "/.lmstudio/models/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf"
M4 = H + "/odysseus-local/models/Qwen3.5-4B-Q4_K_M.gguf"
D = [0, 4096, 8192, 12288, 15360]


def mk(n, title, m, args, depths):
    L = [f"title: {title}", f"args: -m {m} {args} -t 6 -fa 1 -p 0 -n 32 -r 2"]
    L += [f"d{d} | build-live | CUDA_INJECTION64_PATH={INJ} CUPTI_TRACE_OUT={T}/{n}-d{d}.tsv | -d {d}" for d in depths]
    L += [f"d{d}-plain | build-live | | -d {d}" for d in depths]
    open(os.path.join(ROOT, "results", f"compare-prof-{n}.plan"), "w").write("\n".join(L) + "\n")


mk("7b", "Profile: 7B decode by context depth, CUPTI trace then untraced", M7, "-ngl 99 -ctk q8_0 -ctv q4_0", D)
mk("9b", "Profile: 9B, 25 layers on GPU, decode by context depth, CUPTI trace then untraced", M9, "-ngl 25 -ctk q8_0 -ctv q8_0", D)
mk("4b", "Profile: 4B decode by context depth, CUPTI trace then untraced", M4, "-ngl 99 -ctk q8_0 -ctv q8_0", [0, 8192])
