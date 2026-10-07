#!/usr/bin/env python3
"""Writes the big context-ceiling plans, one per model: ctxplan.py [model ...]  (no argument: every model, smallest first).
Run one with: ctxprobe.py run full-<model>   Watch it with watch.py (it follows the newest run).
Variant groups: kv (every KV type pair at -ub 512), ub (-ub sweep on q8/q8 and q4/q4), vec (vector kernel forced with GGML_CUDA_FATTN_VEC_ALL),
ngram (n-gram speculation types and sizes on q8/q8). Every row also does the copy test (speed, fidelity, draft acceptance).
Context lists stop at the model's trained context length (gguf), except where the plan says otherwise."""
import sys
from paths import rp

O, L = "/home/wrekt/odysseus-local/models/", "/home/wrekt/.lmstudio/models/"
# name: (gguf, base args, native ctx, extra ladder past 131072)
MODELS = {
    "deepseek1.3b": (O + "deepseek-coder-1.3b-base-Q8_0.gguf", "-ngl 99", 16384, []),
    "minicpm2b":    (O + "MiniCPM5-2B-Q4_K_M.gguf", "-ngl 99", 131072, []),
    "gemma4-e2b":   (O + "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf", "-ngl 99", 131072, []),
    "qwen2.5-3b":   (L + "Crimsonwasp/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf", "-ngl 99", 32768, []),
    "qwen35-4b":    (O + "Qwen3.5-4B-Q4_K_M.gguf", "-ngl 99", 262144, [163840, 196608, 262144]),
    "r1-7b":        (L + "mradermacher/DeepSeek-R1-Distill-Qwen-7B-Uncensored-i1-GGUF/DeepSeek-R1-Distill-Qwen-7B-Uncensored.i1-Q4_K_S.gguf", "-ngl 99", 131072, []),
}
LADDER = [8192, 16384, 32768, 49152, 65536, 81920, 98304, 114688, 131072]
KV = [("f16", "f16"), ("q8_0", "q8_0"), ("q5_1", "q5_1"), ("q5_0", "q5_0"), ("q4_1", "q4_1"), ("q8_0", "q4_0"), ("q4_0", "q8_0"), ("q4_0", "q4_0")]
NGRAM = [("simple default n12 m48", "--spec-type ngram-simple"),
         ("simple n4 m16", "--spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 16"),
         ("simple n6 m24", "--spec-type ngram-simple --spec-ngram-simple-size-n 6 --spec-ngram-simple-size-m 24"),
         ("simple n8 m32", "--spec-type ngram-simple --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 32"),
         ("simple n16 m64", "--spec-type ngram-simple --spec-ngram-simple-size-n 16 --spec-ngram-simple-size-m 64"),
         ("map-k default", "--spec-type ngram-map-k"),
         ("map-k n6 m24", "--spec-type ngram-map-k --spec-ngram-map-k-size-n 6 --spec-ngram-map-k-size-m 24"),
         ("map-k4v default", "--spec-type ngram-map-k4v"),
         ("map-k4v n6 m24", "--spec-type ngram-map-k4v --spec-ngram-map-k4v-size-n 6 --spec-ngram-map-k4v-size-m 24"),
         ("mod default", "--spec-type ngram-mod"),
         ("mod match12 n24-32", "--spec-type ngram-mod --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 24 --spec-ngram-mod-n-max 32"),
         ("cache default", "--spec-type ngram-cache")]
BUILD = "build-allq"   # has every flash-attention KV pair and the GGML_CUDA_FATTN_VEC_ALL switch


def plan(name):
    gguf, base, native, extra = MODELS[name]
    full = [c for c in LADDER if c <= native]
    if native > 131072:
        full += extra
    short = [c for c in (16384, 65536, 114688, 131072) if c <= native] or [native]
    short = sorted(set(short + ([native] if native <= 131072 and native not in short else [])))
    v = []
    kvs = lambda k, vv: f"-ctk {k} -ctv {vv}"
    for k, vv in KV:
        v.append((f"kv {k}/{vv} ub512", f"-m {gguf} {base} {kvs(k, vv)}", full))
    for ub in (256, 128, 64, 32):
        for k, vv in (("q8_0", "q8_0"), ("q4_0", "q4_0")):
            v.append((f"ub {k}/{vv} ub{ub}", f"-m {gguf} {base} {kvs(k, vv)} -ub {ub}", full))
    for k, vv, ub in (("q8_0", "q8_0", 512), ("q8_0", "q4_0", 512), ("q4_0", "q4_0", 512), ("q4_0", "q4_0", 128), ("q4_0", "q4_0", 64)):
        v.append((f"vec {k}/{vv} ub{ub}", f"-m {gguf} {base} {kvs(k, vv)} -ub {ub} ENV:GGML_CUDA_FATTN_VEC_ALL=1", full))
    for label, flags in NGRAM:
        v.append((f"ngram {label} q8/q8", f"-m {gguf} {base} {kvs('q8_0', 'q8_0')} {flags}", short))
    if name == "qwythos-9b":
        v.insert(0, ("kv f16/f16 past 24K", f"-m {gguf} {base} {kvs('f16', 'f16')}", [24576, 28672, 32768, 40960, 49152]))
    lines = [f"title: {name} context ceiling: KV types, -ub, forced vector kernel, n-gram speculation, copy and key recall at every step",
             "args: -fa on --parallel 1 -cram 0 --no-mmproj"]
    lines += [f"{lab} | {BUILD} | {a} | {','.join(map(str, c))}" for lab, a, c in v]
    open(rp(f"ctx-full-{name}.plan"), "w").write("\n".join(lines) + "\n")
    return len(v)


SLIM_LADDER = {"qwen35-4b": [32768, 65536, 98304, 114688, 131072, 163840, 196608, 262144],
               "r1-7b": [16384, 32768, 49152, 65536, 81920, 98304, 114688, 131072]}


def slim(name):
    """The narrowed plan for models where VRAM decides the ceiling: KV pairs at -ub 512 and 128, forced vector kernel on two pairs, key recall only."""
    gguf, base, _, _ = MODELS[name]
    ctxs = ",".join(map(str, SLIM_LADDER[name]))
    v = [(f"{k}/{vv} ub{ub}", "") for ub in (512, 128) for k, vv in (("f16", "f16"), ("q8_0", "q8_0"), ("q8_0", "q4_0"), ("q4_0", "q4_0"))]
    v += [(f"vec {k}/{vv} ub{ub}", "ENV:GGML_CUDA_FATTN_VEC_ALL=1") for k, vv, ub in (("q8_0", "q4_0", 512), ("q4_0", "q4_0", 128))]
    lines = [f"title: {name} context ceiling: KV pairs, -ub 512 and 128, forced vector kernel, key recall at every step",
             "args: -fa on --parallel 1 -cram 0 --no-mmproj"]
    for lab, env in v:
        pair = lab.replace("vec ", "").split()[0].split("/")
        ub = lab.split("ub")[-1]
        lines.append(f"{lab} | {BUILD} | -m {gguf} {base} -ctk {pair[0]} -ctv {pair[1]} -ub {ub} NOCOPY {env} | {ctxs}")
    open(rp(f"ctx-slim-{name}.plan"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__" and sys.argv[1:2] == ["slim"]:
    for n in SLIM_LADDER:
        slim(n)
        print("slim", n)
elif __name__ == "__main__":
    for n in (sys.argv[1:] or MODELS):
        print(n, plan(n), "variants")
