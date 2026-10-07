#!/usr/bin/env python3
"""Per-token cost model from a cupti_trace.cpp trace.
analyze.py <trace.tsv> [--json out.json] [--seq]   prints the decode-token table; --seq dumps the event order of one token.
A token is the span between two logits downloads (device-to-host copy of n_vocab*4 bytes).
Only spans that use the matrix-vector kernel and no batch matmul kernel count as decode tokens."""
import json, re, subprocess, sys, collections

CBID_H = "/opt/cuda/targets/x86_64-linux/include/cupti_runtime_cbid.h"


def cbids():
    out = {}
    for m in re.finditer(r"CUPTI_RUNTIME_TRACE_CBID_(\w+?)(?:_v\d+)?\s*=\s*(\d+)", open(CBID_H).read()):
        out[int(m[2])] = m[1]
    return out


def demangle(names):
    p = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True)
    return dict(zip(names, p.stdout.splitlines()))


def short(n):
    """Kernel name with template arguments, without the parameter list."""
    n = re.sub(r"_INTERNAL_\w+?_cu_\w+::", "", n)
    depth, out = 0, ""
    for ch in n:
        if ch == "(" and depth == 0 and out and not out.endswith("<") and "<" not in out[-12:] and out.count("<") == out.count(">"):
            break
        out += ch
    return re.sub(r"^void ", "", out).strip()


def load(path):
    K, C, Y, M, R = [], [], [], [], []
    for ln in open(path, errors="replace"):
        f = ln.rstrip("\n").split("\t")
        if f[0] == "K":
            K.append((int(f[1]), int(f[2]), f[5], f[6], int(f[7]), int(f[8]), int(f[9]), int(f[10]), f[12], int(f[11])))
        elif f[0] == "C":
            C.append((int(f[1]), int(f[2]), int(f[5]), int(f[6])))
        elif f[0] == "Y":
            Y.append((int(f[1]), int(f[2]), int(f[5])))
        elif f[0] == "M":
            M.append((int(f[1]), int(f[3]), int(f[4]), int(f[5])))
        elif f[0] == "R":
            R.append((int(f[1]), int(f[2]), int(f[4])))
    for a in (K, C, Y, M, R):
        a.sort()
    return K, C, Y, M, R


def union(iv):
    """Total covered time of a sorted interval list."""
    tot, cur_s, cur_e = 0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            tot += (cur_e - cur_s) if cur_e is not None else 0
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + ((cur_e - cur_s) if cur_e is not None else 0)


def within(a, s, e, key=lambda x: x[0]):
    import bisect
    lo = bisect.bisect_left(a, s, key=key)
    hi = bisect.bisect_left(a, e, key=key)
    return a[lo:hi]


def analyze(path, skip=4):
    K, C, Y, M, R = load(path)
    names = demangle(sorted({k[8] for k in K}))
    cb = cbids()
    # logits download: the device-to-host size above 64 KiB that bounds the most decode spans
    d2h = collections.Counter(c[3] for c in C if c[2] == 2 and c[3] > 65536)
    if not d2h:
        sys.exit("no logits download found")
    toks, nlog = [], 0
    for size, _ in d2h.most_common(4):
        marks = [c[1] for c in C if c[2] == 2 and c[3] == size]
        cand = []
        for s, e in zip(marks, marks[1:]):
            ks = within(K, s, e)
            if not ks:
                continue
            nm = [names[k[8]] for k in ks]
            if any("mul_mat_vec" in n for n in nm) and not any(n.startswith("void mul_mat_q") for n in nm):
                cand.append((s, e))
        if len(cand) > len(toks):
            toks, nlog = cand, size
    # drop the first tokens of the file: warm-up and graph capture
    toks = toks[skip:]
    # drop spans much longer than the median: they hold a context refill between repetitions
    med = sorted(e - s for s, e in toks)[len(toks) // 2]
    toks = [(s, e) for s, e in toks if e - s < 1.5 * med]
    n = len(toks)
    agg = collections.defaultdict(lambda: [0, 0, set(), set(), 0, 0, 0])
    wall = busy = 0
    sync_n = sync_ns = 0
    cpy = collections.defaultdict(lambda: [0, 0, 0])
    api = collections.defaultdict(lambda: [0, 0])
    mem = collections.defaultdict(lambda: [0, 0])
    gaps = []
    for s, e in toks:
        ks = within(K, s, e)
        wall += e - s
        busy += union([(k[0], k[1]) for k in ks])
        for a, b in zip(ks, ks[1:]):
            gaps.append(b[0] - a[1])
        for k in ks:
            a = agg[short(names[k[8]])]
            a[0] += 1; a[1] += k[1] - k[0]; a[2].add(k[2]); a[3].add(k[3]); a[4] = max(a[4], k[4] + k[5]); a[5] = max(a[5], k[6])
            a[6] += max(0, k[0] - k[9]) if k[9] else 0
        for y in within(Y, s, e):
            sync_n += 1; sync_ns += y[1] - y[0]
        for c in within(C, s, e):
            x = cpy[c[2]]; x[0] += 1; x[1] += c[3]; x[2] += c[1] - c[0]
        for r in within(R, s, e):
            x = api[cb.get(r[2], str(r[2]))]; x[0] += 1; x[1] += r[1] - r[0]
        for m in within(M, s, e):
            x = mem[(m[1], m[2])]; x[0] += 1; x[1] += m[3]
    gaps.sort()
    res = {
        "trace": path, "tokens": n, "logits_bytes": nlog,
        "wall_ms": wall / n / 1e6, "gpu_busy_ms": busy / n / 1e6,
        "kernels_per_token": sum(a[0] for a in agg.values()) / n,
        "sync_per_token": sync_n / n, "sync_ms": sync_ns / n / 1e6,
        "gap_us_median": gaps[len(gaps) // 2] / 1e3 if gaps else 0,
        "gap_ms_total": sum(g for g in gaps if g > 0) / n / 1e6,
        "kernels": sorted(({"name": k, "calls": a[0] / n, "ms": a[1] / n / 1e6, "us_per_call": a[1] / a[0] / 1e3,
                            "grid": sorted(a[2])[-3:], "block": sorted(a[3])[-2:], "shmem": a[4], "local": a[5]}
                           for k, a in agg.items()), key=lambda x: -x["ms"]),
        "memcpy": {{1: "host_to_device", 2: "device_to_host", 8: "device_to_device"}.get(k, str(k)):
                   {"calls": v[0] / n, "bytes": v[1] / n, "ms": v[2] / n / 1e6} for k, v in cpy.items()},
        "api": {k: {"calls": v[0] / n, "ms": v[1] / n / 1e6} for k, v in sorted(api.items(), key=lambda x: -x[1][1])},
        "mem_events": {f"op{k[0]}_kind{k[1]}": {"calls": v[0] / n, "bytes": v[1] / n} for k, v in mem.items()},
    }
    return res, (K, C, Y, R, names, cb, toks)


def show(r):
    print(f"{r['trace']}\n  decode tokens {r['tokens']}   wall {r['wall_ms']:.2f} ms/token ({1000 / r['wall_ms']:.1f} t/s)   GPU busy {r['gpu_busy_ms']:.2f} ms ({100 * r['gpu_busy_ms'] / r['wall_ms']:.0f}%)")
    print(f"  kernels/token {r['kernels_per_token']:.0f}   stream syncs/token {r['sync_per_token']:.1f} ({r['sync_ms']:.2f} ms waited)   median gap between kernels {r['gap_us_median']:.1f} us, idle gaps {r['gap_ms_total']:.2f} ms/token")
    print(f"  {'kernel':<74}{'calls':>7}{'ms/tok':>8}{'%busy':>7}{'us/call':>9}  grid / block / shmem")
    tot = sum(k["ms"] for k in r["kernels"])
    for k in r["kernels"]:
        print(f"  {k['name'][:73]:<74}{k['calls']:>7.1f}{k['ms']:>8.3f}{100 * k['ms'] / tot:>7.1f}{k['us_per_call']:>9.1f}  {k['grid'][-1]} / {k['block'][-1]} / {k['shmem']}")
    for k, v in r["memcpy"].items():
        print(f"  memcpy {k}: {v['calls']:.1f}/token, {v['bytes'] / 1024:.1f} KiB, {v['ms']:.3f} ms")
    print("  runtime API (CPU side): " + ", ".join(f"{k} {v['calls']:.1f}x {v['ms']:.3f}ms" for k, v in list(r["api"].items())[:8]))
    if r["mem_events"]:
        print("  memory events/token: " + ", ".join(f"{k} {v['calls']:.2f}x" for k, v in r["mem_events"].items()))


def seq(ctx):
    """Event order of one decode token: kernel runs merged, with syncs and copies in place."""
    K, C, Y, R, names, cb, toks = ctx
    s, e = toks[len(toks) // 2]
    ev = [(k[0], "K", short(names[k[8]])[:60], k[1] - k[0]) for k in within(K, s, e)]
    ev += [(y[0], "SYNC", "", y[1] - y[0]) for y in within(Y, s, e)]
    ev += [(c[0], "COPY", f"kind{c[2]} {c[3]} B", c[1] - c[0]) for c in within(C, s, e)]
    ev += [(r[0], "API", cb.get(r[2], str(r[2])), r[1] - r[0]) for r in within(R, s, e) if cb.get(r[2], "") not in ("cudaGetLastError", "cudaGetDevice", "cudaStreamSynchronize", "cudaMemcpyAsync")]
    ev.sort()
    last, run = None, 0
    for t, kind, what, dur in ev:
        key = (kind, what)
        if key == last:
            run += 1
            continue
        if last and run:
            print(f"            ... x{run + 1}")
        print(f"  {(t - s) / 1e6:8.3f} ms  {kind:<5}{what}  {dur / 1e3:.1f} us")
        last, run = key, 0


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    res, ctx = analyze(a[0])
    if "--seq" in sys.argv:
        seq(ctx)
    else:
        show(res)
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
