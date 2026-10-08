#!/usr/bin/env python3
"""Every stat across every run, joined per gguf file. allchart.py prints a speed table.
Reads every results csv (llama-bench compare, ppl, ngram spec, ctx probe) plus the plan/log timings.
embed-unified.py calls collect() to feed the All Stats window in UNIFIED.html."""
import csv, glob, json, os, re, statistics as st
from collections import defaultdict
from paths import ROOT

LT = os.path.join(ROOT, "local-tune")
RES = os.path.join(LT, "results")
DONE = re.compile(r"^\d\d:\d\d:\d\d DONE (.+?) took=(\d+)s")
CTXF = "label,ctx,status,fail,vram_base,vram_peak,gpu_model,gpu_kv,gpu_compute,cpu_model,cpu_kv,tg0,tg_ctx,pp_ctx,keys,prompt_tokens,secs,copy_tg,copy_q,spec_acc".split(",")
M = defaultdict(lambda: defaultdict(list))


def stem(path):
    return os.path.basename(path)[:-5] if path.endswith(".gguf") else path


def plan(p):
    out, default = {}, None
    for ln in open(p, errors="replace").read().splitlines():
        m = re.search(r"-m\s+(\S+\.gguf)", ln)
        if ln.startswith(("title:", "args:", "prompts:")) or not ln.strip():
            default = default or (stem(m[1]) if m else None)
            continue
        out[ln.split("|")[0].strip()] = stem(m[1]) if m else default
    return out


def rows(p, **kw):
    try:
        return list(csv.reader(open(p, errors="replace")))
    except Exception:
        return []


def collect():
    M.clear()
    for p in glob.glob(RES + "/**/*.csv", recursive=True):
        b = os.path.basename(p)
        pl = p[:-4] + ".plan"
        labels = plan(pl) if os.path.exists(pl) else {}
        if b.startswith(("compare-", "base")):
            for r in rows(p):
                try:
                    if not r[6].endswith(".gguf"): continue
                    m = stem(r[6]); np_, ng, ts, sd = int(r[-8]), int(r[-7]), float(r[-2]), float(r[-1])
                except (ValueError, IndexError):
                    continue
                k = "pp" if np_ and not ng else "tg"
                M[m][k].append(ts); M[m][k + "cv"].append(sd / ts if ts else 0); M[m]["runs"].append(1)
        elif b.startswith("ppl-"):
            for r in rows(p):
                m = labels.get(r[0]) if r else None
                if m and len(r) >= 2:
                    try: M[m]["ppl"].append(float(r[1]))
                    except ValueError: pass
        elif b.startswith("ctx-"):
            for r in csv.DictReader(open(p), fieldnames=CTXF):
                m = labels.get(r["label"])
                if not m: continue
                ok = r["status"] == "ok"
                M[m]["ctxrun"].append(1 if ok else 0)
                if ok:
                    M[m]["ctxmax"].append(int(r["ctx"]))
                    k = re.match(r"(\d+)/(\d+)", r.get("keys") or "")
                    if k and int(k[2]): M[m]["recall"].append(int(k[1]) / int(k[2]))
        elif b.startswith("spec-ngram-"):
            name = re.sub(r"[^a-z0-9]", "", b[11:-4].lower())
            by = defaultdict(list)
            for r in csv.DictReader(open(p)):
                by[(r["variant"], r["prompt"])].append(float(r["tg_tps"]))
            tgt = None
            for lab, m in labels.items():
                tgt = m
            if not tgt:
                tgt = min((m for m in M if name and name in re.sub(r"[^a-z0-9]", "", m.lower())), key=len, default=None)
            if tgt:
                for (v, pr), x in by.items():
                    if v != "none" and (("none", pr) in by):
                        M[tgt]["spec"].append(st.mean(x) / st.mean(by[("none", pr)]))
    for pl in glob.glob(RES + "/**/*.plan", recursive=True):
        lg, labels = pl[:-5] + ".log", plan(pl)
        if os.path.exists(lg):
            for ln in open(lg, errors="replace"):
                d = DONE.match(ln)
                m = labels.get(d[1].strip()) if d else None
                if m: M[m]["secs"].append(int(d[2]))
    out = []
    for m, d in M.items():
        if not (d["tg"] or d["pp"]): continue
        f = lambda k, fn: round(fn(d[k]), 3) if d[k] else None
        out.append({"model": m, "runs": len(d["runs"]), "tg": f("tg", max), "tgmed": f("tg", st.median), "pp": f("pp", max),
                    "cv": f("tgcv", st.median), "ppl": f("ppl", min), "ctxmax": max(d["ctxmax"]) if d["ctxmax"] else None,
                    "ctxok": f("ctxrun", st.mean), "recall": f("recall", st.mean), "spec": f("spec", max), "secs": sum(d["secs"]) or None})
    return out


def main():
    for r in sorted(collect(), key=lambda r: -(r['tg'] or 0)):
        print(f"{r['model']:<42}{r['tg'] or 0:>8.1f} tg {r['pp'] or 0:>8.0f} pp")


if __name__ == "__main__":
    main()
