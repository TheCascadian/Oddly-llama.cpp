#!/usr/bin/env python3
"""Refresh the data block inside UNIFIED.html.
embed-unified.py    re-reads the docs and every results csv/plan, rewrites the block between DATA:BEGIN and DATA:END.
The page renders the file browser, Notepad and the MiniCPM lab from that block, so rerun it after new results land."""
import csv, glob, io, json, os, re, statistics, time
from collections import defaultdict
from paths import ROOT
import allchart

LT = os.path.join(ROOT, "local-tune")
HTML = os.path.join(LT, "UNIFIED.html")
CAP = 160_000
FIELDS = "label,ctx,status,fail,vram_base,vram_peak,gpu_model,gpu_kv,gpu_compute,cpu_model,cpu_kv,tg0,tg_ctx,pp_ctx,keys,prompt_tokens,secs,copy_tg,copy_q,spec_acc".split(",")

VARIANT_NOTE = {
    "baseline-Q4_K_M": "stock Q4_K_M recipe, no imatrix",
    "H5-imatrix-q4km": "imatrix only, same size",
    "H1b1-ffn_down-q6to-q5": "ffn_down q6_K layers to q5_K",
    "H1b2-ffn_down-q6to-q4": "ffn_down q6_K layers to q4_K",
    "H3c-output-q4K": "output.weight to q4_K",
    "H6a-token_embd-q5K": "token_embd to q5_K",
    "H7a-attn_v-q6to-q5": "attn_v q6_K layers to q5_K",
    "H7b-attn_v-q6to-q4": "attn_v q6_K layers to q4_K",
    "H10-stack-H1b2-H3c-H7b": "stack of H1b2 + H3c + H7b",
    "H11-imx-iq3s-ffn": "imatrix + ffn gate/up/down as IQ3_S",
}


def text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        s = f.read()
    return s if len(s) <= CAP else s[:CAP] + "\n... truncated at %d bytes ...\n" % CAP


def files():
    out = {}
    for n in ("UNIFIED.md", "HISTORY.md", "README.md"):
        out[n] = text(os.path.join(LT, n))
    for ext in ("csv", "plan"):
        for p in sorted(glob.glob(os.path.join(LT, "results", "**", "*." + ext), recursive=True)):
            out[os.path.relpath(p, LT)] = text(p)
    return out


def variants():
    sp = os.path.join(LT, "results/throughput/compare-mc2b-variants.csv")
    pp = os.path.join(LT, "results/quality/ppl-mc2b-variants.csv")
    if not os.path.exists(sp):
        return []
    ppl = {}
    if os.path.exists(pp):
        for r in csv.reader(open(pp)):
            if len(r) >= 3:
                ppl[r[0]] = (float(r[1]), float(r[2]))
    seen, out = {}, []
    for r in csv.reader(open(sp)):
        tag, size, n_prompt, n_gen = r[0], int(r[8]), int(r[-8]), int(r[-7])
        d = seen.setdefault(tag, {"tag": tag, "size": size, "note": VARIANT_NOTE.get(tag, "")})
        d["pp" if n_prompt else "tg"] = [float(r[-2]), float(r[-1])]
    for d in seen.values():
        d["ppl"] = ppl.get(d["tag"])
        out.append(d)
    return out


def specs():
    out = {}
    for p in sorted(glob.glob(os.path.join(LT, "results/speculative/spec-ngram-*.csv"))):
        name = os.path.basename(p)[len("spec-ngram-"):-4]
        tg, dn, da, md5 = defaultdict(list), defaultdict(int), defaultdict(int), defaultdict(set)
        for r in csv.DictReader(open(p)):
            k = (r["variant"], r["prompt"])
            tg[k].append(float(r["tg_tps"])); dn[k] += int(r["draft_n"]); da[k] += int(r["draft_accepted"]); md5[k].add(r["text_md5"])
        out[name] = [[v, pr, round(statistics.mean(x), 1), dn[(v, pr)], da[(v, pr)], len(md5[(v, pr)])] for (v, pr), x in tg.items()]
    return out


def ctxs():
    out = {}
    for p in sorted(glob.glob(os.path.join(LT, "results/context/ctx-*.csv"))):
        name = os.path.basename(p)[4:-4]
        rows = []
        for r in csv.DictReader(open(p), fieldnames=FIELDS):
            f = lambda k: float(r[k]) if r.get(k) not in (None, "") else None
            rows.append([r["label"], int(r["ctx"]), r["status"], f("vram_peak"), f("gpu_kv"), f("gpu_compute"), f("tg_ctx"), f("pp_ctx"), r.get("keys", "")])
        out[name] = rows
    return out


def main():
    data = {"built": time.strftime("%Y-%m-%d %H:%M"), "files": files(), "var": variants(), "spec": specs(), "ctx": ctxs(), "all": allchart.collect()}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--")
    html = open(HTML, encoding="utf-8").read()
    new = '<!--DATA:BEGIN--><script id="data" type="application/json">' + blob + "</script><!--DATA:END-->"
    html, n = re.subn(r"<!--DATA:BEGIN-->.*?<!--DATA:END-->", lambda m: new, html, flags=re.S)
    if n != 1:
        raise SystemExit("markers not found")
    open(HTML, "w", encoding="utf-8").write(html)
    print(f"{len(data['files'])} files, {len(data['var'])} variants, {len(blob)//1024} KiB embedded")


if __name__ == "__main__":
    main()
