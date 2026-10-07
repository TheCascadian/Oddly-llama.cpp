#!/usr/bin/env python3
"""Context ceiling probe with a live view in the style of watch.py.
ctxprobe.py run <name>    runs results/ctx-<name>.plan and writes ctx-<name>.csv/.log (server logs in results/ctx-<name>/)
ctxprobe.py <name>        live view of that run
Put LOADONLY in the args to skip the requests: the row then only shows the buffers and the VRAM after loading.
Plan lines: "title: text", "args: common llama-server args", then one variant per line: "label | build-dir | server args | ctx list".
Per context size the server starts with -c N, answers a short prompt (decode speed at 0K), then reads a prompt that fills the context
with corpus text and three hidden codes at 10%, 50% and 90%, and must repeat them (decode speed and retrieval at the ceiling).
The ctx list is tried in ascending order and a variant stops at its first failure."""
import csv, difflib, json, os, random, re, signal, subprocess, sys, threading, time, urllib.request
import watch
from watch import G, Y, RED, C, D, B, X
from paths import rp, ROOT, SCRIPTS

PORT = 8811
FIELDS = "label,ctx,status,fail,vram_base,vram_peak,gpu_model,gpu_kv,gpu_compute,cpu_model,cpu_kv,tg0,tg_ctx,pp_ctx,keys,prompt_tokens,secs,copy_tg,copy_q,spec_acc".split(",")
KEYS = ("ALPHA", "BETA", "GAMMA")


def plan(name):
    title, args, variants = name, "", []
    for ln in watch.read(f"ctx-{name}.plan"):
        if ln.startswith("title:"):
            title = ln[6:].strip()
        elif ln.startswith("args:"):
            args = ln[5:].strip()
        elif "|" in ln:
            variants.append([p.strip() for p in ln.split("|")] + [""] * 4)
    return title, args, variants


def post(path, body, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def used_mib():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(o.split()[0])


def sizes(log):
    """MiB per buffer kind from the -lv 4 load log."""
    s = dict(gpu_model=0, gpu_kv=0, gpu_compute=0, cpu_model=0, cpu_kv=0)
    for m in re.finditer(r"(\w+) (model buffer|KV buffer|RS buffer|compute buffer) size =\s+([\d.]+) MiB", log):
        dev, kind, mib = m[1], m[2], float(m[3])
        gpu = dev.startswith("CUDA") and "Host" not in dev
        key = "model" if kind == "model buffer" else "compute" if kind == "compute buffer" else "kv"
        if gpu:
            s["gpu_" + key] += mib
        elif key != "compute" and dev.startswith("CPU"):
            s["cpu_" + key] += mib
    return {k: round(v) for k, v in s.items()}


def build_prompt(ctx, corpus, tok_per_char, codes):
    n_chars = int((ctx - 320) / tok_per_char)
    text = (corpus * (n_chars // len(corpus) + 1))[:n_chars]
    lines = text.split("\n")
    for key, frac in zip(KEYS, (0.1, 0.5, 0.9)):
        lines.insert(int(len(lines) * frac), f"\nNote: the secret code {key} is {codes[key]}.\n")
    return "\n".join(lines) + "\n\n---\nQuestion: the text above hides three secret codes named ALPHA, BETA and GAMMA. Repeat each one.\nALPHA="


COPY_LINES = 6


def copy_prompt(prompt):
    """Same prefix as the retrieval prompt (so the slot cache is reused), a different question: copy the first lines word for word.
    Returns (prompt, expected text). Copying is where n-gram speculation drafts well, and the match against the source is a quality check."""
    body = prompt.rsplit("\n\n---\nQuestion:", 1)[0]
    first = [l for l in body.split("\n") if l.strip()][:COPY_LINES]
    return body + f"\n\n---\nTask: copy the first {COPY_LINES} lines of the text above exactly, word for word, and nothing else.\nOutput:\n", "\n".join(first)

def probe(label, build, args, ctx, say, writer):
    load_only = "LOADONLY" in args.split()
    toks = [a for a in args.split() if a != "LOADONLY"]
    envx = dict(a[4:].split("=", 1) for a in toks if a.startswith("ENV:"))   # ENV:NAME=VALUE in a variant sets an environment variable for the server
    no_copy = "NOCOPY" in toks
    args = " ".join(a for a in toks if not a.startswith("ENV:") and a != "NOCOPY")
    row = dict.fromkeys(FIELDS, ""); row.update(label=label, ctx=ctx, status="fail")
    t0 = time.time()
    d = rp(f"ctx-{CUR}")
    os.makedirs(d, exist_ok=True)
    logp = os.path.join(d, f"{re.sub(r'\W+', '_', label)}-{ctx}.log")
    env = dict(os.environ, PATH="/opt/cuda/bin:" + os.environ["PATH"], **envx)
    row["vram_base"] = base = used_mib()
    srv = subprocess.Popen([f"{ROOT}/{build}/bin/llama-server", "--host", "127.0.0.1", "--port", str(PORT), "-lv", "4", "-c", str(ctx)] + args.split(),
                           env=env, cwd=ROOT, stdout=open(logp, "w"), stderr=subprocess.STDOUT)
    stop, peak = threading.Event(), [base]
    def sample():
        while not stop.wait(1):
            peak[0] = max(peak[0], used_mib())
    threading.Thread(target=sample, daemon=True).start()
    def fail_kind():
        log = open(logp, errors="replace").read()
        if re.search(r"out of memory|failed to allocate|cudaMalloc failed|unable to allocate", log, re.I): return "allocation"
        if re.search(r"CUDA error", log): return "cuda error"
        return "died" if srv.poll() is not None else "timeout"
    try:
        for _ in range(300):
            try:
                if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok": break
            except Exception:
                pass
            if srv.poll() is not None: break
            time.sleep(1)
        else:
            row["fail"] = "timeout"; return row
        if srv.poll() is not None:
            row["fail"] = fail_kind(); return row
        row.update(sizes(open(logp, errors="replace").read()))
        if load_only:
            time.sleep(4); row["status"] = "ok"; return row
        say(f"  loaded gpu_model={row['gpu_model']} kv={row['gpu_kv']} compute={row['gpu_compute']} cpu_model={row['cpu_model']} cpu_kv={row['cpu_kv']}")
        r = post("/completion", {"prompt": "Count upward: 1, 2, 3, 4, 5,", "n_predict": 64, "temperature": 0, "ignore_eos": True, "cache_prompt": False}, 300)
        row["tg0"] = round(r["timings"]["predicted_per_second"], 1)
        corpus = open(rp("ppl-corpus.txt"), errors="replace").read()
        n = len(post("/tokenize", {"content": corpus[:60000]}, 120)["tokens"])
        tpc = n / 60000
        rnd = random.Random(ctx)
        codes = {k: str(rnd.randint(10000, 99999)) for k in KEYS}
        for _ in range(5):
            prompt = build_prompt(ctx, corpus, tpc, codes)
            n = len(post("/tokenize", {"content": prompt}, 300)["tokens"])
            if ctx - 700 <= n <= ctx - 200:
                break
            tpc *= n / (ctx - 450)
        say(f"  prompt sent, {n} tokens")
        r = post("/completion", {"prompt": prompt, "n_predict": 40, "temperature": 0, "cache_prompt": True}, 3600)
        t = r["timings"]
        out = r["content"]
        found = [codes[k] in out for k in KEYS]
        row.update(tg_ctx=round(t["predicted_per_second"], 1), pp_ctx=round(t["prompt_per_second"], 1), prompt_tokens=r.get("tokens_evaluated", t.get("prompt_n", "")),
                   keys=f"{sum(found)}/3", status="ok")
        say(f"  answer {out[:80]!r} codes {codes}")
        if not no_copy:
            cp, want = copy_prompt(prompt)
            r2 = post("/completion", {"prompt": cp, "n_predict": 256, "temperature": 0, "cache_prompt": True}, 3600)
            t2, got = r2["timings"], r2["content"].strip()
            row["copy_tg"] = round(t2["predicted_per_second"], 1)
            sm = difflib.SequenceMatcher(None, got[:len(want) * 2], want, autojunk=False)
            row["copy_q"] = round(sum(b.size for b in sm.get_matching_blocks()) / max(len(want), 1), 3)   # share of the source text found in order in the answer
            if t2.get("draft_n"):
                row["spec_acc"] = round(t2["draft_n_accepted"] / t2["draft_n"], 2)
            say(f"  copy {got[:70]!r}")
            say(f"  copy tg={row['copy_tg']} quality={row['copy_q']} acc={row['spec_acc'] or '-'} prompt_n={t2.get('prompt_n')}")
    except Exception as e:
        row["fail"] = fail_kind() if srv.poll() is not None or "refused" in str(e) or "closed" in str(e) else "timeout"
        say(f"  exception {type(e).__name__} {str(e)[:100]}")
    finally:
        stop.set()
        row["vram_peak"] = peak[0]
        if srv.poll() is None:
            srv.send_signal(signal.SIGINT)
            try: srv.wait(30)
            except subprocess.TimeoutExpired: srv.kill(); srv.wait()
        row["secs"] = round(time.time() - t0)
        time.sleep(3)
    return row


def run(name):
    global CUR
    CUR = name
    _, args, variants = plan(name)
    out = open(rp(f"ctx-{name}.csv"), "w")
    w = csv.DictWriter(out, FIELDS); log = open(rp(f"ctx-{name}.log"), "w")
    def say(msg):
        log.write(f"{time.strftime('%T')} {msg}\n"); log.flush()
    t0 = time.time()
    for label, build, extra, ctxs, *_ in variants:
        say(f"START {label}"); t1 = time.time()
        for ctx in [int(c) for c in ctxs.split(",")]:
            say(f"  CTX {ctx}")
            row = probe(label, build, f"{args} {extra}", ctx, say, w)
            w.writerow(row); out.flush()
            say(f"  RESULT {ctx} {row['status']} {row['fail']} peak={row['vram_peak']}")
            say(f"  TEST tg@{ctx} took={row['secs']}s")
            if row["status"] != "ok":
                break
        say(f"DONE {label} took={time.time() - t1:.0f}s")
    say(f"ALLDONE total={time.time() - t0:.0f}s")


def predict(pts, ctx):
    """Seconds for context ctx from finished (ctx, secs) points: next-difference extrapolation (quadratic through the last
    three points, linear through two, proportional through one). Returns None with no points."""
    pts = sorted(pts)[-3:]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0][1] * ctx / pts[0][0]
    if len(pts) == 2:
        (x0, y0), (x1, y1) = pts
        return y1 + (y1 - y0) / (x1 - x0) * (ctx - x1)
    (x0, y0), (x1, y1), (x2, y2) = pts
    d1, d2 = (y1 - y0) / (x1 - x0), (y2 - y1) / (x2 - x1)
    curv = (d2 - d1) / (x2 - x0)   # second divided difference
    return y2 + d2 * (ctx - x2) + curv * (ctx - x2) * (ctx - x1)


def render(name):
    title, args, variants = plan(name)
    log = watch.read(f"ctx-{name}.log")
    rows = list(csv.DictReader(watch.read(f"ctx-{name}.csv"), FIELDS))
    cur, curctx, done = None, None, set()
    for ln in log:
        m = re.match(r"\S+ START (.+)", ln)
        if m: cur = m[1]
        m = re.match(r"\S+ +CTX (\d+)", ln)
        if m: curctx = int(m[1])
        m = re.match(r"\S+ DONE (.+)", ln)
        if m: done.add(m[1])
    finished = any("ALLDONE" in l for l in log)
    L = [f"{B}  CONTEXT CEILING{X}  {D}{title}{X}", f"  {D}llama-server {args}{X}", ""]
    L += watch.gpu_panel() + [""]
    W = 10
    hdr = f"  {'Variant':<26}{'ctx':>7} {'result':<12}{'VRAM peak':>10}{'free':>6}{'gpu mdl':>8}{'gpu kv':>7}{'cmpute':>7}{'cpu MiB':>8}{'t/s 0K':>8}{'t/s ctx':>8}{'pp t/s':>7}{'codes':>6}{'cp t/s':>7}{'cp q':>6}{'acc':>5}{'time':>6}"
    L += [f"{B}{hdr}{X}", " " + "─" * (len(hdr) - 1)]
    for v in variants:
        mine = [r for r in rows if r["label"] == v[0]]
        for r in mine:
            ok = r["status"] == "ok"
            res = f"{G}ok{X}" + " " * 10 if ok else f"{RED}{r['fail']:<12}{X}"
            peak = int(r["vram_peak"] or 0)
            L.append(f"  {v[0]:<26}{int(r['ctx']):>7} {res}{peak:>8} M{6144 - peak:>5}{r['gpu_model']:>8}{r['gpu_kv']:>7}{r['gpu_compute']:>7}{int(float(r['cpu_model'] or 0) + float(r['cpu_kv'] or 0)):>8}"
                     f"{r['tg0']:>8}{r['tg_ctx']:>8}{r['pp_ctx']:>7}{r['keys']:>6}{(r.get('copy_tg') or '-'):>7}{(r.get('copy_q') or '-'):>6}{(r.get('spec_acc') or '-'):>5}{r['secs']:>5}s")
        if v[0] == cur and v[0] not in done and not finished:
            start = max((i for i, l in enumerate(log) if re.search(rf" START {re.escape(v[0])}$", l)), default=0)
            pts = [(int(m[1]), float(m[2])) for l in log[start:] if (m := re.search(r"TEST tg@(\d+) took=(\d+)s", l))]
            tail = ""
            if curctx:
                try:
                    t0 = next(l[:8] for l in reversed(log) if re.search(rf" +CTX {curctx}$", l))
                    spent = (watch.secs(time.strftime("%H:%M:%S")) - watch.secs(t0)) % 86400
                except StopIteration:
                    spent = 0
                p = predict(pts, curctx)
                if p:
                    tail = f"  {D}{watch.dur(int(spent))} in, predicted {watch.dur(int(p))}, ends {watch.clock(max(p - spent, 0))}{X}"
            L.append(f"  {watch.ACT}▶ {v[0]:<24}{curctx or '':>7} running{X}{tail}")
            steps = [int(x) for x in v[3].split(",")] if len(v) > 3 and v[3] else []
            nxt = next((x for x in steps if x > (curctx or 0)), None)
            if nxt and pts:
                cur_p = predict(pts, curctx) if curctx and curctx not in dict(pts) else None
                n_p = predict(pts + ([(curctx, cur_p)] if cur_p else []), nxt)
                diffs = ", ".join(f"+{int(q[1] - p_[1])}s" for p_, q in zip(pts, pts[1:]))
                L.append(f"    {D}steps {', '.join(f'{int(s)}s' for _, s in pts)}  diffs {diffs}  next {nxt}: ~{watch.dur(int(n_p))}{X}")
    L += ["", f"  {D}free = 6144 - peak (peak includes the desktop and other processes: base is logged per row). codes = hidden codes repeated correctly. Ctrl+C closes this view; the run keeps going.{X}"]
    L += [""] + watch.render_timing(log)
    return "\n".join(L), finished


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    if a and a[0] == "run":
        run(a[1])
    else:
        try:
            while True:
                text, fin = render(a[0])
                sys.stdout.write(("" if "--once" in sys.argv else "\033[H\033[J") + text + "\n"); sys.stdout.flush()
                if fin or "--once" in sys.argv:
                    break
                time.sleep(2)
        except KeyboardInterrupt:
            pass
