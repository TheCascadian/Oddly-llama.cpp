#!/usr/bin/env python3
"""End-to-end benchmark of every model the local gateway serves, in one view.
suite.py run [name] [--skip=model,model] [--only=model,model]
                      phase 1: llama-bench per generative model with its own gateway settings (via compare.py)
                      phase 2: each model as served, through the gateway: load time, VRAM, one stream, all slots, encoders
                      phase 3: job checks (jobs.py) for the jobs named on the model's models.conf line
suite.py [name]       live view (watch.py follows it too)
Models come from the running gateway (gateway.py, GET /conf), so there is one model list. GATEWAY_URL sets its address.
Files: results/compare-<name>.*, results/suite-<name>.csv/.log and suite-<name>.conf.json (the model list as it was run)."""
import csv, json, os, re, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import compare, jobs, watch
from watch import G, Y, RED, C, D, B, X
from paths import rp, ROOT, SCRIPTS

GW = os.environ.get("GATEWAY_URL", "http://localhost:8700")
CORES = len(set(re.findall(r"core id\s*:\s*(\d+)", open("/proc/cpuinfo").read()))) or os.cpu_count()
BENCH = f"-t {CORES} -fa 1 -d 0,4096,8192 -p 512 -n 64 -r 3"
ESSAY = "Write a detailed 400 word essay about how rivers shape the land."
CODE = "import math\n\ndef primes_up_to(n):\n    \"\"\"Return every prime number up to n.\"\"\"\n"
WORDS = "river stone apple engine cloud violin harbor copper meadow lantern tiger saddle".split()


def models(skip=(), only=(), saved=None):
    """The gateway's model list. saved = a run name: the list stored by that run, so old runs render as they were."""
    f = rp(f"suite-{saved}.conf.json")
    conf = json.load(open(f)) if saved and os.path.exists(f) else json.load(urllib.request.urlopen(GW + "/conf", timeout=5))
    out = []
    for m in conf:
        if m["name"] in skip or (only and m["name"] not in only):
            continue
        a = m["args"].split()
        val = lambda f, d: next((a[i + 1] for i, x in enumerate(a[:-1]) if x == f), d)
        spec = val("--spec-type", "")
        out.append(dict(m, ngl=val("-ngl", "99"), ctk=val("-ctk", "f16"), ctv=val("-ctv", "f16"), spec=spec, drafted=spec.startswith("draft"),
                        gb=os.path.getsize(m["path"]) / 2**30 if os.path.exists(m["path"]) else 0))
    return out


def post(path, body, timeout=1800):
    req = urllib.request.Request(GW + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def vram():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])


def unload_all():
    for n in json.load(urllib.request.urlopen(GW + "/status"))["loaded"]:
        post(f"/unload/{n}", {})
    time.sleep(2)


def gen(m, max_tokens=200):
    """One generation as served; returns (tokens written, tokens/s while writing, ms until the first token)."""
    if m["type"] == "completion":
        j = post("/v1/completions", dict(model=m["name"], prompt=CODE, max_tokens=max_tokens, temperature=0))
    else:
        j = post("/v1/chat/completions", dict(model=m["name"], messages=[{"role": "user", "content": ESSAY}], max_tokens=max_tokens, temperature=0.3))
    t = j.get("timings", {})
    return t.get("predicted_n", 0), t.get("predicted_per_second", 0), t.get("prompt_ms", 0)


def bench_set(ms):
    """Generative models for llama-bench. It cannot run a drafter, so a draft variant of a gguf already listed is left out."""
    return [m for m in ms if m["type"] in ("llm", "completion") and not m["drafted"]]


def run(name, skip=(), only=()):
    ms = models(skip, only)
    json.dump(models(), open(rp(f"suite-{name}.conf.json"), "w"), indent=1)
    gens = bench_set(ms)
    with open(rp(f"compare-{name}.plan"), "w") as f:
        f.write(f"title: every gateway model with its own settings\nargs: {BENCH}\n")
        f.writelines(f"{m['name']} | build-live | | -m {m['path']} -ngl {m['ngl']} -ctk {m['ctk']} -ctv {m['ctv']}\n" for m in gens)
    out = open(rp(f"suite-{name}.csv"), "w")
    log = open(rp(f"suite-{name}.log"), "w")
    def say(msg):
        log.write(f"{time.strftime('%T')} {msg}\n"); log.flush()
    def put(model, metric, value):
        out.write(f"{model},{metric},{value}\n"); out.flush()
    t0 = time.time()
    say("SKIP " + ",".join(skip)); say("ONLY " + ",".join(only))
    say("PHASE bench"); unload_all()
    compare.run(name)
    say("PHASE served")
    for m in ms:
        n = m["name"]; say(f"START {n}"); t1 = time.time()
        try:
            unload_all(); base = vram()
            t = time.time(); post(f"/load/{n}", {}); put(n, "load_s", round(time.time() - t, 1))
            if m["type"] == "embedding":
                texts = [f"{WORDS[i % 12]} {WORDS[(i * 5 + 1) % 12]} number {i}: a short sentence used to measure embedding speed." for i in range(512)]
                t = time.time()
                dim = [len(post("/v1/embeddings", dict(model=n, input=texts[i:i + 32]))["data"][0]["embedding"]) for i in range(0, 512, 32)][0]
                put(n, "items_s", round(512 / (time.time() - t), 1)); put(n, "note", f"{dim} dimensions")
            elif m["type"] == "rerank":
                docs = [f"Note {i}: the {WORDS[i % 12]} is next to the {WORDS[(i * 7 + 3) % 12]}." for i in range(127)] + ["Restart the gateway with systemctl --user restart odysseus-gateway."]
                t = time.time(); res = post("/v1/rerank", dict(model=n, query="how do I restart the gateway", documents=docs))["results"]
                put(n, "items_s", round(128 / (time.time() - t), 1))
                put(n, "note", "right document ranked first" if max(res, key=lambda r: r["relevance_score"])["index"] == 127 else "WRONG document ranked first")
            else:
                gen(m, 16)   # warm up
                k, tps, first = gen(m); put(n, "one_tps", round(tps, 1)); put(n, "first_ms", round(first))
                if m["par"] > 1:
                    t = time.time()
                    with ThreadPoolExecutor(m["par"]) as ex:
                        res = list(ex.map(lambda _: gen(m), range(m["par"])))
                    put(n, "all_tps", round(sum(r[0] for r in res) / (time.time() - t), 1))
                if m["par"] > 1 and m["type"] == "llm":   # the swarm job: one forced label per item
                    body = lambda w: dict(model=n, messages=[{"role": "user", "content": f"Is this alive or not alive? {w}"}], max_tokens=8, temperature=0,
                                          grammar='root ::= "alive" | "not alive"', chat_template_kwargs={"enable_thinking": False})
                    t = time.time()
                    with ThreadPoolExecutor(m["par"]) as ex:
                        list(ex.map(lambda w: post("/v1/chat/completions", body(w)), WORDS * 8))
                    put(n, "items_s", round(96 / (time.time() - t), 1)); put(n, "note", "forced-label classify")
            put(n, "vram", vram() - base)
        except Exception as e:
            put(n, "note", "FAILED " + str(e)[:60].replace(",", ";")); say(f"ERROR {n} {e}")
        say(f"DONE {n} took={time.time() - t1:.0f}s")
    say("PHASE jobs")
    for m in ms:
        for j in m["jobs"]:
            say(f"JOB {m['name']} {j}")
            try:
                unload_all()
                score, note = jobs.JOBS[j](post, m, models())
            except Exception as e:
                score, note = "FAIL", str(e)[:80]
            put(m["name"], "job_" + j, f"{score},{note.replace(',', ';')}"); say(f"JOBDONE {m['name']} {j}")
    unload_all()
    say(f"ALLDONE total={time.time() - t0:.0f}s")


def render(name):
    log, blog = watch.read(f"suite-{name}.log"), watch.read(f"compare-{name}.log")
    skip = next((l.split("SKIP ")[1].split(",") for l in log if "SKIP " in l), [])
    only = [x for x in next((l.split("ONLY ")[1].split(",") for l in log if "ONLY " in l), []) if x]
    ms = models(skip, only, saved=name); gens = [m["name"] for m in bench_set(ms)]
    jlist = [(m["name"], j) for m in ms for j in m["jobs"]]
    jdone = {tuple(l.split()[2:4]) for l in log if " JOBDONE " in l}
    jcur = next((tuple(l.split()[2:4]) for l in reversed(log) if " JOB " in l), None)
    finished = any("ALLDONE" in l for l in log)
    phase = next((l.split("PHASE ")[1] for l in reversed(log) if "PHASE " in l), "starting")
    # phase 1: llama-bench rows written by compare.run
    res, tests = {}, [(t, d) for d in (0, 4096, 8192) for t in ("pp", "tg")]
    for r in csv.reader(watch.read(f"compare-{name}.csv")):
        try:
            res[(r[0], "pp" if int(r[-8]) else "tg", int(r[-6]))] = (float(r[-2]), float(r[-1]))
        except (ValueError, IndexError):
            continue
    cur, done, peak, took = None, set(), {}, {}
    for ln in blog:
        m = re.match(r"\S+ (START|DONE) (.+?)(?: took=(\d+)s)?$", ln)
        if m and m[1] == "START": cur = m[2]
        if m and m[1] == "DONE": done.add(m[2]); took[m[2]] = int(m[3] or 0)
        m = re.search(r"vram=(\d+) MiB", ln)
        if m and cur: peak[cur] = max(peak.get(cur, 0), int(m[1]))
    # phase 2: served rows
    sv, scur, sdone = {}, None, set()
    for r in csv.reader(watch.read(f"suite-{name}.csv")):
        if len(r) >= 3: sv[(r[0], r[1])] = ",".join(r[2:])
    for ln in log:
        m = re.match(r"\S+ (START|DONE) (\S+)", ln)
        if m and m[1] == "START": scur = m[2]
        if m and m[1] == "DONE": sdone.add(m[2]); took[m[2]] = took.get(m[2], 0) + int((re.search(r"took=(\d+)s", ln) or [0, 0])[1])
    steps, n = len(gens) + len(ms) + len(jlist), len(done) + len(sdone) + len(jdone)
    stamps = [l[:8] for l in log if re.match(r"\d\d:\d\d:\d\d ", l)]
    el = (watch.secs(stamps[-1] if finished else time.strftime("%T")) - watch.secs(stamps[0])) % 86400 if stamps else 0
    now = "finished" if finished else (f"speed test: {cur}" if phase == "bench" else f"job check: {' '.join(jcur)}" if phase == "jobs" and jcur else f"as served: {scur}") if (cur or scur) else "starting"
    L = [f"{B}  FULL MODEL SUITE{X}  {D}{name}  ·  {'chosen models' if only else 'every model'} in models.conf  ·  {watch.machine()}{X}", "",
         f"  Progress  {watch.bar(n / max(steps, 1), 40, G if finished else C)} {B}{n}/{steps}{X} steps   {G + '✔ ALL DONE' if finished else Y + '● ' + now}{X}   {D}elapsed{X} {B}{watch.dur(el)}{X}", ""]
    ram = next((m[1] for m in (re.search(r"ram_avail=(\d+)MB", l) for l in reversed(blog)) if m), "-")
    L += watch.gpu_panel(ram) + [""]
    L += [f"  {B}MODELS{X}  {D}as configured in the gateway{X}", "",
          f"{B}  {'Model':<24}{'Type':<11}{'File':>8}{'GPU layers':>12}{'KV cache':>12}{'Slots':>7}{'Context per slot':>18}{X}", " " + "─" * 93]
    for m in ms:
        kv = f"{m['ctk']}/{m['ctv']}" if m["type"] in ("llm", "completion") else "-"
        L.append(f"  {m['name']:<24}{m['type']:<11}{m['gb']:>6.2f}GB{m['ngl']:>12}{kv:>12}{m['par']:>7}{m['ctx']:>12}{' shared' if m['shared'] else '       '}  {D}{m['spec']}{X}")
    if skip != [""] and skip:
        L.append(f"  {D}left out of this run: {', '.join(skip)}{X}")
    L += ["", f"  {B}1. RAW SPEED{X}  {D}llama-bench, tokens/s, average of 3 runs ± spread. pp = reading a 512-token prompt, tg = writing. @Nk = N thousand tokens already in context{X}", ""]
    hdr = f"  {'Model':<24}|" + "".join(f"{t}@{d // 1024}k".rjust(14) for t, d in tests) + f" |{'Peak VRAM':>10}{'Time':>8}"
    L += [f"{B}{hdr}{X}", " " + "─" * (len(hdr) - 1)]
    for g in gens:
        cells = "".join(f"{G}{res[(g, t, d)][0]:8.1f}{X}{D} ±{res[(g, t, d)][1]:<4.1f}{X}" if (g, t, d) in res else f"{RED if g in done else D}{'FAIL' if g in done else '.':>14}{X}" for t, d in tests)
        mark = f"{Y}▶{X}" if g == cur and g not in done else " "
        L.append(f" {mark}{g:<24}|{cells} |{(str(peak[g]) + ' MiB') if g in peak else '-':>10}{watch.dur(took[g]) if g in done else '-':>8}")
    top = max([v[0] for (g, t, d), v in res.items() if t == "tg"] or [1])
    L += ["", f"  {D}Writing speed at empty context (solid) and at 8k context (light), same scale{X}"]
    for g in gens:
        a, b = res.get((g, "tg", 0)), res.get((g, "tg", 8192))
        if a:
            L.append(f"  {g:<24} {watch.bar(a[0] / top, 50, G)} {B}{a[0]:6.1f}{X}")
            L.append(f"  {'':<24} {watch.bar(b[0] / top, 50, C) if b else '':<50} {(format(b[0], '6.1f') + f'  {D}keeps {b[0] / a[0] * 100:.0f}%{X}') if b else ''}")
    L += ["", f"  {B}2. AS SERVED{X}  {D}through the gateway with the shipped settings. One stream = a single chat. All slots = every slot busy at once, total tokens/s{X}", ""]
    hdr = f"  {'Model':<24}|{'Load':>7}{'VRAM':>10}{'One stream':>12}{'First token':>13}{'All slots':>11}{'Gain':>7}{'Items/s':>9} | Notes"
    L += [f"{B}{hdr}{X}", " " + "─" * (len(hdr) + 22)]
    f = lambda m, k, unit="": (sv[(m, k)] + unit) if (m, k) in sv else ("." if m not in sdone else "-")
    for m in ms:
        nm = m["name"]; one, al = sv.get((nm, "one_tps")), sv.get((nm, "all_tps"))
        gain = f"{float(al) / float(one):.1f}x" if one and al and float(one) else ("." if nm not in sdone else "-")
        note = sv.get((nm, "note"), "")
        mark = f"{Y}▶{X}" if nm == scur and nm not in sdone and phase == "served" else " "
        L.append(f" {mark}{nm:<24}|{f(nm, 'load_s', ' s'):>7}{f(nm, 'vram', ' MiB'):>10}{G}{f(nm, 'one_tps'):>12}{X}{f(nm, 'first_ms', ' ms'):>13}{G}{f(nm, 'all_tps'):>11}{X}{gain:>7}{C}{f(nm, 'items_s'):>9}{X} | {RED if 'FAIL' in note or 'WRONG' in note else D}{note}{X}")
    if jlist:
        L += ["", f"  {B}3. JOB CHECKS{X}  {D}small fixed tests for the jobs named on each model's models.conf line (jobs.py){X}", "",
              f"{B}  {'Model':<24}| {'Job':<18}{'Score':>8}   Notes{X}", " " + "─" * 110]
        for nm, j in jlist:
            v = sv.get((nm, "job_" + j), "")
            score, _, note = v.partition(",")
            good = re.fullmatch(r"(\d+)/(\d+)", score)
            col = RED if score == "FAIL" or "DIFFERS" in note or (good and int(good[1]) * 4 < int(good[2]) * 3) else G
            mark = f"{Y}▶{X}" if (nm, j) == jcur and (nm, j) not in jdone else " "
            L.append(f" {mark}{nm:<24}| {jobs.TITLE[j]:<18}{col}{score or '.':>8}{X}   {D}{note}{X}")
    L += ["", f"  {D}Items/s: embedding model = texts embedded, reranker = documents scored, multi-slot chat model = items classified with a forced label.{X}",
          f"  {D}. = not run yet   FAIL = did not complete   Ctrl+C closes this view; the benchmark keeps running.{X}"]
    return "\n".join(L), finished


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    if a and a[0] == "run":
        opt = lambda k: next((x[len(k) + 3:].split(",") for x in sys.argv if x.startswith(f"--{k}=")), ())
        run(a[1] if len(a) > 1 else "e2e", opt("skip"), opt("only"))
    else:
        try:
            while True:
                text, fin = render(a[0] if a else "e2e")
                sys.stdout.write(("" if "--once" in sys.argv else "\033[H\033[J") + text + "\n"); sys.stdout.flush()
                if fin or "--once" in sys.argv:
                    break
                time.sleep(2)
        except KeyboardInterrupt:
            pass
