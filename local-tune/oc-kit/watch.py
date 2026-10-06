#!/usr/bin/env python3
"""Live dashboard for the local-tune runners. Read-only; run: local-tune/watch.py
Without a label it follows the newest run: kvmatrix.sh, compare.py, spectest.py or ppl.py.
With results/sequence.txt (lines: "bench|kv <label> <title>") it also shows every stage of a queued run.
local-tune/watch.py <label> shows a saved run: a compare plan, a spectest label or the KV table of a finished stage."""
import csv, os, re, sys, time

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CONFIGS = [(k, n) for k in ("f16", "q8_0", "q4_0") for n in (0, 1)]
DEPTHS = (0, 8192, 16384)
G, Y, RED, C, D, B, X = "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"


def read(name):
    try:
        return open(os.path.join(R, name), errors="replace").read().splitlines()
    except OSError:
        return []


def results(stem="kvmatrix"):
    out = {}
    for r in csv.reader(read(stem + ".csv")):
        if len(r) < 12 or not r[-2].replace(".", "").isdigit():
            continue
        try:
            kv, nk = r[0], int(r[1][-1])
            pp, tg, d, ts = int(r[-8]), int(r[-7]), int(r[-6]), float(r[-2])
        except ValueError:
            continue
        out[(kv, nk, "pp" if pp else "tg", d)] = ts
    return out


def status(log):
    cur, done, failed, vram, ram, peak = None, set(), set(), "-", "-", {}
    for ln in log:
        m = re.search(r"START ctk=ctv=(\S+) nkvo=(\d)", ln)
        if m:
            cur = (m[1], int(m[2]))
        m = re.search(r"DONE (\S+) nkvo=(\d)", ln)
        if m:
            done.add((m[1], int(m[2])))
        if re.search(r"out of memory|failed to create|error", ln, re.I) and cur:
            failed.add(cur)
        m = re.search(r"vram=(\d+) MiB ram_avail=(\d+)MB", ln)
        if m and cur:
            vram, ram = int(m[1]), int(m[2])
            peak[cur] = max(peak.get(cur, 0), vram)
    return cur, done, failed, vram, ram, peak


def gpu():
    import subprocess
    q = "utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.max.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current,fan.speed"
    try:
        o = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
        return [float(x) if x.strip().replace(".", "").isdigit() else 0 for x in o.split(",")]
    except Exception:
        return None


HIST = []
W = 10  # width of every number column
SPARK = " ▁▂▃▄▅▆▇█"


def bar(frac, width=24, color=C):
    n = max(0, min(width, round(frac * width)))
    return f"{color}{'█' * n}{D}{'░' * (width - n)}{X}"


def heat(v, lo, hi):
    return G if v < lo else (Y if v < hi else RED)


def cell(v, bad):
    if v is not None:
        return f"{G}{v:{W}.1f}{X}"
    return f"{RED}{'FAIL':>{W}}{X}" if bad else f"{D}{'.':>{W}}{X}"


def secs(t):
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s


def dur(s):
    return f"{s // 60}m{s % 60:02d}s"


def timing(log):
    """Rows as [label, total secs or None while running, {depth: secs}] from START/TEST/DONE log lines."""
    out, t0 = [], 0
    for ln in log:
        m = re.match(r"(\d\d:\d\d:\d\d) START (.+)", ln)
        if m:
            k = re.match(r"ctk=ctv=(\S+) nkvo=(\d)", m[2])
            out.append([f"{k[1]} KV in {'RAM' if int(k[2]) else 'GPU'}" if k else m[2], None, {}]); t0 = secs(m[1])
            continue
        if not out:
            continue
        m = re.search(r"TEST \w+@(\d+) took=(\d+)s", ln)
        if m:
            out[-1][2][int(m[1])] = out[-1][2].get(int(m[1]), 0) + int(m[2])
        m = re.match(r"(\d\d:\d\d:\d\d) DONE", ln)
        if m:
            k = re.search(r"took=(\d+)s", ln)
            out[-1][1] = int(k[1]) if k else (secs(m[1]) - t0) % 86400
    return out


def phase_secs(name):
    """Total run time of one log, None when it has no finished rows."""
    t = [r[1] for r in timing(read(name)) if r[1] is not None]
    return sum(t) if t else None


def render_timing(log):
    rows = timing(log)
    if not any(r[1] is not None or r[2] for r in rows):
        return []
    depths = sorted({d for r in rows for d in r[2]})
    L = [f"  {B}TIME{X}  {D}how long each row took, split by tokens already in context. Each part includes loading the model or filling the context{X}", ""]
    def line(c1, parts, tot):
        return f"  {c1:<28}|" + "".join((dur(parts[d]) if d in parts else ".").rjust(10) for d in depths) + f" |{tot:>11}"
    hdr = f"  {'Row':<28}|" + "".join(f"@{d // 1024}k".rjust(10) for d in depths) + f" |{'Row total':>11}"
    L += [f"{B}{hdr}{X}", " " + "─" * (len(hdr) - 1)]
    for label, tot, parts in rows:
        L.append(line(label[:27], parts, dur(tot) if tot is not None else "running"))
    allp = {d: sum(r[2].get(d, 0) for r in rows) for d in depths if any(d in r[2] for r in rows)}
    L += [f"{B}" + line("All rows", allp, dur(sum(r[1] or 0 for r in rows))) + X, ""]
    return L


def mtime(name):
    try:
        return os.path.getmtime(os.path.join(R, name))
    except OSError:
        return 0


def stages():
    """Steps from sequence.txt as (kind, label, short, title, state); state is done, run or wait."""
    out, prev, running = [], 0, False
    for ln in read("sequence.txt"):
        m = re.match(r"(bench|kv) (\S+) (\S+) \| (.+)", ln)
        if not m:
            continue
        kind, label, short, title = m.groups()
        f = f"{label}.csv" if kind == "bench" else f"kvmatrix-{label}.csv"
        done = len(read(f)) >= 8 if kind == "bench" else mtime(f) > prev
        state = "done" if done else ("wait" if running else "run")
        running = running or not done
        prev = max(prev, mtime(f)) if done else prev
        out.append((kind, label, short, title, state))
    return out


KIND = {"bench": "Speed test", "kv": "Long-context test"}
WHERE = {"99": "all on GPU", "0": "CPU only"}


def bench_rows(label):
    out = {}
    for r in csv.reader(read(label + ".csv")):
        try:
            name = re.split(r"[-.][Qq]\d", os.path.basename(r[5]))[0][:20].rstrip("-")
            out[(f"{name}, {WHERE.get(r[17], r[17] + ' layers on GPU')}", "pp" if int(r[-8]) else "tg")] = float(r[-2])
        except (ValueError, IndexError):
            continue
    return out


def render_steps(st, kv_done):
    mark = {"done": f"{G}\u2714{X}", "run": f"{Y}\u25b6{X}", "wait": f"{D}.{X}"}
    now = next((i for i, s in enumerate(st, 1) if s[4] == "run"), None)
    L = [f"  {B}STEPS{X}  {D}" + (f"step {now} of {len(st)}" if now else "all finished") + X, ""]
    for i, (kind, label, short, title, state) in enumerate(st, 1):
        prog = ""
        if state == "run":
            prog = f"   {Y}{len(read(label + '.csv'))} of 8 runs done{X}" if kind == "bench" else f"   {Y}{kv_done} of 6 rows done{X}"
        text = f"{KIND[kind]:<18}{title}"
        took = phase_secs(f"{label}.log" if kind == "bench" else f"kvmatrix-{label}.log") if state == "done" else None
        prog = prog or (f"   {D}took {dur(took)}{X}" if took is not None else "")
        L.append(f"  {mark[state]} {i}  " + (f"{B}{text}{X}" if state == "run" else (text if state == "done" else f"{D}{text}{X}")) + prog)
    return L + [""]


def render_bench(st):
    cols = [(s[2], bench_rows(s[1])) for s in st if s[0] == "bench" and read(s[1] + ".csv")]
    if not cols:
        return []
    keys = []
    for _, rows in cols:
        for k in rows:
            if k[0] not in keys:
                keys.append(k[0])
    L = [f"  {B}SPEED TEST RESULTS{X}  {D}tokens/sec, higher is better{X}", ""]
    for t, name in (("pp", "Reading a 512-token prompt"), ("tg", "Writing a 128-token answer")):
        L.append(f"  {B}{name:<44}" + "".join(f"{c:>{W}}" for c, _ in cols) + X)
        for k in keys:
            L.append(f"  {k:<44}" + "".join(cell(rows.get((k, t)), False) for _, rows in cols))
        L.append("")
    logs = [read(s[1] + ".log") for s in st if s[0] == "bench" and s[4] != "wait"]
    return L + (render_timing([l for l in logs if l][-1]) if any(logs) else [])


def gpu_panel(ram="-"):
    g = gpu()
    L = [f"  {B}GPU LIVE{X}"]
    if g:
        HIST.append(g[0]); del HIST[:-40]
        used, tot = g[2], g[3]
        spark = "".join(SPARK[min(8, int(u / 100 * 8.99))] for u in HIST)
        L += [f"  Video memory  {bar(used / tot, 24, heat(used / tot, .7, .9))} {B}{used:.0f}{X} / {tot:.0f} MiB",
              f"  GPU busy      {bar(g[0] / 100, 24, heat(g[0], 101, 101))} {B}{g[0]:.0f}%{X}   {D}history{X} {C}{spark}{X}",
              f"  Memory busy   {bar(g[1] / 100, 24)} {B}{g[1]:.0f}%{X}",
              f"  Power         {bar(g[5] / max(g[6], 1), 24, heat(g[5] / max(g[6], 1), .8, .95))} {B}{g[5]:.0f}{X} / {g[6]:.0f} W",
              f"  Temperature   {heat(g[4], 70, 83)}{g[4]:.0f} °C{X}      Fan {g[12]:.0f}%      Core clock {B}{g[7]:.0f}{X}/{g[8]:.0f} MHz      Mem clock {g[9]:.0f} MHz",
              f"  PCIe link     Gen{g[10]:.0f} x{g[11]:.0f}      {D}System RAM free:{X} {B}{ram} MB{X}"]
    else:
        L.append("  (nvidia-smi unavailable)")
    return L


def render(stem="kvmatrix", st=()):
    log = read(stem + ".log")
    res = results(stem)
    cur, done, failed, vram, ram, peak = status(log)
    finished = any("ALLDONE" in l for l in log)
    n = len(done)
    stamps = [l[:8] for l in log if re.match(r"\d\d:\d\d:\d\d (START|DONE)", l)]
    el = eta = ""
    if stamps:
        e = (secs(time.strftime("%H:%M:%S")) - secs(stamps[0])) % 86400
        el = f"{e // 60}m{e % 60:02d}s"
        if n and not finished:
            r = e / n * (6 - n)
            eta = f"   ~{int(r // 60)}m left"
    L = [f"{B}  KV-CACHE BENCHMARK{X}  {D}DeepSeek-R1 7B Q4_K_S  ·  GTX 1660 Ti 6GB  ·  Ryzen 5 7600X{X}", ""]
    live = next((x for x in st if x[4] == "run"), None)
    if st and stem == "kvmatrix":
        if not (live and live[0] == "kv" and mtime("kvmatrix.log") > max([mtime(f"{x[1]}.csv" if x[0] == "bench" else f"kvmatrix-{x[1]}.csv") for x in st if x[4] == "done"] or [0])):
            log, res, cur, done, failed, peak, finished, n, el, eta = [], {}, None, set(), set(), {}, False, 0, "", ""
        L += render_steps(st, n)
    state = f"{G}✔ ALL DONE{X}" if finished else (f"{Y}● running: {cur[0]} KV in {'RAM' if cur[1] else 'GPU'}{X}" if cur else "starting…")
    if not st:
        L += [f"  Progress  {bar(n / 6, 30)} {B}{n}/6{X}  {state}", f"  Elapsed   {el}{eta}", ""]
    L += gpu_panel(ram)
    L.append("")
    if st and not (live and live[0] == "kv") and stem == "kvmatrix":
        return "\n".join(L + render_bench(st) + [f"  {D}Ctrl+C closes this view; the benchmark keeps running.{X}"])
    L += [f"  {B}RESULTS{X}  {D}tokens/sec, higher is better. pp = reading prompt, tg = writing answer, @Nk = N thousand tokens already in context{X}", ""]
    # one column spec shared by header, rule and rows => identical widths
    C1, C2, C3 = 9, 7, 11  # KV type, Cache in, Peak VRAM
    def line(c1, c2, nums, c3, mark=" "):
        return f" {mark}{c1:<{C1}}{c2:<{C2}}|" + "".join(nums) + f" | {c3:<{C3}}"
    hdr = line("KV type", "Cache", [f"{t}@{d // 1024}k".rjust(W) for t in ("pp", "tg") for d in DEPTHS], "Peak VRAM")
    L += [f"{B}{hdr}{X}", " " + "─" * (len(hdr) - 1)]
    for kv, nk in CONFIGS:
        bad = (kv, nk) in failed
        m = f"{Y}▶{X}" if cur == (kv, nk) and not finished else " "
        nums = [cell(res.get((kv, nk, t, d)), bad) for t in ("pp", "tg") for d in DEPTHS]
        pk = f"{peak[(kv, nk)]:.0f} MiB" if (kv, nk) in peak else "-"
        L.append(line(kv, "RAM" if nk else "GPU", nums, pk, m))
    L += [""] + render_timing(log)
    L += [f"  {D}. = not run yet   FAIL = out of memory   Ctrl+C closes this view; the benchmark keeps running.{X}"]
    return "\n".join(L)


def newest():
    """(kind, name) of the run whose files changed last; kind is kv, compare, spec or ppl."""
    best = ("kv", None, max(mtime("kvmatrix.log"), mtime("kvmatrix.csv")))
    for f in os.listdir(R):
        # spec logs are named per variant, so only the csv gives the label
        m = re.match(r"compare-(.+?)\.(?:log|csv)$|spec-(.+?)\.csv$|ppl-(.+?)\.(?:log|csv)$|(?!kvmatrix|compare-|spec-|ppl-)(.+?)\.log$", f)
        if m and mtime(f) > best[2]:
            best = ("compare" if m[1] else "spec" if m[2] else "ppl" if m[3] else "bench", m[1] or m[2] or m[3] or m[4], mtime(f))
    return best[:2]


def push_view(name):
    """gpu-push.py run: phase, step table with deltas against the baseline, trend, live clocks."""
    log = read(name + ".log")
    base = tg_best = None
    steps, phase, run, stage, t_run, start, done = [], "starting", None, "", None, None, False
    def hms(t):
        h, m, s = (int(x) for x in t.split(":"))
        return h * 3600 + m * 60 + s
    for ln in log:
        t, _, msg = ln.partition(" ")
        if not re.match(r"\d\d:\d\d:\d\d$", t):
            continue
        start = start if start is not None else hms(t)
        if msg.startswith("PHASE "):
            phase = msg[6:]
        elif msg.startswith("RUN "):
            run, stage, t_run = msg[4:], "starting", hms(t)
        elif msg.startswith("STAGE "):
            stage = msg[6:]
        elif msg.startswith("TEST "):
            m = re.match(r"TEST (\S+) (PASS|FAIL) tg=([\d.]+) t/s ppl=(\S*)(?: sm=(\d+) mem=(\d+) MHz (\d+) W (\d+) C)?\s*(.*)", msg)
            if m:
                steps.append((m[1], m[2], float(m[3]), m[4], m[5], m[6], m[7], m[8], m[9]))
                run = None
                if m[1] == "baseline" or (m[1].startswith("verify") and base is None):
                    base = float(m[3])
        elif msg.startswith("ALLDONE"):
            done = True
    now = hms(time.strftime("%T")) if start is not None else 0
    L = [f"  {B}GPU PUSH{X}  {D}{name}  ·  how far memory and core clock can go before a test fails{X}", ""]
    ok = [s for s in steps if s[1] == "PASS" and s[0] != "baseline"]
    fails = [s for s in steps if s[1] == "FAIL"]
    L.append(f"  Phase      {B}{'finished' if done else phase}{X}      Elapsed {B}{dur(max(0, now - start)) if start is not None else '-'}{X}")
    if run and not done:
        L.append(f"  Now        {Y}{run}{X}  {D}{stage}, {now - t_run}s into this step (a step takes about 50 s){X}")
    best = max((s[2] for s in ok), default=None)
    L += ["", f"  Baseline decode   {B}{base if base else '-'}{X} t/s     Best so far   {G}{B}{best if best else '-'}{X} t/s" + (f"  {G}({(best / base - 1) * 100:+.1f}%){X}" if best and base else ""),
          f"  Steps passed      {G}{len(ok)}{X}     failed {RED if fails else ''}{len(fails)}{X}"]
    if fails:
        L.append(f"  {RED}Stopped at {fails[-1][0]}: {fails[-1][8]}{X}")
    for kind in ("mem", "core"):
        k = [s for s in ok if s[0].startswith(kind + "+")]
        if k:
            L.append(f"  Last good {kind:<5} {G}{B}{k[-1][0].split('+')[1]}{X} MHz offset")
    L += ["", f"  {B}{'Step':<12}{'Result':<8}{'Decode':>9}{'vs base':>9}{'Core':>7}{'Memory':>8}{'Power':>7}{'Temp':>6}  Perplexity   Notes{X}", " " + "─" * 100]
    spark = ""
    for n, r, tg, ppl, sm, mem, w, c, note in steps[-24:]:
        col = G if r == "PASS" else RED
        dl = f"{(tg / base - 1) * 100:+.1f}%" if base else "-"
        L.append(f"  {n:<12}{col}{r:<8}{X}{tg:>9.1f}{col}{dl:>9}{X}{(sm or '-'):>7}{(mem or '-'):>8}{((w or '-') + ' W') if w else '-':>7}{((c or '-') + ' C') if c else '-':>6}  {ppl:<11}  {D}{note}{X}")
    if ok:
        lo, hi = min(s[2] for s in ok), max(s[2] for s in ok)
        spark = "".join(SPARK[min(8, int((s[2] - lo) / max(hi - lo, 1e-9) * 8))] for s in ok)
        L += ["", f"  Decode trend over passed steps  {C}{spark}{X}  {D}{lo:.1f} to {hi:.1f} t/s{X}"]
    L += ["", f"  {D}A step fails on: a hang or crash, a new kernel Xid, perplexity not equal to the baseline, decode >7% under the best, or 83 C.{X}",
          f"  {D}The saved setting is the last good step minus one step, then a soak test must pass.{X}", ""]
    return "\n".join(L + gpu_panel())


def other_view(kind, name):
    if kind == "bench" and name.startswith("gpupush-"):
        return push_view(name), 0
    if kind == "bench":
        log = read(name + ".log")
        return ("\n".join(render_bench([("bench", name, name, 0, "done")]) or ["  " + l for l in log[-12:]]) + "\n\n" + "\n".join(gpu_panel()), 0)
    import compare, ppl, spectest
    return {"compare": compare, "spec": spectest, "ppl": ppl}[kind].render(name)


if __name__ == "__main__":
    once = "--once" in sys.argv
    saved = [a for a in sys.argv[1:] if not a.startswith("-")]
    stem = f"kvmatrix-{saved[0]}" if saved else "kvmatrix"
    clear = "" if once else "\033[H\033[J"
    try:
        while True:
            kind, name = newest() if not saved else next(((k, saved[0]) for k, f in (("compare", f"compare-{saved[0]}.plan"), ("spec", f"spec-{saved[0]}.csv"), ("ppl", f"ppl-{saved[0]}.plan")) if mtime(f)), ("kv", None))
            if kind != "kv":
                # stays open after a run ends, so it moves on to the next run when one starts
                sys.stdout.write(clear + other_view(kind, name)[0] + "\n"); sys.stdout.flush()
                if once or saved:
                    break
                time.sleep(2)
                continue
            st = stages()
            sys.stdout.write(clear + render(stem, st) + "\n")
            sys.stdout.flush()
            if once or saved or (all(x[4] == "done" for x in st) if st else any("ALLDONE" in l for l in read("kvmatrix.log"))):
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass
