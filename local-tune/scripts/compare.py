#!/usr/bin/env python3
"""Side-by-side llama-bench comparison with a live view in the style of watch.py.
compare.py run <name>    runs results/compare-<name>.plan and writes compare-<name>.csv/.log
compare.py <name>        live view of that run
Plan lines: "title: text", "args: common llama-bench args", then one variant per line: "label | build-dir | ENV=1 ... | extra args"."""
import csv, os, re, subprocess, sys, threading, time
import watch
from watch import G, Y, RED, C, D, B, X, R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plan(name):
    title, args, variants = name, "", []
    for ln in watch.read(f"compare-{name}.plan"):
        if ln.startswith("title:"):
            title = ln[6:].strip()
        elif ln.startswith("args:"):
            args = ln[5:].strip()
        elif "|" in ln:
            variants.append([p.strip() for p in ln.split("|")] + [""] * 3)
    return title, args, variants


def run(name):
    _, args, variants = plan(name)
    out = open(os.path.join(R, f"compare-{name}.csv"), "w")
    log = open(os.path.join(R, f"compare-{name}.log"), "w")
    def say(msg):
        log.write(f"{time.strftime('%T')} {msg}\n"); log.flush()
    t0 = time.time()
    for label, build, env, extra, *_ in variants:
        say(f"START {label}"); t1 = tp = time.time()
        stop = threading.Event()
        def sample():
            while not stop.wait(2):
                v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
                m = re.search(r"MemAvailable:\s+(\d+)", open("/proc/meminfo").read())
                say(f"  vram={v} MiB ram_avail={int(m[1]) // 1024}MB")
        threading.Thread(target=sample, daemon=True).start()
        e = dict(os.environ, PATH="/opt/cuda/bin:" + os.environ["PATH"], **dict(kv.split("=", 1) for kv in env.split()))
        cmd = [f"{ROOT}/{build}/bin/llama-bench", "-o", "csv"] + args.split() + extra.split()
        p = subprocess.Popen(cmd, env=e, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        for ln in p.stdout:
            if ln.startswith('"'):
                out.write(f'"{label}",' + ln); out.flush()
                f = ln.replace('"', "").split(",")
                say(f"  TEST {'pp' if int(f[-8]) else 'tg'}@{f[-6]} took={time.time() - tp:.0f}s"); tp = time.time()
            elif re.search(r"CUDA error|out of memory|failed to", ln):
                say("ERROR " + ln.strip()[:160])
        p.wait(); stop.set()
        say(f"DONE {label} took={time.time() - t1:.0f}s")
    say(f"ALLDONE total={time.time() - t0:.0f}s")


def render(name):
    title, args, variants = plan(name)
    res, tests = {}, []
    opt = lambda f, d: (re.search(rf"(?:^| ){f} (\S+)", args) or [0, d])[1]
    for d in opt("-d", "0").split(","):
        tests += [(t, int(d)) for t, f, dflt in (("pp", "-p", "512"), ("tg", "-n", "128")) if int(opt(f, dflt))]
    for r in csv.reader(watch.read(f"compare-{name}.csv")):
        try:
            t = ("pp" if int(r[-8]) else "tg", int(r[-6]))
            res[(r[0], t)] = (float(r[-2]), float(r[-1]))
        except (ValueError, IndexError):
            continue
        if t not in tests:
            tests.append(t)
    cur, done, bad, peak, ram, lowram = None, set(), {}, {}, "-", {}
    log = watch.read(f"compare-{name}.log")
    for ln in log:
        m = re.match(r"\S+ (START|DONE|ERROR) (.+?)(?: took=\d+s)?$", ln)
        if m and m[1] == "START": cur = m[2]
        if m and m[1] == "DONE": done.add(m[2])
        if m and m[1] == "ERROR": bad[cur] = "out of memory" if "memory" in m[2] or "CUDA error" in m[2] else "failed"
        m = re.search(r"vram=(\d+) MiB ram_avail=(\d+)MB", ln)
        if m and cur:
            peak[cur] = max(peak.get(cur, 0), int(m[1])); ram = int(m[2]); lowram[cur] = min(lowram.get(cur, 10**9), ram)
    finished = any("ALLDONE" in l for l in log)
    L = [f"{B}  KERNEL COMPARISON{X}  {D}{title}{X}", f"  {D}llama-bench {args}{X}", ""]
    mark = {"done": f"{G}✔{X}", "run": f"{Y}▶{X}", "wait": f"{D}.{X}"}
    L.append(f"  {B}STEPS{X}  {D}" + ("all finished" if finished else f"step {min(len(done) + 1, len(variants))} of {len(variants)}") + X)
    L.append("")
    for i, v in enumerate(variants, 1):
        st = "done" if v[0] in done else ("run" if v[0] == cur else "wait")
        text = f"{v[0]:<28}{D}{v[2] or 'no env vars'}{X}"
        L.append(f"  {mark[st]} {i}  " + (f"{B}{text}{X}" if st == "run" else text))
    L += [""] + watch.gpu_panel(ram) + [""]
    L += [f"  {B}RESULTS{X}  {D}tokens/sec, higher is better. Each value is the average of the repetitions, +/- is their spread. @Nk = N thousand tokens already in context{X}", ""]
    W = 15
    hdr = f"  {'Variant':<28}|" + "".join(f"{t}@{d // 1024}k".rjust(W) for t, d in tests) + f" | {'Peak VRAM':<10}{'Low RAM':<10}Errors"
    L += [f"{B}{hdr}{X}", " " + "─" * (len(hdr) - 1)]
    for v in variants:
        cells = ""
        for t in tests:
            x = res.get((v[0], t))
            cells += f"{G}{x[0]:8.1f}{X}{D} ±{x[1]:<5.1f}{X}" if x else (f"{RED}{'FAIL':>{W}}{X}" if v[0] in bad and v[0] in done else f"{D}{'.':>{W}}{X}")
        pk = f"{peak[v[0]]} MiB" if v[0] in peak else "-"
        lr = f"{lowram[v[0]]} MB" if v[0] in lowram else "-"
        err = f"{RED}{bad[v[0]]}{X}" if v[0] in bad else (f"{G}none{X}" if v[0] in done else "")
        L.append(f" {Y + chr(0x25b6) + X if v[0] == cur and not finished else ' '}{v[0]:<28}|{cells} | {pk:<10}{lr:<10}{err}")
    L += [""] + watch.render_timing(log)
    L += [f"  {D}. = not run yet   FAIL = did not complete   Ctrl+C closes this view; the benchmark keeps running.{X}"]
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
