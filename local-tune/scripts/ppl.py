#!/usr/bin/env python3
"""Perplexity comparison with a live view in the style of watch.py. Use it when a change touches the KV format.
ppl.py run <name>    runs results/ppl-<name>.plan and writes ppl-<name>.csv/.log
ppl.py <name>        live view of that run
Plan lines: "title: text", "args: common llama-perplexity args", then one variant per line: "label | build-dir | extra args"."""
import os, re, subprocess, sys, time
import watch
from watch import G, Y, RED, C, D, B, X
from paths import rp, ROOT, SCRIPTS



def plan(name):
    title, args, variants = name, "", []
    for ln in watch.read(f"ppl-{name}.plan"):
        if ln.startswith("title:"):
            title = ln[6:].strip()
        elif ln.startswith("args:"):
            args = ln[5:].strip()
        elif "|" in ln:
            variants.append([p.strip() for p in ln.split("|")] + [""])
    return title, args, variants


def run(name):
    _, args, variants = plan(name)
    out = open(rp(f"ppl-{name}.csv"), "w")
    log = open(rp(f"ppl-{name}.log"), "w")
    def say(msg):
        log.write(f"{time.strftime('%T')} {msg}\n"); log.flush()
    t0 = time.time()
    for label, build, extra, *_ in variants:
        say(f"START {label}"); t1 = time.time()
        p = subprocess.Popen([os.path.join(ROOT, build, "bin", "llama-perplexity")] + args.split() + extra.split(), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        buf, final = "", None
        # progress is printed as "[n]value," without line ends, so read it in small pieces
        while True:
            ch = p.stdout.read(64)
            if not ch:
                break
            buf += ch
            for n, v in re.findall(r"\[(\d+)\](\d+\.\d+),", buf):
                say(f"  CHUNK {n} ppl={v}")
            buf = re.sub(r"\[\d+\]\d+\.\d+,", "", buf)[-400:]
            m = re.search(r"Final estimate: PPL = (\d+\.\d+) \+/- (\d+\.\d+)", buf)
            if m:
                final = (m[1], m[2])
        p.wait()
        out.write(f"{label},{final[0] if final else ''},{final[1] if final else ''}\n"); out.flush()
        say(f"DONE {label} took={time.time() - t1:.0f}s" + ("" if final else " FAILED"))
    say(f"ALLDONE total={time.time() - t0:.0f}s")


def render(name):
    title, args, variants = plan(name)
    log = watch.read(f"ppl-{name}.log")
    res = {r.split(",")[0]: r.split(",")[1:] for r in watch.read(f"ppl-{name}.csv") if "," in r}
    total = re.search(r"--chunks (\d+)", args)
    cur, prog, took = None, {}, {}
    for ln in log:
        m = re.search(r"START (.*)", ln)
        if m:
            cur = m[1].strip()
        m = re.search(r"CHUNK (\d+) ppl=(\S+)", ln)
        if m and cur:
            prog[cur] = (int(m[1]), float(m[2]))
        m = re.search(r"DONE (.*?) took=(\d+)s", ln)
        if m:
            took[m[1].strip()] = int(m[2]); cur = None
    done = any("ALLDONE" in l for l in log)
    L = [f"{B}{title}{X}", f"{D}{args}{X}", "", f"  {'variant':<30}{'chunks':>9}{'perplexity':>14}{'+/-':>8}{'vs first':>10}{'time':>7}"]
    base = None
    for label, *_ in variants:
        n, v = prog.get(label, (0, 0))
        ch = f"{n}/{total[1]}" if total else str(n)
        if label in res and res[label][0]:
            p, e = float(res[label][0]), float(res[label][1])
            base = p if base is None else base
            d = (p / base - 1) * 100
            col = G if abs(p - base) <= e else Y if d < 1 else RED
            L.append(f"  {label:<30}{ch:>9}{p:>14.4f}{e:>8.4f}{col}{d:>+9.2f}%{X}{took.get(label, 0):>6}s")
        elif label in res:
            L.append(f"  {label:<30}{ch:>9}{RED}{'FAIL':>14}{X}")
        elif label == cur:
            L.append(f"  {C}{label:<30}{ch:>9}{v:>14.4f}{X}   running")
        else:
            L.append(f"  {D}{label:<30}{'.':>9}{X}")
    L += ["", f"  {D}green = inside the error of the first row. Lower perplexity is better. " + ("Run finished." if done else "Ctrl+C closes this view; the run keeps going.") + X]
    return "\n".join(L), done


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        while True:
            text, done = render(sys.argv[1])
            sys.stdout.write("\033[H\033[J" + text + "\n"); sys.stdout.flush()
            if done:
                break
            time.sleep(2)
