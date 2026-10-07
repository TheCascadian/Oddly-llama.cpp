#!/usr/bin/env python3
"""One entry point for the local-tune checks. It compares the candidate build (build-exp) with the shipped one (build-live).
lab.py              status of every suite and the live view of the step that runs now
lab.py auto         same view, and it watches the source files: on a change it rebuilds build-exp, runs the suites
                    that cover the changed files and writes ledger.html again
lab.py run [suite]  build and run one suite, or all of them, now
lab.py accept       take the current source as checked without a run
Suites are in suites.conf: "name | watched paths (globs from the repo root) | steps".
Steps: ops:<test-backend-ops op>, compare:<plan name>, ppl:<plan name>, spec:<label>:<model>:<server args>."""
import csv, glob, hashlib, json, os, re, subprocess, sys, threading, time
import watch
from watch import G, Y, RED, C, D, B, X, R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATE = os.path.join(R, "lab-state.json")
SHIPPED, CAND = "build-live", "build-exp"
TARGETS = ["llama-server", "llama-bench", "llama-perplexity", "test-backend-ops"]
LIMIT = 3.0  # percent; a smaller difference between the two builds counts as no change


def suites():
    out = []
    for ln in open(os.path.join(HERE, "suites.conf")):
        p = [x.strip() for x in ln.split("|")]
        if len(p) == 3 and not ln.startswith("#"):
            out.append((p[0], p[1].split(), p[2].split()))
    return out


def digest(paths):
    h = hashlib.sha256()
    for f in sorted(f for p in paths for f in glob.glob(os.path.join(ROOT, p))):
        h.update(f.encode()); h.update(open(f, "rb").read())
    return h.hexdigest()[:16]


def load():
    try:
        return json.load(open(STATE))
    except (OSError, ValueError):
        return {}


def save(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def pairs(rows):
    """rows: (build, args, metric, value). Returns {metric: percent change of the candidate against the shipped build}."""
    acc = {}
    for b, a, m, v in rows:
        acc.setdefault((a, m), {}).setdefault(b, []).append(v)
    out = {}
    for (a, m), d in acc.items():
        if SHIPPED in d and CAND in d:
            out.setdefault(m, []).append((sum(d[CAND]) / len(d[CAND])) / (sum(d[SHIPPED]) / len(d[SHIPPED])) * 100 - 100)
    return {m: min(v, key=lambda x: x) if min(v) < -LIMIT else max(v, key=abs) for m, v in out.items()}


def judge_compare(name):
    import compare
    _, _, variants = compare.plan(name)
    by = {v[0]: (v[1], v[2] + " " + v[3]) for v in variants}
    rows, seen = [], set()
    for r in csv.reader(watch.read(f"compare-{name}.csv")):
        try:
            rows.append(by[r[0]] + (f"{'pp' if int(r[-8]) else 'tg'}@{int(r[-6]) // 1024}k", float(r[-2]))); seen.add((r[0], rows[-1][2]))
        except (ValueError, IndexError, KeyError):
            continue
    tests = {t for _, t in seen}
    lost = [v[0] for v in variants if v[1] == CAND and any((v[0], t) not in seen for t in tests)]
    d = pairs(rows)
    if lost:
        return "fail", "did not complete: " + ", ".join(lost), d
    if not d:
        return "fail", "no result", d
    worst, best = min(d.values()), max(d.values())
    text = ", ".join(f"{m} {v:+.1f}%" for m, v in sorted(d.items(), key=lambda x: (x[0][:2], int(x[0][3:-1]))))
    return ("slower" if worst < -LIMIT else "faster" if best > LIMIT else "same"), text, d


def judge_ppl(name):
    import ppl
    _, _, variants = ppl.plan(name)
    res = {r.split(",")[0]: r.split(",")[1:] for r in watch.read(f"ppl-{name}.csv") if "," in r}
    try:
        base, err = float(res[variants[0][0]][0]), float(res[variants[0][0]][1])
        vals = [(v[0], float(res[v[0]][0])) for v in variants[1:]]
    except (KeyError, ValueError, IndexError):
        return "fail", "did not complete", {}
    bad = [l for l, p in vals if p - base > err]
    text = f"{base:.3f} shipped; " + ", ".join(f"{l} {p:.3f}" for l, p in vals)
    return ("fail" if bad else "same"), text, {l: p / base * 100 - 100 for l, p in vals}


def judge_spec(label):
    acc = {}
    for r in csv.DictReader(watch.read(f"spec-{label}.csv")):
        acc.setdefault(r["variant"], []).append(float(r["tg_tps"]))
    if len(acc) < 2:
        return "fail", "did not complete", {}
    v = [sum(x) / len(x) for x in acc.values()]
    d = {k: x / v[0] * 100 - 100 for k, x in zip(list(acc)[1:], v[1:])}
    return ("slower" if min(d.values()) < -LIMIT else "faster" if max(d.values()) > LIMIT else "same"), ", ".join(f"{k} {x:+.1f}%" for k, x in d.items()), d


def run_step(step):
    kind, _, arg = step.partition(":")
    py = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if kind == "ops":
        p = subprocess.run([f"{ROOT}/{CAND}/bin/test-backend-ops", "-o", arg], cwd=ROOT, capture_output=True, text=True, errors="replace")
        open(os.path.join(R, f"lab-ops-{arg}.log"), "w").write(p.stdout + p.stderr)
        ok = p.returncode == 0 and "backends passed" in p.stdout
        return ("same" if ok else "fail"), (re.findall(r"\d+/\d+ backends passed", p.stdout) or ["failed"])[-1], {}
    if kind == "compare":
        py(os.path.join(HERE, "compare.py"), "run", arg)
        return judge_compare(arg)
    if kind == "ppl":
        py(os.path.join(HERE, "ppl.py"), "run", arg)
        return judge_ppl(arg)
    if kind == "spec":
        label, model, args = arg.split(":", 2)
        py(os.path.join(HERE, "spectest.py"), CAND, label, os.path.expanduser(model), *args.replace("_", " ").split())
        return judge_spec(label)
    return "fail", "unknown step kind", {}


def build(st):
    st["running"] = "build"; st["build"] = {"at": time.time(), "ok": None}; save(st)
    env = dict(os.environ, PATH="/opt/cuda/bin:" + os.environ["PATH"])
    with open(os.path.join(R, "lab-build.log"), "w") as log:
        ok = subprocess.run(["cmake", "--build", CAND, "-j", str(os.cpu_count()), "--target", *TARGETS], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode == 0
    st["build"] = {"at": time.time(), "ok": ok}; st["running"] = None; save(st)
    return ok


def run_suites(names):
    """Builds the candidate, runs every step of the named suites once and writes the page again."""
    st = load()
    todo = [s for s in suites() if s[0] in names]
    for n, paths, steps in todo:
        st[n] = {"hash": digest(paths), "state": "queued", "at": time.time(), "steps": {s: ["queued", "", {}] for s in steps}}
    save(st)
    if not build(st):
        for n, _, _ in todo:
            st[n]["state"] = "build failed"
        save(st)
        return
    done = {}
    for n, _, steps in todo:
        st[n]["state"] = "running"
        for s in steps:
            if s not in done:
                st["running"] = s; st[n]["steps"][s][0] = "running"; save(st)
                done[s] = list(run_step(s))
            st[n]["steps"][s] = done[s]; st["running"] = None; save(st)
        kinds = [v[0] for v in st[n]["steps"].values()]
        st[n]["state"] = next((k for k in ("fail", "slower", "faster") if k in kinds), "same")
        st[n]["at"] = time.time(); save(st)
    subprocess.run([sys.executable, os.path.join(HERE, "ledger.py")], cwd=ROOT, stdout=subprocess.DEVNULL)


def stale(st):
    return [n for n, paths, _ in suites() if st.get(n, {}).get("hash") != digest(paths)]


def accept():
    st = load()
    for n, paths, steps in suites():
        st.setdefault(n, {"state": "not run", "at": None, "steps": {}})["hash"] = digest(paths)
    save(st)


def watcher():
    """Runs the suites of changed files. A change must stay the same for one poll, so a half-saved edit does not start a build."""
    seen = None
    while True:
        st = load()
        for n, paths, _ in suites():  # first start: the current source is the reference
            if n not in st:
                st[n] = {"hash": digest(paths), "state": "not run", "at": None, "steps": {}}; save(st)
        now = {n: digest(p) for n, p, _ in suites()}
        changed = [n for n in now if st[n]["hash"] != now[n]]
        if changed and seen == now:
            run_suites(changed)
        seen = now
        time.sleep(3)


COL = {"same": G, "faster": G, "slower": RED, "fail": RED, "build failed": RED, "running": Y, "queued": Y, "not run": D}
WORD = {"same": "no change", "faster": "FASTER", "slower": "SLOWER", "fail": "FAILED"}


def render(auto):
    st, L = load(), []
    L.append(f"{B}  LOCAL-TUNE LAB{X}  {D}candidate {CAND} against shipped {SHIPPED}   " + ("watching the source files" if auto else "view only; start 'lab.py auto' to run on changes") + X)
    b = st.get("build")
    if st.get("running") == "build":
        L.append(f"  {Y}building {CAND} ...{X}  {D}log: results/lab-build.log{X}")
    elif b and b["ok"] is False:
        L.append(f"  {RED}last build failed{X}  {D}see results/lab-build.log{X}")
    L += ["", f"  {B}{'Suite':<12}{'State':<14}{'Last run':<15}Watched files{X}", "  " + "─" * 96]
    for n, paths, steps in suites():
        s = st.get(n, {})
        state = s.get("state", "not run")
        if s.get("hash") and s["hash"] != digest(paths) and state not in ("running", "queued"):
            state = "changed"
        at = time.strftime("%d %b %H:%M", time.localtime(s["at"])) if s.get("at") else "-"
        L.append(f"  {B}{n:<12}{X}{COL.get(state, Y)}{WORD.get(state, state):<14}{X}{at:<15}{D}{' '.join(os.path.basename(p) for p in paths)[:70]}{X}")
        for k in steps:
            r = s.get("steps", {}).get(k, ["not run", "", {}])
            L.append(f"    {COL.get(r[0], D)}{WORD.get(r[0], r[0]):<10}{X} {k.split(':')[0] + ':' + k.split(':')[1]:<26}{D}{r[1][:110]}{X}")
    L += ["", f"  {D}no change = inside {LIMIT:.0f}% of the shipped build. Page: local-tune/ledger.html{X}", ""]
    run = st.get("running")
    kind, name = (run.split(":")[0], run.split(":")[1]) if run and run.split(":")[0] in ("compare", "ppl", "spec") else watch.newest()
    if kind != "kv":
        L += ["  " + "═" * 96, watch.other_view(kind, name)[0]]
    return "\n".join(L)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["run"]:
        run_suites(a[1:] or [s[0] for s in suites()])
        print(render(False))
    elif a[:1] == ["accept"]:
        accept()
    else:
        auto = a[:1] == ["auto"]
        if auto:
            threading.Thread(target=watcher, daemon=True).start()
        try:
            while True:
                sys.stdout.write(("" if "--once" in a else "\033[H\033[J") + render(auto) + "\n"); sys.stdout.flush()
                if "--once" in a:
                    break
                time.sleep(2)
        except KeyboardInterrupt:
            pass
