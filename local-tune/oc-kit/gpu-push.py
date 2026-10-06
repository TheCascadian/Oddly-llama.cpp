#!/usr/bin/env python3
"""Find the highest stable GPU memory and core clock offsets for the GTX 1660 Ti, with a stability test at every step.
usage: local-tune/gpu-push.py [run] [--resume] [--power 120] [--mem-max 1500] [--mem-step 100] [--core-max 300] [--core-step 15] [--soak 10] [--margin 1]
       local-tune/gpu-push.py apply      set the offsets saved in results/gpu-oc.json (they are lost at reboot)
       local-tune/gpu-push.py reset      offsets back to 0
Run as your normal user. It asks for sudo once and uses it only to call NVML (offset set calls). The view is: python3 local-tune/watch.py
Each step sets the offset, then runs the 7B model (llama-bench decode at 4K context, llama-perplexity on 4 chunks) and checks:
  - the process exits cleanly, and the kernel log has no new NVRM Xid line
  - perplexity equals the baseline digit for digit (the run is deterministic, a flipped bit moves it)
  - decode speed did not fall under the best so far (GDDR6 retries bad transfers, which shows as lost speed before it shows as errors)
  - temperature stays under 83 C
The first failure stops that sweep. The saved value is the last good step minus --margin steps, then a soak test (--soak minutes) must pass.
--resume starts from results/gpu-oc.json: it sets --power, applies the saved offsets, re-tests them against the saved perplexity (stepping core, then memory, down if they no longer hold),
then only sweeps upward from there. The saved file changes only after the final soak passes; the old one is copied to gpu-oc.json.<stamp>.
The offsets live in the driver only: a crash or a reboot returns them to 0. A hard hang needs a power button press, so save your work first."""
import atexit, ctypes, json, os, re, signal, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
STATE = os.path.join(RES, "gpu-oc.json")
BUILD = os.path.join(ROOT, "build-live")
MODEL = next((os.path.join(d, f) for d, _, fs in os.walk(os.path.expanduser("~/.lmstudio/models")) for f in fs if f == "DeepSeek-R1-Distill-Qwen-7B-Uncensored.i1-Q4_K_S.gguf"), "")
CORPUS = os.path.join(RES, "ppl-corpus.txt")
STAMP = time.strftime("%m%d-%H%M")
LOG = os.path.join(RES, f"gpupush-{STAMP}.log")
T0 = time.time()


def nvml():
    lib = ctypes.CDLL("libnvidia-ml.so.1")
    assert lib.nvmlInit_v2() == 0, "NVML init failed (needs root)"
    h = ctypes.c_void_p()
    assert lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) == 0
    return lib, h


def set_offset(kind, mhz):
    """root only. kind is mem or core; the driver may refuse a value."""
    lib, h = nvml()
    fn = lib.nvmlDeviceSetMemClkVfOffset if kind == "mem" else lib.nvmlDeviceSetGpcClkVfOffset
    return fn(h, ctypes.c_int(int(mhz)))


def sudo_run(cmd):
    return subprocess.run(["sudo", "-n"] + cmd, capture_output=True).returncode


def sudo_set(kind, mhz):
    return subprocess.run(["sudo", "-n", sys.executable, os.path.abspath(__file__), "_set", kind, str(mhz)]).returncode


def log(msg):
    line = f"{time.strftime('%T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def smi(q):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [float(x) if re.fullmatch(r"[\d.]+", x.strip()) else x.strip() for x in out.split(",")]


def xid_count():
    out = subprocess.run("journalctl -k --no-pager -o cat 2>/dev/null | grep -c 'NVRM: Xid'", shell=True, capture_output=True, text=True).stdout
    return int(out.strip() or 0)


SAMPLE = {}


def sampler(stop):
    """Poll clocks, power and temperature while a workload runs; keeps the sample with the highest power."""
    best = None
    while not stop.is_set():
        c = smi("clocks.sm,clocks.mem,power.draw,temperature.gpu")
        if len(c) == 4 and all(isinstance(x, float) for x in c):
            SAMPLE["t"] = max(SAMPLE.get("t", 0), c[3])
            if best is None or c[2] > best[2]:
                best = c
                SAMPLE["clk"] = c
        stop.wait(2)


def workload():
    """(ok, tg t/s, ppl text, note). Runs bench then perplexity; both with a hard timeout."""
    xid0 = xid_count()
    note = ""
    SAMPLE.clear()
    stop = threading.Event()
    th = threading.Thread(target=sampler, args=(stop,), daemon=True)
    th.start()
    log("STAGE decode benchmark")
    try:
        b = subprocess.run([f"{BUILD}/bin/llama-bench", "-m", MODEL, "-ngl", "99", "-ctk", "q8_0", "-ctv", "q4_0", "-fa", "1", "-t", "6",
                            "-p", "0", "-n", "128", "-d", "4096", "-r", "3", "-o", "csv"], capture_output=True, text=True, timeout=300)
        rows = [r for r in b.stdout.splitlines() if r.startswith('"')]
        tg = float(rows[-1].split(",")[-2].strip('"')) if b.returncode == 0 and rows else 0.0
        log("STAGE perplexity check")
        p = subprocess.run([f"{BUILD}/bin/llama-perplexity", "-m", MODEL, "-f", CORPUS, "-c", "4096", "-b", "512", "--chunks", "4", "-ngl", "99",
                            "-ctk", "q8_0", "-ctv", "q4_0", "-fa", "on", "-t", "6"], capture_output=True, text=True, timeout=600)
        m = re.search(r"Final estimate: PPL = ([\d.]+)", p.stdout + p.stderr)
        ppl = m.group(1) if m else ""
    except subprocess.TimeoutExpired:
        return False, 0.0, "", "timeout (hang)"
    finally:
        stop.set()
        th.join()
    t = SAMPLE.get("t", 0)
    if xid_count() > xid0:
        note = "kernel Xid error"
    elif not ppl or tg == 0.0:
        note = "run failed"
    elif t >= 83:
        note = f"temperature {t:.0f} C"
    return note == "", tg, ppl, note


def step(label, base):
    log(f"RUN {label}")
    ok, tg, ppl, note = workload()
    if ok and base.get("ppl") and ppl != base["ppl"]:
        ok, note = False, f"perplexity {ppl} differs from baseline {base['ppl']}"
    if ok and tg < base.get("tg", 0) * 0.93:
        ok, note = False, f"decode {tg:.1f} t/s fell under the best {base['tg']:.1f} (memory retries)"
    clk = SAMPLE.get("clk") or smi("clocks.sm,clocks.mem,power.draw,temperature.gpu")
    log(f"TEST {label} {'PASS' if ok else 'FAIL'} tg={tg:.1f} t/s ppl={ppl} sm={clk[0]:.0f} mem={clk[1]:.0f} MHz {clk[2]:.0f} W {clk[3]:.0f} C {note}")
    return ok, tg, ppl


def sweep(kind, start, stop, stepv, base, good0=0):
    """Raise the offset until a step fails. Returns the last good offset (good0 if the first step fails)."""
    good, off = good0, start
    while off <= stop:
        if sudo_set(kind, off) != 0:
            log(f"{kind} offset {off} refused by the driver, stopping")
            break
        ok, tg, ppl = step(f"{kind}+{off}", base)
        if not ok:
            break
        good, base["tg"] = off, max(base["tg"], tg)
        off += stepv
    sudo_set(kind, good)
    return good


def soak(minutes, base):
    end = time.time() + minutes * 60
    n = 0
    while time.time() < end:
        n += 1
        ok, tg, ppl = step(f"soak#{n}", {"ppl": base["ppl"], "tg": base["tg"] * 0.98})
        if not ok:
            return False
    return True


def reset():
    for k in ("mem", "core"):
        sudo_set(k, 0)


def main(args):
    opt = {"--mem-max": 1500, "--mem-step": 100, "--core-max": 300, "--core-step": 15, "--soak": 10, "--margin": 1, "--power": 0}
    for i, a in enumerate(args):
        if a in opt:
            opt[a] = int(args[i + 1])
    if not (MODEL and os.path.exists(CORPUS) and os.path.exists(f"{BUILD}/bin/llama-bench")):
        sys.exit("missing model, ppl corpus or build-live")
    open(LOG, "w").close()
    log(f"START gpu push, results in {LOG}")
    # live view
    if "local-tune/watch.py" not in subprocess.run(["ps", "-C", "python3", "-o", "args="], capture_output=True, text=True).stdout:
        subprocess.run(["setsid", "-f", "konsole", "--workdir", ROOT, "-e", "bash", "-c", "while true; do python3 local-tune/watch.py; sleep 2; done"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if subprocess.run(["sudo", "-v"]).returncode:
        sys.exit("sudo is needed")
    subprocess.Popen(["bash", "-c", f"while kill -0 {os.getpid()} 2>/dev/null; do sudo -n true; sleep 50; done"])
    gw = "odysseus-gateway.service"
    if subprocess.run(["systemctl", "--user", "is-active", "--quiet", gw]).returncode == 0:
        subprocess.run(["systemctl", "--user", "stop", gw])
        subprocess.run(["pkill", "-x", "llama-server"])
        atexit.register(lambda: subprocess.run(["systemctl", "--user", "start", gw]))
    atexit.register(lambda: (reset(), log("offsets reset to 0")) if os.environ.get("KEEP") != "1" else None)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: sys.exit(1))
    subprocess.run(["sudo", "-n", "nvidia-smi", "-pm", "1"], capture_output=True)
    if sudo_set("mem", 0) != 0:
        sys.exit("the driver does not accept NVML offset calls here")
    resume = "--resume" in args
    old = json.load(open(STATE)) if resume else None
    if resume:
        open(STATE + f".{old['stamp']}", "w").write(json.dumps(old))
    if opt["--power"]:
        sudo_run(["nvidia-smi", "-pl", str(opt["--power"])])
        log(f"power limit set to {opt['--power']} W")
    log(f"power limit now {smi('power.limit')[0]:.0f} W")
    mem_use = core_use = 0
    if resume:
        mem_use, core_use = old["mem"], old["core"]
        log(f"RESUME from saved mem +{mem_use} core +{core_use} (stamp {old['stamp']}, perplexity {old['ppl']})")
        sudo_set("mem", mem_use)
        sudo_set("core", core_use)
        log("PHASE verify saved settings")
        base = {"ppl": old["ppl"], "tg": 0.0}
        for _ in range(12):
            res = [step(f"verify-m{mem_use}-c{core_use}", base) for _ in range(2)]
            if all(r[0] for r in res):
                base["tg"] = max(r[1] for r in res)
                break
            # the saved pair no longer holds (for example with the new power limit): step core down, then memory
            if core_use >= opt["--core-step"]:
                core_use -= opt["--core-step"]
            elif mem_use >= opt["--mem-step"]:
                mem_use -= opt["--mem-step"]
            else:
                sys.exit("even stock offsets fail the verify, stopping")
            sudo_set("mem", mem_use)
            sudo_set("core", core_use)
            base["tg"] = 0.0
        else:
            sys.exit("could not find a passing pair")
        start_mem, start_core = mem_use, core_use
    else:
        reset()
        log("PHASE baseline")
        log("baseline: three runs at stock clocks, they must agree on perplexity")
        res = [workload() for _ in range(3)]
        ppls = {r[2] for r in res}
        if len(ppls) != 1 or not all(r[0] for r in res):
            sys.exit(f"baseline is not stable or not deterministic ({ppls}, {[r[3] for r in res]}). Fix that first")
        base = {"ppl": ppls.pop(), "tg": max(r[1] for r in res)}
        log(f"TEST baseline PASS tg={base['tg']:.1f} t/s ppl={base['ppl']}")
        start_mem = start_core = 0
    ms, cs = opt["--mem-step"], opt["--core-step"]
    log("PHASE memory offset sweep")
    mem = sweep("mem", start_mem + ms, opt["--mem-max"], ms, base, good0=start_mem)
    # a new value gets the safety margin; no progress keeps the saved one (it already has its margin and a soak)
    mem_use = max(start_mem, mem - opt["--margin"] * ms) if mem > start_mem else start_mem
    sudo_set("mem", mem_use)
    log(f"memory: last good +{mem}, using +{mem_use}")
    log("PHASE core offset sweep")
    core = sweep("core", start_core + cs, opt["--core-max"], cs, base, good0=start_core)
    core_use = max(start_core, core - opt["--margin"] * cs) if core > start_core else start_core
    sudo_set("core", core_use)
    log(f"core: last good +{core}, using +{core_use}")
    log("PHASE soak test")
    passed = False
    while True:
        log(f"soak {opt['--soak']} min at mem +{mem_use} core +{core_use}")
        if soak(opt["--soak"], base):
            passed = True
            break
        if core_use >= cs:
            core_use -= cs
        elif mem_use >= ms:
            mem_use -= ms
        else:
            break
        sudo_set("mem", mem_use)
        sudo_set("core", core_use)
    if passed:
        json.dump({"mem": mem_use, "core": core_use, "stamp": STAMP, "tg": base["tg"], "ppl": base["ppl"], "power": int(smi("power.limit")[0])}, open(STATE, "w"))
        log(f"ALLDONE total={int(time.time() - T0)}s saved mem +{mem_use} core +{core_use} to {STATE}")
    else:
        log("ALLDONE the soak never passed, nothing saved, the old file is untouched")
    log("Offsets are active now and are lost at reboot. 'gpu-push.py apply' sets the saved ones again.")
    os.environ["KEEP"] = "1" if passed else "0"


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["_set"]:
        sys.exit(1 if set_offset(a[1], a[2]) else 0)
    if a[:1] == ["reset"]:
        reset()
    elif a[:1] == ["apply"]:
        s = json.load(open(STATE))
        s.get("power") and sudo_run(["nvidia-smi", "-pl", str(s["power"])])
        sudo_set("mem", s["mem"]), sudo_set("core", s["core"])
        print(f"applied mem +{s['mem']} core +{s['core']}")
    else:
        main(a[1:] if a[:1] == ["run"] else a)
