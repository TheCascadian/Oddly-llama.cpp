#!/usr/bin/env python3
"""Decode speed at context depth through the gateway, for llama-server settings that are not in models.conf yet.
served.py run <name>    runs results/compare-<name>.plan on a private gateway (port 8701) and writes compare-<name>.csv/.log
The files have the layout compare.py writes, so compare.py <name> and watch.py show the run.
Plan lines: "title: text", "args: -m <gguf> -d <depths> -p 0 -n <tokens> -r <repetitions> [--soak <cycles>]",
then one variant per line: "label | gateway | | llama-server args" (-m <gguf> there replaces the one in args).
Per depth: a chat whose prompt is that many tokens of results/ppl-corpus.txt, then -n tokens written; the value is the
server's own predicted_per_second. --soak: that many chats with a different prompt each, filling the whole context
(column tg@16k); a failed load or request is logged as ERROR. The real gateway and its models.conf are not touched."""
import json, os, re, statistics, subprocess, sys, tempfile, threading, time, urllib.request
import compare
from paths import rp, ROOT, SCRIPTS, RESULTS

HERE = os.path.dirname(os.path.abspath(__file__))  # scripts/
PORT = 8701


def call(url, body=None, timeout=1800):
    req = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def run(name):
    _, args, variants = compare.plan(name)
    opt = lambda f, d: (re.search(rf"(?:^| ){f} (\S+)", args) or [0, d])[1]
    gguf, depths, n, reps, soak = opt("-m", ""), [int(d) for d in opt("-d", "0").split(",")], int(opt("-n", "128")), int(opt("-r", "3")), int(opt("--soak", "0"))
    ctx = max((int(m[1]) for v in variants for m in [re.search(r"(?:^| )-c (\d+)", v[3])] if m), default=16384)
    corpus = open(rp("ppl-corpus.txt"), errors="replace").read()
    home = tempfile.mkdtemp(prefix="served-")
    names = [f"v{i}" for i in range(len(variants))]
    with open(os.path.join(home, "models.conf"), "w") as f:
        for i, (nm, v) in enumerate(zip(names, variants)):   # a variant may name its own gguf with -m
            own = re.search(r"(?:^| )-m (\S+)", v[3])
            f.write(f"{nm}|{os.path.abspath(own[1]) if own else gguf}|{v[3].replace(own[0], '') if own else v[3]}|llm|0|{8301 + i}|0\n")
    gw = subprocess.Popen([sys.executable, os.path.join(HERE, "gateway.py")], env=dict(os.environ, GATEWAY_HOME=home, GATEWAY_PORT=str(PORT), GATEWAY_BINDS="127.0.0.1"))
    url = f"http://127.0.0.1:{PORT}"
    out = open(rp(f"compare-{name}.csv"), "w")
    log = open(rp(f"compare-{name}.log"), "w")
    def say(msg):
        log.write(f"{time.strftime('%T')} {msg}\n"); log.flush()
    def chat(nm, text, max_tokens):
        j = call(f"{url}/m/{nm}/v1/chat/completions", dict(model=nm, max_tokens=max_tokens, temperature=0,
                 messages=[{"role": "user", "content": text + "\n\nContinue the text above in the same style."}]))
        return j["timings"]["predicted_per_second"], j["usage"]["prompt_tokens"], j["timings"]["predicted_n"]
    t0 = time.time(); time.sleep(1)
    try:
        for i, (nm, v) in enumerate(zip(names, variants)):
            label = v[0]; say(f"START {label}"); t1 = time.time()
            stop = threading.Event()
            def sample():
                while not stop.wait(2):
                    g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
                    m = re.search(r"MemAvailable:\s+(\d+)", open("/proc/meminfo").read())
                    say(f"  vram={g} MiB ram_avail={int(m[1]) // 1024}MB")
            threading.Thread(target=sample, daemon=True).start()
            try:
                call(f"{url}/load/{nm}", {})
                srv = f"http://127.0.0.1:{8301 + i}"
                toks = call(srv + "/tokenize", dict(content=corpus[:300000]))["tokens"]
                piece = lambda start, k: call(srv + "/detokenize", dict(tokens=toks[start:start + k]))["content"] if k else ""
                for d in depths:
                    tp = time.time(); text = piece(0, max(d - 40, 0)); vals = []
                    for _ in range(reps):
                        tps, pt, got = chat(nm, text, n); vals.append(tps)
                    out.write(f'"{label}",0,{got},{d},{pt},0,{reps},{statistics.mean(vals):.4f},{statistics.pstdev(vals):.4f}\n'); out.flush()
                    say(f"  TEST tg@{d} prompt_tokens={pt} took={time.time() - tp:.0f}s")
                if soak:
                    tp = time.time(); vals = []
                    for c in range(soak):
                        tps, pt, got = chat(nm, piece(2000 + c * 9000, ctx - 300), 240); vals.append(tps)
                        say(f"  SOAK cycle {c + 1}/{soak} prompt_tokens={pt} wrote={got}")
                    out.write(f'"{label}",0,240,{ctx},{pt},0,{soak},{statistics.mean(vals):.4f},{statistics.pstdev(vals):.4f}\n'); out.flush()
                    say(f"  TEST tg@{ctx} took={time.time() - tp:.0f}s")
            except Exception as e:
                body = e.read().decode(errors="replace") if hasattr(e, "read") else ""
                tail = open(os.path.join(home, "logs", nm + ".log"), errors="replace").read()[-3000:] if os.path.exists(os.path.join(home, "logs", nm + ".log")) else ""
                say("ERROR " + ("out of memory " if "out of memory" in (body + tail).lower() else "") + (str(e) + " " + body).replace("\n", " ")[:160])
            try: call(f"{url}/unload/{nm}", {})
            except Exception: pass
            time.sleep(3); stop.set()
            say(f"DONE {label} took={time.time() - t1:.0f}s")
    finally:
        gw.terminate()
    say(f"ALLDONE total={time.time() - t0:.0f}s")


if __name__ == "__main__":
    run(sys.argv[2])
