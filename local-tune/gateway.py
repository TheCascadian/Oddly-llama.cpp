#!/usr/bin/env python3
"""Local model gateway: OpenAI-compatible front for the GGUF models in models.conf.
Starts a local llama-server process on the first request for a model, evicts least-recently-used
models when VRAM is short, and unloads idle ones. Stdlib only.

  /v1/models, /v1/chat/completions, /v1/completions, /v1/embeddings   any model (by "model" field)
  /m/<name>/v1/...      same, with the model fixed (lets Odysseus keep one endpoint per model)
  /m/<name>/slots       context size per slot, from models.conf (answered without loading the model)
  /status               loaded models, VRAM        POST /load/<name>   POST /unload/<name>
  /conf                 models.conf as JSON with the slot layout of each model (what suite.py reads)

models.conf is the one list of models: this gateway, suite.py and assess.py all read it. Format in models.example.conf.
Settings (environment): GATEWAY_HOME (folder with models.conf and logs/, default: the folder this file is started from),
GATEWAY_CONF, GATEWAY_PORT, GATEWAY_BINDS (comma-separated addresses), LLAMA_SERVER.
"""
import http.client, json, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Not resolved: started through a symlink, the symlink's folder is the home, so the code can live in the repo.
HERE = Path(os.environ.get("GATEWAY_HOME") or Path(__file__).absolute().parent).expanduser()
# The build of this checkout (rebuild: local-tune/build.sh build-live). Override with LLAMA_SERVER.
BIN = os.environ.get("LLAMA_SERVER", str(Path(__file__).resolve().parents[1] / "build-live/bin/llama-server"))
LOGS = HERE / "logs"
PORT = int(os.environ.get("GATEWAY_PORT", 8700))
# localhost + docker0 (reachable from containers, not the LAN)
BINDS = os.environ.get("GATEWAY_BINDS", "127.0.0.1,172.17.0.1").split(",")
LOAD_TIMEOUT = 300
MARGIN_MB = 250

def load_conf():
    out = {}
    for line in Path(os.environ.get("GATEWAY_CONF") or HERE / "models.conf").expanduser().read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        n, path, args, typ, vram, port, idle, *jobs = [x.strip() for x in line.split("|")]
        out[n] = dict(name=n, path=os.path.expanduser(path), args=args, type=typ, vram=int(vram), port=int(port), idle=int(idle),
                      jobs=jobs[0].split() if jobs else [])   # jobs: what suite.py checks the model on, unused here
    return out

CONF = load_conf()
lock = threading.Condition()
state = {n: dict(last=0.0, inflight=0) for n in CONF}

def sh(*a, check=False):
    return subprocess.run(a, capture_output=True, text=True, check=check)

procs = {}   # name -> Popen of its llama-server

def running(n):
    p = procs.get(n)
    return p is not None and p.poll() is None

def healthy(n):
    try:
        c = http.client.HTTPConnection("127.0.0.1", CONF[n]["port"], timeout=3)
        c.request("GET", "/health"); ok = c.getresponse().status == 200; c.close(); return ok
    except Exception:
        return False

def free_mb():
    r = sh("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits")
    try: return int(r.stdout.strip().splitlines()[0])
    except Exception: return 10**6

def unload(n):
    p = procs.pop(n, None)
    if p is None or p.poll() is not None: return
    p.terminate()
    try: p.wait(10)
    except subprocess.TimeoutExpired: p.kill(); p.wait()

def log_tail(n):
    try: return "".join((LOGS / f"{n}.log").read_text(errors="replace").splitlines(True)[-8:])[-800:]
    except OSError: return ""

def start(n):
    m = CONF[n]; p = Path(m["path"])
    if not p.is_file():
        raise RuntimeError(f"model file missing: {p}")
    if not Path(BIN).is_file():
        raise RuntimeError(f"llama-server missing: {BIN}")
    unload(n)
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / f"{n}.log", "w") as lf:   # one log per model, replaced on every load
        procs[n] = subprocess.Popen([BIN, "-m", str(p), "--alias", n, "--host", "127.0.0.1", "--port", str(m["port"]), *m["args"].split()],
                                    stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < LOAD_TIMEOUT:
        if healthy(n): return
        if not running(n):
            raise RuntimeError("llama-server exited while loading:\n" + log_tail(n))
        time.sleep(0.5)
    unload(n)
    raise RuntimeError("load timeout")

def ensure(n):
    """Return once model n is serving; caller must hold no locks. Marks inflight."""
    if n not in CONF: raise KeyError(n)
    with lock:
        state[n]["inflight"] += 1
        try:
            if running(n) and healthy(n):
                state[n]["last"] = time.time(); return
            need = CONF[n]["vram"]
            deadline = time.time() + LOAD_TIMEOUT
            while need and free_mb() - MARGIN_MB < need:
                if running(n): break
                cands = [k for k in CONF if k != n and running(k) and CONF[k]["vram"] and state[k]["inflight"] == 0]
                if not cands:
                    if not any(running(k) for k in CONF if k != n):   # nothing of ours holds VRAM: waiting cannot help
                        raise RuntimeError(f"{n} is listed with {need} MiB but only {free_mb() - MARGIN_MB} MiB are free with nothing loaded; lower its vram field in models.conf or free the GPU")
                    if time.time() > deadline: raise RuntimeError("not enough free VRAM and nothing evictable")
                    lock.wait(2); continue
                v = min(cands, key=lambda k: (CONF[k]["idle"] == 0, state[k]["last"]))
                unload(v); time.sleep(2)
            start(n)
            state[n]["last"] = time.time()
        except BaseException:
            state[n]["inflight"] -= 1; raise

def release(n):
    with lock:
        state[n]["inflight"] -= 1; state[n]["last"] = time.time(); lock.notify_all()

def reaper():
    while True:
        time.sleep(30)
        with lock:
            for n, m in CONF.items():
                if m["idle"] and running(n) and state[n]["inflight"] == 0 and time.time() - state[n]["last"] > m["idle"]:
                    unload(n)

def status():
    with lock:
        loaded = [n for n in CONF if running(n)]
        return dict(free_vram_mb=free_mb(), loaded=loaded,
                    models={n: dict(type=m["type"], loaded=n in loaded, inflight=state[n]["inflight"], vram_est_mb=m["vram"]) for n, m in CONF.items()})

def slots(n):
    """What llama-server's /slots would say for model n: -c is split across --parallel slots unless -kvu shares one pool."""
    a = CONF[n]["args"].split()
    val = lambda *flags, d: next((int(a[i + 1]) for i, x in enumerate(a[:-1]) if x in flags), d)
    par = max(1, val("--parallel", "-np", d=1)); ctx = val("-c", "--ctx-size", d=4096)
    per = ctx if "-kvu" in a or "--kv-unified" in a else ctx // par
    return [dict(id=i, n_ctx=per, is_processing=False) for i in range(par)]

def describe():
    out = []
    for n, m in CONF.items():
        s = slots(n); a = m["args"].split()
        out.append(dict(m, par=len(s), ctx=s[0]["n_ctx"], shared="-kvu" in a or "--kv-unified" in a))
    return out

HOP = {"transfer-encoding", "connection", "keep-alive", "content-length"}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send_json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def model_list(self, only=None):
        return dict(object="list", data=[dict(id=n, object="model", owned_by="local", type=m["type"]) for n, m in CONF.items() if not only or n == only])

    def route(self):
        path = self.path.split("?")[0]
        fixed = None
        if path.startswith("/m/"):
            _, _, fixed, rest = path.split("/", 3); path = "/" + rest
            if fixed not in CONF: return self.send_json(404, {"error": f"unknown model {fixed}"})
        return fixed, path

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/status": return self.send_json(200, status())
        if p == "/conf": return self.send_json(200, describe())
        r = self.route()
        if r is None or not isinstance(r, tuple): return
        fixed, path = r
        if path in ("/v1/models", "/models"): return self.send_json(200, self.model_list(fixed))
        if path in ("/health", "/"): return self.send_json(200, {"status": "ok"})
        if path == "/slots" and fixed: return self.send_json(200, slots(fixed))
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        p = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n)
        try:
            if p.startswith("/load/") or p.startswith("/unload/"):
                act, name = p.strip("/").split("/", 1)
                if name not in CONF: return self.send_json(404, {"error": "unknown model"})
                if act == "load": ensure(name); release(name)
                else: unload(name)
                return self.send_json(200, status())
            r = self.route()
            if r is None or not isinstance(r, tuple): return
            fixed, path = r
            data = json.loads(body or b"{}")
            name = fixed or data.get("model")
            if name not in CONF: return self.send_json(404, {"error": f"unknown model '{name}'", "available": list(CONF)})
            data["model"] = name
            body = json.dumps(data).encode()
        except Exception as e:
            return self.send_json(400, {"error": str(e)})
        try:
            ensure(name)
        except Exception as e:
            return self.send_json(503, {"error": f"could not load {name}: {e}"})
        try:
            c = http.client.HTTPConnection("127.0.0.1", CONF[name]["port"], timeout=1800)
            c.request("POST", path, body, {"Content-Type": "application/json"})
            resp = c.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP: self.send_header(k, v)
            self.end_headers()
            while chunk := resp.read(4096):
                self.wfile.write(chunk); self.wfile.flush()
            c.close()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try: self.send_json(502, {"error": str(e)})
            except Exception: pass
        finally:
            release(name)

class S(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    for n in CONF:   # remove containers left over from the Docker version of this gateway
        try: sh("docker", "rm", "-f", f"llama-gw-{n}", f"llama-{n}")
        except OSError: pass
    threading.Thread(target=reaper, daemon=True).start()
    for b in BINDS:
        try:
            s = S((b, PORT), H); threading.Thread(target=s.serve_forever, daemon=True).start(); print("listening", b, PORT, flush=True)
        except OSError as e:
            print("bind failed", b, e, file=sys.stderr, flush=True)
    while True: time.sleep(3600)
