#!/usr/bin/env python3
"""gpumon - btop/htop-style monitor for the GPU and the local-tune model gateway (NVIDIA, stdlib only).

Pages (1/2/3 or Tab):
  1 GPU      utilization, VRAM, sensors, GPU processes (llama-server rows are labelled with their model)
  2 Gateway  gateway process, clients, every model in models.conf (loaded state, pid, VRAM, RAM, context, slots, speed),
             per-slot view of what each loaded model is doing right now (active chats, context fill)
  3 Speed    tokens/s (generation + prompt), time to first token, per-model sparklines, recent requests

Data: nvidia-smi -q -x, the gateway's /status and /conf, each llama-server's /slots, and the per-model logs in
<gateway home>/logs (print_timing lines give tokens/s, prompt eval time ~ TTFT, context used at slot release).
Options: gpumon.py [--gateway http://127.0.0.1:8700] [--home <gateway home with logs/>]
Keys: q quit | 1 2 3 Tab page | m/p/n sort GPU procs | r reverse | +/- refresh | space pause
"""
import collections, curses, http.client, json, os, pwd, re, subprocess, sys, threading, time
import xml.etree.ElementTree as ET

HIST = 300
CLK = os.sysconf("SC_CLK_TCK")
BLOCKS = " ▁▂▃▄▅▆▇█"
GW_URL = "http://127.0.0.1:8700"
GW_HOME = None
PAGES = ["GPU", "Gateway", "Speed"]


# ---------------------------------------------------------------- helpers
def num(s):
    try:
        return float(str(s).split()[0])
    except Exception:
        return None


def http_json(port, path, timeout=1.5, host="127.0.0.1"):
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        c.request("GET", path)
        r = c.getresponse()
        return json.loads(r.read())
    finally:
        c.close()


def sh(*a, timeout=4):
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def fmt_t(s):
    s = int(s)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return (f"{d}d " if d else "") + f"{h:02d}:{m:02d}:{s:02d}"


def mb(v):
    return "N/A" if v is None else (f"{v/1024:.2f}G" if v >= 1024 else f"{v:.0f}M")


def tok(v):
    return "-" if v is None else (f"{v/1000:.1f}k" if v >= 1000 else f"{v:.0f}")


def ms(v):
    return "-" if v is None else (f"{v/1000:.2f}s" if v >= 1000 else f"{v:.0f}ms")


def rate(v):
    return "-" if v is None else f"{v:.1f}"


def color_for(pct):
    return 2 if pct < 50 else (3 if pct < 80 else 4)


def ago(t):
    d = max(0, time.time() - t)
    return f"{d:.0f}s" if d < 90 else (f"{d/60:.0f}m" if d < 5400 else f"{d/3600:.1f}h")


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


_uptime_cache = [0, 0]
def uptime():
    if time.time() - _uptime_cache[0] > 0.5:
        with open("/proc/uptime") as f:
            _uptime_cache[1] = float(f.read().split()[0])
        _uptime_cache[0] = time.time()
    return _uptime_cache[1]


_pc = {}
def procinfo(pid):
    """(user, runtime_seconds, cmdline, start_epoch, rss_mb) from /proc."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            st = f.read().rsplit(")", 1)[1].split()
        start = int(st[19]) / CLK
        run = max(0, uptime() - start)
        rss = int(st[21]) * os.sysconf("SC_PAGE_SIZE") / 1048576
        if pid not in _pc:
            uid = os.stat(f"/proc/{pid}").st_uid
            try:
                user = pwd.getpwuid(uid).pw_name
            except KeyError:
                user = str(uid)
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
            _pc[pid] = (user, cmd)
        return _pc[pid][0], run, _pc[pid][1], time.time() - run, rss
    except Exception:
        return "?", 0, "", time.time(), 0


# ---------------------------------------------------------------- GPU
def query_gpus():
    out = sh("nvidia-smi", "-q", "-x", timeout=5)
    root = ET.fromstring(out)
    gpus = []
    for g in root.findall("gpu"):
        t = lambda p: (g.findtext(p) or "N/A").strip()
        procs = []
        for p in g.findall("processes/process_info"):
            procs.append(dict(pid=int(p.findtext("pid")), type=(p.findtext("type") or "?").strip(),
                              name=(p.findtext("process_name") or "?").strip(), mem=num(p.findtext("used_memory") or "") or 0))
        gpus.append(dict(
            name=t("product_name"), driver=root.findtext("driver_version"), cuda=root.findtext("cuda_version"),
            util=num(t("utilization/gpu_util")), mutil=num(t("utilization/memory_util")),
            enc=num(t("utilization/encoder_util")), dec=num(t("utilization/decoder_util")),
            used=num(t("fb_memory_usage/used")), total=num(t("fb_memory_usage/total")),
            temp=num(t("temperature/gpu_temp")), tlim=num(t("temperature/gpu_temp_slow_threshold")),
            pw=num(t("gpu_power_readings/power_draw")) or num(t("gpu_power_readings/instant_power_draw")),
            plim=num(t("gpu_power_readings/current_power_limit")),
            sm=num(t("clocks/sm_clock")), msm=num(t("max_clocks/sm_clock")),
            mem=num(t("clocks/mem_clock")), mmem=num(t("max_clocks/mem_clock")),
            fan=num(t("fan_speed")), pstate=t("performance_state"),
            gen=t("pci/pci_gpu_link_info/pcie_gen/current_link_gen"), width=t("pci/pci_gpu_link_info/link_widths/current_link_width"),
            procs=procs))
    return gpus


# ---------------------------------------------------------------- llama-server log tail
TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")
RX_ID = re.compile(r"id\s+(\d+) \| task (-?\d+) \| (.*)")
RX_PE = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens .*?([\d.]+) tokens per second")
RX_EV = re.compile(r"^\s*eval time =\s*([\d.]+) ms /\s*(\d+) tokens .*?([\d.]+) tokens per second")
RX_TOT = re.compile(r"total time =\s*([\d.]+) ms /\s*(\d+) tokens")
RX_PP = re.compile(r"prompt processing, n_tokens =\s*(\d+), progress = ([\d.]+)")
RX_TG = re.compile(r"n_gen =\s*(\d+), tg =\s*([\d.]+) t/s")
RX_REL = re.compile(r"stop processing: n_tokens = (\d+)")


class LogTail:
    """Incremental reader of one model's llama-server log; the gateway replaces the file on every load."""
    def __init__(self, path):
        self.path, self.ino, self.pos = path, None, 0
        self.reqs = collections.deque(maxlen=300)   # finished requests
        self.pend = {}                              # task -> partial record
        self.live = {}                              # slot -> dict(phase, ...) while a slot is working
        self.slot_ctx = {}                          # slot -> n_tokens at last release (the chat held in that slot)
        self.slot_last = {}                         # slot -> wall time of last release
        self.t0 = None
        self.loads = 0

    def update(self, t0):
        try:
            st = os.stat(self.path)
        except OSError:
            return
        if st.st_ino != self.ino or st.st_size < self.pos:   # new load: start over
            self.__init__(self.path)
            self.ino = st.st_ino
            self.loads += 1
            self.pos = max(0, st.st_size - 4_000_000) if self.loads == 1 else 0
        self.t0 = t0
        if st.st_size == self.pos:
            return
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read(8_000_000)
        cut = data.rfind(b"\n") + 1
        self.pos += cut
        for line in data[:cut].decode(errors="replace").splitlines():
            self.feed(line)

    def feed(self, line):
        m = TS.match(line)
        if not m:
            return
        ts = int(m[1]) * 60 + int(m[2]) + int(m[3]) / 1e3 + int(m[4]) / 1e6
        wall = (self.t0 or time.time()) + ts
        m = RX_ID.search(line)
        if not m:
            return
        slot, task, rest = int(m[1]), int(m[2]), m[3]
        if task < 0:
            return
        p = self.pend.setdefault(task, dict(task=task, slot=slot, start=wall, pe_ms=None, pe_n=None, pp=None, ev_ms=None, ev_n=None, tg=None, tot_ms=None, tot_n=None))
        if "processing task" in rest:
            p["start"] = wall
            self.live[slot] = dict(task=task, phase="queued", since=wall)
        elif (m := RX_PP.search(rest)):
            self.live[slot] = dict(task=task, phase="prompt", n=int(m[1]), prog=float(m[2]), since=self.live.get(slot, {}).get("since", wall))
        elif (m := RX_TG.search(rest)):
            self.live[slot] = dict(task=task, phase="gen", n=int(m[1]), tg=float(m[2]), since=self.live.get(slot, {}).get("since", wall))
        elif (m := RX_PE.search(rest)):
            p["pe_ms"], p["pe_n"], p["pp"] = float(m[1]), int(m[2]), float(m[3])
        elif (m := RX_EV.search(rest)):
            p["ev_ms"], p["ev_n"], p["tg"] = float(m[1]), int(m[2]), float(m[3])
        elif (m := RX_TOT.search(rest)):
            p["tot_ms"], p["tot_n"] = float(m[1]), int(m[2])
            p["end"] = wall
        elif (m := RX_REL.search(rest)):
            self.slot_ctx[slot] = int(m[1])
            self.slot_last[slot] = wall
            self.live.pop(slot, None)
            p = self.pend.pop(task, None)
            if p and p.get("tot_ms") is not None:
                self.reqs.append(p)
        if len(self.pend) > 200:
            self.pend.pop(next(iter(self.pend)))


# ---------------------------------------------------------------- gateway + models collector
class Collector(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.snap = {}
        self.dt = 1.0
        self.paused = False
        self.tails = {}
        self.gwport = int(GW_URL.rsplit(":", 1)[1].split("/")[0])
        self.hist = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.deque(maxlen=HIST)))

    def listeners(self):
        """port -> pid for every listening TCP socket."""
        out = {}
        for line in sh("ss", "-ltnpH").splitlines():
            f = line.split()
            m = re.search(r"pid=(\d+)", line)
            if len(f) >= 4 and m:
                out[int(f[3].rsplit(":", 1)[1])] = int(m[1])
        return out

    def gateway(self, ports):
        gw = dict(url=GW_URL, up=False, pid=ports.get(self.gwport))
        t = time.time()
        try:
            st = http_json(self.gwport, "/status", timeout=3)
            gw["latency"] = (time.time() - t) * 1000
            gw["up"] = True
            gw["status"] = st
            gw["conf"] = {c["name"]: c for c in http_json(self.gwport, "/conf", timeout=3)}
        except Exception as e:
            gw["err"] = str(e)
        if gw["pid"]:
            u, run, cmd, start, rss = procinfo(gw["pid"])
            gw.update(user=u, run=run, cmd=cmd, rss=rss)
            home = GW_HOME
            if not home:
                try:
                    env = open(f"/proc/{gw['pid']}/environ", "rb").read().split(b"\0")
                    home = next((e[12:].decode() for e in env if e.startswith(b"GATEWAY_HOME=")), None)
                except OSError:
                    pass
                if not home:
                    script = next((a for a in cmd.split() if a.endswith("gateway.py")), "")
                    home = os.path.dirname(os.path.abspath(script)) if script else None
                    try:
                        home = os.readlink(f"/proc/{gw['pid']}/cwd") if not home or not os.path.isdir(home + "/logs") else home
                    except OSError:
                        pass
            gw["home"] = home
        # connections into the gateway: who is talking to it
        peers = collections.Counter()
        for line in sh("ss", "-tnH", "state", "established", f"( sport = :{self.gwport} )").splitlines():
            f = line.split()
            if len(f) >= 4:
                peers[f[3].rsplit(":", 1)[0]] += 1
        gw["peers"] = peers
        return gw

    def agents(self):
        """Processes and containers that look like clients of the gateway (agent swarm, Odysseus, ...)."""
        out = []
        pat = re.compile(r"swarm_server|odysseus|opencode|aider|openhands|agent_loop|hermes|goose", re.I)
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if pat.search(cmd) and "gpumon" not in cmd and "llama-server" not in cmd and "polkit" not in cmd and "gateway.py" not in cmd and "docker-proxy" not in cmd:
                u, run, c, _, rss = procinfo(int(d))
                out.append(dict(pid=int(d), user=u, run=run, cmd=c, rss=rss))
        return out

    def containers(self):
        rows = []
        for line in sh("docker", "ps", "--format", "{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}", timeout=3).splitlines():
            f = line.split("|")
            if len(f) >= 3:
                rows.append(f)
        return rows

    def models(self, gw, ports, gpus):
        st, conf = gw.get("status"), gw.get("conf")
        if not st:
            return []
        gpumem = {p["pid"]: p["mem"] for g in gpus for p in g["procs"]}
        home = gw.get("home")
        rows = []
        for n, c in conf.items():
            s = st["models"].get(n, {})
            r = dict(name=n, type=c["type"], loaded=s.get("loaded", False), inflight=s.get("inflight", 0), est=c["vram"], port=c["port"],
                     idle=c["idle"], ctx=c["ctx"], par=c["par"], shared=c["shared"], args=c["args"], jobs=c.get("jobs", []))
            a = c["args"].split()
            val = lambda *fl: next((a[i + 1] for i, x in enumerate(a[:-1]) if x in fl), None)
            r["ngl"], r["ctk"], r["ctv"] = val("-ngl", "--n-gpu-layers"), val("-ctk", "--cache-type-k"), val("-ctv", "--cache-type-v")
            r["fa"] = val("-fa", "--flash-attn")
            r["spec"] = val("--spec-type")
            r["size_gb"] = os.path.getsize(c["path"]) / 1e9 if os.path.isfile(c["path"]) else None
            r["gguf"] = os.path.basename(c["path"])
            if r["loaded"]:
                pid = ports.get(c["port"])
                r["pid"] = pid
                if pid:
                    u, run, cmd, start, rss = procinfo(pid)
                    r.update(run=run, start=start, rss=rss, vram=gpumem.get(pid))
                    slots = []
                    try:
                        slots = http_json(c["port"], "/slots", timeout=1.0)
                    except Exception:
                        pass
                    r["slots"] = slots if isinstance(slots, list) else []
                    if home:
                        tl = self.tails.setdefault(n, LogTail(os.path.join(home, "logs", f"{n}.log")))
                        tl.update(start)
                        r["tail"] = tl
            rows.append(r)
        return rows

    def run(self):
        while True:
            if self.paused:
                time.sleep(0.2)
                continue
            t0 = time.time()
            snap = dict(ts=t0)
            try:
                snap["gpus"] = query_gpus()
            except Exception as e:
                snap["gpus"] = []
                snap["gpu_err"] = str(e)
            try:
                ports = self.listeners()
                gw = self.gateway(ports)
                snap["gw"] = gw
                snap["models"] = self.models(gw, ports, snap["gpus"])
                snap["agents"] = self.agents()
                snap["containers"] = self.containers() if int(t0) % 3 == 0 or "containers" not in self.snap else self.snap["containers"]
            except Exception as e:
                snap["gw"] = dict(up=False, err=f"{type(e).__name__}: {e}", url=GW_URL)
                snap["models"], snap["agents"], snap["containers"] = [], [], []
            # histories
            for gi, g in enumerate(snap["gpus"]):
                for k in ("util", "mutil", "used", "temp", "pw", "sm"):
                    self.hist[f"gpu{gi}"][k].append(g[k] or 0)
            for m in snap["models"]:
                tl = m.get("tail")
                live = [l["tg"] for l in tl.live.values() if l.get("phase") == "gen"] if tl else []
                self.hist["m:" + m["name"]]["live_tg"].append(sum(live) if live else 0)
                busy = sum(1 for s in m.get("slots", []) if s.get("is_processing"))
                self.hist["m:" + m["name"]]["busy"].append(busy)
            snap["hist"] = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in self.hist.items()}
            with self.lock:
                self.snap = snap
            time.sleep(max(0.05, self.dt - (time.time() - t0)))


# ---------------------------------------------------------------- UI
class UI:
    def __init__(s, scr, col):
        s.scr, s.col = scr, col
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        for i, c in enumerate([curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED,
                               curses.COLOR_MAGENTA, curses.COLOR_BLUE], 1):
            curses.init_pair(i, c, -1)
        curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_GREEN)
        curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_CYAN)
        s.sort, s.rev, s.page = "mem", True, 0

    def put(s, y, x, txt, attr=0):
        H, W = s.scr.getmaxyx()
        if 0 <= y < H and 0 <= x < W:
            try:
                s.scr.addstr(y, x, txt[: W - x - (1 if y == H - 1 else 0)], attr)
            except curses.error:
                pass

    def C(s, n, *extra):
        a = curses.color_pair(n)
        for e in extra:
            a |= e
        return a

    def box(s, y, x, h, w, title=""):
        a = s.C(6)
        s.put(y, x, "╭" + "─" * (w - 2) + "╮", a)
        for i in range(1, h - 1):
            s.put(y + i, x, "│", a)
            s.put(y + i, x + w - 1, "│", a)
        s.put(y + h - 1, x, "╰" + "─" * (w - 2) + "╯", a)
        if title:
            s.put(y, x + 2, "┤", a)
            s.put(y, x + 3, title, curses.A_BOLD)
            s.put(y, x + 3 + len(title), "├", a)

    def bar(s, y, x, w, pct, label="", col=None):
        pct = max(0, min(100, pct or 0))
        n = int(w * pct / 100)
        s.put(y, x, "■" * n, s.C(col or color_for(pct), curses.A_BOLD))
        s.put(y, x + n, "·" * (w - n), curses.A_DIM)
        if label:
            s.put(y, x + w + 1, label)

    def spark(s, y, x, w, series, maxv=None, col=2):
        data = list(series)[-w:]
        data = [0] * (w - len(data)) + data
        mx = maxv or max(max(data), 1e-9)
        s.put(y, x, "".join(BLOCKS[max(0, min(8, int(round(min(1, (v or 0) / mx) * 8))))] for v in data), s.C(col))

    def graph(s, y, x, w, h, series, maxv, col):
        data = list(series)[-w:]
        data = [0] * (w - len(data)) + data
        for r in range(h):
            lo = (h - 1 - r) / h
            row = ""
            for v in data:
                f = max(0, min(1, (v or 0) / maxv)) if maxv else 0
                row += BLOCKS[max(0, min(8, int(round((f - lo) * h * 8))))]
            s.put(y + r, x, row, s.C(col))

    def table(s, y, cols, rows, W, maxrows, colr=None):
        """cols: [(title, width, align)]; rows: list of lists of str (+ optional per-row attr via colr(row_index))."""
        x, line = 2, ""
        for t, w, al in cols:
            line += (f"{t:>{w}}" if al == ">" else f"{t:<{w}}") + " "
        s.put(y, x, line.ljust(W - 4), s.C(7))
        for i, row in enumerate(rows[:maxrows]):
            line = ""
            for (t, w, al), v in zip(cols, row):
                v = str(v)
                v = v[: w] if len(v) > w else v
                line += (f"{v:>{w}}" if al == ">" else f"{v:<{w}}") + " "
            s.put(y + 1 + i, x, line, colr(i) if colr else 0)
        return y + 1 + min(len(rows), maxrows)

    # ---- chrome
    def tabs(s, W, snap):
        x = 0
        s.put(0, x, " gpumon ", s.C(7, curses.A_BOLD))
        x += 9
        for i, p in enumerate(PAGES):
            lab = f" {i+1}:{p} "
            s.put(0, x, lab, s.C(8, curses.A_BOLD) if i == s.page else curses.A_DIM)
            x += len(lab) + 1
        gw = snap.get("gw", {})
        ok = gw.get("up")
        s.put(0, x + 1, "gateway " + (f"UP {gw['latency']:.0f}ms" if ok else "DOWN"), s.C(2 if ok else 4, curses.A_BOLD))
        g = (snap.get("gpus") or [None])[0]
        if g:
            s.put(0, x + 18, f"{g['name']} · driver {g['driver']} · cuda {g['cuda']} · {g['pstate']} · PCIe gen{g['gen']} x{g['width']}", curses.A_DIM)

    def footer(s, H, W):
        s.put(H - 1, 0, (" q quit  1/2/3/Tab page  m/p/n sort  r reverse  +/- rate  space pause "
                         f"| {1/s.col.dt:.1f}Hz{' PAUSED' if s.col.paused else ''} ").ljust(W - 1), s.C(7))

    def vram_line(s, y, W, g):
        vp = 100 * (g["used"] or 0) / g["total"] if g["total"] else 0
        s.put(y, 1, "VRAM", curses.A_BOLD)
        s.bar(y, 6, max(10, W // 2 - 30), vp, f"{mb(g['used'])}/{mb(g['total'])} {vp:3.0f}%")
        x = 6 + max(10, W // 2 - 30) + 24
        s.put(y, x, "GPU", curses.A_BOLD)
        s.bar(y, x + 4, 14, g["util"], f"{(g['util'] or 0):3.0f}%")
        s.put(y, x + 24, f"{(g['temp'] or 0):.0f}°C  {(g['pw'] or 0):.0f}/{(g['plim'] or 0):.0f}W  sm {(g['sm'] or 0):.0f}MHz", curses.A_DIM)

    # ---- page 1: GPU
    def page_gpu(s, snap, y, H, W):
        for gi, g in enumerate(snap["gpus"]):
            hh = snap["hist"].get(f"gpu{gi}", {})
            vp = 100 * (g["used"] or 0) / g["total"] if g["total"] else 0
            gh = 6
            lw = W // 2
            rw = W - lw
            s.box(y, 0, gh + 4, lw, f"GPU{gi} core")
            iw = lw - 4
            bw = iw - 14
            s.put(y + 1, 2, "GPU  ", curses.A_BOLD); s.bar(y + 1, 7, bw, g["util"], f"{(g['util'] or 0):3.0f}%")
            s.put(y + 2, 2, "MCtl ", curses.A_BOLD); s.bar(y + 2, 7, bw, g["mutil"], f"{(g['mutil'] or 0):3.0f}%")
            s.graph(y + 3, 2, iw, gh, hh.get("util", []), 100, 1)
            s.put(y + 3 + gh, 2, "util %", curses.A_DIM)
            s.box(y, lw, gh + 4, rw, "VRAM")
            iw2, bw2 = rw - 4, rw - 4 - 22
            s.put(y + 1, lw + 2, "VRAM ", curses.A_BOLD)
            s.bar(y + 1, lw + 7, bw2, vp, f"{mb(g['used'])}/{mb(g['total'])} {vp:3.0f}%")
            ptot = sum(p["mem"] for p in g["procs"])
            s.put(y + 2, lw + 2, f"free {mb((g['total'] or 0) - (g['used'] or 0))}   by processes {mb(ptot)}", curses.A_DIM)
            s.graph(y + 3, lw + 2, iw2, gh, hh.get("used", []), g["total"] or 1, 5)
            s.put(y + 3 + gh, lw + 2, "VRAM used over time", curses.A_DIM)
            y += gh + 4
            s.box(y, 0, 5, W, "sensors")
            cols = [("Temp", f"{(g['temp'] or 0):.0f}°C", color_for(100 * (g['temp'] or 0) / (g['tlim'] or 90))),
                    ("Power", f"{(g['pw'] or 0):.0f}/{(g['plim'] or 0):.0f}W", color_for(100 * (g['pw'] or 0) / (g['plim'] or 1))),
                    ("Fan", "N/A" if g["fan"] is None else f"{g['fan']:.0f}%", 2),
                    ("SM clk", f"{(g['sm'] or 0):.0f}/{(g['msm'] or 0):.0f}MHz", 1),
                    ("Mem clk", f"{(g['mem'] or 0):.0f}/{(g['mmem'] or 0):.0f}MHz", 1),
                    ("Enc/Dec", f"{(g['enc'] or 0):.0f}/{(g['dec'] or 0):.0f}%", 1)]
            cw = (W - 4) // len(cols)
            for i, (k, v, c) in enumerate(cols):
                s.put(y + 1, 2 + i * cw, k, curses.A_DIM)
                s.put(y + 2, 2 + i * cw, v, s.C(c, curses.A_BOLD))
            for i, key, mx in ((0, "temp", g["tlim"] or 100), (1, "pw", g["plim"] or 100), (3, "sm", g["msm"] or 2000)):
                s.spark(y + 3, 2 + i * cw, min(cw - 2, 14), hh.get(key, []), mx, 2)
            y += 5
            names = {m["pid"]: m for m in snap["models"] if m.get("pid")}
            procs = []
            for p in g["procs"]:
                u, run, cmd, _, _ = procinfo(p["pid"])
                m = names.get(p["pid"])
                label = f"llama-server ▸ {m['name']} :{m['port']}  [{m['type']}]" if m else (cmd or p["name"])
                procs.append(dict(p, user=u, run=run, cmd=label, model=bool(m)))
            keyf = {"mem": lambda p: p["mem"], "pid": lambda p: p["pid"], "run": lambda p: p["run"]}[s.sort]
            procs.sort(key=keyf, reverse=s.rev)
            avail = H - y - 2
            if avail < 3:
                continue
            nrows = min(avail, len(procs) + 4)
            s.box(y, 0, nrows, W, f"GPU processes ({len(procs)}) · sort:{s.sort}{'↓' if s.rev else '↑'}")
            s.put(y + 1, 2, f"{'PID':>8} {'USER':<10} {'T':<3} {'VRAM':>8} {'VRAM%':>6} {'':<10} {'RUNTIME':>12}  COMMAND".ljust(W - 4), s.C(7))
            for i, p in enumerate(procs[: nrows - 3]):
                pct = 100 * p["mem"] / g["total"] if g["total"] else 0
                yy = y + 2 + i
                s.put(yy, 2, f"{p['pid']:>8} {p['user'][:10]:<10} {p['type'][:3]:<3} {mb(p['mem']):>8} {pct:5.1f}% ")
                s.bar(yy, 42, 9, pct)
                s.put(yy, 51, f"{fmt_t(p['run']):>12}  ")
                s.put(yy, 65, p["cmd"], s.C(5 if p["model"] else 1, curses.A_BOLD if p["model"] else 0))
            if not procs:
                s.put(y + 2, 2, "no processes using the GPU", curses.A_DIM)
            y += nrows

    # ---- shared per-model stats
    @staticmethod
    def mstats(m, window=120):
        tl = m.get("tail")
        if not tl:
            return {}
        now = time.time()
        rs = [r for r in tl.reqs if now - r.get("end", 0) < window and r.get("ev_n")]
        allr = [r for r in tl.reqs if r.get("ev_n")]
        last = tl.reqs[-1] if tl.reqs else None
        live = [l["tg"] for l in tl.live.values() if l.get("phase") == "gen"]
        return dict(tg=avg([r["tg"] for r in rs]), pp=avg([r["pp"] for r in rs if r["pe_n"] and r["pe_n"] > 32]),
                    ttft=avg([r["pe_ms"] for r in rs]), n=len(tl.reqs), nrecent=len(rs), last=last,
                    tg_all=avg([r["tg"] for r in allr]), live_tg=sum(live) if live else None,
                    gen_tokens=sum(r["ev_n"] or 0 for r in tl.reqs), prompt_tokens=sum(r["pe_n"] or 0 for r in tl.reqs))

    # ---- page 2: gateway
    def page_gateway(s, snap, y, H, W):
        gw, models = snap["gw"], snap["models"]
        if snap["gpus"]:
            s.vram_line(y, W, snap["gpus"][0]); y += 1
        if not gw.get("up"):
            s.box(y, 0, 4, W, "gateway")
            s.put(y + 1, 2, f"gateway {gw.get('url')} is not answering: {gw.get('err','')}", s.C(4, curses.A_BOLD))
            return
        st = gw["status"]
        loaded = [m for m in models if m["loaded"]]
        infl = sum(m["inflight"] for m in models)
        busy = sum(1 for m in loaded for sl in m.get("slots", []) if sl.get("is_processing"))
        slots_total = sum(len(m.get("slots", [])) for m in loaded)
        conf_est = sum(m["est"] for m in loaded)
        s.box(y, 0, 8, W, "gateway")
        script = next((a for a in gw.get("cmd", "").split() if a.endswith(".py")), "")
        l1 = [("URL", gw["url"]), ("PID", gw.get("pid") or "?"), ("user", gw.get("user", "?")), ("uptime", fmt_t(gw.get("run", 0))),
              ("RSS", mb(gw.get("rss"))), ("latency", f"{gw['latency']:.0f}ms")]
        x = 2
        for k, v in l1:
            s.put(y + 1, x, k + " ", curses.A_DIM); s.put(y + 1, x + len(k) + 1, str(v), curses.A_BOLD)
            x += len(k) + len(str(v)) + 4
        s.put(y + 2, 2, f"script {script}   home {gw.get('home')}", curses.A_DIM)
        s.put(y + 3, 2, "models", curses.A_DIM); s.put(y + 3, 9, f"{len(loaded)} loaded / {len(models)} configured", curses.A_BOLD)
        s.put(y + 3, 40, "in-flight", curses.A_DIM); s.put(y + 3, 50, str(infl), s.C(3 if infl else 2, curses.A_BOLD))
        s.put(y + 3, 56, "busy slots", curses.A_DIM); s.put(y + 3, 67, f"{busy}/{slots_total}", s.C(3 if busy else 2, curses.A_BOLD))
        s.put(y + 3, 78, "free VRAM (gateway view)", curses.A_DIM); s.put(y + 3, 103, mb(st["free_vram_mb"]), curses.A_BOLD)
        s.put(y + 4, 2, "est. VRAM of loaded", curses.A_DIM); s.put(y + 4, 22, mb(conf_est), curses.A_BOLD)
        tot_tg = sum((s.mstats(m).get("live_tg") or 0) for m in loaded)
        s.put(y + 4, 40, "live generation", curses.A_DIM); s.put(y + 4, 56, f"{tot_tg:.1f} tok/s", s.C(2, curses.A_BOLD))
        peers = ", ".join(f"{a}×{n}" for a, n in gw["peers"].most_common(5)) or "none"
        s.put(y + 5, 2, "clients connected", curses.A_DIM); s.put(y + 5, 20, peers)
        ag = snap["agents"]
        s.put(y + 6, 2, "agents / clients", curses.A_DIM)
        desc = [f"{os.path.basename(a['cmd'].split()[1] if len(a['cmd'].split()) > 1 and 'python' in a['cmd'].split()[0] else a['cmd'].split()[0] if a['cmd'] else '?')}#{a['pid']} ({fmt_t(a['run'])})" for a in ag]
        desc += [f"docker:{c[0]} ({c[2]})" for c in snap["containers"]]
        s.put(y + 6, 19, "  ".join(desc) or "none detected", s.C(5))
        y += 8
        # models table
        n = len(models)
        s.box(y, 0, min(H - y - 1, n + 3), W, f"models ({len(loaded)}/{n} in memory)")
        cols = [("MODEL", 22, "<"), ("TYPE", 10, "<"), ("STATE", 8, "<"), ("PID", 7, ">"), ("PORT", 5, ">"), ("VRAM", 7, ">"), ("EST", 6, ">"),
                ("RAM", 7, ">"), ("CTX/slot", 8, ">"), ("SLOTS", 6, ">"), ("INFL", 4, ">"), ("TG t/s", 7, ">"), ("PP t/s", 7, ">"), ("TTFT", 7, ">"),
                ("UP", 10, ">"), ("IDLE", 5, ">")]
        order = sorted(models, key=lambda m: (not m["loaded"], -(m.get("vram") or 0), m["name"]))
        rows = []
        for m in order:
            ms_ = s.mstats(m)
            sl = m.get("slots", [])
            b = sum(1 for x in sl if x.get("is_processing"))
            state = "BUSY" if b or m["inflight"] else ("loaded" if m["loaded"] else "-")
            tg = ms_.get("live_tg") or ms_.get("tg")
            rows.append([m["name"], m["type"], state, m.get("pid", "-"), m["port"], mb(m.get("vram")) if m["loaded"] else "-", mb(m["est"]),
                         mb(m.get("rss")) if m["loaded"] else "-", tok(m["ctx"]), f"{b}/{len(sl) or m['par']}", m["inflight"], rate(tg),
                         rate(ms_.get("pp")), ms(ms_.get("ttft")), fmt_t(m["run"]) if m.get("run") else "-", f"{m['idle']//60}m" if m["idle"] else "never"])
        def colr(i):
            m = order[i]
            if not m["loaded"]:
                return curses.A_DIM
            return s.C(3, curses.A_BOLD) if (m["inflight"] or any(x.get("is_processing") for x in m.get("slots", []))) else s.C(2)
        y = s.table(y + 1, cols, rows, W, H - y - 4, colr)
        y += 2
        # slots / active chats
        avail = H - y - 1
        if avail < 4 or not loaded:
            return
        rows, cl = [], []
        for m in loaded:
            tl = m.get("tail")
            for sl in m.get("slots", []):
                sid = sl["id"]
                nt = (sl.get("next_token") or [{}])[0]
                live = tl.live.get(sid) if tl else None
                held = tl.slot_ctx.get(sid, 0) if tl else 0
                proc = sl.get("is_processing")
                if proc:
                    ph = live.get("phase") if live else ("prompt" if not nt.get("n_decoded") else "gen")
                    used = (sl.get("n_prompt_tokens") or 0) + (nt.get("n_decoded") or 0)
                    used = max(used, held)
                    state = {"prompt": "prefill", "gen": "decode", "queued": "queued"}.get(ph, ph)
                    prog = f"{live['prog']*100:.0f}%" if live and ph == "prompt" else (f"{live['n']} tok" if live and ph == "gen" else "")
                    spd = rate(live.get("tg")) if live and ph == "gen" else "-"
                    since = ago(live["since"]) if live else "-"
                    task = sl.get("id_task")
                elif held:
                    state, used, prog, spd, since, task = "cached", held, "", "-", ago(tl.slot_last.get(sid, time.time())), sl.get("id_task")
                else:
                    state, used, prog, spd, since, task = "free", 0, "", "-", "-", None
                if state == "free" and len([r for r in rows if r[0] == m["name"]]) >= 1 and m["par"] > 4:
                    continue   # do not list dozens of empty slots
                rows.append([m["name"], sid, state, task if task is not None else "-", prog, tok(used), tok(sl.get("n_ctx")), f"{100*used/sl['n_ctx']:.0f}%" if sl.get("n_ctx") else "-", spd, since])
                cl.append(state)
        s.box(y, 0, min(avail, len(rows) + 3), W, f"slots · active chats ({sum(1 for c in cl if c in ('prefill','decode','queued'))} working, {sum(1 for c in cl if c=='cached')} held)")
        cols = [("MODEL", 22, "<"), ("SLOT", 4, ">"), ("STATE", 8, "<"), ("TASK", 6, ">"), ("PROGRESS", 9, ">"), ("CTX USED", 8, ">"), ("CTX MAX", 8, ">"), ("FILL", 5, ">"), ("TG t/s", 7, ">"), ("SINCE", 6, ">")]
        yy = s.table(y + 1, cols, rows, W, avail - 3, lambda i: s.C(3, curses.A_BOLD) if cl[i] in ("prefill", "decode", "queued") else (s.C(1) if cl[i] == "cached" else curses.A_DIM))
        for i in range(min(len(rows), avail - 3)):
            try:
                pct = float(rows[i][7].rstrip("%"))
            except ValueError:
                pct = 0
            s.bar(yy - min(len(rows), avail - 3) + i, 2 + 22 + 5 + 9 + 7 + 10 + 9 + 9 + 6 + 8 + 7 + 1, 12, pct)

    # ---- page 3: speed
    def page_speed(s, snap, y, H, W):
        models, hist = snap["models"], snap["hist"]
        if snap["gpus"]:
            s.vram_line(y, W, snap["gpus"][0]); y += 1
        loaded = [m for m in models if m["loaded"] and m.get("tail")]
        if not loaded:
            s.box(y, 0, 3, W, "speed")
            s.put(y + 1, 2, "no loaded model with a readable log (is the gateway home right? --home <dir with logs/>)", curses.A_DIM)
            return
        tot_live = sum((s.mstats(m).get("live_tg") or 0) for m in loaded)
        allreq = [(m["name"], r) for m in loaded for r in m["tail"].reqs]
        now = time.time()
        rpm = sum(1 for _, r in allreq if now - r.get("end", 0) < 60)
        gen_tok = sum(s.mstats(m).get("gen_tokens", 0) for m in loaded)
        pp_tok = sum(s.mstats(m).get("prompt_tokens", 0) for m in loaded)
        s.box(y, 0, 3, W, "throughput")
        s.put(y + 1, 2, "live generation ", curses.A_DIM); s.put(y + 1, 18, f"{tot_live:6.1f} tok/s", s.C(2, curses.A_BOLD))
        s.put(y + 1, 38, "requests/min ", curses.A_DIM); s.put(y + 1, 51, str(rpm), curses.A_BOLD)
        s.put(y + 1, 58, "logged requests ", curses.A_DIM); s.put(y + 1, 74, str(len(allreq)), curses.A_BOLD)
        s.put(y + 1, 82, "tokens: prompt ", curses.A_DIM); s.put(y + 1, 97, tok(pp_tok), curses.A_BOLD)
        s.put(y + 1, 106, "generated ", curses.A_DIM); s.put(y + 1, 116, tok(gen_tok), curses.A_BOLD)
        y += 3
        # per-model panels
        ph = 4
        sw = max(10, W - 66)
        s.box(y, 0, len(loaded) * 1 + 3 + 0, W, "per model (last 2 min of finished requests)")
        cols = [("MODEL", 22, "<"), ("NOW tg", 7, ">"), ("avg tg", 7, ">"), ("avg pp", 8, ">"), ("TTFT~", 7, ">"), ("REQS", 5, ">"), ("LAST p/g tok", 13, ">"), ("tg history", sw, "<")]
        rows = []
        stats = []
        for m in loaded:
            st = s.mstats(m)
            stats.append(st)
            last = st.get("last")
            rows.append([m["name"], rate(st.get("live_tg")), rate(st.get("tg")), rate(st.get("pp")), ms(st.get("ttft")), st.get("n", 0),
                         f"{last['pe_n']}/{last['ev_n']}" if last else "-", ""])
        yy = s.table(y + 1, cols, rows, W, H - y - 4)
        for i, m in enumerate(loaded):
            series = [r["tg"] for r in m["tail"].reqs if r.get("tg")][-sw:]
            s.spark(y + 2 + i, 2 + sum(c[1] + 1 for c in cols[:-1]), sw, series, None, 2)
        y += len(loaded) + 3
        # recent requests
        avail = H - y - 1
        if avail < 4:
            return
        recent = sorted(allreq, key=lambda x: x[1].get("end", 0), reverse=True)
        s.box(y, 0, min(avail, len(recent) + 3), W, "recent requests (newest first) · TTFT~ = prompt eval time, excludes queueing")
        cols = [("AGO", 5, ">"), ("MODEL", 22, "<"), ("SLOT", 4, ">"), ("TASK", 6, ">"), ("PROMPT", 7, ">"), ("GEN", 6, ">"), ("TTFT~", 8, ">"), ("PP t/s", 8, ">"), ("TG t/s", 8, ">"), ("TOTAL", 8, ">"), ("CTX END", 8, ">")]
        rows = []
        for n, r in recent:
            rows.append([ago(r.get("end", 0)), n, r["slot"], r["task"], tok(r["pe_n"]), tok(r["ev_n"]), ms(r["pe_ms"]), rate(r["pp"]), rate(r["tg"]), ms(r["tot_ms"]), tok(r["tot_n"])])
        s.table(y + 1, cols, rows, W, avail - 3, lambda i: s.C(2) if i < 3 else 0)

    def draw(s, snap):
        s.scr.erase()
        H, W = s.scr.getmaxyx()
        if W < 80 or H < 16:
            s.put(0, 0, "terminal too small (need 80x16)")
            return
        s.tabs(W, snap)
        try:
            if not snap.get("gpus") and s.page == 0:
                s.put(2, 2, f"nvidia-smi failed: {snap.get('gpu_err', '')}", s.C(4))
            elif s.page == 0:
                s.page_gpu(snap, 1, H, W)
            elif s.page == 1:
                s.page_gateway(snap, 1, H, W)
            else:
                s.page_speed(snap, 1, H, W)
        except Exception as e:
            s.put(2, 2, f"draw error: {type(e).__name__}: {e}", s.C(4))
        s.footer(H, W)

    def run(s):
        s.scr.timeout(150)
        while True:
            with s.col.lock:
                snap = s.col.snap
            if snap:
                s.draw(snap)
            else:
                s.scr.erase(); s.put(0, 0, "collecting...");
            s.scr.refresh()
            k = s.scr.getch()
            if k in (ord("q"), 27):
                return
            elif k in (ord("1"), ord("2"), ord("3")): s.page = k - ord("1")
            elif k == 9: s.page = (s.page + 1) % len(PAGES)
            elif k == curses.KEY_BTAB: s.page = (s.page - 1) % len(PAGES)
            elif k == ord("m"): s.sort = "mem"
            elif k == ord("p"): s.sort = "pid"
            elif k == ord("n"): s.sort = "run"
            elif k == ord("r"): s.rev = not s.rev
            elif k == ord("+"): s.col.dt = max(0.25, s.col.dt / 2)
            elif k == ord("-"): s.col.dt = min(8, s.col.dt * 2)
            elif k == ord(" "): s.col.paused = not s.col.paused
            elif k == curses.KEY_RESIZE: s.scr.clear()


def main():
    global GW_URL, GW_HOME
    a = sys.argv[1:]
    for i, x in enumerate(a):
        if x == "--gateway" and i + 1 < len(a):
            GW_URL = a[i + 1].rstrip("/")
        elif x == "--home" and i + 1 < len(a):
            GW_HOME = os.path.expanduser(a[i + 1])
    if "--dump" in a:   # one collection pass, printed: for checking without a terminal
        c = Collector()
        gpus = query_gpus()
        ports = c.listeners()
        gw = c.gateway(ports)
        for m in c.models(gw, ports, gpus):
            tl = m.pop("tail", None)
            print(m["name"], m["loaded"], m.get("pid"), m.get("vram"), m.get("rss"), "slots", len(m.get("slots", [])),
                  "reqs", len(tl.reqs) if tl else None, UI.mstats(dict(tail=tl)) if tl else "")
        print("home", gw.get("home"), "peers", dict(gw["peers"]), "agents", [(x["pid"], x["cmd"][:60]) for x in c.agents()], c.containers())
        return
    col = Collector()
    col.start()
    curses.wrapper(lambda scr: UI(scr, col).run())


if __name__ == "__main__":
    main()
