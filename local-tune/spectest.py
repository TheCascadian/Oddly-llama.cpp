#!/usr/bin/env python3
"""Speculative decoding A/B on llama-server (llama-bench cannot test it).
usage: local-tune/spectest.py <build-dir> <label> <model.gguf> <server args...>   (run from repo root)
live view: local-tune/watch.py (newest run) or local-tune/watch.py <label>
Runs the server once per variant, sends each prompt REPS times at temperature 0 with thinking off and writes results/spec-<label>.csv.
Prompts: "rewrite" repeats a source file with one rename (much repeated text), "free" is an open question (little repeated text),
"edit" adds a parameter to one function and "refactor" changes every function (both print the full file again).
"complete" is the first half of the file as a raw prompt for a completion model.
Optional results/spec-<label>.plan: "title: text", "prompts: names", "mode: completion" (raw prompt, no chat template),
"parallel: N" (N requests at once, speed is all tokens over wall time), then one variant per line: "name | ENV=1 ... server args". The first variant is the baseline."""
import csv, hashlib, json, os, re, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORT, REPS, N_PREDICT = 8299, 3, 400
VARIANTS = [("none", []), ("ngram-simple", ["--spec-type", "ngram-simple"]), ("ngram-mod", ["--spec-type", "ngram-mod"])]
CODE = "".join(open(os.path.join(HERE, "watch.py")).readlines()[4:45])
PROMPTS = {
    "rewrite": "Print this Python code again in full. Change only one thing: rename the function `read` to `read_lines`. Output only the code.\n\n```python\n" + CODE + "```",
    "free": "Explain in about 300 words how a hash table handles collisions.",
    "edit": "Change the function `gpu` so that it takes a parameter `timeout` with default 3 and passes it to subprocess.run. Print the full updated code, nothing else.\n\n```python\n" + CODE + "```",
    "complete": "".join(CODE.splitlines(True)[:20]),
    "refactor": "Add type hints and a one-line docstring to every function in this code. Print the full updated code, nothing else.\n\n```python\n" + CODE + "```",
}


def plan(label):
    """(title, prompt names, variants, options) from results/spec-<label>.plan; without a plan the three built-in variants on rewrite and free."""
    title, prompts, variants, opts = "", ["rewrite", "free"], [], {"mode": "chat", "parallel": "1"}
    try:
        lines = open(os.path.join(HERE, "results", f"spec-{label}.plan")).read().splitlines()
    except OSError:
        return title, prompts, VARIANTS, opts
    for ln in lines:
        key, _, val = ln.partition(":")
        if "|" in ln:
            name, args = ln.split("|", 1)
            variants.append((name.strip(), args.split()))
        elif key == "title":
            title = val.strip()
        elif key == "prompts":
            prompts = val.split()
        elif key in opts:
            opts[key] = val.strip()
    return title, prompts, variants, opts


def ask(prompt, chat):
    """One request; returns (timings, text)."""
    if not chat:
        r = post("/completion", {"prompt": prompt, "temperature": 0, "seed": 1, "n_predict": N_PREDICT, "cache_prompt": False})
        return r["timings"], r["content"]
    r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "seed": 1, "max_tokens": N_PREDICT, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}})
    msg = r["choices"][0]["message"]
    return r["timings"], (msg.get("reasoning_content") or "") + (msg.get("content") or "")


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def main():
    build, label, model, *args = sys.argv[1:]
    f = open(os.path.join(HERE, "results", f"spec-{label}.csv"), "w")
    out = csv.writer(f)
    out.writerow(["variant", "prompt", "rep", "n_predicted", "tg_tps", "draft_n", "draft_accepted", "text_md5"])
    _, prompts, variants, opts = plan(label)
    par = int(opts["parallel"])
    for name, extra in variants:
        env = dict(x.split("=", 1) for x in extra if re.match(r"[A-Z0-9_]+=", x))
        extra = [x for x in extra if not re.match(r"[A-Z0-9_]+=", x)]
        cmd = [f"{ROOT}/{build}/bin/llama-server", "-m", model, "--port", str(PORT), "--parallel", str(par)] + args + extra
        log = open(os.path.join(HERE, "results", f"spec-{label}-{name.replace(' ', '_')}.log"), "w")
        srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PATH="/opt/cuda/bin:" + os.environ["PATH"], **env))
        try:
            for _ in range(120):
                time.sleep(1)
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                    break
                except OSError:
                    if srv.poll() is not None:
                        raise SystemExit(f"server stopped for variant {name}, see {log.name}")
            for pname in prompts:
                prompt = PROMPTS[pname]
                for rep in range(REPS):
                    t0 = time.time()
                    with ThreadPoolExecutor(par) as ex:
                        res = list(ex.map(lambda _: ask(prompt, opts["mode"] == "chat"), range(par)))
                    n = sum(t["predicted_n"] for t, _ in res)
                    tps = res[0][0]["predicted_per_second"] if par == 1 else n / (time.time() - t0)
                    text = "".join(x for _, x in res)
                    row = [name, pname, rep, n, f"{tps:.2f}", sum(t.get("draft_n", 0) for t, _ in res), sum(t.get("draft_n_accepted", 0) for t, _ in res), hashlib.md5(text.encode()).hexdigest()[:8]]
                    out.writerow(row); f.flush(); print(*row, flush=True)
        finally:
            srv.terminate(); srv.wait()


def render(label):
    """Text view of results/spec-<label>.csv and whether the run is complete."""
    import watch
    from watch import G, Y, RED, D, B, X
    rows = list(csv.DictReader(watch.read(f"spec-{label}.csv")))
    title, prompts, variants, opts = plan(label)
    base_name = variants[0][0]
    total = len(variants) * len(prompts) * REPS
    L = [f"{B}  SPECULATIVE DECODING TEST{X}  {D}{label}  {title}  ·  llama-server, temperature 0, {N_PREDICT} tokens per answer, {REPS} repetitions" + (f", {opts['parallel']} requests at once" if opts["parallel"] != "1" else "") + f"{X}", ""]
    L += [f"  Progress  {watch.bar(len(rows) / total, 30)} {B}{len(rows)}/{total}{X}  " + (f"{G}✔ ALL DONE{X}" if len(rows) >= total else f"{Y}● running{X}"), ""]
    L += watch.gpu_panel() + [""]
    L += [f"  {B}RESULTS{X}  {D}writing speed in tokens/sec, higher is better. rewrite = repeat a file with one rename, edit = change one function, refactor = change every function, free = open question{X}", ""]
    hdr = f"  {'Variant':<26}{'Prompt':<9}|{'t/s':>9}{'vs base':>9}{'drafted':>9}{'accepted':>10} | Same text as {base_name}"
    L += [f"{B}{hdr}{X}", " " + "─" * (len(hdr) - 1)]
    def pick(v, p):
        return [r for r in rows if r["variant"] == v and r["prompt"] == p]
    for v, _ in variants:
        for p in prompts:
            r, base = pick(v, p), pick(base_name, p)
            if not r:
                L.append(f"  {v:<26}{p:<9}|{D}{'.':>9}{X}")
                continue
            tps = sum(float(x["tg_tps"]) for x in r) / len(r)
            b = sum(float(x["tg_tps"]) for x in base) / len(base) if base else 0
            dn, da = sum(int(x["draft_n"]) for x in r), sum(int(x["draft_accepted"]) for x in r)
            same = {x["text_md5"] for x in r} <= {x["text_md5"] for x in base}
            L.append(f"  {v:<26}{p:<9}|{G}{tps:9.1f}{X}{(f'{tps / b:.2f}x' if b else '-'):>9}{dn // len(r):>9}{(f'{100 * da // dn}%' if dn else '-'):>10} | " + (f"{G}yes{X}" if same else f"{RED}no{X}"))
    L += ["", f"  {D}. = not run yet   Ctrl+C closes this view; the test keeps running.{X}"]
    return "\n".join(L), len(rows) >= total


if __name__ == "__main__":
    main()
