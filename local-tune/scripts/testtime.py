#!/usr/bin/env python3
"""Total test time per model. testtime.py [-k]
Walks every results/**/<kind>-<name>.plan, finds the model each plan line runs (-m <dir>/<file>.gguf),
and adds up the "DONE <label> took=<n>s" lines of the matching .log. -k also splits the totals by test kind.
Plans without a log, and runs still in progress (no ALLDONE), are counted as observed so far and flagged."""
import collections, glob, os, re, sys
from paths import ROOT

RES = os.path.join(ROOT, "local-tune", "results")
DONE = re.compile(r"^\d\d:\d\d:\d\d DONE (.+?) took=(\d+)s")


def model_of(line):
    m = re.search(r"-m\s+\S*/([^/\s]+)/([^/\s]+)\.gguf", line)
    if not m:
        return None
    return m[2] if m[1].lower() == "models" else m[1]


def plan_labels(plan):
    """label -> model for each plan line; a bare line (no -m) falls back to the model named in the title/args."""
    labels, default = {}, None
    for ln in open(plan, errors="replace").read().splitlines():
        if ln.startswith(("title:", "args:", "prompts:")) or not ln.strip():
            default = default or model_of(ln)
            continue
        labels[ln.split("|")[0].strip()] = model_of(ln) or default
    return labels


def main():
    by_kind = "-k" in sys.argv
    tot, flag = collections.defaultdict(lambda: collections.defaultdict(int)), set()
    for plan in sorted(glob.glob(os.path.join(RES, "**", "*.plan"), recursive=True)):
        kind = os.path.basename(plan).split("-")[0]
        log = plan[:-5] + ".log"
        if not os.path.exists(log):
            continue
        lines = open(log, errors="replace").read().splitlines()
        labels = plan_labels(plan)
        if not any("ALLDONE" in l for l in lines):
            flag.add(os.path.basename(plan))
        for ln in lines:
            m = DONE.match(ln)
            if m:
                model = labels.get(m[1].strip()) or next((v for k, v in labels.items() if k and m[1].startswith(k)), None) or "(unknown)"
                tot[model][kind] += int(m[2])
    h = lambda s: f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"
    for model, kinds in sorted(tot.items(), key=lambda x: -sum(x[1].values())):
        print(f"{model:<28}{h(sum(kinds.values())):>12}")
        if by_kind:
            for k, s in sorted(kinds.items(), key=lambda x: -x[1]):
                print(f"    {k:<24}{h(s):>12}")
    if flag:
        print("still running or unfinished (partial):", ", ".join(sorted(flag)))


if __name__ == "__main__":
    main()
