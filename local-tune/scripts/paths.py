"""Where things live under local-tune/. Every script asks here, so the layout is in one place.

scripts/   code (this folder)         config/   assess.conf, suites.conf, models.example.conf (models.conf is per machine, ignored)
results/   measurements, one sub-folder per family (see family())   img/  generated figures   archive/  superseded documents
"""
import os, re

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.dirname(SCRIPTS)            # local-tune/
ROOT = os.path.dirname(LOCAL)               # repository root
CONFIG = os.path.join(LOCAL, "config")
RESULTS = os.path.join(LOCAL, "results")
IMG = os.path.join(LOCAL, "img")
ARCHIVE = os.path.join(LOCAL, "archive")

# files that stay directly in results/: shared inputs and run state, not measurements
LOOSE = {"ppl-corpus.txt", "lab-state.json", "assess.json", "sequence.txt"}

# first match wins; the name is the file (or folder) name inside results/
FAMILIES = [
    (r"prof-|compare-prof-|compare-[es]\d-|s2-7b-", "profiling"),
    (r"ctx-", "context"),
    (r"spec-", "speculative"),
    (r"ppl-", "quality"),
    (r"kvmatrix-|compare-kvmix-", "kv"),
    (r"gpu|full-", "hardware"),
    (r"suite-", "suites"),
    (r"lab-", "lab"),
]
DEFAULT_FAMILY = "throughput"


def family(name):
    """Sub-folder of results/ for a file name; '' for the loose files."""
    if name in LOOSE:
        return ""
    for pat, fam in FAMILIES:
        if re.match(pat, name):
            return fam
    return DEFAULT_FAMILY


def rp(name):
    """Path of a result file by name. The folder is created when it does not exist, so this works for writing too."""
    d = os.path.join(RESULTS, family(name))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def rfiles():
    """Every file name in results/ (all families), for views that scan the folder."""
    out = []
    for _, _, files in os.walk(RESULTS):
        out += files
    return out


if __name__ == "__main__":   # for shell scripts: python3 paths.py <result file name>  prints its path
    import sys
    print(rp(sys.argv[1]))
