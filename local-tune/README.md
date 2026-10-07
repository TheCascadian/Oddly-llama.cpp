# local-tune

Tuning a llama.cpp fork for a Ryzen 5 7600X + GTX 1660 Ti (6 GB, Turing, no tensor cores). Three kept code changes (GTX 16xx kernel selection, qwen35 GDN CPU path, q8_0 K / q4_0 V pair), a GPU overclock and gateway settings. Prompt reading is 2.9x to 4.2x faster, 9B code edits 7.0x with ngram speculation, and the 9B holds 81,920 tokens of context at q4/q4 `-ub 128`.

Everything measured is in **[UNIFIED.md](UNIFIED.md)** (page version: [UNIFIED.html](UNIFIED.html)). Commit timeline: [HISTORY.md](HISTORY.md).

## Run it yourself

```
cp local-tune/config/models.example.conf local-tune/config/models.conf   # list your GGUF files
python3 local-tune/scripts/gateway.py &                 # serves them on :8700
python3 local-tune/scripts/watch.py                     # live view, second terminal
python3 local-tune/scripts/suite.py run base            # speed, served speed, job checks
python3 local-tune/scripts/suite.py run next
python3 local-tune/scripts/assess.py next base          # writes archive/ASSESSMENT.md and img/*.svg
python3 local-tune/scripts/ctxprobe.py run <plan>       # context ceiling
```

## Directory map

| Path | Contents |
|---|---|
| `scripts/` | all code: gateway, suite, assess, compare, ppl, spectest, lab, ctxprobe, gpu-push, paths.py |
| `config/` | `assess.conf`, `suites.conf`, `models.example.conf` (`models.conf` is git-ignored) |
| `results/<family>/` | throughput, kv, speculative, quality, profiling, context, hardware, suites, lab |
| `prof/` | CUPTI tracer, plans, analysis |
| `img/` | generated charts |
| `archive/` | superseded documents, each with a header |
| `oc-kit/`, `oc-kit.*` | standalone overclock kit for another machine |
| `odysseus/` | Odysseus agent-loop patch |
| `gpu-push.py` | shim for `gpu-oc.service` |
