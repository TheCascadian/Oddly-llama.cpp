# Commit timeline

Split out of [UNIFIED.md](UNIFIED.md) §13. Times are local, 2026. Newest first.

| Commit | When | Subject |
|---|---|---|
| `153501659` | 10-07 04:33 | local-tune: archive superseded docs with headers |
| `77877ccf9` | 10-07 04:33 | local-tune: keep gpu-push.py shim for gpu-oc.service |
| `a748ab534` | 10-07 | gitignore: follow config/ and results families |
| `bb2d01bdc` | 10-07 04:33 | local-tune: resolve result files by family through scripts/paths.py and fix script paths |
| `648f6c3f7` | 10-07 04:31 | local-tune: group results by family (throughput, kv, speculative, quality, profiling, context, hardware, suites, lab) |
| `d1f4e46c9` | 10-07 04:31 | local-tune: move scripts to scripts/ and config files to config/ (paths fixed in the next commits) |
| `6316d717c` | 10-07 04:28 | local-tune: snapshot before consolidation - discovery report, profiling tools, context probe, round 1-2 results |
| `d954ac6f4` | 10-06 17:40 | Edit .gitignore to clean repo |
| `b47645cdd` | 10-06 17:33 | README: rename to Wrekt-llama.cpp, link llama.cpp and Oddly-llama.cpp, add the model and guard results, fold the upstream README |
| `f1e7c17df` | 10-06 17:23 | local-tune: one-token guard verdict and the backend plan with its trials (none kept) |
| `481d872d6` | 10-06 17:23 | local-tune: gateway in the repo, end-to-end suite with job checks, and an assessment report for the added models |
| `d0ab9a678` | 10-06 14:18 | local-tune: GPU overclock sweep (+16% decode), results report, and a branch README for local-1660ti |
| `f7c1ca43d` | 10-06 10:43 | local-tune: results page leads with a plain-language summary, technical detail collapsed |
| `45d2e13e9` | 10-06 10:36 | local-tune: document the q8_0 K / q4_0 V pair as change 3 and the 7B gateway setting |
| `bafd6a875` | 10-06 10:25 | local-tune: KV precision trials, perplexity runner, and lab.py to run the suites on source changes |
| `afbf9a20c` | 10-06 10:25 | CUDA: build the q8_0 K / q4_0 V attention pair by default |
| `328ff74bd` | 10-06 05:46 | local-tune: close out the backlog leads - ngram-simple on the 7B with long answers, draft types, blocked items |
| `e42c95cda` | 10-06 05:29 | local-tune: ledger.py builds the results page with a decision ledger and trial charts |
| `6b3297755` | 10-06 05:20 | local-tune: backlog trials - GPU layer counts, ngram-simple on code edits, draft model, switches with no gain |
| `cf14ecd26` | 10-05 23:24 | local-tune: record the trial plan, the baseline correction and the untested backlog |
| `f40988912` | 10-05 23:01 | local-tune: document the GTX 16xx kernel selection change |
| `efb8f377e` | 10-05 22:51 | local-tune: KV matrix and kernel comparison scripts with live views, timing logs and results |
| `a260d811e` | 10-05 22:51 | cuda: use non-tensor-core kernels on Turing GPUs without tensor cores |
| `155d646a0` | 10-05 19:34 | local-tune: build/bench scripts and results for Ryzen 7600X + GTX 1660 Ti |
| `52b730eb4` | 10-05 19:34 | qwen35: skip fused raw-gate GDN path on CPU-resident layers on x86 |

The commit `d1f4e46c9` leaves script paths broken on purpose; `bb2d01bdc` and later fix them. Consolidation commits for UNIFIED.md, HISTORY.md, UNIFIED.html and README follow this table in `git log`.
