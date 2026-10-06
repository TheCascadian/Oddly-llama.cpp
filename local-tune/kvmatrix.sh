#!/usr/bin/env bash
# KV-cache matrix on R1-distill-7B Q4_K_S: type x depth x kv-offload.
# usage: [KV_REPS=n] local-tune/kvmatrix.sh <build-dir>   (run from repo root)
# live view: tail -f local-tune/results/kvmatrix.log ; data: kvmatrix.csv
# The log also gets one "TEST pp|tg@depth took=Ns" line per result and "took=" on every DONE line.
B=${1:-build-base}; R=local-tune/results
M=$(ls ~/.lmstudio/models/mradermacher/DeepSeek-R1-Distill-Qwen-7B-Uncensored-i1-GGUF/*Q4_K_S.gguf)
export PATH=/opt/cuda/bin:$PATH
# Copies result lines to the csv and logs how long each one took (model load and context fill included).
TIMER='BEGIN{t=systime()} /^"/{print p $0 >> csv; fflush(csv); n=systime(); a=$(NF-7); d=$(NF-5); gsub(/"/,"",a); gsub(/"/,"",d); printf "%s   TEST %s@%s took=%ds\n", strftime("%T"), (a>0?"pp":"tg"), d, n-t >> lf; fflush(lf); t=n}'
T0=$(date +%s)
: > $R/kvmatrix.log; : > $R/kvmatrix.csv
for kv in f16 q8_0 q4_0; do for nk in 0 1; do
  echo "$(date +%T) START ctk=ctv=$kv nkvo=$nk" | tee -a $R/kvmatrix.log; T1=$(date +%s)
  ( while sleep 2; do echo "$(date +%T)   vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) ram_avail=$(free -m | awk '/Mem/{print $7}')MB" >> $R/kvmatrix.log; done ) & S=$!
  $B/bin/llama-bench -m "$M" -ngl 99 -t 6 -fa 1 -ctk $kv -ctv $kv -nkvo $nk \
    -d 0,8192,16384 -p 512 -n 32 -r ${KV_REPS:-1} -o csv 2>&1 \
    | tee -a $R/kvmatrix.log | awk -F, -v p="$kv,nkvo$nk," -v csv=$R/kvmatrix.csv -v lf=$R/kvmatrix.log "$TIMER"
  kill $S
  echo "$(date +%T) DONE $kv nkvo=$nk took=$(( $(date +%s) - T1 ))s" | tee -a $R/kvmatrix.log
done; done
echo "ALLDONE total=$(( $(date +%s) - T0 ))s" >> $R/kvmatrix.log
