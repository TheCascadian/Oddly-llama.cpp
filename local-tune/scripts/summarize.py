#!/usr/bin/env python3
# Summarize header-less llama-bench CSVs written by bench.sh
import csv,sys,os
for f in sys.argv[1:]:
    print('==',os.path.basename(f))
    for r in csv.reader(open(f)):
        pp,tg=int(r[-8]),int(r[-7])
        print(f"{os.path.basename(r[5])[:26]:26} ngl={r[17]:>3} t={r[11]:>2} {'pp%d'%pp if pp else 'tg%d'%tg:>6} {float(r[-2]):8.1f} t/s")
